#!/usr/bin/env python3
"""
Bash Script Generator for Model Evaluation Pipeline

This script generates standardized bash evaluation scripts for the ReMIX project.
It supports both MIMIC and CheXpert+ datasets with BioViL and GLORIA models.

Author: ReMIX Project
Date: 2025-11-03

Usage:
    python generate_evaluation_script.py --config config.json
    python generate_evaluation_script.py --interactive

The script generates bash files that orchestrate the evaluation pipeline:
    1. Generate embeddings (general_*_embedding_library.py)
    2. Calculate cosine similarities (general_*_cosine_similarity.py)
    3. Compute accuracy metrics (general_*_accuracy.py)
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional


class EvaluationScriptGenerator:
    """Generates bash evaluation scripts for model testing."""

    def __init__(self):
        self.template = self._get_template()

    def generate_script(
        self,
        dataset: str,
        model_type: str,
        models: List[str],
        base_dir: str,
        labels_path: str,
        script_name: Optional[str] = None,
        split_type: str = "test",
        num_iterations: int = 1000,
        skip_embeddings: bool = False,
        label_type: Optional[str] = None,
        output_suffix: Optional[str] = None,
        additional_flags: Optional[List[str]] = None,
        frontal_impression_only: bool = True,
    ) -> str:
        """
        Generate a bash evaluation script.

        Args:
            dataset: Dataset name ('mimic' or 'chexpertplus')
            model_type: Model type ('biovil' or 'gloria')
            models: List of model names to evaluate
            base_dir: Base output directory for results
            labels_path: Path to labels file
            script_name: Optional custom script name
            split_type: Data split ('test', 'validate', 'train')
            num_iterations: Number of resampling iterations
            skip_embeddings: Whether to skip embedding generation
            label_type: Label type for MIMIC (e.g., 'convirt')
            output_suffix: Suffix for output subdirectory
            additional_flags: Additional flags for embedding script
            frontal_impression_only: Whether to use frontal impressions only

        Returns:
            Generated bash script content
        """
        dataset = dataset.lower()
        model_type = model_type.lower()

        # Validate inputs
        if dataset not in ["mimic", "chexpertplus"]:
            raise ValueError(
                f"Invalid dataset: {dataset}. Must be 'mimic' or 'chexpertplus'"
            )
        if model_type not in ["biovil", "gloria"]:
            raise ValueError(
                f"Invalid model_type: {model_type}. Must be 'biovil' or 'gloria'"
            )

        # Generate script name if not provided
        if script_name is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            script_name = f"{timestamp}_{model_type}_{dataset}_evaluation.sh"

        # Prepare template variables
        template_vars = {
            "dataset": dataset,
            "dataset_upper": dataset.upper() if dataset == "mimic" else "CheXpert+",
            "model_type": model_type,
            "model_type_upper": model_type.upper(),
            "models": models,
            "base_dir": base_dir,
            "labels_path": labels_path,
            "split_type": split_type,
            "num_iterations": num_iterations,
            "skip_embeddings": skip_embeddings,
            "label_type": label_type,
            "output_suffix": output_suffix,
            "has_label_type": label_type is not None,
            "has_output_suffix": output_suffix is not None,
            "frontal_impression_only": frontal_impression_only,
            "additional_flags": additional_flags or [],
            "eval_scripts_dir": "/opt/gpudata/imadejski/search-model/remix/evaluation",
            "model_checkpoints_dir": "/opt/gpudata/remix",
        }

        return self._render_template(template_vars)

    def _render_template(self, vars: Dict) -> str:
        """Render the bash script template with provided variables."""

        # Build model array
        models_str = "\n".join([f'\t"{model}"' for model in vars["models"]])

        # Build embedding script section
        if vars["skip_embeddings"]:
            embedding_check = """
\t# Always reuse existing embeddings
\tif [ -f "$output_file" ]; then
\t\tlog "Reusing existing embeddings for $model_name: $output_file"
\t\techo "$output_file"
\t\treturn 0
\tfi

\tif [ "$SKIP_EMBEDDINGS" = true ]; then
\t\tlog "✗ Embedding file not found for $model_name and SKIP_EMBEDDINGS=true: $output_file"
\t\treturn 1
\tfi"""
        else:
            embedding_check = """
\t# Check if embeddings already exist
\tif [ -f "$output_file" ]; then
\t\tlog "Reusing existing embeddings for $model_name: $output_file"
\t\techo "$output_file"
\t\treturn 0
\tfi"""

        # Build embedding flags
        embedding_flags = []
        if vars["frontal_impression_only"]:
            embedding_flags.append("--frontal_impression_only")
        embedding_flags.extend(vars["additional_flags"])
        embedding_flags_str = (
            " \\\n\t\t".join(embedding_flags) if embedding_flags else ""
        )
        if embedding_flags_str:
            embedding_flags_str = " \\\n\t\t" + embedding_flags_str

        # Build cosine similarity section
        if vars["has_label_type"]:
            cosine_output_suffix = f"_${{LABEL_TYPE}}"
            if vars.get("output_suffix"):
                cosine_output_suffix += f"_{vars['output_suffix']}"
            cosine_label_flag = '\t\t--label_type "$LABEL_TYPE"'
        else:
            cosine_output_suffix = ""
            cosine_label_flag = ""

        # Build accuracy section
        if vars["has_label_type"]:
            accuracy_output_suffix = f"_${{LABEL_TYPE}}"
            if vars.get("output_suffix"):
                accuracy_output_suffix += f"_{vars['output_suffix']}"
            accuracy_label_flag = '\t\t--label-type "$LABEL_TYPE"'
        else:
            accuracy_output_suffix = ""
            accuracy_label_flag = ""

        # Build directory structure
        if vars["has_output_suffix"]:
            dir_structure = f'''
\t# Create model directories
\tlocal model_dir="${{BASE_DIR}}/${{model_name}}"              # existing embeddings live here
\tlocal model_dir_{vars['output_suffix']}="${{OUT_BASE_DIR}}/${{model_name}}"  # {vars['output_suffix']} outputs go here
\tmkdir -p "$model_dir"
\tmkdir -p "$model_dir_{vars['output_suffix']}"
\tlog "Created/verified directories: $model_dir (embeddings), $model_dir_{vars['output_suffix']} ({vars['output_suffix']} outputs)"'''
            embedding_dir = "$model_dir"
            output_dir = f'$model_dir_{vars["output_suffix"]}'
        else:
            dir_structure = '''
\t# Create model directory
\tlocal model_dir="${BASE_DIR}/${model_name}"
\tmkdir -p "$model_dir"
\tlog "Created/verified directory: $model_dir"'''
            embedding_dir = "$model_dir"
            output_dir = "$model_dir"

        # Build configuration section
        config_lines = [
            'BASE_DIR="' + vars["base_dir"] + '"',
            'EVAL_SCRIPTS_DIR="' + vars["eval_scripts_dir"] + '"',
            'MODEL_CHECKPOINTS_DIR="' + vars["model_checkpoints_dir"] + '"',
            'LABELS_PATH="' + vars["labels_path"] + '"',
            'SPLIT_TYPE="' + vars["split_type"] + '"',
        ]

        if vars["has_label_type"]:
            config_lines.append(f'LABEL_TYPE="{vars["label_type"]}"')

        config_lines.append(f'NUM_ITERATIONS={vars["num_iterations"]}')

        if vars["skip_embeddings"]:
            config_lines.append(
                "# Reuse existing embeddings; do not rerun the embedding script"
            )
            config_lines.append("SKIP_EMBEDDINGS=true")

        if vars["has_output_suffix"]:
            config_lines.append(
                f'# Output under a separate subdirectory to mark {vars["output_suffix"]}'
            )
            config_lines.append(f'OUT_SUFFIX="{vars["output_suffix"]}"')
            config_lines.append('OUT_BASE_DIR="${BASE_DIR}/${OUT_SUFFIX}"')

        config_str = "\n".join(config_lines)

        # Build main log section
        main_logs = [
            f'\tlog "Starting {vars["model_type_upper"]} model evaluation process on {vars["dataset_upper"]} data..."',
        ]

        if vars["has_output_suffix"]:
            main_logs.extend(
                [
                    '\tlog "Embeddings base directory (reused): $BASE_DIR"',
                    f'\tlog "Output base directory ({vars["output_suffix"]}): $OUT_BASE_DIR"',
                ]
            )
        else:
            main_logs.append('\tlog "Base directory: $BASE_DIR"')

        main_logs.extend(
            [
                '\tlog "Evaluation scripts directory: $EVAL_SCRIPTS_DIR"',
                '\tlog "Model checkpoints directory: $MODEL_CHECKPOINTS_DIR"',
                '\tlog "Labels file: $LABELS_PATH"',
                '\tlog "Split type: $SPLIT_TYPE"',
            ]
        )

        if vars["has_label_type"]:
            main_logs.append('\tlog "Label type: $LABEL_TYPE"')

        main_logs.extend(
            [
                '\tlog "Number of iterations: $NUM_ITERATIONS"',
                f'\tlog "Total {vars["model_type_upper"]} models to process: ${{#MODELS[@]}}"',
            ]
        )

        main_logs_str = "\n".join(main_logs)

        # Build main directory setup
        if vars["has_output_suffix"]:
            main_dirs = '''
\t# Ensure base directories exist
\tmkdir -p "$BASE_DIR"
\tmkdir -p "$OUT_BASE_DIR"
\tcd "$OUT_BASE_DIR"'''
        else:
            main_dirs = '''
\t# Ensure base directory exists
\tmkdir -p "$BASE_DIR"
\tcd "$BASE_DIR"'''

        # Build help examples
        help_examples = [
            f'\t$0                                                          # Process all {vars["model_type_upper"]} models with default settings',
            "\t$0 --split-type validate                                    # Use validation split",
            f'\t$0 --model {vars["models"][0] if vars["models"] else "MODEL_NAME"}      # Process only one model',
        ]

        if vars["has_label_type"]:
            help_examples.append(
                f'\t$0 --label-type {vars["label_type"]} --iterations 500                    # Use {vars["label_type"]} labels with 500 iterations'
            )
        else:
            help_examples.append(
                "\t$0 --iterations 500                                               # Use 500 resampling iterations"
            )

        help_examples.append(
            "\t$0 --labels-path /path/to/custom/labels.csv                 # Use custom labels file"
        )

        help_examples_str = "\n".join(['echo "' + ex + '"' for ex in help_examples])

        # Build option parsing
        option_cases = [
            "\t\t--split-type)",
            '\t\t\tSPLIT_TYPE="$2"',
            "\t\t\tshift 2",
            "\t\t\t;;",
        ]

        if vars["has_label_type"]:
            option_cases.extend(
                [
                    "\t\t--label-type)",
                    '\t\t\tLABEL_TYPE="$2"',
                    "\t\t\tshift 2",
                    "\t\t\t;;",
                ]
            )

        option_cases.extend(
            [
                "\t\t--iterations)",
                '\t\t\tNUM_ITERATIONS="$2"',
                "\t\t\tshift 2",
                "\t\t\t;;",
                "\t\t--labels-path)",
                '\t\t\tLABELS_PATH="$2"',
                "\t\t\tshift 2",
                "\t\t\t;;",
                "\t\t--model)",
                '\t\t\tSINGLE_MODEL="$2"',
                "\t\t\tshift 2",
                "\t\t\t;;",
                "\t\t--list-models)",
                f'\t\t\techo "Available {vars["model_type_upper"]} models:"',
                '\t\t\tfor model in "${MODELS[@]}"; do',
                '\t\t\t\techo "  - $model"',
                "\t\t\tdone",
                "\t\t\texit 0",
                "\t\t\t;;",
                "\t\t--help|-h)",
                "\t\t\tshow_help",
                "\t\t\texit 0",
                "\t\t\t;;",
                "\t\t*)",
                '\t\t\techo "Unknown option: $1"',
                "\t\t\tshow_help",
                "\t\t\texit 1",
                "\t\t\t;;",
            ]
        )

        option_cases_str = "\n".join(option_cases)

        # Build help options
        help_options = [
            '\techo "  --split-type SPLIT    Set the data split type (default: '
            + vars["split_type"]
            + ')"',
        ]

        if vars["has_label_type"]:
            help_options.append(
                '\techo "  --label-type TYPE     Set the label type (default: '
                + vars["label_type"]
                + ')"'
            )

        help_options.extend(
            [
                '\techo "  --iterations NUM      Set number of resampling iterations (default: '
                + str(vars["num_iterations"])
                + ')"',
                '\techo "  --labels-path PATH    Set custom labels file path (default: $LABELS_PATH)"',
                '\techo "  --model MODEL_NAME    Process only a specific model"',
                f'\techo "  --list-models         List all available {vars["model_type_upper"]} models"',
                '\techo "  --help               Show this help message"',
            ]
        )

        help_options_str = "\n".join(help_options)

        # Build single model section
        if vars["has_output_suffix"]:
            single_model_dirs = '''
\t\tmkdir -p "$BASE_DIR"
\t\tmkdir -p "$OUT_BASE_DIR"
\t\tcd "$OUT_BASE_DIR"'''
        else:
            single_model_dirs = '''
\t\tmkdir -p "$BASE_DIR"
\t\tcd "$BASE_DIR"'''

        single_model_section = f"""# If single model specified, process only that model
if [ -n "$SINGLE_MODEL" ]; then
\t# Check if model is in the list
\tif [[ " ${{MODELS[@]}} " =~ " ${{SINGLE_MODEL}} " ]]; then
\t\tlog "Processing single model: $SINGLE_MODEL"
{single_model_dirs}
\t\tprocess_model "$SINGLE_MODEL"
\t\texit $?
\telse
\t\tlog "Error: Model '$SINGLE_MODEL' not found in the {vars["model_type_upper"]} model list"
\t\tlog "Use --list-models to see available {vars["model_type_upper"]} models"
\t\texit 1
\tfi
fi"""

        # Generate the complete script
        script = f'''#!/bin/bash

# {vars["model_type_upper"]} model evaluation script for ReMIX project on {vars["dataset_upper"]} data
# This script creates folders for each {vars["model_type_upper"]} model and runs all evaluation scripts:
# 1. general_{vars["dataset"]}_embedding_library.py
# 2. general_{vars["dataset"]}_cosine_similarity.py
# 3. general_{vars["dataset"]}_accuracy.py
#
# Generated by: generate_evaluation_script.py
# Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

set -e  # Exit on any error

# Configuration
{config_str}

# {vars["model_type_upper"]} model configurations array ({vars["dataset_upper"]}-trained models)
declare -a MODELS=(
{models_str}
)

# Function to log with timestamp
log() {{
\techo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}}

# Function to find model checkpoint path
find_model_checkpoint() {{
\tlocal model_name="$1"

\t# Standard path for Hugging Face format models
\tlocal checkpoint_path="${{MODEL_CHECKPOINTS_DIR}}/${{model_name}}"

\t# Check if it's a directory with Hugging Face model files
\tif [ -d "$checkpoint_path" ]; then
\t\t# Check for required Hugging Face model files
\t\tif [ -f "$checkpoint_path/config.json" ] && [ -f "$checkpoint_path/model.safetensors" ]; then
\t\t\techo "$checkpoint_path"
\t\t\treturn
\t\tfi
\tfi

\t# Check if it's a single file (for .ckpt files)
\tif [ -f "${{checkpoint_path}}.ckpt" ]; then
\t\techo "${{checkpoint_path}}.ckpt"
\t\treturn
\tfi

\t# Check alternative locations
\tif [ -f "$checkpoint_path/last.ckpt" ]; then
\t\techo "$checkpoint_path/last.ckpt"
\t\treturn
\tfi

\tif [ -f "$checkpoint_path/checkpoints/last.ckpt" ]; then
\t\techo "$checkpoint_path/checkpoints/last.ckpt"
\t\treturn
\tfi

\tlog "WARNING: Checkpoint not found for $model_name at $checkpoint_path"
\tlog "  Looked for: config.json + model.safetensors (Hugging Face format)"
\tlog "  Also looked for: .ckpt files and alternative locations"
\techo "CHECKPOINT_NOT_FOUND"
}}

# Function to run embedding library script
run_embedding_script() {{
\tlocal model_name="$1"
\tlocal model_checkpoint="$2"
\tlocal output_dir="$3"

\tlocal output_file="${{output_dir}}/${{model_name}}_embeddings_${{SPLIT_TYPE}}.csv"
{embedding_check}

\tlog "Running embedding script for $model_name..."

\tpython "${{EVAL_SCRIPTS_DIR}}/general_{vars["dataset"]}_embedding_library.py" \\
\t\t"$model_checkpoint" \\
\t\t"$output_file" \\
\t\t"$SPLIT_TYPE"{embedding_flags_str}

\tif [ $? -eq 0 ]; then
\t\tlog "✓ Embedding script completed successfully for $model_name"
\t\techo "$output_file"
\t\treturn 0
\telse
\t\tlog "✗ Embedding script failed for $model_name"
\t\treturn 1
\tfi
}}

# Function to run cosine similarity script
run_cosine_script() {{
\tlocal model_name="$1"
\tlocal model_checkpoint="$2"
\tlocal embedding_file="$3"
\tlocal output_dir="$4"

\tlocal output_file="${{output_dir}}/${{model_name}}_cosine_similarity_${{SPLIT_TYPE}}{cosine_output_suffix}.csv"

\t# Check if output already exists
\tif [ -f "$output_file" ]; then
\t\tlog "Skipping cosine similarity script for $model_name: output file already exists"
\t\techo "$output_file"
\t\treturn 0
\tfi

\tlog "Running cosine similarity script for $model_name..."

\tpython "${{EVAL_SCRIPTS_DIR}}/general_{vars["dataset"]}_cosine_similarity.py" \\
\t\t"$model_checkpoint" \\
\t\t"$embedding_file" \\
\t\t"$output_file" \\
\t\t"$SPLIT_TYPE"{' \\' if cosine_label_flag else ''}
{cosine_label_flag}

\tif [ $? -eq 0 ]; then
\t\tlog "✓ Cosine similarity script completed successfully for $model_name"
\t\techo "$output_file"
\t\treturn 0
\telse
\t\tlog "✗ Cosine similarity script failed for $model_name"
\t\treturn 1
\tfi
}}

# Function to run accuracy resampling script
run_accuracy_script() {{
\tlocal model_name="$1"
\tlocal cosine_file="$2"
\tlocal model_dir="$3"

\t# When resampling is enabled, the script creates its own file names
\tlocal output_results="${{model_dir}}/${{model_name}}_accuracy_results_${{SPLIT_TYPE}}_resampling{accuracy_output_suffix}.csv"
\tlocal output_all_results="${{model_dir}}/${{model_name}}_accuracy_results_${{SPLIT_TYPE}}_all_resampling{accuracy_output_suffix}.csv"

\t# Skip if output files already exist (only need to check the main results file)
\tif [ -f "$output_results" ]; then
\t\tlog "Skipping accuracy resampling script for $model_name: output files already exist"
\t\treturn 0
\tfi

\tlog "Running accuracy resampling script for $model_name..."

\t# The resampling script creates consolidated output files
\tpython "${{EVAL_SCRIPTS_DIR}}/general_{vars["dataset"]}_accuracy.py" \\
\t\t--cosine-path "$cosine_file" \\
\t\t--labels-path "$LABELS_PATH" \\
\t\t--output-results-path "${{model_dir}}/${{model_name}}_accuracy_results_${{SPLIT_TYPE}}{accuracy_output_suffix if not accuracy_output_suffix.startswith('_') else accuracy_output_suffix[1:]}.csv" \\
\t\t--split-type "$SPLIT_TYPE" \\{' \\' if accuracy_label_flag else ''}
{accuracy_label_flag}{' \\' if accuracy_label_flag else ''}
\t\t--num-iterations "$NUM_ITERATIONS" \\
\t\t--resampling

\tif [ $? -eq 0 ]; then
\t\tlog "✓ Accuracy resampling script completed successfully for $model_name"
\t\treturn 0
\telse
\t\tlog "✗ Accuracy resampling script failed for $model_name"
\t\treturn 1
\tfi
}}

# Function to run accuracy script (non-resampling)
run_accuracy_no_resample_script() {{
\tlocal model_name="$1"
\tlocal cosine_file="$2"
\tlocal model_dir="$3"

\t# Output file for consolidated non-resampling accuracy script
\tlocal output_results="${{model_dir}}/${{model_name}}_accuracy_results_${{SPLIT_TYPE}}{accuracy_output_suffix}_no_resample.csv"

\t# Skip if output file already exists
\tif [ -f "$output_results" ]; then
\t\tlog "Skipping accuracy (no resample) script for $model_name: output file already exists"
\t\treturn 0
\tfi

\tlog "Running accuracy (no resample) script for $model_name..."

\t# Use the same script but without the --resampling flag
\tpython "${{EVAL_SCRIPTS_DIR}}/general_{vars["dataset"]}_accuracy.py" \\
\t\t--cosine-path "$cosine_file" \\
\t\t--labels-path "$LABELS_PATH" \\
\t\t--output-results-path "$output_results" \\
\t\t--split-type "$SPLIT_TYPE"{' \\' if accuracy_label_flag else ''}
{accuracy_label_flag}

\tif [ $? -eq 0 ]; then
\t\tlog "✓ Accuracy (no resample) script completed successfully for $model_name"
\t\treturn 0
\telse
\t\tlog "✗ Accuracy (no resample) script failed for $model_name"
\t\treturn 1
\tfi
}}

# Function to process a single model
process_model() {{
\tlocal model_name="$1"

\tlog "=================================================="
\tlog "Processing model: $model_name"
\tlog "=================================================="
{dir_structure}

\t# Find model checkpoint
\tlocal model_checkpoint
\tmodel_checkpoint=$(find_model_checkpoint "$model_name")

\tif [ "$model_checkpoint" == "CHECKPOINT_NOT_FOUND" ]; then
\t\tlog "✗ Skipping $model_name: checkpoint not found"
\t\treturn 1
\tfi

\tlog "Using checkpoint: $model_checkpoint"

\t# Step 1: Generate embeddings
\tlocal embedding_file
\tembedding_file=$(run_embedding_script "$model_name" "$model_checkpoint" "{embedding_dir}")
\tif [ $? -ne 0 ]; then
\t\tlog "✗ Failed at embedding step for $model_name"
\t\treturn 1
\tfi

\t# Step 2: Calculate cosine similarities
\tlocal cosine_file
\tcosine_file=$(run_cosine_script "$model_name" "$model_checkpoint" "$embedding_file" "{output_dir}")
\tif [ $? -ne 0 ]; then
\t\tlog "✗ Failed at cosine similarity step for $model_name"
\t\treturn 1
\tfi

\t# Step 3: Calculate accuracy metrics (with resampling)
\tif ! run_accuracy_script "$model_name" "$cosine_file" "{output_dir}"; then
\t\tlog "✗ Failed at accuracy resampling step for $model_name"
\t\treturn 1
\tfi

\t# Step 4: Calculate accuracy metrics (without resampling)
\tif ! run_accuracy_no_resample_script "$model_name" "$cosine_file" "{output_dir}"; then
\t\tlog "✗ Failed at accuracy (no resample) step for $model_name"
\t\treturn 1
\tfi

\tlog "✓ All evaluation steps completed successfully for $model_name"
\treturn 0
}}

# Main execution
main() {{
{main_logs_str}
{main_dirs}

\t# Track results
\tlocal successful_models=()
\tlocal failed_models=()

\t# Process each model
\tfor model in "${{MODELS[@]}}"; do
\t\tif process_model "$model"; then
\t\t\tsuccessful_models+=("$model")
\t\telse
\t\t\tfailed_models+=("$model")
\t\tfi
\t\tlog "--------------------"
\tdone

\t# Summary
\tlog "=================================================="
\tlog "EVALUATION SUMMARY"
\tlog "=================================================="
\tlog "Successfully processed models (${{#successful_models[@]}}):"
\tfor model in "${{successful_models[@]}}"; do
\t\tlog "  ✓ $model"
\tdone

\tif [ ${{#failed_models[@]}} -gt 0 ]; then
\t\tlog ""
\t\tlog "Failed models (${{#failed_models[@]}}):"
\t\tfor model in "${{failed_models[@]}}"; do
\t\t\tlog "  ✗ $model"
\t\tdone
\tfi

\tlog ""
\tlog "Evaluation process completed!"
\tlog "Results are stored in individual model folders under: {'$OUT_BASE_DIR' if vars['has_output_suffix'] else '$BASE_DIR'}"
}}

# Help function
show_help() {{
\techo "Usage: $0 [OPTIONS]"
\techo ""
\techo "{vars["model_type_upper"]} Model Evaluation Script - Tests {vars["model_type_upper"]} models on {vars["dataset_upper"]} data"
\techo ""
\techo "Options:"
{help_options_str}
\techo ""
\techo "Examples:"
{help_examples_str}
}}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
\tcase $1 in
{option_cases_str}
\tesac
done

{single_model_section}

# Run main function
main
'''

        return script

    def _get_template(self) -> str:
        """Get the bash script template (placeholder for future enhancements)."""
        return ""  # Template is generated programmatically

    def save_script(self, content: str, output_path: str) -> None:
        """Save the generated script to a file."""
        with open(output_path, "w") as f:
            f.write(content)

        # Make executable
        os.chmod(output_path, 0o755)
        print(f"✓ Generated script saved to: {output_path}")


def interactive_mode():
    """Run the generator in interactive mode."""
    print("=" * 60)
    print("Model Evaluation Script Generator - Interactive Mode")
    print("=" * 60)
    print()

    # Dataset
    dataset = input("Dataset (mimic/chexpertplus): ").strip().lower()
    while dataset not in ["mimic", "chexpertplus"]:
        print("Invalid dataset. Must be 'mimic' or 'chexpertplus'")
        dataset = input("Dataset (mimic/chexpertplus): ").strip().lower()

    # Model type
    model_type = input("Model type (biovil/gloria): ").strip().lower()
    while model_type not in ["biovil", "gloria"]:
        print("Invalid model type. Must be 'biovil' or 'gloria'")
        model_type = input("Model type (biovil/gloria): ").strip().lower()

    # Models
    print("\nEnter model names (one per line, empty line to finish):")
    models = []
    while True:
        model = input("> ").strip()
        if not model:
            break
        models.append(model)

    if not models:
        print("Error: At least one model is required")
        return

    # Paths
    base_dir = input("\nBase output directory: ").strip()
    labels_path = input("Labels file path: ").strip()

    # Optional parameters
    split_type = input("Split type [test]: ").strip() or "test"
    num_iterations = input("Number of iterations [1000]: ").strip()
    num_iterations = int(num_iterations) if num_iterations else 1000

    skip_embeddings = (
        input("Skip embeddings generation? (y/n) [n]: ").strip().lower() == "y"
    )

    label_type = None
    if dataset == "mimic":
        label_type = input("Label type (optional, e.g., 'convirt'): ").strip() or None

    output_suffix = input("Output subdirectory suffix (optional): ").strip() or None

    frontal_impression = input("Frontal impression only? (y/n) [y]: ").strip().lower()
    frontal_impression_only = frontal_impression != "n"

    # Generate script name
    timestamp = datetime.now().strftime("%Y%m%d")
    script_name = f"{timestamp}_{model_type}_{dataset}_evaluation.sh"
    script_name = input(f"\nScript name [{script_name}]: ").strip() or script_name

    # Generate
    print("\nGenerating script...")
    generator = EvaluationScriptGenerator()

    script_content = generator.generate_script(
        dataset=dataset,
        model_type=model_type,
        models=models,
        base_dir=base_dir,
        labels_path=labels_path,
        script_name=script_name,
        split_type=split_type,
        num_iterations=num_iterations,
        skip_embeddings=skip_embeddings,
        label_type=label_type,
        output_suffix=output_suffix,
        frontal_impression_only=frontal_impression_only,
    )

    # Save
    output_dir = "/opt/gpudata/imadejski/search-model/remix/evaluation/bash_scripts"
    output_path = os.path.join(output_dir, script_name)

    os.makedirs(output_dir, exist_ok=True)
    generator.save_script(script_content, output_path)

    print(f"\n✓ Script generated successfully!")
    print(f"  Location: {output_path}")
    print(f"  Run with: bash {output_path}")


def config_mode(config_path: str):
    """Generate script from a JSON configuration file."""
    with open(config_path, "r") as f:
        config = json.load(f)

    generator = EvaluationScriptGenerator()

    # Validate required fields
    required = ["dataset", "model_type", "models", "base_dir", "labels_path"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    script_content = generator.generate_script(**config)

    # Determine output path
    script_name = config.get("script_name")
    if not script_name:
        timestamp = datetime.now().strftime("%Y%m%d")
        script_name = (
            f"{timestamp}_{config['model_type']}_{config['dataset']}_evaluation.sh"
        )

    output_dir = config.get(
        "output_dir",
        "/opt/gpudata/imadejski/search-model/remix/evaluation/bash_scripts",
    )
    output_path = os.path.join(output_dir, script_name)

    os.makedirs(output_dir, exist_ok=True)
    generator.save_script(script_content, output_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate bash evaluation scripts for model testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python generate_evaluation_script.py --interactive

  # From JSON config
  python generate_evaluation_script.py --config config.json

  # Direct generation
  python generate_evaluation_script.py \\
    --dataset mimic \\
    --model-type biovil \\
    --models "model1" "model2" "model3" \\
    --base-dir "/path/to/output" \\
    --labels-path "/path/to/labels.csv"

Config file format (JSON):
  {
    "dataset": "mimic",
    "model_type": "biovil",
    "models": ["model1", "model2"],
    "base_dir": "/path/to/output",
    "labels_path": "/path/to/labels.csv",
    "split_type": "test",
    "num_iterations": 1000,
    "skip_embeddings": false,
    "label_type": "convirt",
    "output_suffix": "suffix",
    "frontal_impression_only": true
  }
        """,
    )

    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--config", "-c", type=str, help="Path to JSON configuration file"
    )

    # Direct generation arguments
    parser.add_argument(
        "--dataset", type=str, choices=["mimic", "chexpertplus"], help="Dataset name"
    )
    parser.add_argument(
        "--model-type", type=str, choices=["biovil", "gloria"], help="Model type"
    )
    parser.add_argument("--models", nargs="+", help="List of model names")
    parser.add_argument("--base-dir", type=str, help="Base output directory")
    parser.add_argument("--labels-path", type=str, help="Path to labels file")
    parser.add_argument("--script-name", type=str, help="Output script name")
    parser.add_argument(
        "--split-type", type=str, default="test", help="Data split type (default: test)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="Number of resampling iterations (default: 1000)",
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true", help="Skip embedding generation"
    )
    parser.add_argument("--label-type", type=str, help="Label type (for MIMIC)")
    parser.add_argument("--output-suffix", type=str, help="Output subdirectory suffix")
    parser.add_argument(
        "--no-frontal-impression",
        action="store_true",
        help="Do not use frontal impression only",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/gpudata/imadejski/search-model/remix/evaluation/bash_scripts",
        help="Output directory for generated script",
    )

    args = parser.parse_args()

    try:
        if args.interactive:
            interactive_mode()
        elif args.config:
            config_mode(args.config)
        elif (
            args.dataset
            and args.model_type
            and args.models
            and args.base_dir
            and args.labels_path
        ):
            # Direct generation
            generator = EvaluationScriptGenerator()

            script_content = generator.generate_script(
                dataset=args.dataset,
                model_type=args.model_type,
                models=args.models,
                base_dir=args.base_dir,
                labels_path=args.labels_path,
                script_name=args.script_name,
                split_type=args.split_type,
                num_iterations=args.num_iterations,
                skip_embeddings=args.skip_embeddings,
                label_type=args.label_type,
                output_suffix=args.output_suffix,
                frontal_impression_only=not args.no_frontal_impression,
            )

            # Determine output path
            script_name = args.script_name
            if not script_name:
                timestamp = datetime.now().strftime("%Y%m%d")
                script_name = (
                    f"{timestamp}_{args.model_type}_{args.dataset}_evaluation.sh"
                )

            output_path = os.path.join(args.output_dir, script_name)

            os.makedirs(args.output_dir, exist_ok=True)
            generator.save_script(script_content, output_path)
        else:
            parser.print_help()
            print(
                "\nError: Either use --interactive, --config, or provide all required arguments"
            )
            return 1

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
