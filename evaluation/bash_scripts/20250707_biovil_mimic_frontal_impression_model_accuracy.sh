#!/bin/bash

# BioViL model evaluation script for ReMIX project on MIMIC data with impression labels
# This script creates folders for each BioViL model and runs all three evaluation scripts:
# 1. general_mimic_embedding_library.py
# 2. general_mimic_cosine_similarity.py
# 3. general_mimic_accuracy_resampling.py

set -e  # Exit on any error

# Configuration
BASE_DIR="/opt/gpudata/imadejski/search-model/remix/data/20250715_model_evaluation"
EVAL_SCRIPTS_DIR="/opt/gpudata/imadejski/search-model/remix/evaluation"
MODEL_CHECKPOINTS_DIR="/opt/gpudata/remix"
LABELS_PATH="/opt/gpudata/cxr-derived/mimic-impression-labels.csv"
SPLIT_TYPE="test"
LABEL_TYPE="auto"
NUM_ITERATIONS=1000

# BioViL model configurations array (MIMIC-trained models)
declare -a MODELS=(
    "mimic-biovil-frontal-impression-igl_tgl-mlm"
    "mimic-biovil-frontal-impression-igl_tgl-no-mlm"
    "mimic-biovil-frontal-impression-igl_tg-mlm"
    "mimic-biovil-frontal-impression-igl_tg-no-mlm"
    "mimic-biovil-frontal-impression-ig_tgl-mlm"
    "mimic-biovil-frontal-impression-ig_tgl-no-mlm"
    "mimic-biovil-frontal-impression-ig_tg-mlm"
    "mimic-biovil-frontal-impression-ig_tg-no-mlm"
)

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

# Function to find model checkpoint path
find_model_checkpoint() {
    local model_name="$1"

    # Standard path for Hugging Face format models
    local checkpoint_path="${MODEL_CHECKPOINTS_DIR}/${model_name}"

    # Check if it's a directory with Hugging Face model files
    if [ -d "$checkpoint_path" ]; then
        # Check for required Hugging Face model files
        if [ -f "$checkpoint_path/config.json" ] && [ -f "$checkpoint_path/model.safetensors" ]; then
            echo "$checkpoint_path"
            return
        fi
    fi

    # Check if it's a single file (for .ckpt files)
    if [ -f "${checkpoint_path}.ckpt" ]; then
        echo "${checkpoint_path}.ckpt"
        return
    fi

    # Check alternative locations
    if [ -f "$checkpoint_path/last.ckpt" ]; then
        echo "$checkpoint_path/last.ckpt"
        return
    fi

    if [ -f "$checkpoint_path/checkpoints/last.ckpt" ]; then
        echo "$checkpoint_path/checkpoints/last.ckpt"
        return
    fi

    log "WARNING: Checkpoint not found for $model_name at $checkpoint_path"
    log "  Looked for: config.json + model.safetensors (Hugging Face format)"
    log "  Also looked for: .ckpt files and alternative locations"
    echo "CHECKPOINT_NOT_FOUND"
}

# Function to run embedding library script
run_embedding_script() {
    local model_name="$1"
    local model_checkpoint="$2"
    local output_dir="$3"

    local output_file="${output_dir}/${model_name}_embeddings_${SPLIT_TYPE}.csv"

    # Check if output already exists
    if [ -f "$output_file" ]; then
        log "Skipping embedding script for $model_name: output file already exists"
        echo "$output_file"
        return 0
    fi

    log "Running embedding script for $model_name..."

    python "${EVAL_SCRIPTS_DIR}/general_mimic_embedding_library.py" \
        "$model_checkpoint" \
        "$output_file" \
        "$SPLIT_TYPE" \
        --frontal_impression_only

    if [ $? -eq 0 ]; then
        log "✓ Embedding script completed successfully for $model_name"
        echo "$output_file"
        return 0
    else
        log "✗ Embedding script failed for $model_name"
        return 1
    fi
}

# Function to run cosine similarity script
run_cosine_script() {
    local model_name="$1"
    local model_checkpoint="$2"
    local embedding_file="$3"
    local output_dir="$4"

    local output_file="${output_dir}/${model_name}_cosine_similarity_${SPLIT_TYPE}.csv"

    # Check if output already exists
    if [ -f "$output_file" ]; then
        log "Skipping cosine similarity script for $model_name: output file already exists"
        echo "$output_file"
        return 0
    fi

    log "Running cosine similarity script for $model_name..."

    python "${EVAL_SCRIPTS_DIR}/general_mimic_cosine_similarity.py" \
        "$model_checkpoint" \
        "$embedding_file" \
        "$output_file" \
        "$SPLIT_TYPE" \
        --label_type "$LABEL_TYPE"

    if [ $? -eq 0 ]; then
        log "✓ Cosine similarity script completed successfully for $model_name"
        echo "$output_file"
        return 0
    else
        log "✗ Cosine similarity script failed for $model_name"
        return 1
    fi
}

# Function to run accuracy resampling script
run_accuracy_script() {
    local model_name="$1"
    local cosine_file="$2"
    local model_dir="$3"

    # When resampling is enabled, only one combined file is created
    local output_results="${model_dir}/${model_name}_accuracy_results_${SPLIT_TYPE}_resampling_${LABEL_TYPE}.csv"
    local output_all_results="${model_dir}/${model_name}_accuracy_results_${SPLIT_TYPE}_all_resampling_${LABEL_TYPE}.csv"

    # Skip if output files already exist (only need to check the main results file)
    if [ -f "$output_results" ]; then
        log "Skipping accuracy script for $model_name: output files already exist"
        return 0
    fi

    log "Running accuracy resampling script for $model_name..."

    python "${EVAL_SCRIPTS_DIR}/general_mimic_accuracy_resampling.py" \
        --cosine-path "$cosine_file" \
        --labels-path "$LABELS_PATH" \
        --output-results-path "${model_dir}/${model_name}_accuracy_results_${SPLIT_TYPE}.csv" \
        --output-top-k-results-path "${model_dir}/${model_name}_top_k_results_${SPLIT_TYPE}.csv" \
        --output-dcg-results-path "${model_dir}/${model_name}_dcg_results_${SPLIT_TYPE}.csv" \
        --split-type "$SPLIT_TYPE" \
        --label-type "$LABEL_TYPE" \
        --num-iterations "$NUM_ITERATIONS" \
        --resampling

    if [ $? -eq 0 ]; then
        log "✓ Accuracy resampling script completed successfully for $model_name"
        return 0
    else
        log "✗ Accuracy resampling script failed for $model_name"
        return 1
    fi
}

# Function to process a single model
process_model() {
    local model_name="$1"

    log "=================================================="
    log "Processing model: $model_name"
    log "=================================================="

    # Create model directory
    local model_dir="${BASE_DIR}/${model_name}"
    mkdir -p "$model_dir"
    log "Created/verified directory: $model_dir"

    # Find model checkpoint
    local model_checkpoint
    model_checkpoint=$(find_model_checkpoint "$model_name")

    if [ "$model_checkpoint" == "CHECKPOINT_NOT_FOUND" ]; then
        log "✗ Skipping $model_name: checkpoint not found"
        return 1
    fi

    log "Using checkpoint: $model_checkpoint"

    # Step 1: Generate embeddings
    local embedding_file
    embedding_file=$(run_embedding_script "$model_name" "$model_checkpoint" "$model_dir")
    if [ $? -ne 0 ]; then
        log "✗ Failed at embedding step for $model_name"
        return 1
    fi

    # Step 2: Calculate cosine similarities
    local cosine_file
    cosine_file=$(run_cosine_script "$model_name" "$model_checkpoint" "$embedding_file" "$model_dir")
    if [ $? -ne 0 ]; then
        log "✗ Failed at cosine similarity step for $model_name"
        return 1
    fi

    # Step 3: Calculate accuracy metrics
    if ! run_accuracy_script "$model_name" "$cosine_file" "$model_dir"; then
        log "✗ Failed at accuracy step for $model_name"
        return 1
    fi

    log "✓ All evaluation steps completed successfully for $model_name"
    return 0
}

# Main execution
main() {
    log "Starting BioViL model evaluation process on MIMIC data with impression labels..."
    log "Base directory: $BASE_DIR"
    log "Evaluation scripts directory: $EVAL_SCRIPTS_DIR"
    log "Model checkpoints directory: $MODEL_CHECKPOINTS_DIR"
    log "Labels file: $LABELS_PATH"
    log "Split type: $SPLIT_TYPE"
    log "Label type: $LABEL_TYPE"
    log "Number of iterations: $NUM_ITERATIONS"
    log "Total BioViL models to process: ${#MODELS[@]}"

    # Create base directory if it doesn't exist
    mkdir -p "$BASE_DIR"
    cd "$BASE_DIR"

    # Track results
    local successful_models=()
    local failed_models=()

    # Process each model
    for model in "${MODELS[@]}"; do
        if process_model "$model"; then
            successful_models+=("$model")
        else
            failed_models+=("$model")
        fi
        log "--------------------"
    done

    # Summary
    log "=================================================="
    log "EVALUATION SUMMARY"
    log "=================================================="
    log "Successfully processed models (${#successful_models[@]}):"
    for model in "${successful_models[@]}"; do
        log "  ✓ $model"
    done

    if [ ${#failed_models[@]} -gt 0 ]; then
        log ""
        log "Failed models (${#failed_models[@]}):"
        for model in "${failed_models[@]}"; do
            log "  ✗ $model"
        done
    fi

    log ""
    log "Evaluation process completed!"
    log "Results are stored in individual model folders under: $BASE_DIR"
}

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "BioViL Model Evaluation Script - Tests BioViL models on MIMIC data with impression labels"
    echo ""
    echo "Options:"
    echo "  --split-type SPLIT    Set the data split type (default: test)"
    echo "  --label-type TYPE     Set the label type (default: auto)"
    echo "  --iterations NUM      Set number of resampling iterations (default: 1000)"
    echo "  --labels-path PATH    Set custom labels file path (default: $LABELS_PATH)"
    echo "  --model MODEL_NAME    Process only a specific model"
    echo "  --list-models         List all available BioViL models"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                                          # Process all BioViL models with default settings"
    echo "  $0 --split-type validate                                    # Use validation split"
    echo "  $0 --model mimic-biovil-frontal-impression-igl_tgl-mlm      # Process only one model"
    echo "  $0 --label-type convirt --iterations 500                    # Use convirt labels with 500 iterations"
    echo "  $0 --labels-path /path/to/custom/labels.csv                 # Use custom labels file"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --split-type)
            SPLIT_TYPE="$2"
            shift 2
            ;;
        --label-type)
            LABEL_TYPE="$2"
            shift 2
            ;;
        --iterations)
            NUM_ITERATIONS="$2"
            shift 2
            ;;
        --labels-path)
            LABELS_PATH="$2"
            shift 2
            ;;
        --model)
            SINGLE_MODEL="$2"
            shift 2
            ;;
        --list-models)
            echo "Available BioViL models:"
            for model in "${MODELS[@]}"; do
                echo "  - $model"
            done
            exit 0
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# If single model specified, process only that model
if [ -n "$SINGLE_MODEL" ]; then
    # Check if model is in the list
    if [[ " ${MODELS[@]} " =~ " ${SINGLE_MODEL} " ]]; then
        log "Processing single model: $SINGLE_MODEL"
        mkdir -p "$BASE_DIR"
        cd "$BASE_DIR"
        process_model "$SINGLE_MODEL"
        exit $?
    else
        log "Error: Model '$SINGLE_MODEL' not found in the BioViL model list"
        log "Use --list-models to see available BioViL models"
        exit 1
    fi
fi

# Run main function
main
