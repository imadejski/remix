# ReMIX Model Evaluation Pipeline

Evaluation pipeline for testing BioViL and GLORIA models on MIMIC-CXR and CheXpert+ datasets.

## Pipeline Overview

Three-stage evaluation process:
1. **Embeddings** → Generate image embeddings
2. **Cosine Similarity** → Compare image embeddings with label text embeddings
3. **Accuracy** → Calculate metrics with optional resampling

## Quick Start

```bash
# Run an existing script
cd /opt/gpudata/imadejski/search-model/remix/evaluation/bash_scripts
bash 20251027_gloria_chexpert_frontal_impression_only_original_loss.sh

# Or process a single model
bash 20251027_gloria_chexpert_frontal_impression_only_original_loss.sh --model MODEL_NAME

# Generate a new evaluation script
cd /opt/gpudata/imadejski/search-model/remix/evaluation
python generate_evaluation_script.py --interactive
```

---

## Key Configuration Variables

When creating or editing bash scripts, pay attention to these critical variables:

### Required Paths

| Variable | Description | MIMIC Example | CheXpert+ Example |
|----------|-------------|---------------|-------------------|
| `BASE_DIR` | Where outputs are saved | `/opt/gpudata/imadejski/search-model/remix/data/my_eval` | Same |
| `LABELS_PATH` | **Ground truth labels** | `/opt/gpudata/cxr-derived/mimic-impression-labels.csv` | `/opt/gpudata/chexpertplus/impression_fixed.json` |
| `MODEL_CHECKPOINTS_DIR` | Model location | `/opt/gpudata/remix` | `/opt/gpudata/remix` |
| `EVAL_SCRIPTS_DIR` | Evaluation scripts | `/opt/gpudata/imadejski/search-model/remix/evaluation` | Same |

### Important Settings

| Variable | Options | Default | Notes |
|----------|---------|---------|-------|
| `SPLIT_TYPE` | `test`, `validate`, `train` | `test` | Which data split to evaluate |
| `NUM_ITERATIONS` | Any integer | `1000` | Bootstrap iterations for confidence intervals |
| `LABEL_TYPE` | `convirt`, `auto`, `chexpert` | — | **MIMIC only**. Text query style to use. CheXpert+ always uses auto |
| `SKIP_EMBEDDINGS` | `true`, `false` | `false` | Set `true` to reuse existing embeddings |
| `OUT_SUFFIX` | Any string | — | Organizes outputs in subdirectory (optional) |

### Critical: Label Files

**⚠️ Use the correct label file for your dataset:**

- **MIMIC-CXR**: `/opt/gpudata/cxr-derived/mimic-impression-labels.csv`
- **CheXpert+**: `/opt/gpudata/chexpertplus/impression_fixed.json`

**Note:** Labels in these files were recalculated using the CheXpert labeler on the impression section only.

**⚠️ For MIMIC, set `LABEL_TYPE` (controls both ground truth labels AND query expansion):**
- `convirt` = 8 ConVIRT labels + ConVIRT's predefined expanded queries + CheXpert auto-extracted ground truth
- `auto` = 14 labels + automated query expansion patterns + CheXpert auto-extracted ground truth
- `chexpert` = 14 labels + automated query expansion patterns + CheXpert-derived ground truth (not auto-extracted)

---

## Generate New Evaluation Scripts

### Interactive Mode (Recommended)
```bash
python generate_evaluation_script.py --interactive
```

### Command-Line Mode
```bash
python generate_evaluation_script.py \
  --dataset mimic \
  --model-type biovil \
  --models "model1" "model2" \
  --base-dir "/path/to/output" \
  --labels-path "/opt/gpudata/cxr-derived/mimic-impression-labels.csv" \
  --label-type convirt \
  --num-iterations 1000
```

### Generated Script Options
```bash
bash my_script.sh                        # Run all models
bash my_script.sh --model MODEL_NAME      # Single model
bash my_script.sh --split-type validate   # Different split
bash my_script.sh --iterations 500        # Fewer iterations
bash my_script.sh --help                  # Show help
```

---

## Output Files

After running, find results in `BASE_DIR/{model_name}/`:

```
BASE_DIR/
└── model_name/
    ├── model_embeddings_test.csv                    # Stage 1: Image embeddings
    ├── model_cosine_similarity_test.csv             # Stage 2: Similarity scores
    ├── model_accuracy_results_test_no_resample.csv  # Stage 3: Simple metrics
    └── model_accuracy_results_test_resampling.csv   # Stage 3: With confidence intervals
```

**Key file**: `*_resampling.csv` contains mean accuracy, standard deviation, and 95% confidence intervals for each pathology.

---

## Evaluating Base BioVIL/BioVIL-T Models

To evaluate the **pretrained base models** (without fine-tuning), use the HIML scripts instead of the general scripts:

### MIMIC Base Model Evaluation

1. **Generate embeddings** using the base model:
```bash
python himl_mimic_embedding_library.py \
  output_embeddings.csv \
  test \
  --model_type biovil  # or biovil-t
```

2. **Calculate cosine similarities**:
```bash
python himl_mimic_cosine_similarity.py \
  output_embeddings.csv \
  output_cosine.csv \
  test \
  --model_type biovil \
  --label_type auto  # or convirt, raw
```

3. **Calculate accuracy** using the same general script:
```bash
python general_mimic_accuracy.py \
  -c output_cosine.csv \
  --labels-path /opt/gpudata/cxr-derived/mimic-impression-labels.csv \
  -o output_accuracy.csv \
  -s test \
  -r \
  -n 1000 \
  -l auto
```

**Note:** The HIML scripts use the `health_multimodal` library's pretrained BioVIL/BioVIL-T models and are only available for MIMIC-CXR evaluation. For fine-tuned models, use the `general_*` scripts.

---

## Common Patterns

### Reuse Embeddings for Different Experiments

```bash
# First run: Generate embeddings
python generate_evaluation_script.py \
  --dataset mimic --model-type biovil --models "model1" \
  --base-dir "/data/baseline" \
  --labels-path "/opt/gpudata/cxr-derived/mimic-impression-labels.csv"

# Second run: Reuse embeddings, try different label type
python generate_evaluation_script.py \
  --dataset mimic --model-type biovil --models "model1" \
  --base-dir "/data/baseline" \
  --labels-path "/opt/gpudata/cxr-derived/mimic-impression-labels.csv" \
  --skip-embeddings \
  --label-type convirt \
  --output-suffix "convirt_experiment"
```

This creates:
```
/data/baseline/
├── model1/
│   └── model1_embeddings_test.csv  ← Reused
└── convirt_experiment/
    └── model1/
        ├── model1_cosine_similarity_test_convirt.csv
        └── model1_accuracy_results_test_convirt_resampling.csv
```

### Evaluate Subset of Data

See `bash_scripts/20251013_mimic_subset_exclusive_eval.sh` for an example using custom split files.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Checkpoint not found" | Check model exists at `/opt/gpudata/remix/{model_name}/` |
| "Embeddings file missing" | Don't use `SKIP_EMBEDDINGS=true` unless embeddings exist |
| "Label mismatch" | Verify `LABELS_PATH` and `LABEL_TYPE` match your dataset |
| Out of memory | Process models one at a time, or reduce batch size in Python scripts |
| Wrong results | Double-check you're using the correct label file for your dataset |

---

## Evaluation Metrics

The accuracy scripts calculate multiple metrics for each pathology:

| Metric | Description |
|--------|-------------|
| **Accuracy@n** | Precision when retrieving top-n images (n = number of positive cases) |
| **Top-k Accuracy** | Precision at fixed k values (k = 5, 10, 20) |
| **DCG@n / NDCG@n** | Discounted Cumulative Gain at n positive cases (measures ranking quality) |
| **DCG@k / NDCG@k** | DCG at fixed k values (k = 5, 10, 20) |

Each metric is computed with two aggregation methods:
- **max**: Maximum cosine similarity per study
- **mean**: Average cosine similarity per study

---

## Creating Custom Evaluation Subsets

### Exclusive Positive Subsets

For evaluating on exclusively positive cases (e.g., 5 labels × 200 images each):

```bash
python generate_mimic_frontal_exclusive_subset.py \
  --output_csv mimic_frontal_exclusive_test_5x200.csv \
  --labels "Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion" \
  --num_per_label 200 \
  --split_type test \
  --seed 42
```

**Parameters:**
- `--labels`: Comma-separated list of labels to include
- `--num_per_label`: Number of exclusively positive images per label
- `--split_type`: Source split to sample from (`test`, `validate`, or `train`)
- `--seed`: Random seed for reproducibility

**Important:** The script ensures exclusivity - each image is positive for exactly one label.

### Using Custom Splits in Evaluation

Pass your custom split file to the evaluation scripts:

```bash
python general_mimic_embedding_library.py \
  MODEL_CHECKPOINT \
  output_embeddings.csv \
  test \
  --split_file_path /path/to/custom_split.csv
```

Custom split files must have columns: `subject_id`, `study_id`, `dicom_id`, `split`

### Custom Label Lists

Override default labels for specific experiments:

```bash
python general_mimic_accuracy.py \
  -c cosine.csv \
  --labels-path labels.csv \
  -o output.csv \
  -s test \
  --labels "Atelectasis,Cardiomegaly,Edema,Pneumonia"  # Custom subset
```

---

## Model Naming Convention

**BioViL** (MIMIC-trained): `mimic-biovil-frontal-impression-{loss}-{mlm}`
**GLORIA** (CheXpert+-trained): `chexpertplus-gloria-frontal-impression-{loss}-{mlm}`

- **Loss**: `igl_tgl`, `igl_tg`, `ig_tgl`, `ig_tg`
- **MLM**: `mlm` or `no-mlm`

**Examples:**
- `mimic-biovil-frontal-impression-igl_tgl-mlm`
- `chexpertplus-gloria-frontal-impression-ig_tg-no-mlm`

---

## TODO

### High Priority

1. **GLORIA Base Model Evaluation**
   - Create HIML-style scripts for evaluating pretrained GLORIA models (similar to `himl_mimic_*` for BioVIL/BioVIL-T)
   - Support CheXpert+ dataset evaluation with base GLORIA models

2. **Automated Plotting Script**
   - Generate accuracy plots automatically from `*_resampling.csv` files
   - Include confidence intervals as error bars
   - Support comparison plots across multiple models
   - Output formats: PNG, PDF, SVG

3. **CheXpert+ Exclusive Positive Subset**
   - Extend `generate_mimic_frontal_exclusive_subset.py` to support CheXpert+ dataset
   - Ensure compatibility with CheXpert+ file structure and labels
