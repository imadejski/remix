# ReMiX: Retrieval with Multiscale Image-text Cross-alignment

**Cross-modal, cross-scale contrastive learning for chest X-ray and radiology report retrieval**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## Overview

ReMiX is a novel training framework that addresses key limitations in current multimodal approaches by improving **chest X-ray to radiology report retrieval** through cross-modal, cross-scale contrastive learning. By combining global and local alignment strategies across both image and text modalities, ReMiX enables more fine-grained understanding and better alignment of chest X-rays with their corresponding written reports.

### Research Motivation

Current multimodal approaches for biomedical image-text retrieval often fail to capture the complex relationships between chest X-rays and their corresponding radiology reports. Through systematic analysis of large radiology datasets—including embedding, clustering, and quantitative evaluation—we identified that existing methods lack:

1. **Fine-grained text-level local alignment**: Full radiology reports are lengthy and contain multiple findings, while retrieval queries are typically short and focused (e.g., "Findings consistent with Pneumonia"). Standard global alignment treats entire reports as single units, missing the opportunity to align sentence-level chunks that better match the granularity of these direct queries.
2. **Cross-scale representation**: Effective retrieval requires understanding both whole reports and local text chunks at different scales to match diverse query types and improve retrieval accuracy.
3. **Balanced cross-modal learning**: Asymmetric treatment of image and text modalities limits retrieval performance

ReMiX addresses these limitations through a unified cross-modal, cross-scale contrastive training objective that simultaneously optimizes alignment at both global and local levels for both modalities.

### Key Features

- **Cross-Modal Cross-Scale Contrastive Learning**: Simultaneously optimizes four alignment objectives:
  - **Image Global (IG)**: Whole chest X-ray to full report alignment
  - **Text Global (TG)**: Full report to whole chest X-ray alignment
  - **Image Local (IGL)**: Image regions (patches) to specific report findings alignment
  - **Text Local (TGL)**: Sentence-level report chunks to relevant image regions—enabling better matching between short clinical queries and focused report findings
- **Radiology-Specific Design**: Optimized for chest X-ray and radiology report characteristics
- **Flexible Architecture Support**: Compatible with both BioViL (MIMIC-CXR) and GLORIA (CheXpertPlus) backbones
- **Optional MLM Objectives**: Masked language modeling for improved clinical language understanding
- **Comprehensive Evaluation Pipeline**: End-to-end workflow for embedding generation, similarity computation, and retrieval metrics
- **Production-Ready**: Built on PyTorch Lightning with Weights & Biases integration for reproducible research

### Datasets

- **MIMIC-CXR**: 377,110 chest X-ray images paired with full radiology reports
- **CheXpertPlus**: Large-scale chest X-ray dataset with radiology reports and impression-based labels

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended)
- Access to MIMIC-CXR and/or CheXpertPlus datasets

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/imadejski/remix.git
cd remix
```

2. **Install dependencies**
```bash
pip install -e .
```

The project uses `pyproject.toml` for dependency management, which automatically installs all required packages including PyTorch, Lightning, transformers, and health-multimodal libraries.

3. **Configure data paths**

Update dataset paths in configuration files (`configs/*.yaml`) to match your local setup:
- **MIMIC-CXR**: `/opt/gpudata/mimic-cxr/` (chest X-ray images and radiology reports)
- **CheXpertPlus**: `/opt/gpudata/chexpertplus/` (chest X-ray images with radiology reports and impression labels)

---

## Repository Structure

```
remix/
├── configs/                          # Training configuration files
│   ├── mimic-biovil-frontal*.yaml    # BioViL configs for MIMIC-CXR
│   └── chexpertplus-gloria-frontal*.yaml  # GLORIA configs for CheXpertPlus
├── remix/                            # Core library code
│   ├── models/                       # Model implementations
│   │   ├── image_text_multiscale_contraster*.py  # Model architectures
│   │   └── modules/                  # Model components (contraster, ResNet)
│   ├── datasets.py                   # Dataset classes
│   └── utils.py                      # Utility functions
├── evaluation/                       # Evaluation pipeline
│   ├── general_*_embedding_library.py      # Generate embeddings
│   ├── general_*_cosine_similarity.py      # Compute similarities
│   ├── general_*_accuracy.py               # Calculate metrics
│   ├── himl_mimic_*.py                     # Base model evaluation (BioViL/BioViL-T)
│   ├── generate_evaluation_script.py       # Bash script generator
│   ├── generate_mimic_frontal_exclusive_subset.py  # Custom subset creator
│   ├── bash_scripts/                       # Generated evaluation scripts
│   └── README.md                           # Detailed evaluation docs
├── data/                             # Evaluation results and outputs
├── data_analysis/                    # Analysis notebooks
├── scripts_v3/                       # Latest training scripts
│   ├── train-mimic-biovil-frontal-impression-ig-tgl-*.sh    # MIMIC ig-tgl training
│   └── train-chexpertplus-gloria-frontal-impression-ig-tgl-*.sh  # CheXpertPlus ig-tgl training
├── scripts/, scripts_v2/             # Previous training script versions
├── training/                         # Training notebooks and experiments
├── run.py                            # Main training entry point
├── requirements.txt                  # Python dependencies
└── pyproject.toml                    # Project metadata and dependencies
```

### Key Components

#### `remix/` - Core Library
- **`models/`**: Implementation of multiscale contrastive learning models with configurable loss combinations (IG, TG, IGL, TGL)
- **`datasets.py`**: Custom PyTorch datasets for MIMIC-CXR and CheXpertPlus with preprocessing and augmentation
- **`utils.py`**: Tokenizers, data loaders, and utility functions

#### `evaluation/` - Evaluation Pipeline
Complete pipeline for assessing chest X-ray retrieval performance with three stages:
1. **Embedding Generation**: Extract chest X-ray image embeddings from trained models
2. **Cosine Similarity**: Compare image embeddings with clinical text query embeddings (findings, pathologies)
3. **Accuracy Metrics**: Calculate retrieval performance metrics including accuracy@n (precision at n positive cases), top-k accuracy, DCG, and NDCG

The evaluation pipeline quantitatively measures how well models can retrieve relevant chest X-rays given radiology findings or pathology queries.

See [`evaluation/README.md`](evaluation/README.md) for detailed documentation.

#### `configs/` - Training Configurations
YAML files defining hyperparameters, data paths, and training settings. Configure:
- Model architecture (BioViL vs GLORIA)
- Loss combinations (`ig_tg`, `igl_tgl`, etc.)
- Batch size, learning rate, precision
- Data preprocessing options

---

## Quick Start

### Training

Train a chest X-ray retrieval model using PyTorch Lightning CLI:

```bash
# BioViL on MIMIC-CXR with full cross-modal, cross-scale objectives
python run.py fit \
  --config configs/mimic-biovil-frontal.yaml \
  --data.init_args.section impression \
  --data.init_args.mlm_probability 0.15 \
  --model.init_args.loss_combo igl_tgl

# GLORIA on CheXpertPlus with baseline objectives (no MLM)
python run.py fit \
  --config configs/chexpertplus-gloria-frontal.yaml \
  --data.init_args.section impression \
  --data.init_args.mlm_probability 0.0 \
  --model.init_args.loss_combo ig_tg
```

**Loss Combinations:**
- `igl_tgl`: Full cross-modal, cross-scale (all four alignment objectives)
- `ig_tg`: Image global + Text global (baseline approach)
- `igl_tg`: Image cross-scale + Text global only
- `ig_tgl`: Image global + Text cross-scale only

**Training Scripts:**
Pre-configured bash scripts for the full `igl_tgl` model are available in `scripts_v2/`. For other loss combinations, use the command-line interface with `run.py` as shown above.

### Evaluation

Assess chest X-ray retrieval performance by generating and running evaluation scripts:

```bash
cd evaluation

# Interactive script generation
python generate_evaluation_script.py --interactive

# Or use command-line arguments for automated workflow
python generate_evaluation_script.py \
  --dataset mimic \
  --model-type biovil \
  --models "mimic-biovil-frontal-impression-igl_tgl-mlm" \
  --base-dir "/path/to/output" \
  --labels-path "/opt/gpudata/cxr-derived/mimic-impression-labels.csv"

# Run the generated script to evaluate retrieval performance
bash bash_scripts/generated_script.sh
```

The evaluation pipeline tests how accurately models can retrieve chest X-rays matching specific pathology queries (e.g., "Findings consistent with Pneumonia"), producing quantitative metrics including precision@n, top-k accuracy, and NDCG scores.

See [`evaluation/README.md`](evaluation/README.md) for comprehensive evaluation documentation.

---

## Model Naming Convention

Models follow a standardized naming format:

**Format:** `{dataset}-{architecture}-{view}-{section}-{loss}-{mlm}`

**Examples:**
- `mimic-biovil-frontal-impression-igl_tgl-mlm` - Full model with MLM
- `chexpertplus-gloria-frontal-impression-ig_tg-no-mlm` - Baseline without MLM

**Components:**
- `dataset`: `mimic` or `chexpertplus`
- `architecture`: `biovil` or `gloria`
- `view`: `frontal` (frontal X-rays only)
- `section`: `impression` (impression section of reports)
- `loss`: `igl_tgl`, `igl_tg`, `ig_tgl`, `ig_tg`
- `mlm`: `mlm` or `no-mlm`

---

## Project Team

**University of Chicago - Center for Translational Data Science**

- **Irene Madejski** - Co-Lead Developer
- **Steven Song** - Co-Lead Developer
- **Professor Robert L. Grossman** - Principal Investigator

---

## Citation

If you use ReMiX in your research, please cite:

```bibtex
@software{remix2025,
  title={ReMiX: Retrieval with Multiscale Image-text Cross-alignment},
  author={Madejski, Irene* and Song, Steven* and Grossman, Robert L.},
  year={2025},
  organization={Center for Translational Data Science, University of Chicago},
  note={* Equal contribution}
}
```

*\* Equal contribution*

---

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

---

## Contact

For questions or issues, please:
- Open an issue on [GitHub](https://github.com/imadejski/remix/issues)
- Contact: imadejski@uchicago.edu
