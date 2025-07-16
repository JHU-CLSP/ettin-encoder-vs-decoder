# Pre-training Guide

This guide covers the complete pre-training process for Ettin models, including data preparation, training setup, and the adapted ModernBERT recipe used for both encoder and decoder models.

## Overview

Ettin models are trained using a three-phase approach adapted from the ModernBERT training recipe, with identical data and procedures for both encoder and decoder models to enable fair architectural comparisons.

## Training Repository

**📖 Complete training code and detailed instructions**: [https://github.com/orionw/bert24](https://github.com/orionw/bert24)

This repository is a fork of the ModernBERT training codebase, extended with:
- Decoder model support and training objectives
- Ettin-specific configurations for all model sizes
- Data preprocessing pipelines for the three training phases
- Evaluation scripts for GLUE and other discriminative tasks

## Training Phases

### Phase 1: Pre-training (1.7T tokens)
- **Duration**: ~600k steps
- **Data**: Diverse mixture including web text, books, code, and scientific papers
- **Context Length**: 2048 tokens initially, gradually increased
- **Learning Rate**: Peak after warmup

### Phase 2: Mid-training/Extension (250B tokens)  
- **Duration**: ~100k steps
- **Data**: Higher-quality filtered subset with domain balancing
- **Context Length**: Extended to 8k tokens
- **Learning Rate**: decay to half LR from Phase 1

### Phase 3: Decay Phase (50B tokens)
- **Duration**: ~20k steps  
- **Data**: Premium sources (books, academic papers, curated web content)
- **Context Length**: Maintained at 8k tokens
- **Learning Rate**: another decay to 0.02 of the LR

## Data Sources and Mixture

### Core Data Sources
- **DCLM (DataComp for Language Models)**: High-quality web crawl data
- **Dolma v1.7**: Curated training corpus from AI2
- **Scientific Papers**: ArXiv, PubMed, and academic publications
- **Code**: GitHub repositories and programming tutorials
- **Books**: Project Gutenberg and other open book collections

### Data Preprocessing
**in progress, uploading soon!**

All data preprocessing scripts and configurations are available in the [bert24 repository](https://github.com/orionw/bert24). Key preprocessing steps include:

## Model Configurations

### Architecture Scaling

The repository includes configurations for all Ettin model sizes:

| Model Size | Config File | Layers | Hidden Size | Intermediate Size | Attention Heads |
|:-----------|:------------|:-------|:------------|:------------------|:----------------|
| 17M        | `configs/ettin_17m.yaml` | 7 | 256 | 384 | 4 |
| 32M        | `configs/ettin_32m.yaml` | 10 | 384 | 576 | 6 |
| 68M        | `configs/ettin_68m.yaml` | 19 | 512 | 768 | 8 |
| 150M       | `configs/ettin_150m.yaml` | 22 | 768 | 1152 | 12 |
| 400M       | `configs/ettin_400m.yaml` | 28 | 1024 | 2624 | 16 |
| 1B         | `configs/ettin_1b.yaml` | 28 | 1792 | 3840 | 28 |


## Quick Start

### Setup Environment

```bash
# Clone the training repository
git clone https://github.com/orionw/bert24.git
cd bert24

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Setup data directories
TODO
```

### Data Preparation

TODO

### Training Commands


## Cross-Objective Training

The repository also includes scripts for training cross-objective models:

### Decoder → Encoder Conversion



## Hardware Requirements
All models are trained on 4x H100s. Training time is approximately:

1B: 2170 hours to do 2T (~90 days, we did only ~40)
400M: 950 hours (~40 days)
150M: 470 hours (~20 days)
68M: 300 hours (~13 days)
32M: 212 hours (~9 days)
17M: 141 hours (~6 days)

## Links and Resources

- **📖 Training Repository**: [https://github.com/orionw/bert24](https://github.com/orionw/bert24)
- **📊 Training Data**: [HuggingFace Datasets](https://huggingface.co/datasets/jhu-clsp)
- **🔧 Model Configs**: [Configuration Files](https://github.com/orionw/bert24/tree/main/configs)
- **📈 Training Logs**: [Weights & Biases](https://wandb.ai/ettin-project)
- **❓ Support**: [GitHub Issues](https://github.com/orionw/bert24/issues)

---

For detailed implementation specifics, configuration options, and troubleshooting, please refer to the [bert24 repository documentation](https://github.com/orionw/bert24/blob/main/README.md). 