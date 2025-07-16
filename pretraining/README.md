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
- **Learning Rate**: Peak at 6e-4 with warmup and decay

### Phase 2: Mid-training/Extension (250B tokens)  
- **Duration**: ~100k steps
- **Data**: Higher-quality filtered subset with domain balancing
- **Context Length**: Extended to 8192 tokens
- **Learning Rate**: Continued decay from Phase 1

### Phase 3: Decay Phase (50B tokens)
- **Duration**: ~20k steps  
- **Data**: Premium sources (books, academic papers, curated web content)
- **Context Length**: Maintained at 8192 tokens
- **Learning Rate**: Further decay with extended schedule

## Data Sources and Mixture

### Core Data Sources
- **DCLM (DataComp for Language Models)**: High-quality web crawl data
- **Dolma v1.7**: Curated training corpus from AI2
- **Scientific Papers**: ArXiv, PubMed, and academic publications
- **Code**: GitHub repositories and programming tutorials
- **Books**: Project Gutenberg and other open book collections

### Data Preprocessing
All data preprocessing scripts and configurations are available in the [bert24 repository](https://github.com/orionw/bert24). Key preprocessing steps include:

1. **Deduplication**: Near-duplicate removal across all sources
2. **Quality Filtering**: Language detection, quality scoring, and content filtering  
3. **Tokenization**: ModernBERT tokenizer with 50,368 vocabulary size
4. **Sequence Packing**: Efficient packing to maximize context utilization

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
mkdir -p data/processed
mkdir -p checkpoints
```

### Data Preparation

TODO

### Training Commands

#### Start Pre-training (Phase 1)

```bash
# Encoder pre-training
python train.py \
    --config configs/ettin_150m_encoder.yaml \
    --data_dir data/processed/phase1 \
    --output_dir checkpoints/ettin-encoder-150m-phase1 \
    --num_epochs 1 \
    --batch_size 512 \
    --learning_rate 6e-4

# Decoder pre-training  
python train.py \
    --config configs/ettin_150m_decoder.yaml \
    --data_dir data/processed/phase1 \
    --output_dir checkpoints/ettin-decoder-150m-phase1 \
    --num_epochs 1 \
    --batch_size 512 \
    --learning_rate 6e-4
```

#### Continue to Mid-training (Phase 2)

```bash
# Continue encoder training
python train.py \
    --config configs/ettin_150m_encoder.yaml \
    --data_dir data/processed/phase2 \
    --resume_from checkpoints/ettin-encoder-150m-phase1/final \
    --output_dir checkpoints/ettin-encoder-150m-phase2 \
    --context_length 8192 \
    --learning_rate 3e-4
```

## Cross-Objective Training

The repository also includes scripts for training cross-objective models:

### Decoder → Encoder Conversion

```bash
# Continue decoder as encoder (CLM → MLM)
python train_cross_objective.py \
    --source_model checkpoints/ettin-decoder-150m/final \
    --target_objective masked_lm \
    --output_dir checkpoints/ettin-encoder-from-decoder-150m \
    --training_tokens 50B
```

### Encoder → Decoder Conversion

```bash
# Continue encoder as decoder (MLM → CLM)  
python train_cross_objective.py \
    --source_model checkpoints/ettin-encoder-150m/final \
    --target_objective causal_lm \
    --output_dir checkpoints/ettin-decoder-from-encoder-150m \
    --training_tokens 50B
```

## Hardware Requirements
All models are trained on 4x H100s. Training time is approximately:

1B: 2170 hours (~90 days)
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