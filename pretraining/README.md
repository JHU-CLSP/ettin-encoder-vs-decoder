# Pre-training Guide

This guide covers the complete pre-training process for Ettin models, including data preparation, training setup, and the adapted ModernBERT recipe used for both encoder and decoder models.

## Overview

Ettin models are trained using a three-phase approach adapted from the ModernBERT training recipe, with identical data and procedures for both encoder and decoder models to enable fair architectural comparisons.

## Training Repository

**📖 training code**: [https://github.com/orionw/bert24](https://github.com/orionw/bert24)

That repository is a fork of the ModernBERT training codebase, extended with decoder model support and training objectives. You can just clone it and run the command when training.

### Accessing training configs
 - prolong_decay is the final decay
 - de_decay is the context extension

Note: `bert-base-uncased` in the flex_bert config is a no-op. use `load_path` at the bottom to use the checkpoints 

## Training Phases

### Phase 1: Pre-training (1.7T tokens)
- **Duration**: ~600k steps
- **Data**: Diverse mixture including web text, books, code, and scientific papers
- **Context Length**: 1024 tokens initially, gradually increased
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


## Data Preprocessing
You can use the existing data available in Huggingface or create your own. The data should be in MosiacML `streaming` format. For (messy) scripts to do data preprocessing see the README in [Data Processing Guide](data_processing/README.md).

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
```

### Data Preparation

See [Data Processing Guide](data_processing/README.md) for detailed preprocessing instructions or download the data from huggingface, e.g. the [pretraining data here](https://huggingface.co/datasets/jhu-clsp/ettin-pretraining-data).

### Training Commands
The training command will infer the number of GPUs.

`composer main.py $yaml_config_file`

You can run the cross-objective version by using those configs, which are the same but load from the opposite checkpoint. See configs/cross-train for examples

### Decoder → Encoder Conversion
There are a few changes to do: (1) change the tokenizer to work like an encoder (2) change the model class to be modernbert and (3) re-combine the qkv layer. We have some messy scripts to do this and will be uploading them soon, if you need them sooner please open an issue or message us!


## Hardware Requirements
All models are trained on 4x H100s. Training time is approximately:

- 1B: 2170 hours to do 2T (~90 days, we did only ~40)
- 400M: 950 hours (~40 days)
- 150M: 470 hours (~20 days)
- 68M: 300 hours (~13 days)
- 32M: 212 hours (~9 days)
- 17M: 141 hours (~6 days)

## Links and Resources

- **📖 Training Repository**: [https://github.com/orionw/bert24](https://github.com/orionw/bert24)
- **📊 Training Data**: [HuggingFace Datasets](https://huggingface.co/datasets/jhu-clsp)
- **🔧 Model Configs**: [Configuration Files](./pretraining/configs)

--