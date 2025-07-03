# GLUE Evaluation

This directory contains scripts and documentation for evaluating Ettin encoder and decoder models on the GLUE (General Language Understanding Evaluation) benchmark tasks.

## Overview

GLUE is a collection of nine English sentence understanding tasks, designed to evaluate and analyze the general language understanding capabilities of language models. 

## Quick Start

### Installation

```bash
# Install required dependencies
pip install transformers datasets evaluate scikit-learn

# For hyperparameter sweeps (optional)
pip install wandb optuna

# Clone training repository for advanced configs
git clone https://github.com/orionw/bert24.git
cd bert24
pip install -e .
```

### Quick Evaluation Example

TODO

## Links and Resources

- **📊 GLUE Benchmark**: [https://gluebenchmark.com/](https://gluebenchmark.com/)
- **📖 GLUE Paper**: [https://arxiv.org/abs/1804.07461](https://arxiv.org/abs/1804.07461)
- **🤗 HuggingFace GLUE**: [https://huggingface.co/datasets/glue](https://huggingface.co/datasets/glue)
- **📈 Papers With Code**: [https://paperswithcode.com/dataset/glue](https://paperswithcode.com/dataset/glue)
- **📖 Training Repository**: [https://github.com/orionw/bert24](https://github.com/orionw/bert24)

---

For questions about GLUE evaluation, please open an issue in the main [Ettin repository](https://github.com/jhu-clsp/ettin-encoder-vs-decoder/issues). 