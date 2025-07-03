# Retrieval Evaluation

This directory contains scripts and documentation for evaluating Ettin encoder models on retrieval tasks, including fine-tuning on MS MARCO and comprehensive evaluation on MTEB v2 English benchmarks.

## Overview

While Ettin encoders show strong zero-shot retrieval performance, fine-tuning on retrieval-specific data significantly improves their performance on information retrieval tasks. This guide covers the complete pipeline from MS MARCO fine-tuning to comprehensive MTEB evaluation.

## Quick Start

### Installation

```bash
# Install retrieval dependencies
pip install sentence-transformers mteb beir faiss-gpu

# For training (optional)
git clone https://github.com/orionw/bert24.git
cd bert24
pip install -e .
```

### Quick Evaluation Example

```bash
# Evaluate pre-trained Ettin encoder on MTEB
python evaluate_mteb.py \
    --model_name jhu-clsp/ettin-encoder-150m \
    --output_dir results/ettin-encoder-150m-base \
    --tasks retrieval
```


TODO

## Links and Resources

- **🏆 MTEB Leaderboard**: [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- **📊 MS MARCO Dataset**: [https://microsoft.github.io/msmarco/](https://microsoft.github.io/msmarco/)
- **🔧 Sentence Transformers**: [https://www.sbert.net/](https://www.sbert.net/)
- **📈 MTEB Framework**: [https://github.com/embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb)
- **📖 Training Repository**: [https://github.com/orionw/bert24](https://github.com/orionw/bert24)

---

For questions about retrieval evaluation, please open an issue in the main [Ettin repository](https://github.com/jhu-clsp/ettin-encoder-vs-decoder/issues). 