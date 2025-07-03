# Retrieval Evaluation

This document provides a quick overview of retrieval evaluation for Ettin encoder and decoder models. For comprehensive documentation, scripts, and detailed instructions, please see the **[retrieval_eval/ directory](../retrieval_eval/README.md)**.

## Quick Start

### Zero-shot Evaluation
```bash
# Evaluate pre-trained Ettin encoder on MTEB
python evaluate_mteb.py \
    --model_name jhu-clsp/ettin-encoder-150m \
    --output_dir results/ettin-encoder-150m-base \
    --tasks retrieval
```

### Fine-tuning on MS MARCO
```bash
# Fine-tune Ettin encoder on MS MARCO
python scripts/finetune_msmarco.py \
    --model_name jhu-clsp/ettin-encoder-150m \
    --data_dir data/ms_marco_processed \
    --output_dir models/ettin-encoder-150m-msmarco \
    --num_epochs 3 \
    --batch_size 32 \
    --learning_rate 2e-5
```

## Complete Documentation

For detailed instructions, scripts, configurations, and advanced features, please visit:

**📁 [retrieval_eval/README.md](../retrieval_eval/README.md)**

This includes:
- Complete installation and setup instructions
- MS MARCO fine-tuning pipeline
- Comprehensive MTEB evaluation
- Advanced features (hard negative mining, multi-stage training)
- Performance analysis and visualization tools
- Troubleshooting and configuration files 