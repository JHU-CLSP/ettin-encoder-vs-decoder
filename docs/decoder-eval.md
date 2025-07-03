# Decoder Evaluation on Generative Tasks

This guide covers evaluating Ettin decoder models on generative language tasks using the EleutherAI evaluation harness (commit `867413f8677f00f6a817262727cbb041bf36192a`).

## Overview

Ettin decoder models excel at generative tasks and should be evaluated using the standard EleutherAI lm-evaluation-harness. This provides comprehensive evaluation across a wide range of language understanding and generation benchmarks.

## Quick Start

### Installation

```bash
# Clone the specific commit of lm-evaluation-harness
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout 867413f8677f00f6a817262727cbb041bf36192a
pip install -e .
```

### Basic Evaluation

```bash
# Evaluate Ettin decoder on core tasks
lm_eval --model hf \
    --model_args "pretrained=jhu-clsp/ettin-decoder-150m,add_bos_token=True" \
    --tasks hellaswag,arc_easy,arc_challenge,winogrande \
    --device cuda:0 \
    --batch_size 8 \
    --output_path results/ettin-decoder-150m
```


## Multi-Model Evaluation Script

```bash
#!/bin/bash
# evaluate_all_decoders.sh

MODELS=(
    "jhu-clsp/ettin-decoder-17m"
    "jhu-clsp/ettin-decoder-32m"
    "jhu-clsp/ettin-decoder-68m"
    "jhu-clsp/ettin-decoder-150m"
    "jhu-clsp/ettin-decoder-400m"
    "jhu-clsp/ettin-decoder-1b"
)

TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,boolq"

for model in "${MODELS[@]}"; do
    echo "Evaluating $model..."
    output_dir="results/$(basename $model)"
    
    lm_eval --model hf \
        --model_args "pretrained=$model,add_bos_token=True" \
        --tasks $TASKS \
        --device cuda:0 \
        --batch_size 8 \
        --output_path $output_dir \
        --log_samples
done
```

## Checkpoint Evaluation

```bash
# Evaluate specific training checkpoints
lm_eval --model hf \
    --model_args "pretrained=jhu-clsp/ettin-decoder-400m,revision=step590532,add_bos_token=True" \
    --tasks hellaswag,arc_easy \
    --device cuda:0 \
    --batch_size 8 \
    --output_path results/ettin-decoder-400m-step590532
```

## Cross-Objective Model Evaluation
Is done in the same way as the above, since they are decoders now.


## Links and Resources

- **Evaluation Harness**: [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- **Specific Commit**: [867413f8677f00f6a817262727cbb041bf36192a](https://github.com/EleutherAI/lm-evaluation-harness/commit/867413f8677f00f6a817262727cbb041bf36192a)
- **Model Collection**: [jhu-clsp on HuggingFace](https://huggingface.co/jhu-clsp)
- **Documentation**: [lm-eval docs](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs)

---

For issues with decoder evaluation, please refer to the [EleutherAI evaluation harness documentation](https://github.com/EleutherAI/lm-evaluation-harness) or open an issue in the [Ettin repository](https://github.com/jhu-clsp/ettin-encoder-vs-decoder). 