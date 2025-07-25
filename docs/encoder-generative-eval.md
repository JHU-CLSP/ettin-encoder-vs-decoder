# Encoder Evaluation on Generative Tasks

This guide covers evaluating Ettin encoder models on generative language modeling tasks using our modified [lm-evaluation-harness](https://github.com/orionw/lm-evaluation-harness) fork.

## Overview

While encoders are traditionally evaluated on discriminative tasks, evaluating them on generative tasks provides insights into their language modeling capabilities and allows for direct comparison with decoder models on the same benchmarks.

## Quick Start

### Installation

```bash
# Clone our lm-evaluation-harness fork with encoder support
git clone https://github.com/orionw/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

### Basic Usage

```bash
# Evaluate an Ettin encoder on common generative tasks
lm_eval --model hf \
    --model_args "pretrained=jhu-clsp/ettin-encoder-150m,is_encoder=True,add_bos_token=True" \
    --tasks hellaswag,arc_easy,arc_challenge,winogrande \
    --device cuda:0 \
    --batch_size 1 \
    --output_path results/ettin-encoder-150m
```



## Key Modifications

Our fork includes several modifications to enable encoder evaluation on generative tasks, the most notable being that batch size must be 1. Although future could work speed this up, in practice it was fast enough to not deal with the complexity of multi-batch decoding.

## Example Evaluation Scripts

### Single Model Evaluation

```bash
#!/bin/bash
# evaluate_encoder.sh

MODEL_NAME="jhu-clsp/ettin-encoder-400m"
OUTPUT_DIR="results/$(basename $MODEL_NAME)"

lm_eval --model hf \
    --model_args "pretrained=$MODEL_NAME,is_encoder=True,add_bos_token=True" \
    --tasks hellaswag,arc_easy,arc_challenge,winogrande,piqa,boolq \
    --device cuda:0 \
    --batch_size 1 \
    --output_path $OUTPUT_DIR \
    --log_samples \
    --write_out
```

### Multi-Model Comparison

```bash
#!/bin/bash
# compare_encoders.sh

MODELS=(
    "jhu-clsp/ettin-encoder-17m"
    "jhu-clsp/ettin-encoder-32m" 
    "jhu-clsp/ettin-encoder-68m"
    "jhu-clsp/ettin-encoder-150m"
    "jhu-clsp/ettin-encoder-400m"
    "jhu-clsp/ettin-encoder-1b"
)

TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,boolq"

for model in "${MODELS[@]}"; do
    echo "Evaluating $model..."
    output_dir="results/$(basename $model)"
    
    lm_eval --model hf \
        --model_args "pretrained=$model,is_encoder=True,add_bos_token=True" \
        --tasks $TASKS \
        --device cuda:0 \
        --batch_size 1 \
        --output_path $output_dir
done
```

### Cross-Objective Model Evaluation
This is done in the same manner as above, since the model is an encoder now.


### Batch Processing with Different Checkpoints

```python
# evaluate_checkpoints.py
import subprocess
import os

model_base = "jhu-clsp/ettin-encoder-400m"
checkpoints = ["step500000", "step550000", "step599525", "ext1000", "ext2000", "decay100"]

for checkpoint in checkpoints:
    model_name = f"{model_base}@{checkpoint}"
    output_dir = f"results/ettin-encoder-400m-{checkpoint}"
    
    cmd = [
        "lm_eval", "--model", "hf",
        "--model_args", f"pretrained={model_base},revision={checkpoint},is_encoder=True,add_bos_token=True",
        "--tasks", "hellaswag,arc_easy,arc_challenge",
        "--device", "cuda:0",
        "--batch_size", "1",
        "--output_path", output_dir
    ]
    
    print(f"Running evaluation for checkpoint {checkpoint}...")
    subprocess.run(cmd)
```

## Comparison with Decoders

### Head-to-Head Evaluation

```bash
# Compare encoder vs decoder of same size
DECODER_MODEL="jhu-clsp/ettin-decoder-400m"

for model in $ENCODER_MODEL $DECODER_MODEL; do
    echo "Evaluating $model..."
    lm_eval --model hf \
        --model_args pretrained=$model,add_bos_token=True \
        --tasks hellaswag,arc_easy,arc_challenge,winogrande \
        --device cuda:0 \
        --batch_size 8 \ 
        --output_path "results/$(basename $model)"
done
```

## Links and Resources

- **Fork Repository for Encoders**: [https://github.com/orionw/lm-evaluation-harness](https://github.com/orionw/lm-evaluation-harness)
- **Original Harness**: [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- **Model Collection**: [https://huggingface.co/jhu-clsp](https://huggingface.co/jhu-clsp)

---

For questions specific to encoder evaluation or our modifications, please open an issue in the [lm-evaluation-harness fork](https://github.com/orionw/lm-evaluation-harness) or the main [Ettin repository](https://github.com/jhu-clsp/ettin-encoder-vs-decoder). 