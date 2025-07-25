# Retrieval Evaluation

This directory contains scripts and documentation for evaluating Ettin encoder and decoder models on retrieval tasks, including fine-tuning on MS MARCO and evaluation on MTEB v2 English benchmarks.

## Quick Start

### Installation

```bash
# Install retrieval dependencies
pip install sentence-transformers mteb
```


## Training

The `train_st.py` script allows you to fine-tune Ettin models (both encoder and decoder variants) on the MS MARCO dataset for retrieval tasks. The script supports both encoder-only models and decoder models with configurable pooling strategies.

### Usage

```bash
python train_st.py --lr <learning_rate> --model_name <model_path> --model_out_dir <output_directory> --model_suffix <suffix> --accum_steps <steps> --bsize <batch_size> [additional_options]
```

### Required Arguments

- `--lr`: Learning rate (float)
- `--model_name`: Path or name of the base model to fine-tune
- `--model_out_dir`: Directory where trained models will be saved
- `--model_suffix`: Suffix to append to the run name for identification
- `--accum_steps`: Number of gradient accumulation steps (int)
- `--bsize`: Per-device training batch size (int)

### Optional Arguments

- `--gc_bsize`: Gradient cache batch size for CachedMultipleNegativesRankingLoss (default: 64)
- `--warmup_ratio`: Warmup ratio for learning rate scheduling (default: 0.05)
- `--scale`: Temperature scaling parameter for the loss function (default: 20)
- `--pooling`: Pooling strategy - choices: `lasttoken`, `mean`, `weightedmean` (default: `lasttoken`)
- `--fp16`: Enable FP16 mixed precision training
- `--bf16`: Enable BF16 mixed precision training
- `--resume_training`: Resume training from checkpoint
- `--decoder`: Use decoder model architecture instead of encoder

### Pooling Strategies

- **`lasttoken`**: Use the last token's representation (suitable for decoder models)
- **`mean`**: Average all token representations
- **`weightedmean`**: Weighted average of token representations

### Training Examples

#### Encoder Training
```bash
python train_st.py \
    --lr 3e-4 \
    --model_name "jhu-clsp/ettin-encoder-17m" \
    --model_out_dir "./models" \
    --model_suffix "encoder-v1" \
    --bf16
```

#### Decoder Model Training
```bash
python train_st.py \
    --lr 3e-4 \
    --model_name "jhu-clsp/ettin-decoder-17m" \
    --model_out_dir "./models" \
    --model_suffix "decoder-v1" \
    --decoder \
    --pooling lasttoken \
    --bf16
```

### Evaluation Examples
Evaluation was performed with [MTEB](https://github.com/embeddings-benchmark/mteb/tree/main). Please see their documentation for more. To reproduce on MTEB v2 Eng you can use the following:

```bash
import mteb
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "path_to_your_model"
benchmark = mteb.get_benchmark("MTEB(eng, v2)")
evaluation = mteb.MTEB(tasks=benchmark)
results = evaluation.run(model, output_folder=f"results/{model_name}")
```