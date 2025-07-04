# Ettin: an Open Suite of Paired Encoders and Decoders

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-red)](https://github.com/jhu-clsp/ettin-encoder-vs-decoder)
[![Models](https://img.shields.io/badge/🤗%20Hugging%20Face-24%20Models-blue)](https://huggingface.co/jhu-clsp)
[![Data](https://img.shields.io/badge/🤗%20Training%20Data-2T%20Tokens-green)](https://huggingface.co/datasets/jhu-clsp)

> 🎯 **TL;DR**: State-of-the-art paired encoder and decoder models (17M-1B params) trained identically for fair comparison with open data. Encoders beat ModernBERT. Decoders beat Llama 3.2/SmolLM2.

📄 [Paper (Coming Soon)](https://github.com/jhu-clsp/ettin-encoder-vs-decoder) | 🤗 [Model Collection](https://huggingface.co/jhu-clsp) | 📊 [Training Data](https://huggingface.co/datasets/jhu-clsp)

This repository contains the first collection of paired encoder-only and decoder-only models trained with identical data, architecture, and training recipes. Ettin enables fair comparisons between encoder and decoder architectures across multiple scales, providing state-of-the-art performance for open-data models in their respective size categories.

## Table of Contents
- [Performance Highlights](#-performance-highlights)
- [Quick Start](#-quick-start)
- [Model Family](#-model-family)
- [Getting Started](#-getting-started)
- [Training and Evaluation](#-training-and-evaluation)
- [Research Applications](#-research-applications)
- [Training Details](#training-details)
- [FAQ](#-faq)
- [Citation](#citation)

## 📊 Performance Highlights

### Encoder Tasks (vs. ModernBERT)
- **GLUE Average**: 88.9 vs 88.4 (Base), 90.8 vs 90.4 (Large)
- **MTEB v2 English Retrieval**: 45.7 vs 43.9 (Base), 48.4 vs 47.0 (Large)
- **Code Search and Long Context**: Superior performance on CodeSearchNet and MLDR

### Decoder Tasks (vs. SmolLM2 & Llama 3.2)
- **Average Score**: 46.2 vs 45.2 (SmolLM2-135M)
- **1B Model**: 59.0 vs 56.6 (Llama 3.2-1B)
- **Generative Tasks**: Competitive across all model sizes

### Key Finding
**Architecture-specific advantages persist**: A 400M encoder outperforms a 1B decoder on classification tasks, while a 400M decoder outperforms a 1B encoder on generation tasks.

## 🚀 Quick Start

### Installation
```bash
pip install torch>=1.9.0 transformers>=4.21.0
```

### 30-Second Examples

**Encoder for Classification/Embeddings:**
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-150m")
model = AutoModel.from_pretrained("jhu-clsp/ettin-encoder-150m")

# Example: Get embeddings
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1)
```

**Decoder for Text Generation:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-decoder-150m")
model = AutoModelForCausalLM.from_pretrained("jhu-clsp/ettin-decoder-150m")

# Example: Generate text
inputs = tokenizer("The future of AI is", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 🤖 Model Family

### Encoder Models

| Size | Model | Parameters | Best For | Download |
|:-----|:------|:-----------|:---------|:---------|
| XXS | [ettin-encoder-17m](https://huggingface.co/jhu-clsp/ettin-encoder-17m) | 17M | Mobile/Edge devices | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-17m) |
| XS | [ettin-encoder-32m](https://huggingface.co/jhu-clsp/ettin-encoder-32m) | 32M | Fast inference | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-32m) |
| Small | [ettin-encoder-68m](https://huggingface.co/jhu-clsp/ettin-encoder-68m) | 68M | Balanced performance | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-68m) |
| Base | [ettin-encoder-150m](https://huggingface.co/jhu-clsp/ettin-encoder-150m) | 150M | Standard use cases | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-150m) |
| Large | [ettin-encoder-400m](https://huggingface.co/jhu-clsp/ettin-encoder-400m) | 400M | High accuracy needs | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-400m) |
| XL | [ettin-encoder-1b](https://huggingface.co/jhu-clsp/ettin-encoder-1b) | 1B | Best performance | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-1b) |

### Decoder Models

| Size | Model | Parameters | Best For | Download |
|:-----|:------|:-----------|:---------|:---------|
| XXS | [ettin-decoder-17m](https://huggingface.co/jhu-clsp/ettin-decoder-17m) | 17M | Lightweight generation | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-17m) |
| XS | [ettin-decoder-32m](https://huggingface.co/jhu-clsp/ettin-decoder-32m) | 32M | Quick prototyping | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-32m) |
| Small | [ettin-decoder-68m](https://huggingface.co/jhu-clsp/ettin-decoder-68m) | 68M | Efficient generation | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-68m) |
| Base | [ettin-decoder-150m](https://huggingface.co/jhu-clsp/ettin-decoder-150m) | 150M | Standard generation | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-150m) |
| Large | [ettin-decoder-400m](https://huggingface.co/jhu-clsp/ettin-decoder-400m) | 400M | Quality generation | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-400m) |
| XL | [ettin-decoder-1b](https://huggingface.co/jhu-clsp/ettin-decoder-1b) | 1B | Best generation | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-1b) |

### Cross-Objective Models

These models demonstrate what happens when you continue training encoders as decoders (and vice versa). **Important**: Load these models using the architecture they were *converted to*, not their original architecture.

#### Encoders Trained from Decoders (Decoder → MLM)
**Load as encoders** using `AutoModel` or `AutoModelForMaskedLM`:

| Size | Model | Parameters | Description | Download |
|:-----|:------|:-----------|:------------|:---------|
| XXS | [ettin-encoder-from-decoder-17m](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-17m) | 17M | Decoder → MLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-17m) |
| XS | [ettin-encoder-from-decoder-32m](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-32m) | 32M | Decoder → MLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-32m) |
| Small | [ettin-encoder-from-decoder-68m](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-68m) | 68M | Decoder → MLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-68m) |
| Base | [ettin-encoder-from-decoder-150m](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-150m) | 150M | Decoder → MLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-150m) |
| Large | [ettin-encoder-from-decoder-400m](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-400m) | 400M | Decoder → MLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-400m) |
| XL | [ettin-encoder-from-decoder-1b](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-1b) | 1B | Decoder → MLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-1b) |

#### Decoders Trained from Encoders (Encoder → CLM)
**Load as decoders** using `AutoModelForCausalLM`:

| Size | Model | Parameters | Description | Download |
|:-----|:------|:-----------|:------------|:---------|
| XXS | [ettin-decoder-from-encoder-17m](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-17m) | 17M | Encoder → CLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-17m) |
| XS | [ettin-decoder-from-encoder-32m](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-32m) | 32M | Encoder → CLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-32m) |
| Small | [ettin-decoder-from-encoder-68m](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-68m) | 68M | Encoder → CLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-68m) |
| Base | [ettin-decoder-from-encoder-150m](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-150m) | 150M | Encoder → CLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-150m) |
| Large | [ettin-decoder-from-encoder-400m](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-400m) | 400M | Encoder → CLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-400m) |
| XL | [ettin-decoder-from-encoder-1b](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-1b) | 1B | Encoder → CLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-1b) |

### Accessing Training Checkpoints

Beyond the final models, we provide access to intermediate training checkpoints for research and analysis purposes. All raw training checkpoints are available in the [jhu-clsp/ettin-checkpoints](https://huggingface.co/datasets/jhu-clsp/ettin-checkpoints) dataset.

Each model repository contains multiple tagged versions representing different training stages:

- **`step{number}`** - Pretraining phase checkpoints (e.g., `step599525`, `step596528`)
- **`ext{number}`** - Extension/mid-training phase checkpoints (e.g., `ext1000`, `ext2000`) 
- **`decay{number}`** - Decay phase checkpoints (e.g., `decay100`, `decay500`)

```python
from transformers import AutoModelForCausalLM

# Load a specific pretraining checkpoint
model = AutoModelForCausalLM.from_pretrained(
    "jhu-clsp/ettin-decoder-400m", 
    revision="step590532"  # Specific checkpoint tag
)
```

## 🛠️ Getting Started

### Training Data

The complete training dataset is publicly available:

- **Pre-training Data**: [jhu-clsp/ettin-pretraining-data](https://huggingface.co/datasets/jhu-clsp/ettin-pretraining-data) - 1.7T tokens
- **Mid-training Data**: [jhu-clsp/ettin-extension-data](https://huggingface.co/datasets/jhu-clsp/ettin-extension-data) - 250B tokens  
- **Decay Phase Data**: [jhu-clsp/ettin-decay-data](https://huggingface.co/datasets/jhu-clsp/ettin-decay-data) - 100B tokens
- **Training Order**: [jhu-clsp/ettin-data-order](https://huggingface.co/datasets/jhu-clsp/ettin-data-order) - Batch-level training order

### Usage Examples

<details>
<summary><strong>Encoder: Masked Language Modeling</strong></summary>

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load MLM model
tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-150m")
model = AutoModelForMaskedLM.from_pretrained("jhu-clsp/ettin-encoder-150m")

def predict_masked_token(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions for [MASK] tokens
    mask_indices = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)
    predictions = outputs.logits[mask_indices]
    
    # Get top 5 predictions
    top_tokens = torch.topk(predictions, 5, dim=-1)
    return [tokenizer.decode(token) for token in top_tokens.indices[0]]

# Example
masked_text = "The capital of France is [MASK]."
predictions = predict_masked_token(masked_text)
print(f"Predictions: {predictions}")
```

</details>

<details>
<summary><strong>Decoder: Text Generation</strong></summary>
  
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer  
tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-decoder-150m")
model = AutoModelForCausalLM.from_pretrained("jhu-clsp/ettin-decoder-150m")

# Set pad token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_text(prompt, max_length=100, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "The future of artificial intelligence is"
generated = generate_text(prompt)
print(generated)
```

</details>

<details>
<summary><strong>Cross-Objective Models</strong></summary>

```python
# Encoder-from-decoder: Load as encoder
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-from-decoder-150m")
model = AutoModel.from_pretrained("jhu-clsp/ettin-encoder-from-decoder-150m")

# Decoder-from-encoder: Load as decoder  
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-decoder-from-encoder-150m")
model = AutoModelForCausalLM.from_pretrained("jhu-clsp/ettin-decoder-from-encoder-150m")
```

</details>

## 📋 Training and Evaluation

### Pre-training
For details on model pre-training, data preparation, and training recipes:
- **📖 [Pre-training Guide](pretraining/README.md)** - Complete training setup, data mixture, and ModernBERT recipe adaptation

### Evaluation

#### Encoder Evaluation
- **📊 [Encoder on Generative Tasks](docs/encoder-generative-eval.md)** - Evaluating encoders on language modeling tasks using our lm-evaluation-harness fork
- **🔍 [Encoder Retrieval Training](docs/retrieval.md)** - Fine-tuning on MS MARCO and evaluation on MTEB v2 English
- **🎯 [GLUE Evaluation](glue_evaluation/README.md)** - Comprehensive GLUE benchmark evaluation with fine-tuning scripts

#### Decoder Evaluation
- **🎯 [Decoder on Generative Tasks](docs/decoder-eval.md)** - Using EleutherAI evaluation harness (commit `867413f8677f00f6a817262727cbb041bf36192a`) for comprehensive generative task evaluation

#### Bias Evaluation
- **⚖️ [Gender Bias Evaluation](bias_eval/README.md)** - Comprehensive gender bias testing using Winogender dataset gotcha examples. Tests how well models handle counter-stereotypical pronouns in occupational contexts. Supports both encoder (MLM) and decoder (perplexity) evaluation methods.

### Quick Decoder Evaluation Example

```bash
# Clone the specific commit of lm-evaluation-harness
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout 867413f8677f00f6a817262727cbb041bf36192a
pip install -e .

# Run evaluation on Ettin decoder
lm_eval --model hf \
    --model_args pretrained=jhu-clsp/ettin-decoder-150m \
    --tasks hellaswag,arc_easy,arc_challenge,winogrande \
    --device cuda:0 \
    --batch_size 8
```

## 🔬 Research Applications

### What Makes Ettin Unique

Ettin provides the first **controlled comparison** of encoder vs. decoder architectures:

- **Identical Training Data**: Same 2T token mixture across all models
- **Matched Architectures**: Only attention patterns and objectives differ  
- **Open Everything**: Training data, model weights, and batch-level training order
- **Multiple Scales**: Fair comparison from 17M to 1B parameters
- **250+ Checkpoints**: Complete training trajectory analysis

### Key Research Findings

1. **Architecture Specialization Persists**: 
   - Encoders excel at classification/retrieval even vs. larger decoders
   - Decoders excel at generation even vs. larger encoders
   - A 400M encoder beats a 1B decoder on MNLI (89.2 vs 88.2)

2. **Cross-Training Limitations**: 
   - Converting decoder→encoder or encoder→decoder underperforms
   - 50B tokens of continued training insufficient to close gaps
   - Native training objective remains superior

3. **Scaling Insights**: 
   - Performance gaps between architectures widen with size
   - Decoder-from-encoder adaptation scales particularly poorly

### Use Cases for Researchers

- **Architecture Studies**: Compare encoder vs decoder capabilities fairly
- **Training Dynamics**: Analyze 250+ checkpoints with batch-level data ordering  
- **Scaling Laws**: Study how architectural advantages change with scale
- **Transfer Learning**: Investigate cross-objective training effectiveness
- **Replication Studies**: First open replication of ModernBERT training recipe

## Training Details

### Model Architecture

| Parameter | 17M | 32M | 68M | 150M | 400M | 1B |
|:----------|:----|:----|:----|:-----|:-----|:---|
| Layers | 7 | 10 | 19 | 22 | 28 | 28 |
| Hidden Size | 256 | 384 | 512 | 768 | 1024 | 1792 |
| Intermediate Size | 384 | 576 | 768 | 1152 | 2624 | 3840 |
| Attention Heads | 4 | 6 | 8 | 12 | 16 | 28 |

### Training Configuration

**Data:** High-quality mixture including DCLM, Dolma v1.7, scientific papers, code, and curated sources totaling 2T+ tokens

**Architecture Features:**
- Transformer with RoPE, GLU activations, and prenorm layers
- Context length: Up to 8K tokens
- Vocabulary: 50,368 tokens (ModernBERT tokenizer)
- Deep but efficient architectures following MobileLLM principles

**Training Phases:**
- **Pre-training**: 1.7T tokens with diverse data mixture
- **Mid-training**: 250B tokens with higher-quality filtered data and context extension to 8K
- **Decay phase**: 100B tokens with premium data sources

## ❓ FAQ

### Model Loading Issues

**Q: I'm getting an error that ModernBERT-decoder isn't found.**
**A:** Make sure you have the latest version of transformers installed:
```bash
# for the latest version until the official pypi release:
pip install git+https://github.com/huggingface/transformers.git
```

**Q: Which model should I choose for my task?**
**A:** 
- **Classification/Retrieval/Understanding**: Use encoder models
- **Text Generation/Chat/Completion**: Use decoder models  
- **Research on cross-training**: Use cross-objective models
- **Size selection**: Start with 150M for experimentation, scale up to 400M or 1B for production

**Q: How do I access training checkpoints?**
**A:** Each model has multiple git tags for different training stages. Use the `revision` parameter:
```python
model = AutoModel.from_pretrained("jhu-clsp/ettin-encoder-150m", revision="step500000")
```

**Q: Can I continue training these models?**
**A:** Yes! We provide raw checkpoints in the [jhu-clsp/ettin-checkpoints](https://huggingface.co/datasets/jhu-clsp/ettin-checkpoints) dataset that can be loaded into training frameworks.

**Q: What's the difference between cross-objective models and regular models?**
**A:** Cross-objective models started as one architecture (e.g., decoder) and were continued with a different objective (e.g., MLM). They demonstrate the limitations of cross-training and generally underperform native models.

**Q: How do I reproduce the paper results?**
**A:** See our evaluation guides:
- [Encoder Generative Eval](docs/encoder-generative-eval.md)
- [Retrieval Eval](docs/retrieval.md) 
- [GLUE Eval](glue_evaluation/README.md)
- [Decoder Eval](docs/decoder-eval.md)
- [Pre-training](pretraining/README.md)

## Citation

If you use Ettin models in your research, please cite our work:

```bibtex
@misc{weller2025seqvsseq,
      title={Seq vs Seq: An Open Suite of Paired Encoders and Decoders}, 
      author={Orion Weller and Kathryn Ricci and Marc Marone and Antoine Chaffin and Dawn Lawrie and Benjamin Van Durme},
      year={2025},
      note={Paper coming soon},
      url={https://github.com/jhu-clsp/ettin-encoder-vs-decoder}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Contact**: For questions about the models or research, please open an issue or contact the authors.