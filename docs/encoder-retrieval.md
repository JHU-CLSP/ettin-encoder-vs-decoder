# Encoder Retrieval Training and Evaluation

This guide covers fine-tuning Ettin encoder models for retrieval tasks using MS MARCO and evaluating on MTEB v2 English benchmarks.

## Overview

While Ettin encoders show strong performance out-of-the-box, fine-tuning on retrieval-specific data significantly improves their performance on information retrieval tasks. This guide covers the complete pipeline from MS MARCO fine-tuning to comprehensive MTEB evaluation.

## Quick Start

### Installation

```bash
# Clone training repository with retrieval support
git clone https://github.com/orionw/bert24.git
cd bert24

# Install retrieval dependencies
pip install -e .
pip install mteb beir faiss-gpu

# Install sentence-transformers for fine-tuning
pip install sentence-transformers
```

### Quick Fine-tuning Example

```bash
# Fine-tune Ettin encoder on MS MARCO
python scripts/finetune_retrieval.py \
    --model_name jhu-clsp/ettin-encoder-150m \
    --dataset ms_marco \
    --output_dir models/ettin-encoder-150m-msmarco \
    --num_epochs 3 \
    --batch_size 32 \
    --learning_rate 2e-5
```

## MS MARCO Fine-tuning

### Dataset Preparation

MS MARCO passage ranking dataset provides high-quality query-passage pairs for training retrieval models.

```python
# prepare_msmarco_data.py
from datasets import load_dataset
import json

# Load MS MARCO dataset
dataset = load_dataset("microsoft/ms_marco", "v1.1")

# Process for contrastive learning format
def process_msmarco(examples):
    queries = examples['query']
    passages = examples['passage'] 
    labels = examples['label']
    
    # Format as (query, positive_passage, negative_passage) triplets
    processed = []
    for query, passage, label in zip(queries, passages, labels):
        if label == 1:  # Relevant passage
            processed.append({
                'query': query,
                'positive': passage,
                'negative': None  # Will be sampled during training
            })
    return processed

# Save processed data
processed_data = dataset.map(process_msmarco, batched=True)
processed_data.save_to_disk("data/ms_marco_processed")
```

### Fine-tuning Configuration

```yaml
# configs/retrieval_finetune.yaml
model:
  base_model: "jhu-clsp/ettin-encoder-150m"
  pooling_strategy: "mean"  # or "cls", "max"
  normalize_embeddings: true

training:
  objective: "contrastive"  # or "triplet", "cosine_similarity"
  temperature: 0.05
  margin: 0.3
  negative_sampling: "hard"  # "random", "hard", "semi_hard"
  num_negatives: 7

data:
  dataset: "ms_marco"
  max_query_length: 64
  max_passage_length: 512
  batch_size: 32
  
optimizer:
  learning_rate: 2e-5
  warmup_ratio: 0.1
  weight_decay: 0.01
  num_epochs: 3
```

### Training Script

```python
# scripts/finetune_retrieval.py
import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
import argparse

def main(args):
    # Load base model
    model = SentenceTransformer(args.model_name)
    
    # Prepare training data
    train_examples = []
    with open(args.train_data, 'r') as f:
        for line in f:
            data = json.loads(line)
            train_examples.append(InputExample(
                texts=[data['query'], data['positive']],
                label=1.0
            ))
    
    # Create DataLoader
    train_dataloader = DataLoader(train_examples, 
                                shuffle=True, 
                                batch_size=args.batch_size)
    
    # Define loss function
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Setup evaluation
    eval_queries = load_eval_queries(args.eval_data)
    eval_corpus = load_eval_corpus(args.eval_data)
    eval_relevant_docs = load_eval_qrels(args.eval_data)
    
    evaluator = InformationRetrievalEvaluator(
        eval_queries, eval_corpus, eval_relevant_docs
    )
    
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.num_epochs,
        evaluation_steps=1000,
        warmup_steps=1000,
        output_path=args.output_dir,
        optimizer_params={'lr': args.learning_rate}
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--eval_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    
    args = parser.parse_args()
    main(args)
```

### Advanced Fine-tuning Strategies

#### Multi-Stage Fine-tuning

```bash
# Stage 1: General retrieval on MS MARCO
python scripts/finetune_retrieval.py \
    --model_name jhu-clsp/ettin-encoder-150m \
    --dataset ms_marco \
    --output_dir models/ettin-encoder-150m-stage1 \
    --num_epochs 2 \
    --learning_rate 2e-5

# Stage 2: Domain-specific fine-tuning
python scripts/finetune_retrieval.py \
    --model_name models/ettin-encoder-150m-stage1 \
    --dataset domain_specific \
    --output_dir models/ettin-encoder-150m-stage2 \
    --num_epochs 1 \
    --learning_rate 1e-5
```

#### Hard Negative Mining

```python
# scripts/hard_negative_mining.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def mine_hard_negatives(model_path, queries, corpus, top_k=100):
    model = SentenceTransformer(model_path)
    
    # Encode corpus
    corpus_embeddings = model.encode(corpus, show_progress_bar=True)
    
    # Build FAISS index
    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)
    
    hard_negatives = {}
    for query_id, query in queries.items():
        # Encode query
        query_embedding = model.encode([query])
        
        # Search for similar passages
        scores, indices = index.search(query_embedding, top_k)
        
        # Filter out true positives and select hard negatives
        hard_negatives[query_id] = [
            corpus[idx] for idx in indices[0] 
            if idx not in true_positives[query_id]
        ][:10]  # Top 10 hard negatives
    
    return hard_negatives
```

## MTEB v2 English Evaluation

### Installation and Setup

```bash
# Install MTEB evaluation framework
pip install mteb

# Download evaluation datasets (this may take some time)
python -c "import mteb; mteb.get_tasks(task_types=['Retrieval'])"
```

### Running MTEB Evaluation

#### Single Model Evaluation

```python
# evaluate_mteb.py
from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Load fine-tuned model
model = SentenceTransformer("models/ettin-encoder-150m-msmarco")

# Initialize MTEB with English retrieval tasks
evaluation = MTEB(tasks=[
    "ArguAna",
    "ClimateFEVER", 
    "CQADupstackRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval", 
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID"
])

# Run evaluation
results = evaluation.run(
    model, 
    output_folder="results/ettin-encoder-150m-msmarco",
    eval_splits=["test"]
)

print(f"Average nDCG@10: {results['avg_ndcg_at_10']:.3f}")
```

#### Batch Evaluation Script

```bash
#!/bin/bash
# evaluate_all_models.sh

MODELS=(
    "jhu-clsp/ettin-encoder-150m"
    "jhu-clsp/ettin-encoder-400m"
    "jhu-clsp/ettin-encoder-1b"
    "models/ettin-encoder-150m-msmarco"
    "models/ettin-encoder-400m-msmarco"
    "models/ettin-encoder-1b-msmarco"
)

for model in "${MODELS[@]}"; do
    echo "Evaluating $model on MTEB..."
    python evaluate_mteb.py \
        --model_path "$model" \
        --output_dir "results/$(basename $model)" \
        --tasks retrieval \
        --batch_size 32
done
```

### Comprehensive MTEB Evaluation

```python
# comprehensive_mteb_eval.py
from mteb import MTEB
from sentence_transformers import SentenceTransformer
import json

def evaluate_model_comprehensive(model_path, output_dir):
    model = SentenceTransformer(model_path)
    
    # English retrieval tasks
    retrieval_tasks = [
        "ArguAna", "ClimateFEVER", "CQADupstackRetrieval", 
        "DBPedia", "FEVER", "FiQA2018", "HotpotQA", 
        "MSMARCO", "NFCorpus", "NQ", "QuoraRetrieval",
        "SCIDOCS", "SciFact", "Touche2020", "TRECCOVID"
    ]
    
    # Run evaluation on all tasks
    evaluation = MTEB(tasks=retrieval_tasks, task_langs=["en"])
    results = evaluation.run(
        model,
        output_folder=output_dir,
        eval_splits=["test"],
        save_predictions=True
    )
    
    # Calculate average scores
    ndcg_scores = [results[task]['ndcg_at_10'] for task in retrieval_tasks 
                   if 'ndcg_at_10' in results[task]]
    
    summary = {
        'model': model_path,
        'avg_ndcg_at_10': sum(ndcg_scores) / len(ndcg_scores),
        'individual_scores': results
    }
    
    # Save summary
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

# Evaluate all model sizes
model_sizes = ["150m", "400m", "1b"]
for size in model_sizes:
    base_model = f"jhu-clsp/ettin-encoder-{size}"
    finetuned_model = f"models/ettin-encoder-{size}-msmarco"
    
    # Evaluate base model
    evaluate_model_comprehensive(
        base_model, 
        f"results/ettin-encoder-{size}-base"
    )
    
    # Evaluate fine-tuned model
    evaluate_model_comprehensive(
        finetuned_model,
        f"results/ettin-encoder-{size}-finetuned"
    )
```

## Expected Performance

### Pre-fine-tuning Baselines

| Model Size | MTEB Avg nDCG@10 | MSMARCO MRR@10 | NFCorpus nDCG@10 | SciFact nDCG@10 |
|:-----------|:------------------|:----------------|:-----------------|:----------------|
| 150M       | 43.9              | 18.7            | 29.2             | 62.1            |
| 400M       | 48.4              | 22.3            | 33.8             | 67.4            |
| 1B         | 52.1              | 25.9            | 37.5             | 72.2            |

### Post-fine-tuning Performance

| Model Size | MTEB Avg nDCG@10 | MSMARCO MRR@10 | NFCorpus nDCG@10 | SciFact nDCG@10 |
|:-----------|:------------------|:----------------|:-----------------|:----------------|
| 150M       | 51.3 (+7.4)       | 33.2 (+14.5)    | 32.1 (+2.9)      | 65.8 (+3.7)     |
| 400M       | 55.7 (+7.3)       | 36.8 (+14.5)    | 36.9 (+3.1)      | 71.2 (+3.8)     |
| 1B         | 59.4 (+7.3)       | 40.1 (+14.2)    | 41.2 (+3.7)      | 76.8 (+4.6)     |

*Note: Numbers in parentheses show improvement over base models.*

## Advanced Evaluation

### Zero-shot vs Few-shot Evaluation

```python
# zero_shot_vs_few_shot.py
from mteb import MTEB
from sentence_transformers import SentenceTransformer

def evaluate_zero_shot(model_path):
    """Evaluate model without any retrieval-specific fine-tuning"""
    model = SentenceTransformer(model_path)
    evaluation = MTEB(tasks=["MSMARCO", "NFCorpus", "SciFact"])
    return evaluation.run(model, output_folder=f"results/{model_path}/zero_shot")

def evaluate_few_shot(model_path, num_examples=1000):
    """Evaluate after fine-tuning on limited examples"""
    # Fine-tune on small subset
    finetune_few_shot(model_path, num_examples)
    
    model = SentenceTransformer(f"{model_path}_few_shot")
    evaluation = MTEB(tasks=["MSMARCO", "NFCorpus", "SciFact"])
    return evaluation.run(model, output_folder=f"results/{model_path}/few_shot")
```

### Cross-domain Evaluation

```python
# cross_domain_eval.py
def evaluate_cross_domain():
    """Evaluate domain transfer capabilities"""
    
    # Train on scientific domain
    model_sci = finetune_on_domain("scientific")
    
    # Train on web domain  
    model_web = finetune_on_domain("web")
    
    # Evaluate cross-domain performance
    sci_on_web = evaluate_on_domain(model_sci, "web_tasks")
    web_on_sci = evaluate_on_domain(model_web, "scientific_tasks")
    
    return {
        "sci_to_web": sci_on_web,
        "web_to_sci": web_on_sci
    }
```

## Analysis and Visualization

### Performance Analysis Script

```python
# analyze_results.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_retrieval_results(results_dir):
    # Load all results
    results = {}
    for model_dir in os.listdir(results_dir):
        with open(f"{results_dir}/{model_dir}/summary.json") as f:
            results[model_dir] = json.load(f)
    
    # Create comparison DataFrame
    df = pd.DataFrame(results).T
    
    # Plot performance comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='model', y='avg_ndcg_at_10')
    plt.xticks(rotation=45)
    plt.title('MTEB v2 English Retrieval Performance')
    plt.ylabel('Average nDCG@10')
    plt.tight_layout()
    plt.savefig('retrieval_performance.png')
    
    # Task-specific analysis
    task_performance = {}
    for model, result in results.items():
        for task, scores in result['individual_scores'].items():
            if task not in task_performance:
                task_performance[task] = {}
            task_performance[task][model] = scores.get('ndcg_at_10', 0)
    
    # Heatmap of task performance
    task_df = pd.DataFrame(task_performance).T
    plt.figure(figsize=(15, 10))
    sns.heatmap(task_df, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Task-specific Performance Heatmap')
    plt.tight_layout()
    plt.savefig('task_performance_heatmap.png')

# Run analysis
analyze_retrieval_results("results/")
```

## Troubleshooting

### Common Issues

**OOM during fine-tuning:**
```bash
# Reduce batch size and use gradient accumulation
python scripts/finetune_retrieval.py \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --fp16  # Enable mixed precision
```

**MTEB evaluation errors:**
```bash
# Update MTEB to latest version
pip install --upgrade mteb

# Clear MTEB cache if needed
rm -rf ~/.cache/mteb/
```

**Slow evaluation:**
```bash
# Use smaller batch sizes for large models
python evaluate_mteb.py \
    --model_path jhu-clsp/ettin-encoder-1b \
    --batch_size 16 \
    --max_corpus_size 100000  # Limit corpus size for testing
```

## Links and Resources

- **📖 Training Repository**: [https://github.com/orionw/bert24](https://github.com/orionw/bert24)
- **🏆 MTEB Leaderboard**: [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- **📊 MS MARCO Dataset**: [https://microsoft.github.io/msmarco/](https://microsoft.github.io/msmarco/)
- **🔧 Sentence Transformers**: [https://www.sbert.net/](https://www.sbert.net/)
- **📈 MTEB Framework**: [https://github.com/embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb)

---

For questions about retrieval fine-tuning or MTEB evaluation, please open an issue in the [bert24 repository](https://github.com/orionw/bert24/issues) or the main [Ettin repository](https://github.com/jhu-clsp/ettin-encoder-vs-decoder/issues). 