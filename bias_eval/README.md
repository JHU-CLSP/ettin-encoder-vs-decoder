# Bias Evaluation Toolkit

This directory contains tools for evaluating gender bias in language models using the Winogender dataset, specifically focusing on counter-stereotypical examples.

## Overview

The evaluation toolkit tests how well models handle counter-stereotypical gendered pronouns in occupational contexts. It uses the "gotcha" subset of the Winogender dataset, which contains examples where pronouns go against typical gender stereotypes for specific occupations.

## Files

- `eval.py` - Main evaluation script with `WinogenderEvaluator` class
- `batch_eval.py` - Batch evaluation script for multiple models

## Key Features

### Model Support
- **Encoder Models** (MLM): Masks pronouns and evaluates prediction probabilities
- **Decoder Models** (Causal LM): Compares perplexity across gender variants

### Evaluation Metrics
- **Gotcha Preference Rate**: How often the model prefers counter-stereotypical pronouns
- **Gender Distribution**: Breakdown of model predictions by gender
- **Baseline Comparisons**: Against uniform and dataset-based baselines

## Usage

### Single Model Evaluation

```bash
# Evaluate an encoder model
python eval.py --model_name "your-model-name" --model_type encoder --output_path results.json

# Evaluate a decoder model  
python eval.py --model_name "your-model-name" --model_type decoder --output_path results.json

# Show failure examples
python eval.py --model_name "your-model-name" --model_type encoder --show_failures

# Test pronoun replacement logic
python eval.py --model_name "your-model-name" --model_type encoder --test_pronouns
```

### Batch Evaluation

```bash
# Run evaluation on all configured models
python batch_eval.py
```

The batch script will:
- Load existing results to avoid re-evaluation
- Evaluate missing models incrementally
- Save results after each model
- Provide progress updates and summary statistics

## Evaluation Method

### For Encoder Models (MLM)
1. Takes counter-stereotypical sentences from Winogender "gotcha" dataset
2. Masks pronouns in each sentence
3. Evaluates model's pronoun predictions
4. Classifies predictions by gender (male/female/neutral)
5. Calculates preference rates for counter-stereotypical pronouns

### For Decoder Models (Causal LM)
1. Takes counter-stereotypical sentences from Winogender "gotcha" dataset
2. Generates alternative versions with different gender pronouns
3. Compares perplexity across all gender variants
4. Determines which gender version the model "prefers" (lowest perplexity)
5. Measures how often the counter-stereotypical version is preferred

## Example Output

```
GOTCHA EVALUATION SUMMARY
========================================
Model: your-model-name (encoder)
Testing counter-stereotypical vs stereotypical preferences

🎯 GOTCHA PREFERENCE RATE: 0.342
   (How often model prefers counter-stereotypical pronouns)
   Expected baseline: 0.289 (dataset-based)
   Uniform baseline: 0.333 (1/3 random)

Encoder Details:
Total gotcha examples tested: 120

Dataset gender distribution:
  male: 0.400 (40.0%)
  female: 0.350 (35.0%)
  neutral: 0.250 (25.0%)

Model prediction distribution:
  male: 0.425 (42.5%)
  female: 0.308 (30.8%)
  neutral: 0.267 (26.7%)

Gender-specific gotcha preference rates:
  male: 0.354
  female: 0.310
  neutral: 0.367
```

## Interpretation

- **Gotcha Preference Rate > 0.33**: Model shows some ability to handle counter-stereotypical cases
- **Rate ≈ baseline**: Model performs similarly to random chance
- **Rate < baseline**: Model shows strong stereotypical bias
- **Gender-specific rates**: Reveals which gender presentations the model handles better

## Model Configuration

The batch evaluation script is configured to test these model families:
- `ettin-encoder` models (17M to 1B parameters)
- `ettin-decoder` models (17M to 1B parameters)  
- Cross-trained variants (`enc-from-dec`, `dec-from-enc`)

To add new models, edit the `models` list in `batch_eval.py`.

## Dataset

Uses the Winogender dataset "gotcha" split from HuggingFace:
- Dataset: `oskarvanderwal/winogender`
- Split: `gotcha` (counter-stereotypical examples)
- Contains sentences with pronouns that go against occupational stereotypes