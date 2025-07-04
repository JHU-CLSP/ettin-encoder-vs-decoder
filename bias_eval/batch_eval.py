#!/usr/bin/env python3

import json
import os
import sys
import time
from pathlib import Path

# Add the current directory to path so we can import eval
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eval import WinogenderEvaluator

def load_existing_results(results_file):
    """Load existing results if file exists"""
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not load {results_file}, starting fresh")
            return {}
    return {}

def save_results(results, results_file):
    """Save results to file"""
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_file}")

def create_model_key(model_name, model_type):
    """Create a unique key for model + type combination"""
    return f"{model_name}_{model_type}"

def evaluate_model(model_name, model_type, max_samples=None):
    """Evaluate a single model and return results"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name} ({model_type})")
    print(f"{'='*80}")
    
    try:
        evaluator = WinogenderEvaluator(model_name, model_type)
        results = evaluator.run_evaluation(max_samples=max_samples)
        
        # Extract the key metrics for storage
        summary_stats = results['summary_stats']
        
        result_data = {
            'model_name': model_name,
            'model_type': model_type,
            'gotcha_preference_rate': summary_stats['gotcha_preference_rate'],
            'baseline_expected': summary_stats['baseline_expected'],
            'baseline_uniform': summary_stats['baseline_uniform'],
            'gender_distribution': summary_stats['gender_distribution'],
            'prediction_distribution': summary_stats['prediction_distribution'],
            'gender_gotcha_rates': summary_stats['gender_gotcha_rates'],
            'total_examples': summary_stats['gotcha_examples_tested'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add decoder-specific metrics if applicable
        if model_type == "decoder":
            result_data.update({
                'total_samples': summary_stats['total_samples'],
                'average_perplexity': summary_stats['average_perplexity'],
                'gender_perplexity': summary_stats['gender_perplexity']
            })
        
        print(f"\n✅ COMPLETED: {model_name} ({model_type})")
        print(f"   Gotcha preference rate: {summary_stats['gotcha_preference_rate']:.3f}")
        print(f"   Prediction distribution: {summary_stats['prediction_distribution']}")
        
        return result_data
        
    except Exception as e:
        print(f"\n❌ FAILED: {model_name} ({model_type})")
        print(f"   Error: {str(e)}")
        
        return {
            'model_name': model_name,
            'model_type': model_type,
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

def main():
    # Model lists
    models = [
        "blab-jhu/ettin-encoder-17m",
        "blab-jhu/ettin-encoder-32m", 
        "blab-jhu/ettin-encoder-66m",
        "blab-jhu/ettin-encoder-150m",
        "blab-jhu/ettin-encoder-400m",
        "blab-jhu/ettin-encoder-1b",

        # cross-train
        "jhu-clsp/ettin-enc-from-dec-17m",
        "jhu-clsp/ettin-enc-from-dec-66m",
        "jhu-clsp/ettin-enc-from-dec-32m",
        "jhu-clsp/ettin-enc-from-dec-150m",
        "jhu-clsp/ettin-enc-from-dec-400m",
        "jhu-clsp/ettin-enc-from-dec-1b",
    ]
    
    # Convert encoder models to decoder models for decoder evaluation
    decoder_models = []
    for model in models:
        if "ettin-encoder" in model:
            decoder_model = model.replace("ettin-encoder", "ettin-decoder")
            decoder_models.append(decoder_model)
        if "enc-from-dec" in model:
            decoder_model = model.replace("enc-from-dec", "dec-from-enc")
            decoder_models.append(decoder_model)
    
    results_file = "bias_evaluation_results.json"
    
    # Load existing results
    all_results = load_existing_results(results_file)
    
    print(f"Loaded {len(all_results)} existing results from {results_file}")
    
    # Create evaluation plan
    evaluations = []
    
    # Add encoder evaluations
    for model in models:
        key = create_model_key(model, "encoder")
        if key not in all_results:
            evaluations.append((model, "encoder"))
        else:
            print(f"⏭️  Skipping {model} (encoder) - already evaluated")
    
    # Add decoder evaluations
    for model in decoder_models:
        key = create_model_key(model, "decoder")
        if key not in all_results:
            evaluations.append((model, "decoder"))
        else:
            print(f"⏭️  Skipping {model} (decoder) - already evaluated")
    
    print(f"\n📋 Evaluation Plan: {len(evaluations)} models to evaluate")
    
    if not evaluations:
        print("🎉 All models already evaluated!")
        return
    
    # Confirm before starting
    print("\nModels to evaluate:")
    for i, (model, model_type) in enumerate(evaluations, 1):
        print(f"  {i:2d}. {model} ({model_type})")
    
    response = input(f"\nProceed with {len(evaluations)} evaluations? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Aborted.")
        return
    
    # Run evaluations
    for i, (model_name, model_type) in enumerate(evaluations, 1):
        print(f"\n🚀 Progress: {i}/{len(evaluations)}")
        
        # Evaluate model
        result = evaluate_model(model_name, model_type, max_samples=None)
        
        # Save result
        key = create_model_key(model_name, model_type)
        all_results[key] = result
        
        # Save incrementally
        save_results(all_results, results_file)
        
        # Small delay to be nice to system
        if i < len(evaluations):
            time.sleep(2)
    
    print(f"\n🎉 BATCH EVALUATION COMPLETE!")
    print(f"📊 Total results: {len(all_results)}")
    print(f"💾 Results saved to: {results_file}")
    
    # Summary statistics
    successful = sum(1 for r in all_results.values() if 'error' not in r)
    failed = len(all_results) - successful
    
    print(f"\n📈 Summary:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    
    if successful > 0:
        # Show best performing models
        successful_results = [r for r in all_results.values() if 'error' not in r]
        
        encoder_results = [r for r in successful_results if r['model_type'] == 'encoder']
        decoder_results = [r for r in successful_results if r['model_type'] == 'decoder']
        
        if encoder_results:
            best_encoder = max(encoder_results, key=lambda x: x['gotcha_preference_rate'])
            print(f"\n🏆 Best Encoder: {best_encoder['model_name']}")
            print(f"   Gotcha rate: {best_encoder['gotcha_preference_rate']:.3f}")
        
        if decoder_results:
            best_decoder = max(decoder_results, key=lambda x: x['gotcha_preference_rate'])
            print(f"🏆 Best Decoder: {best_decoder['model_name']}")
            print(f"   Gotcha rate: {best_decoder['gotcha_preference_rate']:.3f}")

if __name__ == "__main__":
    main() 