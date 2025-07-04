import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import pandas as pd
from collections import defaultdict
import numpy as np
import re
from tqdm import tqdm
from datasets import load_dataset
import json


class WinogenderEvaluator:
    def __init__(self, model_name, model_type="encoder"):
        """
        Initialize evaluator for Winogender dataset
        
        Args:
            model_name: HuggingFace model name
            model_type: "encoder" for MLM or "decoder" for causal LM
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if model_type == "encoder":
            self.model = AutoModelForMaskedLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        elif model_type == "decoder":
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise ValueError("model_type must be 'encoder' or 'decoder'")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Pronoun mappings
        self.pronoun_mappings = {
            "male": {"NOM": "he", "POSS": "his", "ACC": "him"},
            "female": {"NOM": "she", "POSS": "her", "ACC": "her"}, 
            "neutral": {"NOM": "they", "POSS": "their", "ACC": "them"}
        }
        
        # Load gotcha examples - these are the counter-stereotypical cases we want to test
        self.gotcha_examples = self._load_gotcha_examples()

    def _replace_pronouns(self, sentence, from_gender, to_gender):
        """Replace pronouns in sentence from one gender to another"""
        result = sentence
        
        from_pronouns = self.pronoun_mappings[from_gender]
        to_pronouns = self.pronoun_mappings[to_gender]
        
        # Replace each pronoun type using word boundaries
        for pronoun_type in ["NOM", "POSS", "ACC"]:
            from_pronoun = from_pronouns[pronoun_type]
            to_pronoun = to_pronouns[pronoun_type]
            
            # Use regex with word boundaries to avoid partial word matches
            # Replace both capitalized and lowercase versions
            result = re.sub(r'\b' + re.escape(from_pronoun.capitalize()) + r'\b', 
                          to_pronoun.capitalize(), result)
            result = re.sub(r'\b' + re.escape(from_pronoun) + r'\b', 
                          to_pronoun, result)
        
        # Fix grammar for neutral pronouns
        if to_gender == "neutral":
            result = result.replace("they was", "they were")
            result = result.replace("They was", "They were")
            result = result.replace("they is", "they are") 
            result = result.replace("They is", "They are")
            result = result.replace("they has", "they have")
            result = result.replace("They has", "They have")
            
        return result

    def _load_gotcha_examples(self):
        """Load gotcha examples (counter-stereotypical cases) from Winogender dataset"""
        try:
            # Load only the gotcha split - these are counter-stereotypical examples
            dataset = load_dataset("oskarvanderwal/winogender", "gotcha", split="test")
            
            gotcha_examples = []
            for example in dataset:
                gotcha_examples.append({
                    'sentid': example['sentid'],
                    'sentence': example['sentence'],
                    'pronoun': example['pronoun'], 
                    'occupation': example['occupation'],
                    'participant': example['participant'],
                    'gender': example['gender'],
                    'target': example['target']
                })
            
            print(f"Loaded {len(gotcha_examples)} gotcha examples from Winogender dataset")
            return gotcha_examples
            
        except Exception as e:
            print(f"Warning: Could not load Winogender gotcha dataset: {e}")
            print("Please ensure you have access to the oskarvanderwal/winogender dataset")
            return []

    def run_evaluation(self, output_path=None, max_samples=None):
        """Run evaluation on Winogender gotcha examples only"""
        
        if not self.gotcha_examples:
            raise ValueError("No gotcha examples loaded. Please check dataset access.")
        
        examples_to_test = self.gotcha_examples
        if max_samples:
            examples_to_test = examples_to_test[:max_samples]
        
        results = {
            'mlm_results': [] if self.model_type == "encoder" else None,
            'decoder_results': [] if self.model_type == "decoder" else None,
            'decoder_comparisons': [] if self.model_type == "decoder" else None,
            'mlm_comparisons': [] if self.model_type == "encoder" else None,
            'summary_stats': {}
        }
        
        print(f"Evaluating {len(examples_to_test)} gotcha examples with {self.model_type} model: {self.model_name}")
        
        if self.model_type == "encoder":
            # For encoders, mask pronouns in gotcha sentences and see what model predicts
            for example in tqdm(examples_to_test):
                sentence = example['sentence']
                gotcha_gender = example['gender']
                occupation = example['occupation']
                participant = example['participant']
                sentid = example['sentid']
                
                # Find and mask each pronoun in the gotcha sentence
                mlm_results = self.evaluate_mlm_simple(sentence)
                
                if not mlm_results:
                    continue  # Skip if no pronouns found
                
                # Analyze predictions for each masked pronoun position
                for result in mlm_results:
                    result.update({
                        'sentid': sentid,
                        'occupation': occupation,
                        'participant': participant,
                        'gotcha_gender': gotcha_gender,
                        'sentence': sentence
                    })
                    results['mlm_results'].append(result)
                
                # Aggregate results for this example
                gender_scores = {'male': 0.0, 'female': 0.0, 'neutral': 0.0}
                total_weight = 0.0
                
                for result in mlm_results:
                    # Weight by the total probability mass of gender-specific pronouns
                    weight = result['total_gender_prob']
                    if weight > 0:
                        gender_scores['male'] += result['male_prob'] * weight
                        gender_scores['female'] += result['female_prob'] * weight  
                        gender_scores['neutral'] += result['neutral_prob'] * weight
                        total_weight += weight
                
                # Normalize scores
                if total_weight > 0:
                    for gender in gender_scores:
                        gender_scores[gender] /= total_weight
                
                # Determine model's preferred gender
                predicted_gender = max(gender_scores.keys(), key=lambda g: gender_scores[g])
                
                # Create comparison
                comparison = {
                    'sentid': sentid,
                    'occupation': occupation,
                    'participant': participant,
                    'gotcha_gender': gotcha_gender,
                    'sentence': sentence,
                    'male_prob': gender_scores['male'],
                    'female_prob': gender_scores['female'],
                    'neutral_prob': gender_scores['neutral'],
                    'predicted_gender': predicted_gender,
                    'gotcha_preferred': predicted_gender == gotcha_gender
                }
                
                results['mlm_comparisons'].append(comparison)
        
        elif self.model_type == "decoder":
            # For decoders, compare perplexity across all gender versions  
            for example in tqdm(examples_to_test):
                sentence = example['sentence']
                gotcha_gender = example['gender']
                occupation = example['occupation']
                participant = example['participant']
                sentid = example['sentid']
                
                # Generate sentences for all three genders for comparison
                gender_results = {}
                
                for gender in ["male", "female", "neutral"]:
                    if gender == gotcha_gender:
                        # Use the actual gotcha sentence
                        test_sentence = sentence
                    else:
                        # Generate alternative version by replacing pronouns
                        test_sentence = self._replace_pronouns(sentence, gotcha_gender, gender)
                    
                    sample_id = f"{sentid}.{gender}.comparison"
                    
                    # Evaluate decoder
                    decoder_result = self.evaluate_decoder(test_sentence)
                    decoder_result.update({
                        'sample_id': sample_id,
                        'sentid': sentid,
                        'occupation': occupation,
                        'participant': participant,
                        'gender': gender,
                        'sentence': test_sentence,
                        'gotcha_gender': gotcha_gender,
                        'is_gotcha_version': gender == gotcha_gender
                    })
                    results['decoder_results'].append(decoder_result)
                    gender_results[gender] = decoder_result
                
                # Create comparison
                comparison = {
                    'sentid': sentid,
                    'occupation': occupation,
                    'participant': participant,
                    'gotcha_gender': gotcha_gender,
                    'male_perplexity': gender_results['male']['perplexity'],
                    'female_perplexity': gender_results['female']['perplexity'],
                    'neutral_perplexity': gender_results['neutral']['perplexity']
                }
                
                # Determine model's "prediction" (lowest perplexity)
                perplexities = {
                    'male': gender_results['male']['perplexity'],
                    'female': gender_results['female']['perplexity'],
                    'neutral': gender_results['neutral']['perplexity']
                }
                predicted_gender = min(perplexities.keys(), key=lambda g: perplexities[g])
                comparison['predicted_gender'] = predicted_gender
                
                # Key metric: Did model prefer the gotcha (counter-stereotypical) version?
                comparison['gotcha_preferred'] = predicted_gender == gotcha_gender
                
                results['decoder_comparisons'].append(comparison)
        
        # Calculate summary statistics
        if self.model_type == "encoder":
            self._calculate_mlm_gotcha_stats(results)
        elif self.model_type == "decoder":
            self._calculate_decoder_gotcha_stats(results)
        
        # Save results
        if output_path:
            self._save_results(results, output_path)
        
        return results

    def create_masked_sentence(self, sentence):
        """Create masked version for MLM testing"""
        # Find pronouns to mask
        pronouns = ["he", "she", "they", "him", "her", "them", "his", "their"]
        masked_versions = []
        
        tokens = sentence.split()
        for i, token in enumerate(tokens):
            # Remove punctuation for comparison
            clean_token = re.sub(r'[^\w]', '', token.lower())
            if clean_token in pronouns:
                masked_tokens = tokens.copy()
                # Preserve punctuation
                if token[-1] in ".,!?;:":
                    masked_tokens[i] = self.tokenizer.mask_token + token[-1]
                else:
                    masked_tokens[i] = self.tokenizer.mask_token
                
                masked_sentence = " ".join(masked_tokens)
                masked_versions.append({
                    'masked_sentence': masked_sentence,
                    'target_pronoun': clean_token,
                    'position': i,
                    'original_token': token
                })
        
        return masked_versions

    def evaluate_mlm_simple(self, sentence):
        """Simple MLM evaluation: mask pronouns and classify predictions by gender"""
        # Create masked versions
        masked_versions = self.create_masked_sentence(sentence)
        results = []
        
        # All pronouns by gender for classification
        male_pronouns = set(self.pronoun_mappings['male'].values())
        female_pronouns = set(self.pronoun_mappings['female'].values())
        neutral_pronouns = set(self.pronoun_mappings['neutral'].values())
        all_gender_pronouns = male_pronouns | female_pronouns | neutral_pronouns
        
        for masked_info in masked_versions:
            masked_sentence = masked_info['masked_sentence']
            original_pronoun = masked_info['target_pronoun']
            
            # Tokenize
            inputs = self.tokenizer(masked_sentence, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Find mask position
            mask_token_id = self.tokenizer.mask_token_id
            mask_positions = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]
            
            if len(mask_positions) == 0:
                continue
                
            mask_pos = mask_positions[0]
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits[0, mask_pos]
                probabilities = torch.softmax(predictions, dim=-1)
            
            # Get top predictions
            top_k = 20  # Look at more predictions to capture all pronouns
            top_probs, top_indices = torch.topk(probabilities, top_k)
            top_tokens = [self.tokenizer.decode([idx]).strip().lower() for idx in top_indices]
            
            # Classify predictions by gender
            male_prob = 0.0
            female_prob = 0.0
            neutral_prob = 0.0
            total_gender_prob = 0.0
            
            pronoun_details = []
            
            for token, prob in zip(top_tokens, top_probs):
                prob_val = prob.item()
                if token in all_gender_pronouns:
                    total_gender_prob += prob_val
                    pronoun_details.append((token, prob_val))
                    
                    if token in male_pronouns:
                        male_prob += prob_val
                    elif token in female_pronouns:
                        female_prob += prob_val
                    elif token in neutral_pronouns:
                        neutral_prob += prob_val
            
            # Normalize gender probabilities
            if total_gender_prob > 0:
                male_prob /= total_gender_prob
                female_prob /= total_gender_prob
                neutral_prob /= total_gender_prob
            
            # Determine which gender the original pronoun belongs to
            original_gender = None
            if original_pronoun in male_pronouns:
                original_gender = 'male'
            elif original_pronoun in female_pronouns:
                original_gender = 'female'
            elif original_pronoun in neutral_pronouns:
                original_gender = 'neutral'
            
            # Find rank of original pronoun
            original_rank = None
            original_prob = None
            for rank, (token, prob_val) in enumerate(zip(top_tokens, top_probs)):
                if token == original_pronoun:
                    original_rank = rank + 1
                    original_prob = prob_val.item()
                    break
            
            result = {
                'masked_sentence': masked_sentence,
                'original_pronoun': original_pronoun,
                'original_gender': original_gender,
                'original_rank': original_rank,
                'original_prob': original_prob,
                'male_prob': male_prob,
                'female_prob': female_prob,
                'neutral_prob': neutral_prob,
                'total_gender_prob': total_gender_prob,
                'pronoun_details': pronoun_details,
                'top_predictions': [(token, prob.item()) for token, prob in zip(top_tokens[:10], top_probs[:10])]
            }
            
            results.append(result)
        
        return results

    def evaluate_mlm(self, sentence, target_pronoun):
        """Evaluate masked language model on pronoun prediction"""
        masked_versions = self.create_masked_sentence(sentence)
        results = []
        
        for masked_info in masked_versions:
            if masked_info['target_pronoun'] == target_pronoun.lower():
                masked_sentence = masked_info['masked_sentence']
                
                # Tokenize
                inputs = self.tokenizer(masked_sentence, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Find mask position
                mask_token_id = self.tokenizer.mask_token_id
                mask_positions = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]
                
                if len(mask_positions) == 0:
                    continue
                    
                mask_pos = mask_positions[0]
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = outputs.logits[0, mask_pos]
                    probabilities = torch.softmax(predictions, dim=-1)
                
                # Get top predictions
                top_k = 10
                top_probs, top_indices = torch.topk(probabilities, top_k)
                top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]
                
                # Check if target pronoun is in predictions
                target_rank = None
                target_prob = None
                
                for rank, (token, prob) in enumerate(zip(top_tokens, top_probs)):
                    if token.strip().lower() == target_pronoun.lower():
                        target_rank = rank + 1
                        target_prob = prob.item()
                        break
                
                results.append({
                    'masked_sentence': masked_sentence,
                    'target_pronoun': target_pronoun,
                    'target_rank': target_rank,
                    'target_prob': target_prob,
                    'top_predictions': [(token.strip(), prob.item()) for token, prob in zip(top_tokens[:5], top_probs[:5])]
                })
        
        return results

    def evaluate_decoder(self, sentence, context_length=None):
        """Evaluate causal language model on full sentence perplexity"""
        # Tokenize full sentence for perplexity calculation
        inputs = self.tokenizer(sentence, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Calculate perplexity of full sentence
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            perplexity = torch.exp(outputs.loss).item()
            
            # Also calculate log-likelihood for more detailed analysis
            log_likelihood = -outputs.loss.item() * inputs['input_ids'].shape[1]
        
        # Optional: Generate continuation for qualitative analysis
        continuation_result = None
        if context_length is not None:
            tokens = sentence.split()
            context = " ".join(tokens[:context_length])
            target = " ".join(tokens[context_length:])
            
            context_inputs = self.tokenizer(context, return_tensors="pt")
            context_inputs = {k: v.to(self.device) for k, v in context_inputs.items()}
            
            with torch.no_grad():
                generated = self.model.generate(
                    context_inputs['input_ids'],
                    max_length=context_inputs['input_ids'].shape[1] + len(target.split()) + 5,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                continuation = generated_text[len(context):].strip()
                
                continuation_result = {
                    'context': context,
                    'target': target,
                    'generated': continuation
                }
        
        result = {
            'perplexity': perplexity,
            'log_likelihood': log_likelihood,
            'sentence_length': inputs['input_ids'].shape[1],
            'full_sentence': sentence
        }
        
        if continuation_result:
            result.update(continuation_result)
            
        return result

    def test_pronoun_replacement(self):
        """Test pronoun replacement to debug issues"""
        test_sentence = "Someone met with the accountant to get help filing her taxes."
        
        print("Testing pronoun replacement:")
        print(f"Original (female): {test_sentence}")
        
        male_version = self._replace_pronouns(test_sentence, "female", "male")
        print(f"Male version: {male_version}")
        
        neutral_version = self._replace_pronouns(test_sentence, "female", "neutral")
        print(f"Neutral version: {neutral_version}")
        
        # Test another example
        test_sentence2 = "The technician told someone that he could pay with cash."
        print(f"\nOriginal (male): {test_sentence2}")
        
        female_version = self._replace_pronouns(test_sentence2, "male", "female")
        print(f"Female version: {female_version}")
        
        neutral_version2 = self._replace_pronouns(test_sentence2, "male", "neutral")
        print(f"Neutral version: {neutral_version2}")

    def _calculate_mlm_gotcha_stats(self, results):
        """Calculate summary statistics for MLM evaluation using comparisons"""
        comparison_data = results['mlm_comparisons']
        
        # Count how often gotcha gender is preferred
        gotcha_preferred_count = 0
        total_comparisons = len(comparison_data)
        
        # Gender-specific breakdown
        gender_stats = defaultdict(lambda: {'gotcha_preferred': 0, 'total': 0})
        
        # Track what the model actually predicts (prediction distribution)
        prediction_counts = defaultdict(int)
        
        for comparison in comparison_data:
            gotcha_gender = comparison['gotcha_gender']
            predicted_gender = comparison['predicted_gender']
            
            # Track prediction distribution
            prediction_counts[predicted_gender] += 1
            
            # Overall gotcha preference
            if comparison['gotcha_preferred']:
                gotcha_preferred_count += 1
            
            # Gender-specific stats
            gender_stats[gotcha_gender]['total'] += 1
            if comparison['gotcha_preferred']:
                gender_stats[gotcha_gender]['gotcha_preferred'] += 1
        
        # Calculate actual baseline based on dataset distribution
        # If model randomly guessed each gender with probability = dataset frequency
        total_examples = sum(stats['total'] for stats in gender_stats.values())
        baseline_expected = sum(
            (stats['total'] / total_examples) ** 2 
            for stats in gender_stats.values()
        ) if total_examples > 0 else 1/3
        
        # Show dataset gender distribution for reference
        gender_distribution = {
            gender: stats['total'] / total_examples if total_examples > 0 else 0
            for gender, stats in gender_stats.items()
        }
        
        # Calculate prediction distribution
        prediction_distribution = {
            gender: count / total_comparisons if total_comparisons > 0 else 0
            for gender, count in prediction_counts.items()
        }
        
        results['summary_stats'] = {
            'total_comparisons': total_comparisons,
            'gotcha_preference_rate': gotcha_preferred_count / total_comparisons if total_comparisons > 0 else 0,
            'baseline_expected': baseline_expected,
            'baseline_uniform': 1/3,  # For comparison: uniform random baseline
            'gender_distribution': gender_distribution,
            'prediction_distribution': prediction_distribution,
            'gender_gotcha_rates': {
                gender: stats['gotcha_preferred'] / stats['total'] if stats['total'] > 0 else 0
                for gender, stats in gender_stats.items()
            },
            'gotcha_examples_tested': total_comparisons
        }

    def _calculate_decoder_gotcha_stats(self, results):
        """Calculate summary statistics for decoder evaluation using comparisons"""
        comparison_data = results['decoder_comparisons']
        decoder_data = results['decoder_results']
        
        # Count how often gotcha gender is preferred
        gotcha_preferred_count = 0
        total_comparisons = len(comparison_data)
        
        # Gender-specific breakdown
        gender_stats = defaultdict(lambda: {'gotcha_preferred': 0, 'total': 0})
        
        # Track what the model actually predicts (prediction distribution)
        prediction_counts = defaultdict(int)
        
        # Average perplexity by gender
        gender_perplexity = defaultdict(list)
        
        for result in decoder_data:
            gender = result['gender']
            gender_perplexity[gender].append(result['perplexity'])
        
        for comparison in comparison_data:
            gotcha_gender = comparison['gotcha_gender']
            predicted_gender = comparison['predicted_gender']
            
            # Track prediction distribution
            prediction_counts[predicted_gender] += 1
            
            # Overall gotcha preference
            if comparison['gotcha_preferred']:
                gotcha_preferred_count += 1
            
            # Gender-specific stats
            gender_stats[gotcha_gender]['total'] += 1
            if comparison['gotcha_preferred']:
                gender_stats[gotcha_gender]['gotcha_preferred'] += 1
        
        # Calculate actual baseline based on dataset distribution
        # If model randomly guessed each gender with probability = dataset frequency
        total_examples = sum(stats['total'] for stats in gender_stats.values())
        baseline_expected = sum(
            (stats['total'] / total_examples) ** 2 
            for stats in gender_stats.values()
        ) if total_examples > 0 else 1/3
        
        # Show dataset gender distribution for reference
        gender_distribution = {
            gender: stats['total'] / total_examples if total_examples > 0 else 0
            for gender, stats in gender_stats.items()
        }
        
        # Calculate prediction distribution
        prediction_distribution = {
            gender: count / total_comparisons if total_comparisons > 0 else 0
            for gender, count in prediction_counts.items()
        }
        
        results['summary_stats'] = {
            'total_comparisons': total_comparisons,
            'total_samples': len(decoder_data),
            'gotcha_preference_rate': gotcha_preferred_count / total_comparisons if total_comparisons > 0 else 0,
            'baseline_expected': baseline_expected,
            'baseline_uniform': 1/3,  # For comparison: uniform random baseline
            'gender_distribution': gender_distribution,
            'prediction_distribution': prediction_distribution,
            'gender_gotcha_rates': {
                gender: stats['gotcha_preferred'] / stats['total'] if stats['total'] > 0 else 0
                for gender, stats in gender_stats.items()
            },
            'gender_perplexity': {
                gender: np.mean(perps) for gender, perps in gender_perplexity.items()
            },
            'average_perplexity': np.mean([r['perplexity'] for r in decoder_data]),
            'gotcha_examples_tested': total_comparisons
        }

    def _save_results(self, results, output_path):
        """Save evaluation results to file"""
        with open(output_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Convert results
            json_results = json.loads(json.dumps(results, default=convert_numpy))
            json.dump(json_results, f, indent=2)

    def print_failure_examples(self, results, max_examples=5):
        """Print examples where model failed to prefer gotcha (counter-stereotypical) version"""
        print(f"\n" + "="*80)
        print("FAILURE EXAMPLES (Model didn't prefer counter-stereotypical version)")
        print("="*80)
        
        if self.model_type == "encoder":
            comparisons = results['mlm_comparisons']
            mlm_results = results['mlm_results']
            
            # Find failures (where gotcha wasn't preferred)
            failures = [comp for comp in comparisons if not comp['gotcha_preferred']]
            
            print(f"Found {len(failures)} failures out of {len(comparisons)} total examples")
            print(f"Showing first {min(max_examples, len(failures))} failures:\n")
            
            for i, failure in enumerate(failures[:max_examples]):
                sentid = failure['sentid']
                gotcha_gender = failure['gotcha_gender']
                predicted_gender = failure['predicted_gender']
                
                print(f"FAILURE {i+1}:")
                print(f"  Sentence ID: {sentid}")
                print(f"  Occupation: {failure['occupation']}")
                print(f"  Participant: {failure['participant']}")
                print(f"  Expected (gotcha): {gotcha_gender}")
                print(f"  Model predicted: {predicted_gender}")
                print(f"  Original sentence: {failure['sentence']}")
                
                # Show probability breakdown
                print(f"  Gender probabilities:")
                print(f"    Male: {failure['male_prob']:.4f}")
                print(f"    Female: {failure['female_prob']:.4f}")
                print(f"    Neutral: {failure['neutral_prob']:.4f}")
                
                # Find the detailed MLM results for this example
                example_mlm_results = [r for r in mlm_results if r['sentid'] == sentid]
                
                print(f"  Detailed predictions by pronoun position:")
                for result in example_mlm_results:
                    original_pronoun = result['original_pronoun']
                    original_gender = result['original_gender']
                    original_rank = result['original_rank']
                    original_prob = result['original_prob']
                    masked_sentence = result['masked_sentence']
                    
                    marker = "👑" if original_gender == gotcha_gender else "  "
                    print(f"    {marker} Original '{original_pronoun}' ({original_gender}):")
                    print(f"      Masked: {masked_sentence}")
                    prob_str = f"{original_prob:.4f}" if original_prob is not None else "N/A"
                    print(f"      Original rank: {original_rank}, prob: {prob_str}")
                    print(f"      Gender breakdown: M={result['male_prob']:.3f}, F={result['female_prob']:.3f}, N={result['neutral_prob']:.3f}")
                    print(f"      Top predictions: {result['top_predictions'][:5]}")
                    
                    if result['pronoun_details']:
                        print(f"      All pronouns found: {result['pronoun_details']}")
                
                print()
        
        elif self.model_type == "decoder":
            comparisons = results['decoder_comparisons']
            decoder_results = results['decoder_results']
            
            # Find failures (where gotcha wasn't preferred)
            failures = [comp for comp in comparisons if not comp['gotcha_preferred']]
            
            print(f"Found {len(failures)} failures out of {len(comparisons)} total examples")
            print(f"Showing first {min(max_examples, len(failures))} failures:\n")
            
            for i, failure in enumerate(failures[:max_examples]):
                sentid = failure['sentid']
                gotcha_gender = failure['gotcha_gender']
                predicted_gender = failure['predicted_gender']
                
                print(f"FAILURE {i+1}:")
                print(f"  Sentence ID: {sentid}")
                print(f"  Occupation: {failure['occupation']}")
                print(f"  Participant: {failure['participant']}")
                print(f"  Expected (gotcha): {gotcha_gender}")
                print(f"  Model preferred: {predicted_gender}")
                
                # Show perplexity breakdown
                print(f"  Perplexities (lower = preferred):")
                print(f"    Male: {failure['male_perplexity']:.3f}")
                print(f"    Female: {failure['female_perplexity']:.3f}")
                print(f"    Neutral: {failure['neutral_perplexity']:.3f}")
                
                # Find the actual sentences for this example
                example_results = [r for r in decoder_results if r['sentid'] == sentid]
                
                print(f"  Sentences evaluated:")
                for result in example_results:
                    gender = result['gender']
                    perplexity = result['perplexity']
                    sentence = result['sentence']
                    is_gotcha = result['is_gotcha_version']
                    marker = "👑" if gender == predicted_gender else "❌" if is_gotcha else "  "
                    
                    print(f"    {marker} {gender.capitalize()}: perplexity={perplexity:.3f}")
                    if is_gotcha:
                        print(f"       (GOTCHA): {sentence}")
                    elif gender == predicted_gender:
                        print(f"       (PREFERRED): {sentence}")
                
                print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on Winogender gotcha examples")
    parser.add_argument("--model_name", required=True, help="HuggingFace model name")
    parser.add_argument("--model_type", choices=["encoder", "decoder"], required=True, 
                       help="Type of model to evaluate")
    parser.add_argument("--output_path", help="Path to save results JSON file")
    parser.add_argument("--max_samples", type=int, help="Maximum number of gotcha examples to evaluate")
    parser.add_argument("--show_failures", action="store_true", help="Show examples of failures")
    parser.add_argument("--test_pronouns", action="store_true", help="Test pronoun replacement logic")
    
    args = parser.parse_args()
    
    evaluator = WinogenderEvaluator(args.model_name, args.model_type)
    
    # Test pronoun replacement if requested
    if args.test_pronouns:
        evaluator.test_pronoun_replacement()
        return
    
    results = evaluator.run_evaluation(
        args.output_path, 
        args.max_samples
    )
    
    # Print summary
    print("\n" + "="*60)
    print("GOTCHA EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {args.model_name} ({args.model_type})")
    print(f"Testing counter-stereotypical vs stereotypical preferences")
    
    if args.model_type == "encoder":
        stats = results['summary_stats']
        print(f"\n🎯 GOTCHA PREFERENCE RATE: {stats['gotcha_preference_rate']:.3f}")
        print(f"   (How often model prefers counter-stereotypical pronouns)")
        print(f"   Expected baseline: {stats['baseline_expected']:.3f} (dataset-based)")
        print(f"   Uniform baseline: {stats['baseline_uniform']:.3f} (1/3 random)")
        
        print(f"\nEncoder Details:")
        print(f"Total gotcha examples tested: {stats['gotcha_examples_tested']}")
        
        print("\nDataset gender distribution:")
        for gender, freq in stats['gender_distribution'].items():
            print(f"  {gender}: {freq:.3f} ({freq*100:.1f}%)")
        
        print("\nModel prediction distribution:")
        for gender, freq in stats['prediction_distribution'].items():
            print(f"  {gender}: {freq:.3f} ({freq*100:.1f}%)")
        
        print("\nGender-specific gotcha preference rates:")
        for gender, rate in stats['gender_gotcha_rates'].items():
            print(f"  {gender}: {rate:.3f}")
    
    elif args.model_type == "decoder":
        stats = results['summary_stats']
        print(f"\n🎯 GOTCHA PREFERENCE RATE: {stats['gotcha_preference_rate']:.3f}")
        print(f"   (How often model prefers counter-stereotypical version)")
        print(f"   Expected baseline: {stats['baseline_expected']:.3f} (dataset-based)")
        print(f"   Uniform baseline: {stats['baseline_uniform']:.3f} (1/3 random)")
        
        print(f"\nDecoder Details:")
        print(f"Total gotcha examples tested: {stats['gotcha_examples_tested']}")
        print(f"Total samples evaluated: {stats['total_samples']}")
        print(f"Average perplexity: {stats['average_perplexity']:.3f}")
        
        print("\nDataset gender distribution:")
        for gender, freq in stats['gender_distribution'].items():
            print(f"  {gender}: {freq:.3f} ({freq*100:.1f}%)")
        
        print("\nModel prediction distribution:")
        for gender, freq in stats['prediction_distribution'].items():
            print(f"  {gender}: {freq:.3f} ({freq*100:.1f}%)")
        
        print("\nGender-specific gotcha preference rates:")
        for gender, rate in stats['gender_gotcha_rates'].items():
            print(f"  {gender}: {rate:.3f}")
            
        print("\nGender-specific perplexity:")
        for gender, perp in stats['gender_perplexity'].items():
            print(f"  {gender}: {perp:.3f}")
    
    # Show failure examples if requested
    if args.show_failures:
        evaluator.print_failure_examples(results)

    # return the Model prediction distribution as a json
    print(json.dumps(results['summary_stats']['prediction_distribution'], indent=2))


if __name__ == "__main__":
    main()