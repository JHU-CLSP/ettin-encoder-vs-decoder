import os
import argparse
import json
import logging
import pandas as pd
from tqdm import tqdm
import glob
from streaming.base.util import merge_index

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_index_file(directory):
    index_files = glob.glob(f"{directory}/**/index.json", recursive=True)
    if index_files:
        logging.info(f"Merging {len(index_files)} index files for {directory}")
        merge_index(index_files, directory)
    else:
        logging.warning(f"No index files found in {directory}")

def count_instances_from_index(index_path):
    if not os.path.exists(index_path):
        directory = os.path.dirname(index_path)
        create_index_file(directory)
    
    try:
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        return sum(shard['samples'] for shard in index_data['shards'])
    except Exception as e:
        logging.error(f"Error loading index from {index_path}: {str(e)}")
        return 0

def compare_datasets(chunking_dir, output_csv):
    train_dir = os.path.join(chunking_dir, 'train')
    validation_dir = os.path.join(chunking_dir, 'validation')

    subfolders = [f for f in os.listdir(chunking_dir) if os.path.isdir(os.path.join(chunking_dir, f)) 
                  and f not in ['train', 'validation', '.locks'] and "-FULL" not in f]

    results = []

    for subfolder in tqdm(subfolders, desc="Processing subfolders"):
        main_path = os.path.join(chunking_dir, subfolder, 'index.json')
        train_path = os.path.join(train_dir, subfolder, 'index.json')
        validation_path = os.path.join(validation_dir, subfolder, 'index.json')

        main_count = count_instances_from_index(main_path)
        train_count = count_instances_from_index(train_path)
        validation_count = count_instances_from_index(validation_path)

        train_ratio = train_count / main_count if main_count > 0 else 0
        is_train_ratio_correct = abs(train_ratio - 0.999) <= 0.001 if main_count > 0 else False
        is_total_correct = (train_count + validation_count == main_count)

        result = {
            'Dataset': subfolder,
            'Main instances': main_count,
            'Train instances': train_count,
            'Validation instances': validation_count,
            'Train ratio': train_ratio,
            'Is train ratio correct': is_train_ratio_correct,
            'Is total correct': is_total_correct
        }
        results.append(result)

        logging.info(f"\nDataset: {subfolder}")
        logging.info(f"Main instances: {main_count}")
        logging.info(f"Train instances: {train_count}")
        logging.info(f"Validation instances: {validation_count}")
        logging.info(f"Train ratio: {train_ratio:.4f}")

        if not is_train_ratio_correct:
            logging.warning(f"Train ratio is not approximately 0.999 (99.9%)")
        if not is_total_correct:
            logging.warning(f"Sum of train and validation instances does not equal main instances: {train_count} + {validation_count} != {main_count}")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    logging.info(f"Results saved to {output_csv}")

    # Print summary statistics
    logging.info("\nSummary Statistics:")
    logging.info(df.describe())
    
    # Print datasets with incorrect ratios or totals
    incorrect_ratios = df[~df['Is train ratio correct']]
    incorrect_totals = df[~df['Is total correct']]
    
    if not incorrect_ratios.empty:
        logging.warning("\nDatasets with incorrect train ratios:")
        logging.warning(incorrect_ratios[['Dataset', 'Train ratio']])
    
    if not incorrect_totals.empty:
        logging.warning("\nDatasets with mismatched totals:")
        logging.warning(incorrect_totals[['Dataset', 'Main instances', 'Train instances', 'Validation instances']])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare dataset instances in chunking directory and save results to CSV")
    parser.add_argument("chunking_dir", help="Path to the chunking directory")
    parser.add_argument("--output", default="dataset_comparison_results.csv", help="Output CSV file name")
    args = parser.parse_args()

    compare_datasets(args.chunking_dir, args.output)
    # python compare_train_and_chunked.py data/chunked-olmo-1024-512-128-backfill-nodups
