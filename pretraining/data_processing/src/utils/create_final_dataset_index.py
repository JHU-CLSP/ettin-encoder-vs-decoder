import os
import argparse
import json
import logging
import glob
from streaming.base.util import merge_index
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_main_index(main_dir, split):
    split_dir = os.path.join(main_dir, split)
    index_files = glob.glob(f"{split_dir}/*/index.json")
    if index_files:
        logging.info(f"Merging {len(index_files)} index files for entire {split} set")
        merge_index(index_files, split_dir)
        logging.info(f"Created main index.json in {split_dir}")
    else:
        logging.warning(f"No index files found in {split_dir}")

def count_instances_from_index(index_path):
    try:
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        return sum(shard['samples'] for shard in index_data['shards'])
    except Exception as e:
        logging.error(f"Error loading index from {index_path}: {str(e)}")
        return 0

def create_and_verify_index(main_dir, split):
    # Create the main index.json
    create_main_index(main_dir, split)

    # Verify the count
    main_index_path = os.path.join(main_dir, split, 'index.json')
    main_count = count_instances_from_index(main_index_path)
    logging.info(f"Main {split} count from new index.json: {main_count}")

    # Count instances in individual subdirectories for verification
    split_dir = os.path.join(main_dir, split)
    subfolders = [f for f in os.listdir(split_dir) 
                  if os.path.isdir(os.path.join(split_dir, f)) and f != '.locks']
    
    total_count = 0
    for subfolder in tqdm(subfolders, desc=f"Verifying {split} subfolders"):
        index_path = os.path.join(split_dir, subfolder, 'index.json')
        count = count_instances_from_index(index_path)
        total_count += count
        logging.info(f"Subfolder {subfolder}: {count} instances")

    logging.info(f"Total {split} count from subfolders: {total_count}")

    if main_count == total_count:
        logging.info("Verification successful: Main count matches total subfolder count.")
    else:
        logging.warning(f"Verification failed: Main count ({main_count}) does not match total subfolder count ({total_count}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create main index.json and verify count")
    parser.add_argument("main_dir", help="Path to the main directory containing the split")
    parser.add_argument("--split", default="train", help="Split to process (default: train)")
    args = parser.parse_args()

    create_and_verify_index(args.main_dir, args.split)
    # python create_final_dataset_index.py data/chunked-olmo-1024-512-128-backfill-nodups/