import argparse
import glob
import json
import logging
import os
import random
import shutil
from pathlib import Path
from streaming.base.util import merge_index

def setup_logging(log_level):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def find_all_stats_files(input_path):
    """Find all stats.json files in the input directory and its subdirectories."""
    return list(glob.glob(os.path.join(input_path, "**", "stats.json"), recursive=True))

def get_tokens_from_stats(stats_file):
    """Read token count from a stats.json file."""
    with open(stats_file, 'r') as f:
        data = json.load(f)
        return data["total_tokens_written"]

def copy_data_folder(stats_file, output_dir):
    """Copy the entire data folder containing the stats.json file."""
    # Get the parent directory of the stats.json file
    source_dir = os.path.dirname(stats_file)
    # Create a unique folder name based on the path
    folder_name = os.path.basename(source_dir)
    target_dir = os.path.join(output_dir, folder_name)
    
    # Create the output directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy the entire folder
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        target_item = os.path.join(target_dir, item)
        if os.path.isfile(source_item):
            shutil.copy2(source_item, target_item)
        else:
            shutil.copytree(source_item, target_item, dirs_exist_ok=True)
    
    return target_dir

def sample_tokens(args):
    setup_logging(args.log_level)
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Find all stats.json files
    stats_files = find_all_stats_files(args.input_path)
    logging.info(f"Found {len(stats_files)} stats.json files in {args.input_path}")
    
    if not stats_files:
        logging.error(f"No stats.json files found in {args.input_path}")
        return
    
    # Randomly sample and copy files until we reach the target token count
    total_tokens = 0
    used_files = set()
    copied_folders = []
    
    while total_tokens < args.target_tokens and len(used_files) < len(stats_files):
        # Sample a random file that hasn't been used yet
        available_files = [f for f in stats_files if f not in used_files]
        if not available_files:
            break
            
        chosen_file = random.choice(available_files)
        used_files.add(chosen_file)
        
        # Get token count from the stats file
        file_tokens = get_tokens_from_stats(chosen_file)
        
        # Copy the data folder
        copied_folder = copy_data_folder(chosen_file, args.output_path)
        copied_folders.append(copied_folder)
        
        total_tokens += file_tokens
        logging.info(f"Copied folder: {copied_folder}")
        logging.info(f"Added {file_tokens:,} tokens. Total now: {total_tokens:,} / {args.target_tokens:,}")
    
    # Write summary stats
    summary = {
        "total_tokens": total_tokens,
        "target_tokens": args.target_tokens,
        "num_folders_copied": len(copied_folders),
        "copied_folders": copied_folders
    }
    
    summary_path = os.path.join(args.output_path, "sampling_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Merge index files for the output directory
    index_files = glob.glob(os.path.join(args.output_path, "**/index.json"), recursive=True)
    if index_files:
        logging.info(f"Merging {len(index_files)} index files for output directory")
        merge_index(index_files, args.output_path)
    else:
        logging.warning("No index.json files found to merge")

    logging.info(f"Sampling complete! Summary written to {summary_path}")
    logging.info(f"Final token count: {total_tokens:,} tokens from {len(copied_folders)} folders")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample data folders until reaching a target token count")
    parser.add_argument("-i", "--input_path", type=str, required=True,
                        help="Input directory containing stats.json files in subdirectories")
    parser.add_argument("-o", "--output_path", type=str, required=True,
                        help="Output directory where sampled folders will be copied")
    parser.add_argument("-t", "--target_tokens", type=int, required=True,
                        help="Target number of tokens to sample")
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help="Set the logging level")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    sample_tokens(args)

"""
# Example usage:
python ./bin/sample_too_large_down.py \
    --input_path data/chunked-olmo-8000-512-128-backfill-nodups/allenai-dolmino-mix-1124---train---dclm-tokenized-chunked-8000-512-128-backfill-nodups \
    --output_path data/sample_250B/dclm-dolmino/ \
    --target_tokens 175_500_000_000 \
    --log_level INFO \
    --seed 42
"""