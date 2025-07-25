import argparse
import glob
import json
import logging
import os
import random
import shutil
import uuid
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

def copy_data_folder(stats_file, output_dir, unique_suffix=None):
    """Copy the entire data folder containing the stats.json file."""
    # Get the parent directory of the stats.json file
    source_dir = os.path.dirname(stats_file)
    # Create a folder name based on the path
    folder_name = os.path.basename(source_dir)
    
    # Add a unique suffix if provided (for upsampling)
    if unique_suffix:
        folder_name = f"{folder_name}-upsample-{unique_suffix}"
    
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
    
    # Keep a cache of already read token counts to avoid re-reading the same files
    file_token_cache = {}
    
    logging.info(f"Target token count: {args.target_tokens:,}")
    
    # Phase 1: Sample without replacement (original behavior)
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
        
        # Get token count from the stats file, using cache if available
        if chosen_file in file_token_cache:
            file_tokens = file_token_cache[chosen_file]
        else:
            file_tokens = get_tokens_from_stats(chosen_file)
            file_token_cache[chosen_file] = file_tokens
        
        # Copy the data folder
        copied_folder = copy_data_folder(chosen_file, args.output_path)
        copied_folders.append(copied_folder)
        
        total_tokens += file_tokens
        logging.info(f"Copied folder: {copied_folder}")
        logging.info(f"Added {file_tokens:,} tokens. Total now: {total_tokens:,} / {args.target_tokens:,}")
    
    # Phase 2: Upsample if needed (sample with replacement)
    if total_tokens < args.target_tokens:
        logging.info(f"Finished sampling without replacement. Still need {args.target_tokens - total_tokens:,} more tokens.")
        logging.info(f"Starting upsampling (copying with replacement)...")
        
        # Convert used_files set to a list so we can sample from it
        all_used_files = list(used_files)
        
        upsample_count = 0
        while total_tokens < args.target_tokens:
            # Sample a random file from those we've already used
            chosen_file = random.choice(all_used_files)
            
            # Generate a unique suffix for this copy
            unique_suffix = str(uuid.uuid4())[:8]
            
            # Get token count from the stats file, using cache if available
            if chosen_file in file_token_cache:
                file_tokens = file_token_cache[chosen_file]
            else:
                file_tokens = get_tokens_from_stats(chosen_file)
                file_token_cache[chosen_file] = file_tokens
            
            # Copy the data folder with a unique name
            copied_folder = copy_data_folder(chosen_file, args.output_path, unique_suffix)
            copied_folders.append(copied_folder)
            
            total_tokens += file_tokens
            upsample_count += 1
            
            if upsample_count % 10 == 0 or total_tokens >= args.target_tokens:
                logging.info(f"Upsampled {upsample_count} folders so far")
                logging.info(f"Added {file_tokens:,} tokens. Total now: {total_tokens:,} / {args.target_tokens:,}")
    
    # Write summary stats
    summary = {
        "total_tokens": total_tokens,
        "target_tokens": args.target_tokens,
        "num_unique_folders": len(used_files),
        "num_total_folders_copied": len(copied_folders),
        "num_upsampled_folders": len(copied_folders) - len(used_files),
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
    logging.info(f"Original folders: {len(used_files)}, Upsampled folders: {len(copied_folders) - len(used_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample data folders until reaching a target token count, with upsampling if needed")
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
python ./bin/sample_with_upsampling.py \
    --input_path /path/to/input/data \
    --output_path /path/to/output/data \
    --target_tokens 175_500_000_000 \
    --log_level INFO \
    --seed 42
"""