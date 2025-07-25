import os
import json
import random
import shutil
import argparse
import time
import tqdm
from transformers import set_seed

set_seed(42)

def sample_folders(source_dir, target_tokens, output_dir):
    total_tokens = 0
    selected_folders = []
    
    # Get all subdirectories
    all_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    
    while total_tokens < target_tokens and all_folders:
        # Randomly select a folder
        folder = random.choice(all_folders)
        all_folders.remove(folder)
        
        folder_path = os.path.join(source_dir, folder)
        stats_file = os.path.join(folder_path, 'stats.json')
        
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            folder_tokens = stats['total_tokens_written']
            
            total_tokens += folder_tokens
            selected_folders.append(folder)
            print(f"Selected folder: {folder}, Tokens: {folder_tokens:,}, Total: {total_tokens:,}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Copy selected folders to output directory
    for folder in tqdm.tqdm(selected_folders):
        src = os.path.join(source_dir, folder)
        dst = os.path.join(output_dir, folder)
        print(f"copying {src} to {dst}")
        shutil.copytree(src, dst)
    
    print(f"Sampling complete. Total tokens: {total_tokens}")
    return total_tokens, selected_folders

def main():
    parser = argparse.ArgumentParser(description="Sample folders based on token count.")
    parser.add_argument("source_dir", help="Source directory to sample from")
    parser.add_argument("tokens_to_sample", type=int, help="Number of tokens to sample")
    args = parser.parse_args()
    
    source_dir = args.source_dir
    tokens_to_sample = args.tokens_to_sample
    output_dir = f"{source_dir}-sampled"
    print(f"Writing output to {output_dir}")
    time.sleep(5)
    
    total_tokens, selected_folders = sample_folders(source_dir, tokens_to_sample, output_dir)
    
    print(f"Sampled {len(selected_folders)} folders")
    print(f"Total tokens sampled: {total_tokens}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()

    # example usage:
    #   python sample_from_folders.py "data/chunked-space-1024-512-128-backfill-nodups/mlfoundations-dclm-baseline-1.0-parquet-FULL" 837179337679 # (837,179,337,679)