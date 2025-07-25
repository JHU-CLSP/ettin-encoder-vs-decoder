import argparse
import os
import shutil
import time
import logging
import json
from collections import defaultdict
import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Move chunked folders to a new directory and aggregate stats.")
    parser.add_argument("tokenized_dir", help="Path to the tokenized directory")
    parser.add_argument("chunk_format", help="Format of the chunked folders (e.g., '-chunked-1024-512-128-backfill-nodups')")
    return parser.parse_args()

def infer_tokenizer_name(tokenized_dir):
    if "/" == tokenized_dir[-1]:
        tokenized_dir = tokenized_dir[:-1]
    return os.path.basename(tokenized_dir).split('_')[-1]

def create_chunked_folder(tokenized_dir, chunk_format, tokenizer_name):
    new_folder_name = f"chunked-{tokenizer_name}-{chunk_format}"
    if "/" == tokenized_dir[-1]:
        tokenized_dir = tokenized_dir[:-1]
    new_folder_path = os.path.join(os.path.dirname(tokenized_dir), new_folder_name)
    
    if os.path.exists(new_folder_path):
        return new_folder_path
    
    os.makedirs(new_folder_path)
    logging.info(f"Created new folder: {new_folder_path}")
    
    return new_folder_path

def get_chunked_folders(tokenized_dir, chunk_format):
    chunked_folders = []
    for dataset_folder in tqdm.tqdm(os.listdir(tokenized_dir)):
        dataset_path = os.path.join(tokenized_dir, dataset_folder)
        if dataset_path.endswith(chunk_format):
            chunked_folders.append(dataset_path)
            continue
        if os.path.isdir(dataset_path):
            snapshots_path = os.path.join(dataset_path, 'snapshots')
            if os.path.isdir(snapshots_path):
                for snapshot in os.listdir(snapshots_path):
                    if snapshot.endswith(chunk_format):
                        chunked_folders.append(os.path.join(snapshots_path, snapshot))
    return chunked_folders

def move_chunked_folders(chunked_folders, dest_path):
    for source_path in chunked_folders:
        folder_name = os.path.basename(source_path)
        # get the important part of the folder name
        assert source_path.split("/")[0] == "data"
        if "orionweller" in source_path:
            dataset_name = source_path.split('/')[2].split("--")[-1].replace("_mds_incremental", "")
        else:
            dataset_name = source_path.split('/')[2].split("---")[0]
        dest_item_path = os.path.join(dest_path, dataset_name)
        shutil.move(source_path, dest_item_path)
        logging.info(f"Moved {source_path} to {dest_item_path}")

def merge_stats(stats1, stats2):
    merged = defaultdict(int)
    for d in (stats1, stats2):
        for key, value in d.items():
            if key != "percentiles":
                if isinstance(value, (int, float)):
                    merged[key] += value
                else:
                    # For non-numeric fields, prefer non-empty values
                    merged[key] = merged[key] or value
    return dict(merged)

def merge_stats(stats1, stats2):
    merged = defaultdict(int)
    for d in (stats1, stats2):
        for key, value in d.items():
            if key != "percentiles":
                if isinstance(value, (int, float)):
                    merged[key] += value
                else:
                    # For non-numeric fields, prefer non-empty values
                    merged[key] = merged[key] or value
    return dict(merged)

def aggregate_stats(root_dir):
    overall_stats = {}
    dataset_stats = {}

    for dataset_folder in tqdm.tqdm(os.listdir(root_dir)):
        if dataset_folder.endswith("-FULL"):
            continue
        dataset_path = os.path.join(root_dir, dataset_folder)
        if os.path.isdir(dataset_path):
            dataset_stats[dataset_folder] = {}
            for chunk_folder in os.listdir(dataset_path):
                chunk_path = os.path.join(dataset_path, chunk_folder)
                if os.path.isdir(chunk_path):
                    stats_file = os.path.join(chunk_path, 'stats.json')
                    if os.path.exists(stats_file):
                        with open(stats_file, 'r') as f:
                            chunk_stats = json.load(f)
                        # Merge into dataset stats
                        dataset_stats[dataset_folder] = merge_stats(dataset_stats[dataset_folder], chunk_stats)
                        # Merge into overall stats
                        overall_stats = merge_stats(overall_stats, chunk_stats)

    # Save overall stats
    if overall_stats:
        overall_stats.pop('percentiles', None)
        with open(os.path.join(root_dir, 'stats.json'), 'w') as f:
            json.dump(overall_stats, f, indent=2)
        logging.info(f"Created overall aggregated stats.json in {root_dir}")

    # Save individual dataset stats
    for dataset, stats in dataset_stats.items():
        stats.pop('percentiles', None)
        dataset_stats_file = os.path.join(root_dir, f'{dataset}_stats.json')
        with open(dataset_stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logging.info(f"Created aggregated stats for dataset {dataset} in {dataset_stats_file}")

def main():
    args = parse_arguments()
    
    if not os.path.isdir(args.tokenized_dir):
        logging.error(f"Error: {args.tokenized_dir} is not a valid directory.")
        return
    
    try:
        logging.info(f"Starting process for tokenized directory: {args.tokenized_dir}")
        
        tokenizer_name = infer_tokenizer_name(args.tokenized_dir)
        new_folder_path = create_chunked_folder(args.tokenized_dir, args.chunk_format, tokenizer_name)
        chunked_folders = get_chunked_folders(args.tokenized_dir, args.chunk_format)

        # remove any folder with "dclm" or "cc_en_head" in it
        # chunked_folders = [folder for folder in chunked_folders if "dclm" not in folder and "cc_en_head" not in folder]
        
        if not chunked_folders:
            logging.warning(f"No folders matching the chunk format '{args.chunk_format}' found to move.")
        else:
            logging.info(f"Found {len(chunked_folders)} folders to move.")
            
            move_chunked_folders(chunked_folders, new_folder_path)
            logging.info("Move operation completed successfully.")

        logging.info("Starting stats aggregation...")
        aggregate_stats(new_folder_path)
        logging.info("Stats aggregation completed successfully.")
    
    except FileExistsError as e:
        logging.error(str(e))
    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

# python move_chunks.py data/tokenized_olmo/ 8192-512-32-backfill-nodups