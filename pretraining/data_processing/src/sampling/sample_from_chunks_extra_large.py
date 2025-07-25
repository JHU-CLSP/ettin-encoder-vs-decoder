import os
import json
import logging
import glob
import shutil
from tqdm import tqdm
from streaming import MDSWriter, StreamingDataset
from ettin_data.utils.data_utils import MDS_COLS_OUTPUT_ONLY
from transformers import set_seed
from streaming.base.util import merge_index

set_seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sample_large_dataset(base_dir, tokenized_dir, dataset_name, sample_fraction=0.001):
    dataset_dir = os.path.join(base_dir, dataset_name)
    tokenized_dataset_dir = tokenized_dir # os.path.join(tokenized_dir, dataset_name)

    if os.path.exists(os.path.join(dataset_dir, 'train', 'index.json')) and not args.force_redo:
        logging.info(f"Dataset {dataset_name} already has a train index.json file, skipping")
        return

    stats_file = os.path.join(base_dir, f"{dataset_name}_stats.json")

    logging.info(f"Processing dataset: {dataset_name}")
    logging.info(f"Reading stats file: {stats_file}")

    with open(stats_file, 'r') as f:
        stats = json.load(f)
    total_tokens = stats['total_tokens_written']
    tokens_to_sample = int(total_tokens * sample_fraction)

    logging.info(f"Total tokens: {total_tokens}")
    logging.info(f"Tokens to sample for validation: {tokens_to_sample}")

    main_subfolders = set(os.listdir(dataset_dir))
    tokenized_subfolders = set(os.listdir(tokenized_dataset_dir))
    unique_new_subfolders = list(tokenized_subfolders - main_subfolders)

    logging.info(f"Found {len(unique_new_subfolders)} unique new subfolders")
    breakpoint()

    validation_dir = os.path.join(dataset_dir, 'validation')
    train_dir = os.path.join(dataset_dir, 'train')

    if os.path.exists(validation_dir):
        logging.info(f"Removing existing validation directory: {validation_dir}")
        # shutil.rmtree(validation_dir)
    
    if os.path.exists(train_dir):
        logging.info(f"Removing existing train directory: {train_dir}")
        # shutil.rmtree(train_dir)

    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    tokens_sampled = 0

    for subfolder in tqdm(unique_new_subfolders):
        subfolder_path = os.path.join(tokenized_dataset_dir, subfolder)
        stats_file = os.path.join(subfolder_path, 'stats.json')
        
        with open(stats_file, 'r') as f:
            subfolder_stats = json.load(f)
        
        subfolder_tokens = subfolder_stats['total_tokens_written']
        
        if tokens_sampled + subfolder_tokens <= tokens_to_sample:
            # Copy entire folder
            dst = os.path.join(validation_dir, subfolder)
            shutil.copytree(subfolder_path, dst)
            tokens_sampled += subfolder_tokens
        else:
            # Sample partial folder
            remaining_tokens = tokens_to_sample - tokens_sampled
            dataset = StreamingDataset(local=subfolder_path, shuffle=True, split=None, predownload=8, batch_size=1)
            
            dst = os.path.join(validation_dir, subfolder)
            os.makedirs(dst, exist_ok=True)
            
            with MDSWriter(out=dst, columns=MDS_COLS_OUTPUT_ONLY, compression='zstd') as validation_writer:
                for instance in tqdm(dataset):
                    instance_tokens = len(instance["input_ids"])
                    if tokens_sampled + instance_tokens <= tokens_to_sample:
                        validation_writer.write(instance)
                        tokens_sampled += instance_tokens
                    else:
                        break
            
            break

    logging.info(f"Total tokens sampled to validation: {tokens_sampled}")

    # Copy existing subfolders from main directory to train directory
    for subfolder in tqdm(main_subfolders):
        if subfolder not in ['validation', 'train'] and ".json" not in subfolder:
            # if it is a dir
            if os.path.isdir(os.path.join(dataset_dir, subfolder)):
                src = os.path.join(dataset_dir, subfolder)
                dst = os.path.join(train_dir, subfolder)
                logging.info(f"Copying {src} to {dst}")
                shutil.copytree(src, dst)

    # Merge index files for train section
    index_files = glob.glob(f"{train_dir}/**/index.json", recursive=True)
    logging.info(f"Merging {len(index_files)} index files for train set")
    merge_index(index_files, train_dir)

    # Merge index files for validation section
    index_files = glob.glob(f"{validation_dir}/**/index.json", recursive=True)
    logging.info(f"Merging {len(index_files)} index files for validation set")
    merge_index(index_files, validation_dir)

    logging.info(f"All done for {dataset_name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample large dataset and create validation set")
    parser.add_argument("base_dir", help="Base directory containing the dataset")
    parser.add_argument("tokenized_dir", help="Directory containing the tokenized dataset")
    parser.add_argument("dataset_name", help="Name of the dataset to sample")
    parser.add_argument("--force_redo", action="store_true", help="Force redo the sampling")
    parser.add_argument("--sample_fraction", type=float, default=0.001, help="Fraction of data to sample for validation (default: 0.001)")
    
    args = parser.parse_args()
    
    sample_large_dataset(args.base_dir, args.tokenized_dir, args.dataset_name, args.sample_fraction)

    # example usage
    #   python sample_from_chunks_extra_large.py data/chunked-gemma-1024-512-32-backfill-nodups data/chunked-gemma-1024-512-32-backfill-nodups/dclm-FULL dclm-FULL-sampled --sample_fraction 0.0003333 --force_redo
