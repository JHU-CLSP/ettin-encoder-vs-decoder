import os
import json
import random
import shutil
import logging
import glob
from tqdm import tqdm
from streaming import MDSWriter, StreamingDataset
from src.utils.data_utils import MDS_COLS_OUTPUT_ONLY
from transformers import set_seed
from streaming.base.util import merge_index
from src.tokenization.tokenize_mds_subfolders import cleanup_folder

set_seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sample_dataset(base_dir, dataset_name, sample_fraction=0.001):
    dataset_dir = os.path.join(base_dir, dataset_name)

    # if dataset_dir, train has an index.json file, skip
    if os.path.exists(os.path.join(dataset_dir, 'train', 'index.json')) and not args.force_redo:
        logging.info(f"Dataset {dataset_name} already has a train index.json file, skipping")
        return


    stats_file = os.path.join(base_dir, f"{dataset_name}_stats.json")

    logging.info(f"Processing dataset: {dataset_name}")
    logging.info(f"Reading stats file: {stats_file}")

    # Read the stats file
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    total_tokens = stats['total_tokens_written']
    tokens_to_sample = int(total_tokens * sample_fraction)

    logging.info(f"Total tokens: {total_tokens}")
    logging.info(f"Tokens to sample for validation: {tokens_to_sample}")

    # Randomly select a subfolder
    subfolders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    selected_subfolder = random.choice(subfolders)
    subfolder_path = os.path.join(dataset_dir, selected_subfolder)

    logging.info(f"Randomly selected subfolder: {selected_subfolder}")

    # also load the subfolders stats.json
    with open(os.path.join(subfolder_path, 'stats.json'), 'r') as f:
        subfolder_stats = json.load(f)
    # if the subfolder stats are < tokens_to_sample, throw an error
    if subfolder_stats['total_tokens_written'] < tokens_to_sample:
        raise ValueError(f"Subfolder {selected_subfolder} has less tokens than the required {tokens_to_sample} tokens to sample ({subfolder_stats['total_tokens_written']}), try just taking N folders instead")

    # Create validation and train directories
    validation_dir = os.path.join(dataset_dir, 'validation')
    train_dir = os.path.join(dataset_dir, 'train')

    if os.path.exists(validation_dir):
        # remove it
        logging.info(f"Removing existing validation directory: {validation_dir}")
        shutil.rmtree(validation_dir)
    
    if os.path.exists(train_dir):
        # remove it
        logging.info(f"Removing existing train directory: {train_dir}")
        shutil.rmtree(train_dir)

    logging.info(f"Creating validation directory: {validation_dir}")
    os.makedirs(validation_dir, exist_ok=True)

    logging.info(f"Creating train directory: {train_dir}")
    os.makedirs(train_dir, exist_ok=True)

    # Load the dataset
    logging.info(f"Loading dataset from: {subfolder_path}")
    dataset = StreamingDataset(local=subfolder_path, shuffle=True, split=None, predownload=8, batch_size=1)

    tokens_sampled = 0
    sampled_dir = f"{subfolder_path}-sampled"
    logging.info(f"Writing validation set to: {validation_dir}")
    logging.info(f"Writing sampled set to: {sampled_dir}")

    tokens_to_train = 0
    validation_complete = False
    batch_size = 10000

    with MDSWriter(out=validation_dir, columns=MDS_COLS_OUTPUT_ONLY, compression='zstd') as validation_writer, \
         MDSWriter(out=sampled_dir, columns=MDS_COLS_OUTPUT_ONLY, compression='zstd') as sampled_writer:
        
        for i in tqdm(range(0, len(dataset), batch_size), total=(len(dataset) + batch_size - 1) // batch_size):
            batch = dataset[i:i+batch_size]
            for instance in batch:
                instance_tokens = len(instance["input_ids"])
                
                if not validation_complete and tokens_sampled + instance_tokens <= tokens_to_sample:
                    # Still within validation limit
                    validation_writer.write(instance)
                    tokens_sampled += instance_tokens
                else:
                    # Exceeded validation limit or validation already complete
                    sampled_writer.write(instance)
                    tokens_to_train += instance_tokens
                    validation_complete = True

    logging.info(f"Total tokens sampled to validation: {tokens_sampled}")
    logging.info(f"Total tokens to train: {tokens_to_train}")

    # Copy all train instances except the selected subfolder
    logging.info("Copying subfolders to train directory")
    for subfolder in subfolders:
        if subfolder != selected_subfolder and subfolder not in ['validation', 'train', f"{selected_subfolder}-sampled"]:
            src = os.path.join(dataset_dir, subfolder)
            dst = os.path.join(train_dir, subfolder)
            logging.info(f"Copying {src} to {dst}")
            shutil.copytree(src, dst)

    # Move the -sampled folder to the train directory and remove the -sampled at the end
    dst = os.path.join(train_dir, f"{selected_subfolder}")
    assert not os.path.exists(dst), f"Destination folder {dst} already exists"
    logging.info(f"Moving {sampled_dir} to {dst}")
    shutil.move(sampled_dir, dst)

    logging.info(f"Sampling complete for {dataset_name}")
    logging.info(f"Updated {selected_subfolder} with remaining tokens")
    logging.info(f"Train set: copied all subfolders including updated {selected_subfolder}")

    # now also remove all non .zstd files from the subfolder directory we read in and decompressed
    logging.info(f"Removing all non .zstd files from {subfolder_path}")
    cleanup_folder(selected_subfolder)

    # make a root index.json file for the train section
    index_files = glob.glob(f"{train_dir}/**/index.json", recursive=True)
    print(f"Merging {len(index_files)} index files")
    merge_index(index_files, train_dir)

    logging.info(f"All done for {dataset_name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample dataset and create validation set")
    parser.add_argument("base_dir", help="Base directory containing the dataset")
    parser.add_argument("dataset_name", help="Name of the dataset to sample")
    parser.add_argument("--force_redo", action="store_true", help="Force redo the sampling")
    parser.add_argument("--sample_fraction", type=float, default=0.001, help="Fraction of data to sample for validation (default: 0.001)")
    
    args = parser.parse_args()
    
    sample_dataset(args.base_dir, args.dataset_name, args.sample_fraction)

    # example usage
    #   python sample_from_chunks.py data/chunked-olmo-1024-512-128-backfill-nodups wikipedia
