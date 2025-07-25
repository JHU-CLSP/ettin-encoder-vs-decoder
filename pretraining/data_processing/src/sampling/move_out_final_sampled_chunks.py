import os
import shutil
import glob
import json
import logging
from tqdm import tqdm
from streaming import StreamingDataset
from streaming.base.util import merge_index

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def reorganize_and_index_datasets(main_dir):
    # Create train and validation directories in the main folder
    train_dir = os.path.join(main_dir, 'train')
    validation_dir = os.path.join(main_dir, 'validation')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # Get all dataset names
    dataset_names = [name for name in os.listdir(main_dir) 
                     if os.path.isdir(os.path.join(main_dir, name)) 
                     and name not in ['train', 'validation'] and "-FULL" not in name]

    for dataset_name in tqdm(dataset_names, desc="Processing datasets"):
        dataset_path = os.path.join(main_dir, dataset_name)
        
        # Move train data
        src_train = os.path.join(dataset_path, 'train')
        dst_train = os.path.join(train_dir, dataset_name)
        if os.path.exists(src_train):
            logging.info(f"Moved {src_train} to {dst_train}")
            shutil.move(src_train, dst_train)

        # Move validation data
        src_validation = os.path.join(dataset_path, 'validation')
        dst_validation = os.path.join(validation_dir, dataset_name)
        if os.path.exists(src_validation):
            logging.info(f"Moved {src_validation} to {dst_validation}")
            shutil.move(src_validation, dst_validation)

        # Combine index.json files for the dataset
        for split in ['train']:    
            split_dir = os.path.join(main_dir, split, dataset_name)
            # rename the current index.json file to index.json.old
            if os.path.exists(os.path.join(split_dir, 'index.json')):
                os.rename(os.path.join(split_dir, 'index.json'), os.path.join(split_dir, 'index.json.old'))
            if os.path.exists(split_dir):
                index_files = glob.glob(f"{split_dir}/**/index.json", recursive=True)
                if index_files:
                    logging.info(f"Merging {len(index_files)} index files for {dataset_name} {split}")
                    merge_index(index_files, split_dir)

    # Combine index.json files for the entire dataset
    for split in ['train', 'validation']:
        split_dir = os.path.join(main_dir, split)
        index_files = glob.glob(f"{split_dir}/*/index.json")
        if index_files:
            logging.info(f"Merging {len(index_files)} index files for entire {split} set")
            merge_index(index_files, split_dir)

    # Load the overall dataset and print the number of instances
    for split in ['train', 'validation']:
        split_dir = os.path.join(main_dir, split)
        if os.path.exists(split_dir):
            dataset = StreamingDataset(local=split_dir, predownload=1, batch_size=1)
            num_instances = len(dataset)
            logging.info(f"Number of instances in {split} set: {num_instances}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reorganize datasets and create combined index")
    parser.add_argument("main_dir", help="Main directory containing the datasets")
    
    args = parser.parse_args()
    
    reorganize_and_index_datasets(args.main_dir)

    # Example usage:
    # python move_out_final_sampled_chunks.py data/chunked-olmo-1024-512-128-backfill-nodups