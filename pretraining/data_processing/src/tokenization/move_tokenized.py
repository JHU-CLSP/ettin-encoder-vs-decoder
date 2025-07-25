import argparse
import os
import shutil
import time
import logging
from streaming.base.util import _merge_index_from_root, merge_index

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Move '-tokenized' folders to a new directory.")
    parser.add_argument("folder_path", help="Path to the folder to process")
    parser.add_argument("--tokenizer_name", default="olmo", help="Name of the tokenizer used")
    parser.add_argument("--create-root-index", action="store_true", help="Create an index.json file at the root folder")
    return parser.parse_args()

def create_tokenized_folder(source_path, tokenizer_name):
    current_folder = os.path.basename(os.path.normpath(source_path))
    new_folder_name = f"{current_folder}-tokenized"
    new_folder_path = os.path.join(os.path.dirname(source_path), new_folder_name)

    # replace /data/ with /data/tokenized_{args.tokenizer_name}/
    new_folder_path = new_folder_path.replace("data/text/", f"data/tokenized_{tokenizer_name}/")
    
    logging.info(f"Preparing to create new folder: {new_folder_path}")
    
    if not os.path.exists(new_folder_path):
        logging.info(f"Creating new folder: {new_folder_path}")
        os.makedirs(new_folder_path)
    else:
        logging.warning(f"Folder already exists: {new_folder_path}")
    
    return new_folder_path

def get_tokenized_folders(source_path):
    tokenized_folders = []
    for item in os.listdir(source_path):
        item_path = os.path.join(source_path, item)
        if os.path.isdir(item_path) and item.endswith('-tokenized'):
            tokenized_folders.append(item)
    return tokenized_folders

def get_tokenized_folders_one_level_down(source_path):
    """Find tokenized folders one level deeper in the directory structure."""
    tokenized_folders = []  # Change to list of tuples (parent_folder, tokenized_subfolder)
    
    for item in os.listdir(source_path):
        item_path = os.path.join(source_path, item)
        if os.path.isdir(item_path):
            # Check one level down
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path) and subitem.endswith('-tokenized'):
                    # Store as tuple (parent_folder, tokenized_subfolder)
                    tokenized_folders.append((item, subitem))
    
    return tokenized_folders

def move_tokenized_folders(source_path, dest_path, tokenized_folders):
    for item in tokenized_folders:
        source_item_path = os.path.join(source_path, item)
        dest_item_path = os.path.join(dest_path, item)
        
        logging.info(f"Moving {source_item_path} to {dest_item_path}")
        shutil.move(source_item_path, dest_item_path)
        logging.info(f"Successfully moved {item}")

def main():
    args = parse_arguments()
    
    if not os.path.isdir(args.folder_path):
        logging.error(f"Error: {args.folder_path} is not a valid directory.")
        return
    
    try:
        logging.info(f"Starting process for folder: {args.folder_path}")
        
        new_folder_path = create_tokenized_folder(args.folder_path, args.tokenizer_name)
        tokenized_folders = get_tokenized_folders(args.folder_path)
        # remove the main folder from the list
        basename = os.path.basename(os.path.normpath(args.folder_path))
        if basename + "-tokenized" in tokenized_folders:
            logging.info(f"Removing main folder from list: {basename}")
            tokenized_folders.remove(basename + "-tokenized")
        
        if not tokenized_folders:
            logging.warning("No '-tokenized' folders found to move.")
            # Try to find it one more level down
            logging.info("Looking for tokenized folders one level deeper...")
            deeper_tokenized = get_tokenized_folders_one_level_down(args.folder_path)
            
            if deeper_tokenized:
                logging.info(f"Found {len(deeper_tokenized)} tokenized folders one level deeper.")
                for parent, folder in deeper_tokenized:
                    source_path = os.path.join(args.folder_path, parent)
                    source_item_path = os.path.join(source_path, folder)
                    
                    # Extract base name of parent folder to create a flattened structure
                    parent_base = os.path.basename(os.path.normpath(parent))
                    # Create a new name that combines parent folder and tokenized folder name
                    # to avoid potential name collisions when flattening
                    if folder == parent_base + "-tokenized":
                        # If the folder is already named after its parent, use it as is
                        dest_folder_name = folder
                    else:
                        # Otherwise, prefix with parent name to maintain context
                        dest_folder_name = f"{parent_base}-{folder}"
                    
                    dest_item_path = os.path.join(new_folder_path, dest_folder_name)
                    
                    logging.info(f"Moving {source_item_path} to {dest_item_path}")
                    shutil.move(source_item_path, dest_item_path)
                    logging.info(f"Successfully moved {folder} to flattened structure as {dest_folder_name}")
            else:
                logging.warning("No '-tokenized' folders found one level deeper.")
            
            if args.create_root_index:
                logging.info(f"Creating index.json file at root folder: {new_folder_path}")
                _merge_index_from_root(new_folder_path)
            return
        
        logging.info("Summary of planned actions:")
        for folder in tokenized_folders:
            logging.info(f"  - Move '{folder}' to '{new_folder_path}'")
        
        logging.info("Waiting for 5 seconds before starting the move operation...")
        time.sleep(5)
        
        move_tokenized_folders(args.folder_path, new_folder_path, tokenized_folders)
        logging.info("Operation completed successfully.")

        if args.create_root_index:
            logging.info(f"Creating index.json file at root folder {new_folder_path}...")
            _merge_index_from_root(new_folder_path)
    
    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

    # python move_tokenized.py data/text/HuggingFaceTB-smollm-corpus---train---fineweb-edu-dedup --create-root-index