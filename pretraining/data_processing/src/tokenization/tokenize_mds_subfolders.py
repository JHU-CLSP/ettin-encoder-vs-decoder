import os
import json
import glob
import subprocess
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import shutil
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_folders(path):
    folders = glob.glob(os.path.join(path, "**/**/index*.json"), recursive=True)
    tokenized_folders = glob.glob(os.path.join(path, "**/**-tokenized/index*.json"), recursive=True)
    return list(set(folders)), list(set(tokenized_folders))

def cleanup_folder(folder_path):
    logging.info(f"Cleaning up folder: {folder_path}")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not (file.endswith('.zstd') or file.endswith('.json')):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    # logging.info(f"Deleted file: {file_path}")
                except OSError as e:
                    logging.error(f"Error deleting file {file_path}: {str(e)}")

def process_folder(folder_path, tokenizer: str, has_domains: bool):
    try:
        if "index_old.json" in folder_path.split(os.path.sep)[-1]:
            new_path = folder_path.replace("index_old.json", "index.json")
            try:
                os.rename(folder_path, new_path)
                folder_path = new_path
            except OSError as e:
                error_message = f"Error renaming file {folder_path}: {str(e)}"
                logging.error(error_message)
                return folder_path, error_message
        
        script_path = "src/tokenization/tokenize_mds.py"
        folder_dir = os.path.dirname(folder_path)
        logging.info(f"Processing folder: {folder_dir}")

        # if the folder + "-tokenized" exists, remove it
        tokenized_folder = folder_dir + "-tokenized/"

        if os.path.exists(tokenized_folder):
            try:
                # remove all the contents also
                logging.info(f"Deleted folder: {tokenized_folder}")
                shutil.rmtree(tokenized_folder)
                os.makedirs(tokenized_folder) # make an empty one for the log file
            except OSError as e:
                error_message = f"Error deleting folder {tokenized_folder}: {str(e)}"
                logging.error(error_message)
                return folder_dir, error_message
        else:
            os.makedirs(tokenized_folder)
        
        # Create a log file for this folder's output
        log_file_path = os.path.join(folder_dir, "tokenize_output.log")
        logging.info(f"Logging output to: {log_file_path}")
        cmd = ["python", script_path, "-t", tokenizer, "-d", folder_dir]
        if has_domains:
            cmd.append("--has_domains")
        with open(log_file_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Wait for the process to complete
            process.wait()
        
        if process.returncode != 0:
            with open(log_file_path, "r") as log_file:
                output = log_file.read()
            error_message = f"Error processing folder {folder_dir}. Return code: {process.returncode}\n"
            error_message += f"Output: {output}"
            logging.error(error_message)
            return folder_dir, error_message
        
        logging.info(f"Successfully processed folder: {folder_dir}")
        
        cleanup_folder(folder_dir)
        
        return folder_dir, None
    except Exception as e:
        error_message = f"Unexpected error processing folder {folder_dir}: {str(e)}"
        logging.error(error_message)
        return folder_dir, error_message

def main(root_path, num_processes, tokenizer: str, has_domains: bool = False):
    folders, tokenized_folders = find_folders(root_path)
    logging.info(f"Found {len(folders)} folders and {len(tokenized_folders)} tokenized folders")
    to_skip = set()
    for item in tokenized_folders:
        if os.path.exists(os.path.join("/".join(item.split("/")[:-1]), "num_tokens.json")):
            to_skip.add(item.replace("-tokenized/", "/"))
    logging.info(f"Skipping {len(to_skip)} folders")

    sorted_folders = []
    for item in sorted(folders):
        if item in to_skip:
            logging.info(f"Skipping folder: {item}")
        elif "-tokenized" in item:
            continue
        elif item != os.path.join(root_path, "index.json"):
            sorted_folders.append(item)
        elif item == os.path.join(root_path, "index.json"):
            # logging.info(f"Found the root folder: {item}")
            continue
        else:
            raise ValueError(f"Found the root folder: {item}")

    folders = sorted_folders
    # randomly select 1 folder
    # folders = [folders[0]]
    total_folders = len(folders)
    
    # print(to_skip, tokenized_folders, folders)
    logging.info(f"Found {total_folders} folders to process and skipped {len(to_skip)} tokenized folders")
    # assert False, "Stop here"
    # Adjust num_processes if there are fewer folders
    num_processes = min(num_processes, total_folders)
    logging.info(f"Using {num_processes} processes")

    if total_folders == 0:
        logging.info("No folders to process")
        return
    
    successful = 0
    failed = 0
    failed_folders = []

    
    # add the args.tokenizer to `process_folder` function
    process_folder_partial = partial(process_folder, tokenizer=tokenizer, has_domains=has_domains)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_folder_partial, folder) for folder in folders]
        
        with tqdm(total=total_folders, desc="Processing folders") as pbar:
            for future in as_completed(futures):
                folder, error = future.result()
                if error is None:
                    successful += 1
                else:
                    failed += 1
                    failed_folders.append((folder, error))
                pbar.update(1)
    
    logging.info(f"Processing complete. Successful: {successful}, Failed: {failed}")
    
    if failed_folders:
        logging.info("Failed folders:")
        for folder, error in failed_folders:
            logging.info(f"  {folder}: {error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_path", type=str, required=True)
    parser.add_argument("-n", "--num_processes", type=int, required=True)
    parser.add_argument("-d", "--has_domains", action="store_true")
    parser.add_argument("-t", "--tokenizer", type=str, required=True, default="answerdotai/ModernBERT-base")
    args = parser.parse_args()
    print(args)
    
    main(args.root_path, args.num_processes, args.tokenizer, args.has_domains)

# example usage:
# python tokenize_mds_subfolders.py --root_path data/text/mlfoundations-dclm-baseline-1.0-parquet---train---small/ --num_processes 15 --tokenizer bclavie/olmo_bert_template
