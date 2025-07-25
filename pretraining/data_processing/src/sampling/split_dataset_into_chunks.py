import argparse
import os
import subprocess
from tqdm import tqdm
import multiprocessing as mp
from collections import deque
import logging
import random

random.seed(12345)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_process_subfolder(args_tuple):
    subfolder_path, args = args_tuple
    cmd = [
        "python", "src/ettin_data/sampling/split_dataset_into_chunks_individual.py",
        "--subfolder_path", subfolder_path,
        "-c", str(args.chunk_size),
        "-m", str(args.min_chunk_size),
        "-a", str(args.always_skip_size),
        "--batch_size", str(args.batch_size),
    ]
    if args.enforce_min_chunk_size_zeroth:
        cmd.append("--enforce_min_chunk_size_zeroth")
    if args.backfill:
        cmd.append("--backfill")
    if args.backfill_no_duplicates:
        cmd.append("--backfill_no_duplicates")
    if args.add_eos_token:
        cmd.append("--add_eos_token")
    
    log_file_path = os.path.join(subfolder_path, "chunk_output.log")
    logging.info(f"Logging output to: {log_file_path}")
    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(cmd, 
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True)
            
        # Wait for the process to complete
        process.wait()
        
        if process.returncode != 0:
            with open(log_file_path, "r") as log_file:
                output = log_file.read()
            error_message = f"Error processing folder {subfolder_path}. Return code: {process.returncode}\n"
            error_message += f"Output: {output}"
            logging.error(error_message)
            return subfolder_path, error_message
        else:
            logging.info(f"Successfully processed folder: {subfolder_path}")

    return subfolder_path, None

def process_main_folder(args):
    main_folder = args.source
    # only keep subfolders that have not been processed, e.g. they don't have a stats.json file
    subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]
    to_do_subfolders = []
    chunk_details = f"-tokenized-chunked-{args.chunk_size}-{args.min_chunk_size}-{args.always_skip_size}-{'backfill' if args.backfill else 'no-backfill'}-{'nodups' if args.backfill_no_duplicates else 'dups'}"
    for subfolder in subfolders:
        # print(os.path.join(subfolder.replace("-tokenized", chunk_details)))
        if not os.path.exists(os.path.join(subfolder.replace("-tokenized", chunk_details), "stats.json")):
            to_do_subfolders.append(subfolder)
    subfolders = to_do_subfolders
    print(f"Found {len(subfolders)} subfolders to process.")
    import time
    time.sleep(0.5)

    # sort the subfolders
    try:
        subfolders.sort(key=lambda x: int(x.split("_")[-1].replace("-tokenized", "").replace("-chunked", "")))  
    except Exception as e:
        print(f"Error sorting subfolders: {e}, moving on without sort")
    # save them to file with the name of the main folder basename
    with open(os.path.basename(args.source) + "_subfolders.txt", "w") as f:
        for subfolder in subfolders:
            f.write(f"{subfolder}\n")
            
    subfolder_args = [(subfolder, args) for subfolder in subfolders]

    if args.reverse:
        print(subfolders[:5], subfolders[-5:])
        subfolder_args = subfolder_args[::-1]

    if args.max_subfolders and len(subfolders) > args.max_subfolders: 
        # randomly sample subfolders
        subfolder_args = random.sample(subfolder_args, args.max_subfolders)
    
    failed_subfolders = []
    total_processed = 0
    total_failed = 0
    
    with mp.Pool(processes=args.num_processes) as pool:
        with tqdm(total=len(subfolders), desc="Processing") as pbar:
            for subfolder, error in pool.imap(run_process_subfolder, subfolder_args):
                total_processed += 1
                if error:
                    failed_subfolders.append((subfolder, error))
                    total_failed += 1
                pbar.set_postfix({"Failed": total_failed}, refresh=True)
                pbar.update(1)

    
    if failed_subfolders:
        print("\nThe following subfolders failed to process:")
        for subfolder, error in failed_subfolders:
            print(f"\n{subfolder}")
            print(error)
        print(f"\nTotal failed subfolders: {len(failed_subfolders)}")
        # save to file the names of the failed subfolders
        with open("failed_subfolders.txt", "w") as f:
            for subfolder, error in failed_subfolders:
                f.write(f"{subfolder}\n")
    else:
        print("\nAll subfolders processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, required=True, help="Main folder containing subfolders to process")
    parser.add_argument("-c", "--chunk_size", type=int, default=1024, help="Size of each chunk")
    parser.add_argument("-m", "--min_chunk_size", type=int, default=800, help="Minimum size of a chunk to keep")
    parser.add_argument("-a", "--always_skip_size", type=int, default=500, help="Always skip chunks smaller than this size")
    parser.add_argument("-b", "--backfill", action="store_true", help="Enable backfilling for short final chunks", default=False)
    parser.add_argument("--backfill_no_duplicates", action="store_true", help="Backfill without duplicating tokens", default=False)
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--num_processes", type=int, default=mp.cpu_count(), help="Number of processes to use for multiprocessing")
    parser.add_argument("--enforce_min_chunk_size_zeroth", action="store_true", help="Enforce minimum chunk size for the first chunk", default=False)
    parser.add_argument("--max_subfolders", type=int, help="Maximum number of subfolders to process", default=10000)
    parser.add_argument("--add_eos_token", action="store_true", help="Add eos token to the end of the chunk", default=False)
    parser.add_argument("--reverse", action="store_true", help="Reverse the order of the subfolders", default=False)
    args = parser.parse_args()

    process_main_folder(args)

# Example usage:
# python split_dataset_into_chunks.py -s data/tokenized_olmo/datasets--orionweller--wikipedia_mds_incremental/snapshots/aff2afa7d7274979206600f1b53d7869eebc3dc9-tokenized -c 1024