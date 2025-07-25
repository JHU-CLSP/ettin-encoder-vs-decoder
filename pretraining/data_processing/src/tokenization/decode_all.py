import argparse
import os
import subprocess
from tqdm import tqdm
import multiprocessing as mp
import logging
from streaming import StreamingDataset
from transformers import AutoTokenizer
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def process_shard(shard_path, output_path, tokenizer_name, batch_size):
    # Extract domain from shard_path
    domain = os.path.basename(os.path.dirname(shard_path))
    
    # Setup logger for this shard
    log_file = os.path.splitext(output_path)[0] + '.log'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = setup_logger(os.path.basename(shard_path), log_file)
    
    logger.info(f"Starting to process shard: {shard_path}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load the dataset
    dataset = StreamingDataset(local=shard_path, shuffle=False, split=None)
    
    # Prepare output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Process and write data
    with open(output_path, 'w') as f:
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            input_ids_batch = [item['input_ids'] for item in batch]
            
            # Convert input_ids to text using batch_decode
            texts = tokenizer.batch_decode(input_ids_batch, skip_special_tokens=True)
            fields_in_original = batch[0].keys()
            # drop input_ids and add text
            
            for j, text in enumerate(texts):
                output_item = {
                    'id': batch[j]['id'],
                    'text': text,
                    'domain': domain
                }
                f.write(json.dumps(output_item) + '\n')
            
            logger.info(f"Processed {i+len(batch)} out of {len(dataset)} items")
    
    logger.info(f"Finished processing shard: {shard_path}")

def run_process_shard(args_tuple):
    shard_path, output_path, tokenizer_name, batch_size = args_tuple
    cmd = [
        "python", __file__,
        "--mode", "process_single_shard",
        "--shard_path", shard_path,
        "--output_path", output_path,
        "--tokenizer_name", tokenizer_name,
        "--batch_size", str(batch_size)
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        logging.error(f"Error processing shard {shard_path}. Error: {stderr.decode()}")
    else:
        logging.info(f"Successfully processed shard: {shard_path}")

def process_main_folder(args):
    main_folder = args.source
    output_folder = args.output
    
    # Collect all unique shard paths
    shard_paths = set()
    for root, _, files in os.walk(main_folder):
        for file in files:
            if file.endswith('index.json'):
                # if "tokenized-chunked" in os.path.join(root, file):
                shard_paths.add(root)
    
    shard_paths = list(shard_paths)
    logging.info(f"Found {len(shard_paths)} unique shards to process.")
    
    # Prepare arguments for multiprocessing
    mp_args = []
    for shard_path in shard_paths:
        relative_path = os.path.relpath(shard_path, main_folder)
        output_path = os.path.join(output_folder, os.path.splitext(relative_path)[0] + '.jsonl')
        mp_args.append((shard_path, output_path, args.tokenizer_name, args.batch_size))
    
    # Process shards using multiprocessing
    with mp.Pool(processes=min(50, len(shard_paths))) as pool:
        list(tqdm(pool.imap_unordered(run_process_shard, mp_args), total=len(shard_paths), desc="Processing"))
    
    logging.info("All shards processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["process_main", "process_single_shard"], required=True, help="Mode of operation")
    parser.add_argument("-s", "--source", type=str, help="Main folder containing the train folder")
    parser.add_argument("-o", "--output", type=str, help="Output folder for processed data")
    parser.add_argument("-t", "--tokenizer_name", type=str, default="bclavie/olmo_bert_template", help="Name or path of the tokenizer to use")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--shard_path", type=str, help="Path to single shard for processing")
    parser.add_argument("--output_path", type=str, help="Output path for single shard processing")
    args = parser.parse_args()

    if args.mode == "process_main":
        process_main_folder(args)
    elif args.mode == "process_single_shard":
        process_shard(args.shard_path, args.output_path, args.tokenizer_name, args.batch_size)

    # python decode_all.py -t princeton-nlp/Llama-3-8B-ProLong-64k-Base --mode process_main -s data/chunked-olmo-1024-512-128-backfill-nodups/train -o data/chunked-olmo-1024-512-128-backfill-nodups/train_text
