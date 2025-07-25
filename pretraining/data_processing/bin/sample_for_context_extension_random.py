import argparse
import json
import os
import glob
import random
import logging
from typing import Dict, List, Tuple
from streaming.base.util import _merge_index_from_root, merge_index
from streaming import MDSWriter, StreamingDataset
import streaming
from tqdm import tqdm

random.seed(12345)

MDS_COLS_OUTPUT_ONLY = {
    "input_ids": "ndarray:uint32",
    "id": "str",
}

def setup_logging(log_file: str = 'token_sampling.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample tokens for LLM training")
    parser.add_argument("chunked_data_path", type=str, help="Path to the chunked data folder")
    parser.add_argument("output_path", type=str, help="Path to save the sampled data")
    parser.add_argument("--num_tokens", type=int, default=10_000_000, help="Number of tokens to sample")
    parser.add_argument("--skip_domains", type=str, default="", help="Comma-separated list of domains to skip")
    parser.add_argument("--keep_domain", type=str, default="", help="Comma-separated list of domains to keep")
    parser.add_argument("--log_file", type=str, default="token_sampling.log", help="Path to the log file")
    return parser.parse_args()

def load_dataset_percentages() -> Dict[str, float]:
    return {
        "mlfoundations-dclm-baseline-1.0-parquet": 100.00,
    }

def recalculate_percentages(percentages: Dict[str, float], skip_domains: List[str]) -> Dict[str, float]:
    total = sum(value for key, value in percentages.items() if key not in skip_domains)
    return {key: (value / total) * 100 for key, value in percentages.items() if key not in skip_domains}

def sample_tokens(
    chunked_data_path: str,
    output_path: str,
    num_tokens: int,
    chunk_size: int,
    keep_domain: str
) -> None:
    if keep_domain:
        percentages = {keep_domain: 100}
    else:
        percentages = load_dataset_percentages()
        if skip_domains:
            percentages = recalculate_percentages(percentages, skip_domains)
    
    logging.info(f"Starting token sampling process. Target: {num_tokens} tokens")
    
    total_sampled = 0
    for domain, percentage in tqdm(percentages.items(), desc="Sampling domains"):
        # multiprocess this
        domain_tokens = int(num_tokens * (percentage / 100))
        logging.info(f"Sampling {domain_tokens} tokens from {domain} ({percentage:.2f}%)")
        sampled = sample_domain_tokens(chunked_data_path, output_path, domain, domain_tokens)
        total_sampled += sampled
        logging.info(f"Sampled {sampled} tokens from {domain}")
    
    logging.info(f"Sampling complete. Total sampled: {total_sampled} tokens")

def sample_domain_tokens(
    chunked_data_path: str,
    output_path: str,
    domain: str,
    num_tokens: int,
) -> int:
    domain_path = os.path.join(chunked_data_path, domain)
    shards = list(set(list(os.listdir(domain_path))))
    
    tokens_sampled = 0
    with tqdm(total=num_tokens, desc=f"Sampling {domain}", unit="tokens") as pbar:
        while tokens_sampled < num_tokens:
            random.shuffle(shards)
            for shard in shards:
                shard_path = os.path.join(domain_path, shard)
                token_decile_path = os.path.join(shard_path, "token_decile.json")
                
                if not os.path.exists(token_decile_path):
                    logging.warning(f"Token decile file not found for {shard_path}. Skipping")
                    continue
                    
                # json file with links from 0k -> doc_ids, 1k -> doc_ids, etc.
                with open(token_decile_path, "r") as f:
                    token_deciles = json.load(f)


                # get max key from them, by flattening the lists and taking a max
                all_values = [value for key, value in token_deciles.items()]
                flat_values = [item for sublist in all_values for item in sublist]
                max_key = max(flat_values)
                indexes = list(range(max_key + 1))
                random.shuffle(indexes)
                
                new_tokens = sample_shard_tokens(shard_path, output_path, indexes, num_tokens - tokens_sampled)
                tokens_sampled += new_tokens
                pbar.update(new_tokens)

                if tokens_sampled >= num_tokens:
                    break
      
            logging.warning(f"Exhausted all chunk sizes for {domain}")
    
    return tokens_sampled

def sample_shard_tokens(
    shard_path: str,
    output_path: str,
    indexes: List[int],
    remaining_tokens: int,
) -> int:
    tokens_sampled = 0

    subdomain = os.path.basename(os.path.dirname(shard_path))
    shard_num = os.path.basename(shard_path).split("-")[0] if "code_repos" not in shard_path else "=".join(os.path.basename(shard_path).split("-")[:2])
    # clear the cache
    streaming.base.util.clean_stale_shared_memory()
    reader = StreamingDataset(local=shard_path, shuffle=False, split=None, predownload=8)
    # make sure we can load the first
    assert len(reader[0]["input_ids"])
    write_out_location = os.path.join(output_path, subdomain, shard_num)
    os.makedirs(write_out_location, exist_ok=True)
    with MDSWriter(out=write_out_location, columns=MDS_COLS_OUTPUT_ONLY, compression="zstd") as writer:
        for index in tqdm(indexes, leave=False):
            document = reader[index]
            tokens_sampled += len(document["input_ids"])
            writer.write(document)
            
            if tokens_sampled >= remaining_tokens:
                break
    
    return tokens_sampled

def merge(root_folder):
    # merge them all together by gathering all index.json files
    string_files = list(glob.glob(root_folder + "/*/index.json", recursive=True))
    print(f"Merging {len(string_files)} files")
    if len(string_files):
        print(f"Merging at root folder: {root_folder}, {len(string_files)} files")
        _merge_index_from_root(root_folder)

    print(f"Merged to {root_folder}/index.json")

def main() -> None:
    args = parse_arguments()
    setup_logging(args.log_file)
    
    logging.info(f"Starting script with arguments: {args}")
    
    skip_domains = args.skip_domains.split(",") if args.skip_domains else []
    if skip_domains:
        logging.info(f"Skipping domains: {skip_domains}")
    
    sample_tokens(
        args.chunked_data_path,
        args.output_path,
        args.num_tokens,
        skip_domains,
        args.keep_domain
    )

    # will combine all the shard index.json files into one overall dataset index.json file
    merge(args.output_path)

    logging.info("Script completed successfully")

if __name__ == "__main__":
    main()

    # python ./bin/sample_for_context_extension_random.py ./data/chunked-olmo-8000-512-128-backfill-nodups output_data/ --num_tokens 4_000_000_000 --keep_domain mlfoundations-dclm-baseline-1.0-parquet