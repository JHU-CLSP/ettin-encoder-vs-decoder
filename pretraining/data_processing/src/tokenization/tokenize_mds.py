import argparse
import json
import os
import tqdm
from transformers import AutoTokenizer, set_seed
from streaming import StreamingDataset, MDSWriter
from ettin_data.utils.data_utils import MDS_COLS_PRE_TOKENIZED
import multiprocessing as mp
from functools import partial
import numpy as np
import uuid
import gc
import streaming

def get_uuid():
    return str(uuid.uuid4())

def tokenize_batch(tokenizer, batch):
    tokenized = tokenizer(batch, truncation=False, padding=False, return_tensors="np")
    # Convert each sequence individually to uint32
    input_ids = [np.array(seq, dtype=np.uint32) for seq in tokenized['input_ids']]
    return input_ids

def process_chunk(chunk, tokenizer):
    assert isinstance(chunk, list) and all(isinstance(item, dict) for item in chunk), "Chunk should be a list of dictionaries"
    texts = [item["text"] for item in chunk]
    if "id" not in chunk[0]:
        # create ids from uuids
        ids = [str(uuid.uuid4()) for _ in chunk]
    else:
        ids = [item["id"] for item in chunk]
    input_ids, attention_mask = tokenize_batch(tokenizer, texts)
    assert len(ids) == len(input_ids) == len(attention_mask) == len(texts), f"Length mismatch in chunk. {len(ids)} {len(input_ids)} {len(attention_mask)} {len(texts)}"
    return [{'id': id, 'input_ids': input_id, 'attention_mask': mask, "len": len(input_id)}
            for id, input_id, mask in zip(ids, input_ids, attention_mask)]

def sample_dataset_from_config(args):
    assert os.path.isdir(args.dataset), f"Dataset {args.dataset} does not exist."
    print(f"Using local dataset {args.dataset}...")
    # clean up the shared memory
    streaming.base.util.clean_stale_shared_memory()
    dataset = StreamingDataset(local=args.dataset, shuffle=False, split=None, batch_size=1, keep_zip=False)

    print(f"Using tokenizer model {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, add_prefix_space=True)

    output_dir = args.dataset + "-tokenized"

    num_tokens = 0
    num_truncated_tokens = 0
    batch_size = 5000
    if args.has_domains:
        MDS_COLS_PRE_TOKENIZED["domain"] = "str"

    with MDSWriter(out=output_dir, columns=MDS_COLS_PRE_TOKENIZED, compression='zstd') as mds_writer:
        pbar = tqdm.tqdm(total=len(dataset), desc="Processing samples")
        
        for i in range(0, len(dataset), batch_size):
            # Get a single batch of data
            end = min(i + batch_size, len(dataset))
            chunk = [dataset[k] for k in range(i, end)]
            
            if not chunk:
                break
            
            # Process the chunk directly without multiprocessing
            batch_results = process_chunk(chunk, tokenizer)
            
            # Write results immediately
            for item in batch_results:
                mds_writer.write(item)
                num_tokens += item['len']
                num_truncated_tokens += min(1024, item['len'])
                
                # Clear item from memory
                del item
            
            # Clear results from memory
            del batch_results
            gc.collect()
            
            pbar.update(len(chunk))
            
            if args.debug and pbar.n >= 100 * batch_size:
                break

    print(f"Finished writing with a total of {num_tokens} train tokens.")
    # save a json file in the directory with the number of tokens and the number of truncated tokens
    with open(os.path.join(output_dir, "num_tokens.json"), "w") as f:
        f.write(json.dumps({"num_tokens": num_tokens, "num_truncated_tokens": num_truncated_tokens}, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-t", "--tokenizer", type=str, required=True)
    parser.add_argument("--has_domains", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    set_seed(123456789)
    sample_dataset_from_config(args)

    # python tokenize_mds.py --dataset data/arxiv/ --tokenizer bclavie/olmo_bert_template 