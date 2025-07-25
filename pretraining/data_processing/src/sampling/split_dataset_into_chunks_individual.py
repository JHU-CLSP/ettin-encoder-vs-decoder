import argparse
import os
from tqdm import tqdm
from streaming import MDSWriter, StreamingDataset
import streaming
from ettin_data.utils.data_utils import MDS_COLS_OUTPUT_ONLY
import numpy as np
import gc
import json
from transformers import set_seed
import random
from transformers import AutoTokenizer


set_seed(42)

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", use_fast=True, add_prefix_space=True)
tokenizer.eos_token_id = 50282
tokenizer.bos_token_id = 50281


def enforce_prefix_space(chunk, chunk_size):
    # De-tokenize the entire chunk
    chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
    if chunk_text[0] == " ":
        # return it as is
        return chunk

    # Re-encode with add_prefix_space=True (which is already set in the tokenizer)
    new_tokens = tokenizer.encode(chunk_text, add_special_tokens=True, return_tensors="np")[0]

    # Use the new tokens, making sure we don't exceed the original chunk size
    if len(new_tokens) <= chunk_size:
        chunk = new_tokens
    else:
        # If new tokens are longer, just keep what fits
        chunk = new_tokens[:chunk_size]
    assert len(chunk), f"Chunk is empty: `{chunk_text}`"
    return chunk.astype(np.uint32)

def ensure_special_tokens(chunk, bos_token, eos_token, chunk_size, i):
    chunk = enforce_prefix_space(chunk, chunk_size) #  if i != 0 else chunk
    assert chunk.ndim == 1, f"Chunk is not 1D: {chunk.shape}"
    if chunk[0] != bos_token:
        # print(f"Adding bos token: {BOS_TOKEN}")
        chunk = np.concatenate([[BOS_TOKEN], chunk])
    # if chunk[-1] != EOS_TOKEN:
    #     # print(f"Adding eos token: {EOS_TOKEN}")
    #     chunk = np.concatenate([chunk, [EOS_TOKEN]])
    return chunk.astype(np.uint32)


def chunk_instance(tokens, chunk_size, min_chunk_size, always_skip_size, backfill=False, backfill_no_duplicates=False, enforce_min_chunk_size_zeroth=False, add_eos_token=False):
    chunks = []
    amount_duplicated = 0
    amount_skipped = 0

    if add_eos_token:
        # samples are missing eos, add it
        if tokens[-1] != EOS_TOKEN:
            tokens = np.concatenate([tokens, [EOS_TOKEN]])
    
    # create initial chunks
    chunk_size_for_cls_eos = chunk_size - 2
    initial_chunks = np.array_split(tokens, np.arange(chunk_size_for_cls_eos, len(tokens), chunk_size_for_cls_eos))
    
    for i, chunk in enumerate(initial_chunks):
        # if the chunk is smaller than always_skip_size, skip it - unless it's the first
        if (len(chunk) < always_skip_size and i != 0) or (len(chunk) < always_skip_size and enforce_min_chunk_size_zeroth):
            amount_skipped += len(chunk)
            continue
        
        if len(chunk) < min_chunk_size:
            if backfill and chunks:
                # backfill with the previous chunk
                prev_chunk = chunks[-1]
                if not backfill_no_duplicates:
                    assert False, f"Are you sure you want to backfill with duplicates?"
                    extra_tokens_needed = chunk_size_for_cls_eos - len(chunk)
                    amount_duplicated += extra_tokens_needed
                    backfilled_chunk = np.concatenate([prev_chunk[-extra_tokens_needed:], chunk])
                else:
                    # backfill with the previous chunk, a random amount to keep both instances < args.chunk_size
                    total_len = len(prev_chunk) + len(chunk)
                    # print(f"Total len: {total_len} with {len(prev_chunk)} and {len(chunk)}")
                    random_tokens_needed = np.random.randint(0, chunk_size_for_cls_eos-len(chunk))
                    if random_tokens_needed:
                        # print(f"Random tokens needed: {random_tokens_needed}")
                        backfilled_chunk = np.concatenate([prev_chunk[-random_tokens_needed:], chunk])
                        prev_chunk = prev_chunk[:-random_tokens_needed]
                        chunks[-1] = ensure_special_tokens(prev_chunk.astype(np.uint32), bos_token, eos_token, chunk_size, i)
                        # print(f"Chunks are now lengths: {len(prev_chunk)} and {len(backfilled_chunk)} ")
                    else:
                        backfilled_chunk = chunk

                assert len(backfilled_chunk)
                chunks.append(ensure_special_tokens(backfilled_chunk.astype(np.uint32), bos_token, eos_token, chunk_size, i))
            elif i == 0: # always keep the first chunk
                assert len(chunk)
                chunks.append(ensure_special_tokens(chunk.astype(np.uint32), bos_token, eos_token, chunk_size, i))
            else:
                # below min_chunk_size and no backfill, skip
                assert not backfill
                amount_skipped += len(chunk)
                
        else: # the chunk is large enough to not backfill
            chunks.append(ensure_special_tokens(chunk.astype(np.uint32), bos_token, eos_token, chunk_size, i))
    
    # assert len(chunks)
    return chunks, amount_duplicated, amount_skipped


def process_instance(instance, args):
    chunks, num_dups, num_skipped = chunk_instance(
        instance['input_ids'], 
        args.chunk_size, 
        args.min_chunk_size, 
        args.always_skip_size, 
        args.backfill,
        args.backfill_no_duplicates,
        args.enforce_min_chunk_size_zeroth,
        args.add_eos_token
    )
    return [{'input_ids': chunk, 'id': instance['id'], 'len': len(chunk)} for chunk in chunks], num_dups, num_skipped

def process_batch(batch, args):
    bos_token = 50281
    eos_token = 50282
    results = [process_instance(instance, args) for instance in batch]
    all_chunks = [chunk for result in results for chunk in result[0]]
    # check for special tokens
    assert not len([chunk for chunk in all_chunks if chunk['input_ids'][0] != bos_token or chunk['input_ids'][-1] != eos_token]), f"Special tokens not found in {args.subfolder_path}"
    total_dups = sum(result[1] for result in results)
    total_skipped = sum(result[2] for result in results)
    return all_chunks, total_dups, total_skipped


def process_subfolder(args):
    print(f"Processing subfolder: {args.subfolder_path}")
    
    if os.path.isdir(args.subfolder_path):
        # print(f"Using local dataset {args.subfolder_path}...")
        # streaming.base.util.clean_stale_shared_memory()
        dataset = StreamingDataset(local=args.subfolder_path, shuffle=False, split=None, predownload=8)
    else:
        print(f"Error: {args.subfolder_path} is not a directory.")
        return

    output_repo = args.subfolder_path.replace("-tokenized", f"-tokenized-chunked-{args.chunk_size}-{args.min_chunk_size}-{args.always_skip_size}-{'backfill' if args.backfill else 'no-backfill'}-{'nodups' if args.backfill_no_duplicates else 'dups'}")

    if os.path.exists(output_repo):
        if os.path.exists(os.path.join(output_repo, "stats.json")):
            print(f"Output repo {output_repo} already exists. Skipping...")
            return
        else:
            print(f"Output repo {output_repo} already exists but doesn't have an stats.json file. Deleting...")
            os.system(f"rm -rf {output_repo}")

    os.makedirs(output_repo, exist_ok=True)

    all_duplicated_tokens = 0
    total_tokens_skipped = 0
    distribution = []
    token_decile = {"0k": [], "1k": [], "2k": [], "3k": [], "4k": [], "5k": [], "6k": [], "7k": [], "8k": []}
    instance_i = 0
    with MDSWriter(out=output_repo, columns=MDS_COLS_OUTPUT_ONLY, compression='zstd') as writer:
        for i in tqdm(range(0, len(dataset), args.batch_size), total=(len(dataset) + args.batch_size - 1) // args.batch_size):
            batch = dataset[i:i+args.batch_size]
            processed_chunks, total_dups_batch, total_skipped_batch = process_batch(batch, args)
            all_duplicated_tokens += total_dups_batch
            total_tokens_skipped += total_skipped_batch
            # print(f"Number of chunks: {len(processed_chunks)}")
            # breakpoint()
            for chunk in processed_chunks:
                writer.write(chunk)
                distribution.append(chunk['len'])
                token_decile[f"{chunk['len'] // 1000}k"].append(instance_i)
                instance_i += 1
                
            del processed_chunks
            del batch
            gc.collect()

    print(f"Finished processing {args.subfolder_path}. Output written to {output_repo}")
    print(f"Total duplicated tokens: {all_duplicated_tokens}")
    total_tokens_written = sum(distribution)
    print(f"Total tokens written: {total_tokens_written}")
    print(f"Total tokens skipped: {total_tokens_skipped}")
    try:
        percentiles = np.percentile(distribution, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100])
        percentiles_pretty = {f"{p}th": int(percentiles[i]) for i, p in enumerate([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100])}
    except Exception as e:
        # make it NaN
        percentiles = [np.nan] * 13
        percentiles_pretty = {f"{p}th": percentiles[i] for i, p in enumerate([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100])}
    print(f"Percentiles: {percentiles_pretty}")
    with open(os.path.join(output_repo, "stats.json"), "w") as f:
        json.dump({
            "total_duplicated_tokens": all_duplicated_tokens,
            "total_tokens_written": total_tokens_written,
            "total_tokens_skipped": total_tokens_skipped,
            "percentiles": percentiles_pretty
        }, f)


    with open(os.path.join(output_repo, "token_decile.json"), "w") as f:
        json.dump(token_decile, f)

    del dataset
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfolder_path", type=str, required=True, help="Subfolder to process")
    parser.add_argument("-c", "--chunk_size", type=int, default=1024, help="Size of each chunk")
    parser.add_argument("-m", "--min_chunk_size", type=int, default=800, help="Minimum size of a chunk to keep")
    parser.add_argument("-a", "--always_skip_size", type=int, default=500, help="Always skip chunks smaller than this size")
    parser.add_argument("-b", "--backfill", action="store_true", help="Enable backfilling for short final chunks", default=False)
    parser.add_argument("--backfill_no_duplicates", action="store_true", help="Backfill without duplicating tokens", default=False)
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--enforce_min_chunk_size_zeroth", action="store_true", help="Enforce minimum chunk size for the first chunk", default=False)
    parser.add_argument("--add_eos_token", action="store_true", help="Add eos token to the end of the chunk", default=False)
    args = parser.parse_args()

    process_subfolder(args)

    # nice -n 5 python split_dataset_into_chunks_individual.py --subfolder_path data/tokenized_... -c 1024 -m 512 -a 128 --batch_size 1000 --backfill --backfill_no_duplicates --num_processes 2