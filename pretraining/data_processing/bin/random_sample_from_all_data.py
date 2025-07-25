import os
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import argparse
from streaming.base.format import reader_from_json
from streaming.base.spanner import Spanner
from streaming import MDSWriter
import json
from tqdm import tqdm
from multiprocessing import Pool, Queue, Process
from queue import Empty
from functools import partial

@dataclass
class DataShard:
    dirname: str
    split: str
    samples: int
    raw_data: Any
    
    def validate(self, check_local: bool) -> None:
        pass

def process_shard_batch(indices_chunk: np.ndarray, shards: List[DataShard], 
                       spanner: Spanner, pad_token_id: int) -> List[Dict[str, Any]]:
    """Process a batch of indices in parallel"""
    results = []
    for idx in indices_chunk:
        shard_id, shard_sample_id = spanner[idx]
        shard = shards[shard_id]
        sample = shard[shard_sample_id]
        
        if 'input_ids' in sample:
            if 'id' not in sample:
                sample['id'] = str(idx)
            results.append(sample)
    
    return results

class FastTokenSampler:
    def __init__(
        self, 
        local_path: str,
        split: str,
        batch_size: int = 1024,  # Increased batch size
        num_workers: int = 8,    # Default number of workers
        prefetch_factor: int = 20,
        pad_token_id: int = 50283,
        seed: int = 42
    ):
        self.local_path = local_path
        self.split = split
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        # Initialize dataset
        print(f"Loading shards from {self.local_path} split {self.split}")
        self.shards = self._load_shards()
        print(f"Loaded {len(self.shards)} shards")
        self.samples_per_shard = np.array([shard.samples for shard in self.shards], np.int64)
        self.total_samples = self.samples_per_shard.sum()
        print(f"Total samples: {self.total_samples:,}")
        self.spanner = Spanner(self.samples_per_shard)
        print(f"Spanner initialized")
        
        # One-time shuffle of all indices
        rng = np.random.Generator(np.random.PCG64DXSM(seed))
        self.shuffled_indices = rng.permutation(self.total_samples)
        print("Indices shuffled")

    def _load_shards(self) -> List[DataShard]:
        split_path = os.path.join(self.local_path, self.split)
        index_file_path = os.path.join(split_path, "index.json")
        
        with open(index_file_path) as f:
            obj = json.load(f)
            
        shards = []
        for info in tqdm(obj["shards"], desc="Loading shards"):
            shard = reader_from_json(self.local_path, self.split, info)
            raw_filename = os.path.join(shard.dirname, shard.split, shard.raw_data.basename)
            assert os.path.isfile(raw_filename), f"Raw file {raw_filename} does not exist"
            shards.append(shard)
        return shards

    def _prefetch_worker(self, queue: Queue, worker_id: int):
        """Worker process for prefetching data"""
        process_fn = partial(process_shard_batch, 
                           shards=self.shards, 
                           spanner=self.spanner,
                           pad_token_id=self.pad_token_id)
        
        # Each worker processes every nth batch
        for idx in range(worker_id * self.batch_size, len(self.shuffled_indices), 
                        self.batch_size * self.num_workers):
            batch_end = min(idx + self.batch_size, len(self.shuffled_indices))
            batch_indices = self.shuffled_indices[idx:batch_end]
            if len(batch_indices) > 0:  # Skip empty batches at the end
                results = process_fn(batch_indices)
                queue.put(results)
        
        queue.put(None)  # Signal completion

    def sample_tokens(self, target_tokens: int, output_dir: str):
        """Sample batches until reaching target number of tokens using parallel processing"""
        os.makedirs(output_dir, exist_ok=True)
        
        MDS_COLS_OUTPUT = {
            'input_ids': 'ndarray:uint32',
            'id': 'str',
        }

        print(f"Sampling {target_tokens:,} tokens using {self.num_workers} workers")
        
        # Create queues and processes for prefetching
        queue = Queue(maxsize=self.prefetch_factor * self.num_workers)
        chunk_size = len(self.shuffled_indices) // self.num_workers
        processes = []
        
        # Start prefetch workers
        for i in range(self.num_workers):
            p = Process(target=self._prefetch_worker, args=(queue, i))
            processes.append(p)
            p.start()

        total_tokens = 0
        workers_done = 0
        
        with MDSWriter(out=output_dir, columns=MDS_COLS_OUTPUT, compression="zstd", size_limit="100mb") as writer:
            pbar = tqdm(desc="Sampling tokens", total=target_tokens)
            
            while workers_done < self.num_workers and total_tokens < target_tokens:
                try:
                    batch = queue.get(timeout=60)  # 1-minute timeout
                    if batch is None:
                        workers_done += 1
                        continue
                        
                    for sample in batch:
                        if total_tokens >= target_tokens:
                            break
                            
                        if 'input_ids' in sample:
                            # Count tokens
                            mask = sample['input_ids'] != self.pad_token_id
                            sample_tokens = np.sum(mask)
                            
                            writer.write(sample)
                            total_tokens += sample_tokens
                            pbar.update(sample_tokens)
                            
                except Empty:
                    print("Warning: Queue timeout, checking worker status...")
                    
            pbar.close()
            
        # Clean up processes
        for p in processes:
            p.terminate()
            p.join()
            
        print(f"Finished sampling {total_tokens:,} tokens")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_tokens", type=int, default=75_000_000_000)
    parser.add_argument("--output_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    sampler = FastTokenSampler(
        local_path=args.local_path,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    sampler.sample_tokens(args.target_tokens, args.output_dir)

if __name__ == "__main__":
    main()

"""
# NOTE: this shows how to use the Spanner class to sample from the data when it's decompressed
python ./bin/random_sample_from_all_data.py \
    --local_path data/chunked-olmo-1024-512-32-backfill-nodups \
    --split train \
    --batch_size 10000 \
    --seed 42 \
    --output_dir data/olmo/train \
    --target_tokens 1_600_000_000_000
"""