#!/usr/bin/env python3
"""
JSONL to MDS Converter

This script converts JSONL files to MDS format with support for sharding.
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
import math
from pathlib import Path
from ettin_data.utils.data_utils import MDS_COLS_TEXT
from streaming import MDSWriter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("jsonl_to_mds")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert JSONL files to MDS format")
    
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for MDS files")
    parser.add_argument("--shard", action="store_true", help="Enable sharding of the output")
    parser.add_argument("--shard-size", type=int, default=10000, 
                        help="Number of instances per shard (default: 10000)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()

def count_jsonl_lines(file_path: str) -> int:
    """Count the number of lines in a JSONL file."""
    logger.info(f"Counting lines in {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        count = sum(1 for _ in f)
    logger.info(f"Found {count} lines in the input file")
    return count

def read_jsonl(file_path: str, start: Optional[int] = None, end: Optional[int] = None):
    """Read a JSONL file, optionally from start to end line."""
    logger.debug(f"Reading JSONL from {file_path}" + 
                (f" (lines {start} to {end})" if start is not None else ""))
    
    with open(file_path, 'r', encoding='utf-8') as f:
        if start is not None:
            # Skip to the start line
            for _ in range(start):
                next(f, None)
        
        count = 0
        for line in f:
            yield json.loads(line)
            count += 1
            if end is not None and count >= (end - (start or 0)):
                break

def convert_to_mds(
    input_path: str, 
    output_dir: str, 
    shard: bool = False,
    shard_size: int = 10000
):
    """Convert JSONL to MDS format."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    if shard:
        # Count total instances to determine number of shards
        total_instances = count_jsonl_lines(input_path)
        num_shards = math.ceil(total_instances / shard_size)
        logger.info(f"Sharding into {num_shards} shards with up to {shard_size} instances each")
        
        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = min((shard_idx + 1) * shard_size, total_instances)
            
            # Create shard directory
            shard_dir = os.path.join(output_dir, f"shard_{shard_idx:05d}")
            os.makedirs(shard_dir, exist_ok=True)
            
            logger.info(f"Processing shard {shard_idx+1}/{num_shards} (instances {start_idx}-{end_idx})")
            
            # Process this shard
            with MDSWriter(out=shard_dir, columns=MDS_COLS_TEXT, compression="zstd") as writer:
                for idx, item in enumerate(read_jsonl(input_path, start_idx, end_idx)):
                    if "id" not in item:
                        item["id"] = str(start_idx + idx)
                    elif type(item["id"]) == int:
                        item["id"] = str(item["id"])
                    writer.write(item)
            
            logger.info(f"Completed shard {shard_idx+1}/{num_shards}")
    else:
        # Process all data in a single folder
        logger.info(f"Processing all data to {output_dir}")
        
        with MDSWriter(out=output_dir, columns=MDS_COLS_TEXT, compression='zstd') as train_writer:
            for idx, item in enumerate(read_jsonl(input_path)):
                if "id" not in item:
                    item["id"] = str(idx)
                elif type(item["id"]) == int:
                    item["id"] = str(item["id"])
                train_writer.write(item)
        
        logger.info("Conversion completed successfully")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Starting conversion from {args.input} to {args.output}")
        
    # Perform conversion
    convert_to_mds(
        input_path=args.input,
        output_dir=args.output,
        shard=args.shard,
        shard_size=args.shard_size
    )
    
    logger.info(f"Conversion completed. MDS data written to {args.output}")

if __name__ == "__main__":
    main()

    # example to convert: extracted/txt-files.jsonl
    # python ./bin/jsonl_to_mds.py --input extracted/txt-files.jsonl --output data/text/gutenberg --shard --shard-size 50000 --verbose