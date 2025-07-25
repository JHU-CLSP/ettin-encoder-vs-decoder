import tqdm
import argparse
import random
import json
import datasets
import requests
import math
import os
import gzip
import time
import numpy as np
import multiprocessing
import huggingface_hub
import glob
import tempfile
import gc
import resource
import psutil
from torchdata.stateful_dataloader import StatefulDataLoader
import tracemalloc

from datasets import load_dataset, Dataset, DatasetDict, interleave_datasets
from streaming.base.util import _merge_index_from_root, merge_index
from transformers import set_seed, AutoTokenizer
from streaming import MDSWriter, StreamingDataset
from huggingface_hub import HfFileSystem

from src.initial_dataset_creation.dolma_urls import DOLMA_URLS
from src.utils.data_utils import MDS_COLS_TEXT


set_seed(11111111)

FILES_INFO = None


def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1e9:.2f} GB")


def sample_hf(repo_name, split_name, config_name, resume):
    print(f"Sampling the data with repo {repo_name} and {split_name} and {config_name}...")

    if config_name is not None and split_name:
        dataset = load_dataset(repo_name, config_name, streaming=True)[split_name]
    elif config_name is not None:
        dataset = load_dataset(repo_name, config_name, streaming=True)
    elif split_name is not None:
        dataset = load_dataset(repo_name, streaming=True)[split_name]
    else:
        dataset = load_dataset(repo_name, streaming=True)

    # if loading something with a column that can't be collated:
    # dataset = dataset.remove_columns(["metadata"]) 

    # Add this line to keep only id and text columns
    dataset = dataset.select_columns(['id', 'text'])


    instances_per_subfolder = 30000
    dl = StatefulDataLoader(dataset, num_workers=30, prefetch_factor=1, batch_size=instances_per_subfolder, persistent_workers=False, shuffle=False)

    config_name_dirsafe = config_name.replace("/", "-") if config_name is not None else "default"
    split_name_dirsafe = split_name.replace("/", "-") if split_name is not None else "default"
    tmp_cache_dir = f"data/text/{repo_name.replace('/', '-')}---{split_name_dirsafe}---{config_name_dirsafe}"

    if args.resume:
        with open(args.resume, "r") as f:
            dl_state_dict = json.load(f)
        dl.load_state_dict(dl_state_dict)
        shard_offset = dl_state_dict["_snapshot"]["_snapshot_step"]
        print(f"Resuming from shard {shard_offset}")


        # if the resume file is the biggest resume file
        resume_files = glob.glob(f"{tmp_cache_dir}/dataloader_state_*.json")
        resume_files = [int(f.split("_")[-1].split(".")[0]) for f in resume_files]
        resume_files.sort()
        # for each one, assert that a corresponding split file exists otherwise remove it
        for f in resume_files:
            if not os.path.isfile(f"{tmp_cache_dir}/split_{f}/index.json"):
                print(f"Resume file {f} does not have a corresponding split file, removing it")
                resume_files.remove(f)

        if (shard_offset - 1) == resume_files[-1]:
            # if there exists files past that cur_shard offset, remove them
            shard_files = glob.glob(f"{tmp_cache_dir}/split_*")
            shard_files = [int(f.split("_")[-1]) for f in shard_files]
            shard_files = [f for f in shard_files if f >= shard_offset]
            # sort and print them
            shard_files.sort()
            print(f"Found {len(shard_files)} files past the offset: {shard_files}")
            print(f"Sleeping for 10 seconds to make sure all files are closed")
            time.sleep(10)
            for f in shard_files:
                os.system(f"rm -rf {tmp_cache_dir}/split_{f}")
        else:
            print(f"Shard offset is not the biggest resume file, not removing files...\n\nYou may want to use the biggest resume file: {resume_files[-1]}")
            time.sleep(10)
    else:
        shard_offset = 0


    if not os.path.isfile(os.path.join(tmp_cache_dir, "index.json")):
        print(f"Writing to MDS...")
        # tracemalloc.start()

        is_paused = False
        for shard, batch in enumerate(tqdm.tqdm(dl)):
            gc.collect()

            cur_shard = shard + shard_offset
            cur_out_dir = os.path.join(tmp_cache_dir, f"split_{cur_shard}")
            if cur_shard % 100 == 0 and cur_shard != 0:
                # save the state of the dataset
                dataset_state_dict = dl.state_dict()
                with open(os.path.join(tmp_cache_dir, f"dataloader_state_{cur_shard}.json"), "w") as f:
                    json.dump(dataset_state_dict, f)

                # Your main processing code here
                # snapshot = tracemalloc.take_snapshot()
                # top_stats = snapshot.statistics('lineno')
                # print("[ Top 10 memory users ]")
                # for stat in top_stats[:10]:
                #     print(stat)

            with MDSWriter(out=cur_out_dir, columns=MDS_COLS_TEXT, compression='zstd') as train_writer:
                inst_batch = [dict(zip(batch,t)) for t in zip(*batch.values())]
                for idx, item in enumerate(inst_batch):
                    if "id" not in item:
                        item["id"] = str(idx + cur_shard * instances_per_subfolder)
                    train_writer.write(item)


        print(f"Pushing to HF...")
        # merge them all together by gathering all index.json files
        string_files = list(glob.glob(tmp_cache_dir + "/*/index.json", recursive=True))
        if len(string_files):
            print(f"Merging at root folder: {tmp_cache_dir}, {len(string_files)} files")
            _merge_index_from_root(tmp_cache_dir)

    dataset = StreamingDataset(local=tmp_cache_dir, shuffle=False, split=None, batch_size=1)
    num_instances = len(dataset)
    print(f"Number of instances: {num_instances}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repo_name", type=str, required=True)
    parser.add_argument("-s", "--repo_split", type=str, required=False)
    parser.add_argument("-c", "--repo_config", type=str, required=False)
    parser.add_argument("-m", "--max_shards", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    sample_hf(args.repo_name, args.repo_split, args.repo_config, args.resume)

    # example usage:
    #   python hf_to_mds.py -r HuggingFaceFW/fineweb-edu -c sample-10BT -s train
