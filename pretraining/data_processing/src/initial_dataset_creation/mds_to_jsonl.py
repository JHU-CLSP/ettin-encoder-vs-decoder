import json
import os
import huggingface_hub
import os
import tqdm
import random
from transformers import AutoTokenizer, set_seed
import datasets
import argparse
from streaming import StreamingDataset


from src.utils.data_utils import SOURCE_MAP


def mds_to_jsonl(args):
    source_repo = SOURCE_MAP[args.source]
    assert os.path.isdir(source_repo), f"Source {source_repo} does not exist."
    print(f"Using local dataset {source_repo}...")
    dataset = StreamingDataset(local=source_repo, shuffle=False, split=None, batch_size=1, shuffle_seed=9176)
    out_f = open(args.out_file, "w")
    for instance in tqdm.tqdm(dataset):
        out_f.write(json.dumps(instance) + "\n")


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, required=True)
    parser.add_argument("-o", "--out_file", type=str, required=True)
    args = parser.parse_args()
    
    mds_to_jsonl(args)
                