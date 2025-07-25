import os
import gzip
import numpy as np
import multiprocessing
import huggingface_hub
import glob
import tempfile

from streaming.base.util import merge_index


def merge(root_folder):
    # merge them all together by gathering all index.json files
    string_files = list(glob.glob(root_folder + "/**/index.json", recursive=True))
    print(f"Merging {len(string_files)} files")
    merge_index(string_files, root_folder)

    print(f"Merged to {root_folder}/index.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge all index.json files in a directory")
    parser.add_argument("root_folder", help="Path to the root folder to process")
    args = parser.parse_args()

    merge(args.root_folder)

    # python merge_mds_to_one_index.py data/text/mlfoundations-dclm-baseline-1.0-parquet---train---small/