import argparse
import os
import glob
import shutil
import huggingface_hub

def download_dataset(args):
    print(f"Downloading {args.repo} to {args.cache_dir}")
    root_folder = huggingface_hub.snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        cache_dir=args.cache_dir,
    )
    print(f"Downloaded to {root_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repo", type=str, required=True, help="The HF repo to download")
    parser.add_argument("-c", "--cache_dir", type=str, default="data/text/", help="The cache directory")
    args = parser.parse_args()
    download_dataset(args)
