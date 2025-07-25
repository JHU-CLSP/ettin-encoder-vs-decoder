import huggingface_hub
import os
import argparse


def upload_folder(args):
    print(f"Creating a new repo {args.repo}")
    api = huggingface_hub.HfApi()
    repo_url = api.create_repo(
        args.repo,
        repo_type="dataset",
        exist_ok=True,
    )
    # Upload all the content from the local folder to your remote Space.
    # By default, files are uploaded at the root of the repo
    print(f"Uploading {args.folder} to {args.repo}")
    api.upload_large_folder(
        folder_path=args.folder,
        repo_id=args.repo,
        repo_type="dataset",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a folder to Hugging Face Hub")
    parser.add_argument("-f", "--folder", type=str, help="The folder to upload", required=True)
    parser.add_argument("-r", "--repo", type=str, help="The repo to upload to", required=True)
    args = parser.parse_args()
    upload_folder(args)


    # example usage:
    #   python push_folder_to_hub.py -f downloaded_data/fineweb-edu-350B -r orionweller/fineweb-edu-350B