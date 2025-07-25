import os
import argparse
from huggingface_hub import HfApi, HfFolder
from huggingface_hub.utils import RepositoryNotFoundError

MAX_FOLDERS = 100000  # HF doesn't allow more than this number of folders in a repo

def get_existing_folders(api, repo_id):
    try:
        return set(item.rfilename.split('/')[0] for item in api.list_repo_files(repo_id, repo_type="dataset") if '/' in item.rfilename)
    except RepositoryNotFoundError:
        return set()

def upload_subfolder(api, repo_id, subfolder_path, subfolder_name):
    print(f"Uploading subfolder: {subfolder_name}")
    try:
        api.upload_folder(
            folder_path=subfolder_path,
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=subfolder_name,
            commit_message=f"Upload subfolder: {subfolder_name}",
            create_pr=False,
            multi_commits=True,  # Explicitly use multi_commits
            multi_commits_verbose=True  # Add verbosity for better tracking
        )
        print(f"Successfully uploaded {subfolder_name}")
        return True
    except Exception as e:
        print(f"Error uploading {subfolder_name}: {str(e)}")
        return False

def upload_subfolders(args):
    api = HfApi()
    
    # Create repo if it doesn't exist
    if not args.skip_create:
        try:
            api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
            print(f"Repository {args.repo} created or already exists.")
        except Exception as e:
            print(f"Error creating repository: {str(e)}")
            return

    # Get list of existing folders
    existing_folders = get_existing_folders(api, args.repo)
    print(f"Existing folders: {existing_folders}")

    # Get list of local subfolders
    subfolders = [f for f in os.listdir(args.folder) if os.path.isdir(os.path.join(args.folder, f))]
    
    # Check if number of subfolders exceeds the limit
    if len(subfolders) > MAX_FOLDERS:
        print(f"Error: The number of subfolders ({len(subfolders)}) exceeds the maximum allowed ({MAX_FOLDERS}).")
        return

    # Upload each subfolder
    for subfolder in subfolders:
        if subfolder in existing_folders:
            print(f"Skipping {subfolder} as it already exists in the repository.")
            continue

        subfolder_path = os.path.join(args.folder, subfolder)
        success = upload_subfolder(api, args.repo, subfolder_path, subfolder)
        
        if success:
            existing_folders.add(subfolder)
        else:
            print(f"Failed to upload {subfolder}. Skipping to next subfolder.")

    print("Upload process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload subfolders to Hugging Face Hub")
    parser.add_argument("-f", "--folder", type=str, help="The parent folder containing subfolders to upload", required=True)
    parser.add_argument("-r", "--repo", type=str, help="The repo to upload to", required=True)
    parser.add_argument("--skip_create", action="store_true", help="Skip creating the repository")
    args = parser.parse_args()
    upload_subfolders(args)

    # python upload_dataset_by_subfolders.py -f data/tokenized_olmo/datasets--orionweller--algebraic-stack_mds_incremental/snapshots/5af697376cc89b191fef8b7873280e2c393e8361-tokenized