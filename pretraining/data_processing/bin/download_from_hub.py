import argparse
from huggingface_hub import snapshot_download
import glob
import os

def download_files(repo_id, pattern, local_dir, token=None):
    """
    Download files from a Hugging Face repository that match a specific pattern.
    
    Args:
        repo_id (str): The repository ID (e.g., 'username/repo-name')
        pattern (str): File pattern to match (e.g., '*.txt', 'model/*.safetensors')
        local_dir (str): Local directory to save the files
        token (str, optional): Hugging Face authentication token for private repos
    """
    try:
        # Create the local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        if token is None:
            token = os.environ.get("HF_TOKEN")

        assert token is not None, "HF_TOKEN environment variable is not set"

        # Use the allow_patterns parameter to filter files
        print(f"Downloading files from {repo_id} with pattern `{pattern}` to {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            # allow_patterns=pattern,
            local_dir=local_dir,
            token=token,
            repo_type="dataset",
        )
        
        print(f"Successfully downloaded files matching '{pattern}' to {local_dir}")
        
    except Exception as e:
        print(f"Error downloading files: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Download files from Hugging Face Hub')
    parser.add_argument('--repo', required=True, help='Repository ID (e.g., username/repo-name)')
    parser.add_argument('--pattern', required=True, help='File pattern to match (e.g., *.txt)')
    parser.add_argument('--output', required=True, help='Local directory to save files')
    parser.add_argument('--token', help='Hugging Face authentication token', default=None)
    
    args = parser.parse_args()
    
    download_files(args.repo, args.pattern, args.output, args.token)

if __name__ == "__main__":
    main()
    # this file is most useful for downloading a single dataset from huggingface