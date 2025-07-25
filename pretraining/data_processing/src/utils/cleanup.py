from tokenize_mds_subfolders import cleanup_folder
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_path", type=str, required=True)
    args = parser.parse_args()
    
    cleanup_folder(args.root_path)