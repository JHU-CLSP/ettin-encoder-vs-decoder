from tokenize_mds_subfolders import cleanup_folder
from data_utils import SOURCE_MAP


if __name__ == "__main__":
    for source_dir in SOURCE_MAP.values():
        print(f"Cleaning up {source_dir}-tokenized ...")
        cleanup_folder(source_dir + "-tokenized")