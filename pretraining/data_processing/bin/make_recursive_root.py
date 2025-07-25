import os
import glob
from streaming.base.util import merge_index
import typer

def recursive_merge(current_folder):
    subdirs = [d for d in glob.glob(os.path.join(current_folder, '*')) if os.path.isdir(d)]

    # Check if all direct subdirectories have index.json
    all_subdirs_have_index = all(os.path.exists(os.path.join(d, 'index.json')) for d in subdirs)

    if all_subdirs_have_index:
        # If all subdirectories have index.json, merge them
        index_files = [os.path.join(d, 'index.json') for d in subdirs]
        print(f"Merging {len(index_files)} index files in {current_folder}")
        print(f"Example index file: {index_files[0]}")
        merge_index(index_files, current_folder)
    else:
        # If any subdirectory doesn't have index.json, process it recursively
        for subdir in subdirs:
            if not os.path.exists(os.path.join(subdir, 'index.json')):
                recursive_merge(subdir)

        # After processing subdirectories, check again if we can merge
        if all(os.path.exists(os.path.join(d, 'index.json')) for d in subdirs):
            index_files = [os.path.join(d, 'index.json') for d in subdirs]
            print(f"Merging {len(index_files)} index files in {current_folder}")
            print(f"Example index file: {index_files[0]}")
            merge_index(index_files, current_folder)

app = typer.Typer()

@app.command()
def recursively_make_root(
    folder_path: str = typer.Argument(..., help="Path to the folder to process"),
):
    # Start the recursive merging process
    recursive_merge(folder_path)

if __name__ == "__main__":
    app()

    # NOTE: you can easily combine mds datasets with this simple script
    # python make_recursive_root.py /path/to/folder/of/train 