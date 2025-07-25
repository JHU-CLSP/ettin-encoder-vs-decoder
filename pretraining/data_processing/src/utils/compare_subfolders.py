import os
import argparse
from torch.utils.data import DataLoader
from streaming import StreamingDataset
import os
import tqdm
import streaming


def compare(args):

    # first check that all folders have a index.json file
    for folder in os.listdir(args.local):
        if not os.path.isdir(os.path.join(args.local, folder)):
            continue

        if folder.endswith("-tokenized"):
            continue

        local = os.path.join(args.local, folder)
        if not os.path.exists(os.path.join(local, "index.json")):
            print(f"Folder: {folder} does not have index.json file")
        
        local_tokenized = os.path.join(args.local, folder + "-tokenized")
        if not os.path.exists(os.path.join(local_tokenized, "index.json")):
            print(f"Tokenized folder: {local_tokenized} does not have index.json file")

    # get all folders and sort them
    folders = sorted(os.listdir(args.local))
    if args.skip:
        folders = folders[args.skip:]
    for folder in tqdm.tqdm(folders):
        if not os.path.isdir(os.path.join(args.local, folder)):
            continue

        if folder.endswith("-tokenized"):
            continue

        if folder == "data":
            continue

        streaming.base.util.clean_stale_shared_memory()

        local = os.path.join(args.local, folder)
        dataset = StreamingDataset(local=local, shuffle=False, split=None, batch_size=1, predownload=1)
        len_og = len(dataset)

        local_tokenized = os.path.join(args.local, folder + "-tokenized")
        try:
            dataset_tokenized = StreamingDataset(local=local_tokenized, shuffle=False, split=None, batch_size=1, predownload=1)
            len_tok = len(dataset_tokenized)
        except Exception as e:
            print(f"Error loading tokenized dataset: {e}")
            len_tok = 0

        if len_og != len_tok:
            print(f"Folder: {folder} has different lengths: {len_og} vs {len_tok}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--local", type=str, required=True)
    parser.add_argument("-s", "--skip", type=int, default=0)
    args = parser.parse_args()
    compare(args)

    # NOTE: the skip is there since it only seems to do about 10k before it opens too many files
    # so it allows it to try again from the failure point
    # python compare_subfolders.py -l data/text/datasets--orionweller--dclm-1T-sample/snapshots/e01c4d93f79aacd04361454cc360da67eefab9a3 -s 42000