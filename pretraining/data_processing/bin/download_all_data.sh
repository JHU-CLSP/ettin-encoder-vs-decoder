#!/bin/bash
BASE_DIR=downloaded_data
mkdir -p $BASE_DIR

# this is just useful for bulk downloading
# for name in ./bin/dataset.txt, read in each line and call ./bin/download_folder.py -r $name

while IFS= read -r line
do
    echo "Downloading $line"
    nice -n 10 python ./bin/download_folder.py -r $line
done < bin/datasets.txt