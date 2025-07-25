#!/bin/bash


datasets=(
    # add dataset with local paths here like
    "data/text/datasets--orionweller--wikipedia_mds_incremental/snapshots/aff2afa7d7274979206600f1b53d7869eebc3dc9"
)

for dataset in "${datasets[@]}"
do
    echo "Tokenizing $dataset"
    python src/ettin_data/tokenization/tokenize_mds_subfolders.py -t answerdotai/ModernBERT-base -r $dataset -n 40
    # sometimes you have to run the above multiple times and then run the below
    python src/ettin_data/utils/compare_subfolders.py -l $dataset
    python src/ettin_data/tokenization/move_tokenized.py $dataset --tokenizer_name olmo_space
    python ./bin/make_root.py $dataset --tokenizer_name olmo_space
    python bin/count_tokenized_tokens_from_file.py --dataset_path $dataset-tokenized --tokenizer_name olmo_space
done