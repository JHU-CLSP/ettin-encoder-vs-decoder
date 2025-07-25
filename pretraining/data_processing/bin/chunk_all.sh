#!/bin/bash

datasets=(
    # list datasets here
)


for dataset in "${datasets[@]}"
do
    # change /text/ to /tokenized_olmo/
    dataset=$(echo $dataset | sed 's/text\//tokenized_olmo\//')
    # add -tokenized to the end
    dataset="$dataset-tokenized"
    echo "Chunking $dataset"
    python src/ettin_data/sampling/split_dataset_into_chunks.py -s $dataset -c 8192 -m 512 -a 32 --batch_size 1000 --backfill --backfill_no_duplicates --num_processes 40 --add_eos_token --reverse
done

## after run:
# python src/ettin_data/sampling/move_chunks.py data/tokenized_olmo/ 8192-512-32-backfill-nodups
# python ./bin/compile_final_dataset_stats.py data/chunked-olmo-8192-512-32-backfill-nodups/

