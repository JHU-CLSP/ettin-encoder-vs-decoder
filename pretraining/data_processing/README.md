# Data Processing Guide

⚠️ NOTE: I copied these in from another repo and so I may need to move things around to work with imports (sorry about that, please let me know!)

This directory contains all data preprocessing scripts and configurations used for training Ettin models. The preprocessing pipeline handles the three-phase training approach with different data mixtures and quality levels.

**Prerequisites:** Install dependencies with `pip install -r requirements.txt` before running any data processing scripts.

**Working with Existing Data:**
- You can easily add or remove dataset folders from existing data available on HF.
- After making changes, recombine everything using: `python bin/make_recursive_root.py $dataset`
- This creates a unified MDS dataset at the root level from all subfolders

**Important Tokenization Settings:**
- **For POS/NER tasks:** Use `use_prefix_space=True` during tokenization to preserve word boundaries

**Adding New Text Data:**
If you want to add new text datasets, follow this process:
1. **Convert to MDS format** → `src/initial_dataset_creation/hf_to_mds.py`
2. **Tokenize** → `src/tokenization/tokenize_mds.py` (with appropriate prefix space settings)
3. **Chunk** → `src/sampling/split_dataset_into_chunks.py`
4. **Decompress** → `bin/decompress.py` (if needed for final format)

## Overview

The data processing pipeline consists of several stages:

1. **Initial Data Collection**: Downloading and preparing raw datasets
2. **Tokenization**: Converting text to tokenized format  
3. **Chunking**: Creating training sequences with proper context lengths
4. **Sampling**: Creating balanced mixtures for different training phases

## Directory Structure

```
data_processing/
├── bin/                    # Executable scripts
├── src/                    # Source code modules
│   ├── initial_dataset_creation/  # Dataset download and conversion
│   ├── tokenization/             # Tokenization utilities
│   ├── sampling/                 # Data sampling and chunking
│   └── utils/                    # Utility functions
└── README.md                     # This file
```

## Complete Workflow

### Phase 1: Dataset Creation and Preparation

#### 1.1 Convert Dolma Datasets to MDS Format
```bash
# Convert individual Dolma subset to MDS and upload to HuggingFace
python src/initial_dataset_creation/dolma_to_mds.py -s <DATASET_TAG> -r <HF_REPO_NAME>

# Process all Dolma subsets at once
bash bin/dolma_mds_all.sh
```

#### 1.2 Convert Non-Dolma Datasets  
```bash
# Convert HuggingFace datasets to MDS format
python src/initial_dataset_creation/hf_to_mds.py \
    -r HuggingFaceFW/fineweb-edu \
    -c sample-10BT \
    -s train \
    -u orionweller/fineweb-edu-10B

# For large datasets, use stateful downloading with resume capability
python src/initial_dataset_creation/hf_to_mds.py \
    -r <DATASET> \
    --resume path_to_saved_state.json
```

### Phase 2: Tokenization

#### 2.1 Tokenize MDS Datasets
```bash
# Option 1: Tokenize with subfolders (preferred for Dolma, parallelizes better)
python src/tokenization/tokenize_mds_subfolders.py \
    --root_path DATA_FOLDER \
    --num_processes 50 \
    --tokenizer TOKENIZER

# Option 2: Tokenize single MDS file
python src/tokenization/tokenize_mds.py \
    --tokenizer TOKENIZER \
    -d DATASET
```

#### 2.2 Batch Tokenization
```bash
# Tokenize all datasets
bash bin/tokenize_all.sh

# Verify tokenization and move files (commands in tokenize_all.sh)
# python src/tokenization/move_tokenized.py $dataset --tokenizer_name olmo
```

#### 2.3 Create Root MDS Dataset
```bash
# Make MDS dataset at the root level
python bin/make_recursive_root.py $dataset --tokenizer_name olmo
```

### Phase 3: Chunking and Final Preparation

#### 3.1 Chunk the Tokenized Data
```bash
# Chunk all datasets with specified parameters
bash bin/chunk_all.sh

# This calls split_dataset_into_chunks.py with parameters:
# python src/sampling/split_dataset_into_chunks.py \
#     -s $dataset \
#     -c 1024 \          # chunk size
#     -m 512 \           # max length  
#     -a 128 \           # attention length
#     --batch_size 1000 \
#     --backfill \
#     --backfill_no_duplicates \
#     --num_processes 50
```

#### 3.2 Move Chunks to Organized Directory
```bash
# Move chunks to separate directory with config name
python src/sampling/move_chunks.py \
    data/tokenized_{model_name} \
    {chunked_config_name}  # e.g., 1024-512-128-backfill-nodups
```

#### 3.3 Create Training and Validation Sets
```bash
# For most datasets (will error if subfolder too small - by design)
# NOTE: create_chunked_final_data.sh exists in different directory  
# bash bin/create_chunked_final_data.sh

# For large datasets that error above, manually sample:
python src/sampling/sample_from_chunks_extra_large.py \
    data/chunked-olmo-1024-512-32-backfill-nodups \
    data/chunked-olmo-1024-512-128-backfill-nodups/mlfoundations-dclm-baseline-1.0-parquet-FULL \
    mlfoundations-dclm-baseline-1.0-parquet-sampled \
    --sample_fraction 0.0003333 \
    --force_redo

# NOTE: For very large datasets, first downsample then run extra_large sampling
# python bin/sample_too_large_down.py  # (available in bin/)
```

**⚠️ Important Downsampling Workflow:**
For very large datasets, you **must** follow this order:
1. First run `python bin/sample_too_large_down.py` to downsample
2. Then run `sample_from_chunks_extra_large.py` for final sampling

#### 3.4 Finalize Dataset Structure
```bash
# Move train/validation sets to final organized folder
python src/sampling/move_out_final_sampled_chunks.py \
    data/chunked-olmo-1024-512-32-backfill-nodups
```

#### 3.5 Verify and Create Final Index
```bash
# Create final index.json and verify instance counts
python src/utils/create_final_dataset_index.py \
    data/chunked-olmo-1024-512-128-backfill-nodups/

# Final token stats will be saved in stats.json in the main chunked directory
```

## Key Scripts Reference

### Data Download and Preparation
- `src/initial_dataset_creation/dolma_to_mds.py` - Convert Dolma datasets to MDS
- `src/initial_dataset_creation/hf_to_mds.py` - Convert HuggingFace datasets to MDS  
- `bin/dolma_mds_all.sh` - Batch process all Dolma datasets
- `bin/jsonl_to_mds.py` - Convert JSONL files to MDS format

### Tokenization
- `src/tokenization/tokenize_mds_subfolders.py` - Tokenize with subfolder parallelization
- `src/tokenization/tokenize_mds.py` - Tokenize single MDS dataset
- `bin/tokenize_all.sh` - Batch tokenize all datasets
- `src/tokenization/move_tokenized.py` - Verify and move tokenized files

### Sampling and Chunking  
- `src/sampling/split_dataset_into_chunks.py` - Split tokenized data into chunks
- `bin/chunk_all.sh` - Batch chunk all datasets
- `src/sampling/move_chunks.py` - Move chunks to organized directories
- `src/sampling/sample_from_chunks_extra_large.py` - Sample from very large datasets
- `src/sampling/move_out_final_sampled_chunks.py` - Finalize train/val structure
- `bin/sample_too_large_down.py` - Downsample oversized datasets

### Quality Control and Utilities
- `src/utils/create_final_dataset_index.py` - Create final dataset index
- `bin/make_recursive_root.py` - Create root-level MDS datasets
- `bin/sample_for_context_extension.py` - Sample data for context extension phase
- `bin/count_instances.py` - Count instances in processed datasets


## Usage Examples

### Basic Tokenization
```bash
# Tokenize a single dataset
python src/tokenization/tokenize_mds.py \
    --input_path /path/to/dataset \
    --output_path /path/to/tokenized \
    --tokenizer_name modernbert
```

### Sampling for Context Extension
```bash
# Sample data for 8k context extension  
python bin/sample_for_context_extension.py \
    /path/to/tokenized_data \
    output_dir \
    --num_tokens 10_000_000_000 \
    --chunk_size 8000
```

## Hardware Requirements

This can be quite compute intensive. We used upwards of 10T to download, tokenize, chunk, and prepare the data for the final format. If you are just doing the last phases (decay, context extension) it can be considerably less however.