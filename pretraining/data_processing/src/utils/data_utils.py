

MDS_COLS_TOKENIZED = {
    'input_ids': 'ndarray:uint32',
    'id': 'str'
}

MDS_COLS_TEXT = {
    'text': 'str',
    'id': 'str'
}

MDS_COLS_PRE_TOKENIZED = {
    'input_ids': 'ndarray',
    'id': 'str',
    'len': 'int'
}

MDS_COLS_OUTPUT_ONLY = {
    'input_ids': 'ndarray:uint32',
    'id': 'str',
}


# path to the uploaded datasets on huggingface, not all of these were used
SOURCE_MAP_REMOTE = {
    "books": "orionweller/books_mds_incremental",
    "wiki": "orionweller/wikipedia_mds_incremental",
    "falcon": "orionweller/refinedweb_mds_incremental",
    "c4": "orionweller/c4_mds_incremental",
    "cc_en_head": "orionweller/cc_en_head_mds_incremental",
    "cc_en_tail": "orionweller/cc_en_tail_mds_incremental",
    "cc_en_middle": "orionweller/cc_en_middle_mds_incremental",
    "megawika": "orionweller/megawika_mds_incremental",
    "cc_news": "orionweller/cc_news_mds_incremental",
    "pes2o": "orionweller/pes2o_mds_incremental",
    "tulu_flan": "orionweller/tulu_flan_mds_incremental",
    "starcoder": "orionweller/starcoder_mds_incremental",
    "stackexchange": "orionweller/stackexchange_mds_incremental",
    "arxiv": "orionweller/arxiv_mds_incremental",
    "open_web_math_train": "orionweller/open-web-math_mds_incremental",
    "reddit": "orionweller/reddit_mds_incremental",
    "algebraic_stack_train": "orionweller/algebraic-stack_mds_incremental",
    "caselaw-access-project": "orionweller/caselaw-access-project",
    "fineweb-edu-10B": "orionweller/fineweb-edu-10B",
    "fineweb-edu-350B": "orionweller/fineweb-edu-350B",
    "fineweb-edu-score-2": "orionweller/fineweb-edu-score-2",
}



SOURCE_MAP = {
    # Path to the local downloaded versions of the dataset, not all of these were used
    "case_access_law": "ettin-data/data/text/TeraflopAI-Caselaw_Access_Project---train---default",
    "fineweb-edu": "ettin-data/data/text/HuggingFaceTB-smollm-corpus---train---fineweb-edu-dedup",
    "algebraic_stack_train": "ettin-data/data/text/datasets--orionweller--algebraic-stack_mds_incremental/snapshots/5af697376cc89b191fef8b7873280e2c393e8361",
    "arxiv": "ettin-data/data/text/datasets--orionweller--arxiv_mds_incremental/snapshots/640f80fa7d7ff93226a1f7115f70145fd1f4ead7",
    "books": "ettin-data/data/text/datasets--orionweller--books_mds_incremental/snapshots/502df43dc5445788353f1cf7befdc1a3cbedd6cb",
    "c4": "ettin-data/data/text/datasets--orionweller--c4_mds_incremental/snapshots/fdb71eeccbe17fc95d0e0330dea5f9f0e79c7aaa",
    "cc_en_head": "ettin-data/data/text/datasets--orionweller--cc_en_head_mds_incremental/snapshots/3f13a7e03eef6df4ee62b486ea912eb926e7be91",
    "cc_en_middle": "ettin-data/data/text/datasets--orionweller--cc_en_middle_mds_incremental/snapshots/4e4577a77611d8c6ebcefafa89d64ec7329d8d1b",
    "cc_en_tail": "ettin-data/data/text/datasets--orionweller--cc_en_tail_mds_incremental/snapshots/58bf1c63c23598548a1da42cc3dffe42d2672f80",
    "cc_news": "ettin-data/data/text/datasets--orionweller--cc_news_mds_incremental/snapshots/846a17dd910daf76ffd96fa735a3dd9240736116",
    "megawika": "ettin-data/data/text/datasets--orionweller--megawika_mds_incremental/snapshots/477460d68212afbf7937bbfe0143bf482651b684",
    "open_web_math_train": "ettin-data/data/text/datasets--orionweller--open-web-math_mds_incremental/snapshots/732910d828ea4f7e1ab62a7a787d5e3bd59210b0",
    "pes2o": "ettin-data/data/text/datasets--orionweller--pes2o_mds_incremental/snapshots/71ed50bcfd714e2360c4fcd59d601d4eecc9d1a2",
    "reddit": "ettin-data/data/text/datasets--orionweller--reddit_mds_incremental/snapshots/53d1edb1053ffa4b519ba45d9daed14fd82cfd68",
    "falcon": "ettin-data/data/text/datasets--orionweller--refinedweb_mds_incremental/snapshots/31ce550bcb0c117c0ce058f166795c338a8f6fa1",
    "stackexchange": "ettin-data/data/text/datasets--orionweller--stackexchange_mds_incremental/snapshots/dd95be271cac1709b0dda6776a6e83df93b4d1f0",
    "starcoder": "ettin-data/data/text/datasets--orionweller--starcoder_mds_incremental/snapshots/5cef62e15c251baa36779ebab15a9e5ba3a5d7a6",
    "tulu_flan": "ettin-data/data/text/datasets--orionweller--tulu_flan_mds_incremental/snapshots/7f00a393e1b26ee2d48b65ca26ab61fd6e82786e",
    "wiki": "ettin-data/data/text/datasets--orionweller--wikipedia_mds_incremental/snapshots/aff2afa7d7274979206600f1b53d7869eebc3dc9",
    "fineweb-edu-score-2": "ettin-data/data/text/datasets--orionweller--fineweb-edu-score-2/snapshots/755c506cae00da40a7cbe5d8b7dbcf7f6e171de9/HuggingFaceFW-fineweb-edu-score-2---train---default",
    "dclm": "ettin-data/data/text/mlfoundations-dclm-baseline-1.0-parquet---train---default",
    "cosmopediav2": "ettin-data/data/text/HuggingFaceTB-smollm-corpus---train---cosmopedia-v2"
}

ALL_REPOS_REMOTE = list(SOURCE_MAP_REMOTE.values())
ALL_REPOS = list(SOURCE_MAP.values())