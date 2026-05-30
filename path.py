HOMEPATH = "/path/to/this/repo"

# ── data for amr2text ─────────────────────────────────────────────────────────
TRAIN_CSV = f"{HOMEPATH}/ldc/train.csv"
VAL_CSV   = f"{HOMEPATH}/ldc/dev.csv"
TEST_CSV  = f"{HOMEPATH}/ldc/test.csv"

# data for downstream tasks
DATA_HOME_PATH = f"{HOMEPATH}/dataset"
SCHEMA_DESCRIPTION_PATH = f"{HOMEPATH}/dataset/spider/schema_description.json"

# model checkpoints — point to your local Llama and Qwen directories
MODEL_LLAMA_DIR = "/path/to/Llama-3.1-8B-Instruct"
MODEL_QWEN_DIR  = "/path/to/Qwen3-8B"

# output
OUTPUT_DIR = "/path/to/output"

# evaluation
RAMS_EVAL_PATH  = f"{HOMEPATH}/dataset/rams/test_processed.jsonlines"
SPIDER_TABLE_DIR = f"{HOMEPATH}/dataset/spider/tables.json"
SPIDER_DB_DIR   = f"{HOMEPATH}/dataset/spider/database"
SPIDER_GOLD_DIR = f"{HOMEPATH}/dataset/spider/gold_test.sql"