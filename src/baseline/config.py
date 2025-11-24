"""
Configuration file for the NTO ML competition baseline.
"""

from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from . import constants

# --- DIRECTORIES ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"


# --- PARAMETERS ---
N_SPLITS = 5  # Deprecated: kept for backwards compatibility, not used in temporal split
RANDOM_STATE = 42
TARGET = constants.COL_RELEVANCE  # Multiclass target: 0=cold, 1=planned, 2=read

# --- TEMPORAL SPLIT CONFIG ---
# Ratio of data to use for training (0 < TEMPORAL_SPLIT_RATIO < 1)
# 0.8 means 80% of data points (by timestamp) go to train, 20% to validation
TEMPORAL_SPLIT_RATIO = 0.8

# --- TRAINING CONFIG ---
EARLY_STOPPING_ROUNDS = 50
MODEL_FILENAME_PATTERN = "xgb_fold_{fold}.json"  # Deprecated: kept for backwards compatibility
MODEL_FILENAME = "xgb_model.json"  # Single model filename for temporal split

# --- TF-IDF PARAMETERS ---
TFIDF_MAX_FEATURES = 10
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)

# --- BERT PARAMETERS ---
BERT_MODEL_NAME = constants.BERT_MODEL_NAME
BERT_BATCH_SIZE = 8
BERT_MAX_LENGTH = 512
BERT_EMBEDDING_DIM = 768
BERT_DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"
# Limit GPU memory usage to prevent overheating and OOM errors
BERT_GPU_MEMORY_FRACTION = 0.75


# --- FEATURES ---
CAT_FEATURES = [
    constants.COL_USER_ID,
    constants.COL_BOOK_ID,
    constants.COL_GENDER,
    constants.COL_AGE,
    constants.COL_AUTHOR_ID,
    constants.COL_PUBLICATION_YEAR,
    constants.COL_LANGUAGE,
    constants.COL_PUBLISHER,
]

# --- MODEL PARAMETERS ---
# Changed for Stage 2B: multiclass classification (3 classes) instead of binary
# Classes: 0=cold candidates, 1=planned books, 2=read books
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",  # Fixed typo: was "mloglose"
    "learning_rate": 0.01,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "seed": RANDOM_STATE,
    "tree_method": "hist",  # Fast and memory-efficient
    "grow_policy": "depthwise",
    "max_bin": 255,
    "n_jobs": -1,
    "verbosity": 1,
    # Enable native categorical support if available
    "enable_categorical": True,
}

# XGBoost fit parameters
XGB_FIT_PARAMS = {
    "eval_metric": "mlogloss",
    "verbose": True,
}
