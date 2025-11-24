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
GBM_MODEL_FILENAME_PATTERN = "lgb_fold_{fold}.txt"  # Deprecated: kept for backwards compatibility
GBM_MODEL_FILENAME = "lgb_model.txt"  # Single model filename for temporal split
CATBOOST_MODEL_FILENAME = "catboost_model.cbm"

# --- TF-IDF PARAMETERS ---
TFIDF_MAX_FEATURES = 500
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

# --- MODEL GBM PARAMETERS ---
# Changed for Stage 2B: multiclass classification (3 classes) instead of binary
# Classes: 0=cold candidates, 1=planned books, 2=read books
LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "num_leaves": 31,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
    "boosting_type": "gbdt",
    # Memory optimization parameters to prevent hanging on large datasets
    "max_bin": 255,  # Reduce from default 255 to use less memory (already optimal)
    "force_row_wise": True,  # Use row-wise data loading for better memory efficiency with large datasets
}

# LightGBM's fit method allows for a list of callbacks, including early stopping.
# To use it, we need to specify parameters for the early stopping callback.
LGB_FIT_PARAMS = {
    "eval_metric": "multi_logloss",
    "callbacks": [],  # Placeholder for early stopping callback
}

# --- MODEL CATBOOST PARAMETERS ---
CATBOOST_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1",
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 10,
    "l2_leaf_reg": 3.0,
    "bagging_temperature": 1.0,
    "random_strength": 1.0,
    "border_count": 254,
    "random_seed": RANDOM_STATE,
    "thread_count": -1,
    "verbose": 100,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    "task_type": "GPU" if torch and torch.cuda.is_available() else "CPU",
    "devices": "0" if torch and torch.cuda.is_available() else None,
    # экономия памяти
    "max_bin": 254,
    "used_ram_limit": "12gb",
}

# трейн конфиг
CATBOOST_FIT_KWARGS = {
    "use_best_model": True,
    "plot": False,
}
