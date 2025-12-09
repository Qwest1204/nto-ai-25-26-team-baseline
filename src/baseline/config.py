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
RANDOM_STATE = 56
TARGET = constants.COL_RELEVANCE  # Multiclass target: 0=cold, 1=planned, 2=read

# --- TEMPORAL SPLIT CONFIG ---
# Ratio of data to use for training (0 < TEMPORAL_SPLIT_RATIO < 1)
# 0.8 means 80% of data points (by timestamp) go to train, 20% to validation
TEMPORAL_SPLIT_RATIO = 0.8

# --- TRAINING CONFIG ---
EARLY_STOPPING_ROUNDS = 50
EVAL_METRIC_RANK = "NDCG:top=20"
MODEL_FILENAME_PATTERN = "lgb_fold_{fold}.txt"  # Deprecated: kept for backwards compatibility
MODEL_FILENAME = "catboost_ranker.cbm"  # Single model filename for temporal split

# --- NEGATIVE SAMPLING ---
NEGATIVE_SAMPLES_PER_USER = 1
NEGATIVE_MAX_SAMPLES = 20000

# --- CATEGORY BUCKETING ---
RARE_CATEGORY_MIN_COUNT = 20

# --- NOMIC DIM REDUCTION ---
NOMIC_SVD_DIM = 128

# --- TF-IDF PARAMETERS ---
TFIDF_MAX_FEATURES = 100 #УМЕНЬШИЛ Т К НЕ ТЯНЕТ!
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

# --- NOMIC PARAMETERS ---
NOMIC_MODEL_NAME = constants.NOMIC_MODEL_NAME
NOMIC_BATCH_SIZE = 2
NOMIC_MAX_LENGTH = 8192
NOMIC_EMBEDDING_DIM = 768
NOMIC_DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"
NOMIC_GPU_MEMORY_FRACTION = 0.75
NOMIC_SVD_DIM = 64


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


CATBOOST_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1",
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 10.0,
    "bagging_temperature": 1.0,
    "random_strength": 1.0,
    "max_bin": 254,
    "random_seed": RANDOM_STATE,
    "thread_count": -1,
    #"verbose": 100,
    #"early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    "task_type": "GPU" if torch and torch.cuda.is_available() else "CPU",
    "devices": "0" if torch and torch.cuda.is_available() else None,
    # экономия памяти
    #"used_ram_limit": "12gb",
}

# Ранжирование (CatBoostRanker)
CATBOOST_RANKER_PARAMS = {
    "loss_function": "YetiRankPairwise",
    "eval_metric": EVAL_METRIC_RANK,
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 6,
    "min_data_in_leaf": 32,
    "l2_leaf_reg": 12.0,
    "random_strength": 1.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.7,
    "rsm": 0.7,
    "one_hot_max_size": 1,  # CPU pairwise requires no one-hot
    "max_ctr_complexity": 1,
    "max_bin": 96,
    "random_seed": RANDOM_STATE,
    "thread_count": -1,
    "task_type": "CPU",
    "devices": None,
    "od_type": "Iter",
    "od_wait": EARLY_STOPPING_ROUNDS,
    "metric_period": 100,
}

# тренировка
CATBOOST_RANKER_FIT_KWARGS = {
    "use_best_model": True,
    "verbose": 100,
    "plot": False,
}
