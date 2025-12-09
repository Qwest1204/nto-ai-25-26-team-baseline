"""
Project-wide constants.

This module defines constants that are part of the data schema or project
structure but are not intended to be tuned as hyperparameters.
"""

# --- FILENAMES ---
TRAIN_FILENAME = "train.csv"
TARGETS_FILENAME = "targets.csv"
CANDIDATES_FILENAME = "candidates.csv"
USER_DATA_FILENAME = "users.csv"
BOOK_DATA_FILENAME = "books.csv"
BOOK_GENRES_FILENAME = "book_genres.csv"
GENRES_FILENAME = "genres.csv"
BOOK_DESCRIPTIONS_FILENAME = "book_descriptions.csv"
SUBMISSION_FILENAME = "submission.csv"
TFIDF_VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
BERT_EMBEDDINGS_FILENAME = "bert_embeddings.pkl"
BERT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
NOMIC_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
NOMIC_EMBEDDINGS_FILENAME = "nomic_embeddings.pkl"
PROCESSED_DATA_FILENAME = "processed_features.parquet"

# --- COLUMN NAMES ---
# Main columns
COL_USER_ID = "user_id"
COL_BOOK_ID = "book_id"
COL_TARGET = "has_read"  # Original target from train.csv
COL_RELEVANCE = "relevance"  # New target for multiclass: 0=cold, 1=planned, 2=read
COL_SOURCE = "source"
COL_PREDICTION = "rating_predict"
COL_HAS_READ = "has_read"
COL_TIMESTAMP = "timestamp"
COL_BOOK_ID_LIST = "book_id_list"

# Feature columns (newly created)
# Note: These constants are reused from Stage 1, but will compute mean(has_read) instead of mean(rating)
F_USER_MEAN_RATING = "user_mean_rating"  # Will be mean(has_read) for user
F_USER_RATINGS_COUNT = "user_ratings_count"  # Will be count of interactions for user
F_BOOK_MEAN_RATING = "book_mean_rating"  # Will be mean(has_read) for book
F_BOOK_RATINGS_COUNT = "book_ratings_count"  # Will be count of interactions for book
F_AUTHOR_MEAN_RATING = "author_mean_rating"  # Will be mean(has_read) for author
F_BOOK_GENRES_COUNT = "book_genres_count"
F_USER_BOOK_INTERACTION = "f_user_book_interaction"  # Binary: 1 if (user_id, book_id) in train.csv, else 0
F_USER_LAST_INTERACTION_DAYS = "user_last_interaction_days"
F_BOOK_LAST_INTERACTION_DAYS = "book_last_interaction_days"
F_BOOK_READ_RATE = "book_read_rate"
F_BOOK_PLANNED_RATE = "book_planned_rate"
F_GENRE_READ_RATE = "genre_read_rate"
F_GENRE_PLANNED_RATE = "genre_planned_rate"
F_USER_TOP_GENRE = "user_top_genre"
F_BOOK_MATCH_TOP_GENRE = "book_match_top_genre"
F_USER_MEAN_30D = "user_mean_30d"
F_USER_MEAN_90D = "user_mean_90d"
F_USER_COUNT_30D = "user_count_30d"
F_USER_COUNT_90D = "user_count_90d"
F_BOOK_MEAN_30D = "book_mean_30d"
F_BOOK_MEAN_90D = "book_mean_90d"
F_BOOK_COUNT_30D = "book_count_30d"
F_BOOK_COUNT_90D = "book_count_90d"
F_USER_TFIDF_SIM = "user_tfidf_sim"
F_USER_NOMIC_SIM = "user_nomic_sim"
F_USER_RATINGS_COUNT_LOG = "user_ratings_count_log"
F_BOOK_RATINGS_COUNT_LOG = "book_ratings_count_log"
F_BOOK_GENRES_COUNT_LOG = "book_genres_count_log"
F_USER_COUNT_30D_LOG = "user_count_30d_log"
F_USER_COUNT_90D_LOG = "user_count_90d_log"
F_BOOK_COUNT_30D_LOG = "book_count_30d_log"
F_BOOK_COUNT_90D_LOG = "book_count_90d_log"

# Metadata columns from raw data
COL_GENDER = "gender"
COL_AGE = "age"
COL_AUTHOR_ID = "author_id"
COL_PUBLICATION_YEAR = "publication_year"
COL_LANGUAGE = "language"
COL_PUBLISHER = "publisher"
COL_AVG_RATING = "avg_rating"
COL_GENRE_ID = "genre_id"
COL_DESCRIPTION = "description"


# --- VALUES ---
VAL_SOURCE_TRAIN = "train"
VAL_SOURCE_TEST = "test"

# --- MAGIC NUMBERS ---
MISSING_CAT_VALUE = "-1"
MISSING_NUM_VALUE = -1
PREDICTION_MIN_VALUE = 0
PREDICTION_MAX_VALUE = 2  # Changed: now 3 classes (0, 1, 2) instead of regression
MAX_RANKING_LENGTH = 20

