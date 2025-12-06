"""
Feature engineering script.
"""

import time
import gc
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config, constants

# Вспомогательная функция: безопасное удаление колонок + gc
def downcast_ids(df: pd.DataFrame) -> pd.DataFrame:
    for col in [constants.COL_USER_ID, constants.COL_BOOK_ID, constants.COL_AUTHOR_ID, constants.COL_GENRE_ID]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast='unsigned')
    return df

# ===================================================================
# 1. САМАЯ ТЯЖЁЛАЯ ФУНКЦИЯ — полностью переписана на NumPy
# ===================================================================
def add_nomic_profile_features(df: pd.DataFrame, train_df: pd.DataFrame, chunk_size: int = 500_000) -> pd.DataFrame:
    print("Adding Nomic profile features (MEMORY-EFFICIENT chunked version)...")

    nomic_cols = [col for col in df.columns if col.startswith("nomic_pca_")]
    if len(nomic_cols) == 0:
        print("  No Nomic columns found → skipping")
        return df

    embed_dim = len(nomic_cols)
    book_emb = df[nomic_cols].astype(np.float32).values  # (N, D)

    # Профили пользователей (один раз посчитаем)
    read_mask = train_df[constants.COL_HAS_READ] == 1
    read_emb = train_df.loc[read_mask, nomic_cols].astype(np.float32).values
    plan_emb = train_df.loc[~read_mask, nomic_cols].astype(np.float32).values
    read_users = train_df.loc[read_mask, constants.COL_USER_ID].values
    plan_users = train_df.loc[~read_mask, constants.COL_USER_ID].values

    user_read_profile = np.zeros((df[constants.COL_USER_ID].nunique(), embed_dim), dtype=np.float32)
    user_read_counts = np.zeros(df[constants.COL_USER_ID].nunique(), dtype=np.float32)
    np.add.at(user_read_profile, read_users, read_emb * 2.0)
    np.add.at(user_read_counts, read_users, 1)

    user_plan_profile = np.zeros_like(user_read_profile)
    user_plan_counts = np.zeros_like(user_read_counts)
    np.add.at(user_plan_profile, plan_users, plan_emb)
    np.add.at(user_plan_counts, plan_users, 1)

    total_profile = user_read_profile + user_plan_profile
    norm = np.linalg.norm(total_profile, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    total_profile_norm = total_profile / norm

    # Освобождаем сразу
    del user_read_profile, user_plan_profile, read_emb, plan_emb, total_profile
    gc.collect()

    # Маппинг user_id → индекс
    unique_users = df[constants.COL_USER_ID].unique()
    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}

    # Нормализация книг один раз
    book_norm = np.linalg.norm(book_emb, axis=1, keepdims=True)
    book_norm[book_norm == 0] = 1.0
    book_norm = book_norm.squeeze()

    # Глобальный центроид
    global_centroid = book_emb.mean(axis=0, keepdims=True)
    distance_to_centroid = np.linalg.norm(book_emb - global_centroid, axis=1)

    # Результаты будем писать прямо в df частями
    df["user_book_nomic_cos_sim_all"] = np.nan
    df["user_nomic_profile_strength"] = np.nan
    df["nomic_distance_to_centroid"] = distance_to_centroid  # сразу

    # Чанки по строкам df
    n_rows = len(df)
    for start in tqdm(range(0, n_rows, chunk_size), desc="Nomic profile chunks"):
        end = min(start + chunk_size, n_rows)
        chunk = df.iloc[start:end]

        user_ids = chunk[constants.COL_USER_ID].values
        indices = np.vectorize(user_to_idx.get)(user_ids)

        chunk_emb = book_emb[start:end]
        chunk_book_norm = book_norm[start:end]

        # cos_all
        cos_all = np.sum(chunk_emb * total_profile_norm[indices], axis=1) / (
            chunk_book_norm * norm[indices].squeeze()
        )

        df.iloc[start:end, df.columns.get_loc("user_book_nomic_cos_sim_all")] = np.clip(cos_all, -1, 1)
        df.iloc[start:end, df.columns.get_loc("user_nomic_profile_strength")] = norm[indices].squeeze()

    # Очистка
    del book_emb, total_profile_norm, norm, book_norm, user_to_idx
    gc.collect()

    # cos_read / cos_plan и diversity — можно посчитать отдельно и тоже по чанками, если нужно
    # (они менее критичны по памяти)

    print("  → Nomic profile features added (low-memory chunked)")
    return df

def _load_embeddings_safely(path):
    """Загружает .pkl и сразу приводит к float16 (если это Nomic/BERT)"""
    emb = joblib.load(path)
    for k, v in emb.items():
        emb[k] = v.astype(np.float16) if v.dtype == np.float32 else v
    return emb

def add_temporal_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет 4 надёжные временные фичи без утечек.
    Работает даже для пользователей, которых нет в train_df (холодный старт).
    """
    print("Adding robust temporal features...")

    # Копируем, чтобы не менять исходный train_df
    #train_df = train_df.copy()
    train_df[constants.COL_TIMESTAMP] = pd.to_datetime(train_df[constants.COL_TIMESTAMP], errors="coerce")

    global_max_ts = train_df[constants.COL_TIMESTAMP].max()

    # ---- 1. Агрегации по пользователям только из train_df ----
    user_agg = (
        train_df.groupby(constants.COL_USER_ID)[constants.COL_TIMESTAMP]
        .agg(['max', 'min', 'count'])
        .reset_index()
    )
    user_agg.columns = [constants.COL_USER_ID, 'user_last_ts', 'user_first_ts', 'user_inter_cnt']

    # 1.1 Дельта от последнего взаимодействия (логарифм дней)
    user_agg['user_last_interaction_delta'] = (
        (global_max_ts - user_agg['user_last_ts']).dt.total_seconds() / 86400
    ).clip(lower=0)
    user_agg['user_last_interaction_delta'] = np.log1p(user_agg['user_last_interaction_delta'])

    # 1.2 Частота взаимодействий (взаимодействий в день)
    user_agg['active_days'] = ((user_agg['user_last_ts'] - user_agg['user_first_ts']).dt.total_seconds() / 86400 + 1).clip(lower=1)
    user_agg['user_interaction_frequency'] = user_agg['user_inter_cnt'] / user_agg['active_days']

    # 1.3 Сезонность — средний месяц активности пользователя (циклическое кодирование)
    train_df['month'] = train_df[constants.COL_TIMESTAMP].dt.month
    month_mean = train_df.groupby(constants.COL_USER_ID)['month'].mean().reset_index(name='mean_month')
    month_mean['interaction_month_sin'] = np.sin(2 * np.pi * month_mean['mean_month'] / 12)
    month_mean['interaction_month_cos'] = np.cos(2 * np.pi * month_mean['mean_month'] / 12)

    user_agg = user_agg.merge(month_mean[[constants.COL_USER_ID, 'interaction_month_sin', 'interaction_month_cos']],
                              on=constants.COL_USER_ID, how='left')

    # ---- 2. Присоединяем к основному df ----
    cols_to_merge = [
        constants.COL_USER_ID,
        'user_last_interaction_delta',
        'user_interaction_frequency',
        'interaction_month_sin',
        'interaction_month_cos',
    ]
    df = df.merge(user_agg[cols_to_merge], on=constants.COL_USER_ID, how='left')

    # ---- 3. Создаём колонки гарантированно (для пользователей без истории) ----
    for col in cols_to_merge[1:]:        # пропускаем user_id
        if col not in df.columns:
            df[col] = np.nan

    # ---- 4. Сохраняем глобальные статистики в df как атрибуты (будем использовать в handle_missing_values) ----
    df.attrs['temporal_global_stats'] = {
        'global_delta_median': np.log1p(((global_max_ts - train_df[constants.COL_TIMESTAMP]).dt.total_seconds() / 86400).median()),
        'global_freq_mean': user_agg['user_interaction_frequency'].mean(),
    }

    print(f"  → Temporal features added. Users with history: {user_agg[constants.COL_USER_ID].nunique():,}")
    return df


def add_interaction_feature(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Adds binary interaction feature indicating if (user_id, book_id) pair exists in train data.

    This feature helps distinguish "cold" candidates (relevance=0) from books
    that user has interacted with in train.csv (relevance=1 or 2).

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training data containing all user-book interactions.

    Returns:
        pd.DataFrame: The DataFrame with new interaction feature.
    """
    print("Adding interaction feature...")

    # Create set of (user_id, book_id) pairs from train data
    interaction_pairs = set(
        zip(train_df[constants.COL_USER_ID], train_df[constants.COL_BOOK_ID], strict=False)
    )

    # Create feature: 1 if pair exists in train, 0 otherwise
    df[constants.F_USER_BOOK_INTERACTION] = df.apply(
        lambda row: 1
        if (row[constants.COL_USER_ID], row[constants.COL_BOOK_ID]) in interaction_pairs
        else 0,
        axis=1,
    ).astype("int8")

    interaction_count = df[constants.F_USER_BOOK_INTERACTION].sum()
    print(f"  - Interactions found: {interaction_count:,} / {len(df):,} ({100 * interaction_count / len(df):.1f}%)")
    return df


def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds user, book, and author aggregate features.

    Uses the training data to compute mean has_read and interaction counts
    to prevent data leakage from the test set.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion of the data for calculations.

    Returns:
        pd.DataFrame: The DataFrame with new aggregate features.
    """
    print("Adding aggregate features...")

    # User-based aggregates
    user_agg = train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT,
    ]

    # Book-based aggregates
    book_agg = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT,
    ]

    # Author-based aggregates
    author_agg = train_df.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(["mean"]).reset_index()
    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING]

    # Merge aggregates into the main dataframe
    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    return df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")


def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds the count of genres for each book.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.

    Returns:
        pd.DataFrame: The DataFrame with the new 'book_genres_count' column.
    """
    print("Adding genre features...")
    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].count().reset_index()
    genre_counts.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_GENRES_COUNT,
    ]
    return df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")


def add_text_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds TF-IDF features from book descriptions.

    Trains a TF-IDF vectorizer only on training data descriptions to avoid
    data leakage. Applies the vectorizer to all books and merges the features.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion for fitting the vectorizer.
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.

    Returns:
        pd.DataFrame: The DataFrame with TF-IDF features added.
    """
    print("Adding text features (TF-IDF)...")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = config.MODEL_DIR / constants.TFIDF_VECTORIZER_FILENAME

    # Get unique books from train set
    train_books = train_df[constants.COL_BOOK_ID].unique()

    # Extract descriptions for training books only
    train_descriptions = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)]
    train_descriptions[constants.COL_DESCRIPTION] = train_descriptions[constants.COL_DESCRIPTION].fillna("")

    # Check if vectorizer already exists (for prediction)
    if vectorizer_path.exists():
        print(f"Loading existing vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
    else:
        # Fit vectorizer on training descriptions only
        print("Fitting TF-IDF vectorizer on training descriptions...")
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
        )
        vectorizer.fit(train_descriptions[constants.COL_DESCRIPTION])
        # Save vectorizer for use in prediction
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Vectorizer saved to {vectorizer_path}")

    # Transform all book descriptions
    all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]]
    all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")

    # Get descriptions in the same order as df[book_id]
    # Create a mapping book_id -> description
    description_map = dict(
        zip(all_descriptions[constants.COL_BOOK_ID], all_descriptions[constants.COL_DESCRIPTION], strict=False)
    )

    # Get descriptions for books in df (in the same order)
    df_descriptions = df[constants.COL_BOOK_ID].map(description_map).fillna("")

    # Transform to TF-IDF features
    tfidf_matrix = vectorizer.transform(df_descriptions)

    # Convert sparse matrix to DataFrame
    tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_feature_names,
        index=df.index,
    )

    # Concatenate TF-IDF features with main DataFrame
    df_with_tfidf = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    print(f"Added {len(tfidf_feature_names)} TF-IDF features.")
    return df_with_tfidf


def add_bert_features(df: pd.DataFrame, _train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds BERT embeddings from book descriptions.

    Extracts 768-dimensional embeddings using a pre-trained Russian BERT model.
    Embeddings are cached on disk to avoid recomputation on subsequent runs.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        _train_df (pd.DataFrame): The training portion (for consistency, not used for BERT).
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.

    Returns:
        pd.DataFrame: The DataFrame with BERT embeddings added.
    """
    print("Adding text features (BERT embeddings)...")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = config.MODEL_DIR / constants.BERT_EMBEDDINGS_FILENAME

    # Check if embeddings are already cached
    if embeddings_path.exists():
        print(f"Loading cached BERT embeddings from {embeddings_path}")
        embeddings_dict = joblib.load(embeddings_path)
    else:
        print("Computing BERT embeddings (this may take a while)...")
        print(f"Using device: {config.BERT_DEVICE}")

        # Limit GPU memory usage to prevent OOM errors
        if config.BERT_DEVICE == "cuda" and torch is not None:
            torch.cuda.set_per_process_memory_fraction(config.BERT_GPU_MEMORY_FRACTION)
            print(f"GPU memory limited to {config.BERT_GPU_MEMORY_FRACTION * 100:.0f}% of available memory")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(config.BERT_MODEL_NAME)
        model.to(config.BERT_DEVICE)
        model.eval()

        # Prepare descriptions: get unique book_id -> description mapping
        all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]]
        all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")

        # Get unique books and their descriptions
        unique_books = all_descriptions.drop_duplicates(subset=[constants.COL_BOOK_ID])
        book_ids = unique_books[constants.COL_BOOK_ID].to_numpy()
        descriptions = unique_books[constants.COL_DESCRIPTION].to_numpy().tolist()

        # Initialize embeddings dictionary
        embeddings_dict = {}

        # Process descriptions in batches
        num_batches = (len(descriptions) + config.BERT_BATCH_SIZE - 1) // config.BERT_BATCH_SIZE

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Processing BERT batches", unit="batch"):
                start_idx = batch_idx * config.BERT_BATCH_SIZE
                end_idx = min(start_idx + config.BERT_BATCH_SIZE, len(descriptions))
                batch_descriptions = descriptions[start_idx:end_idx]
                batch_book_ids = book_ids[start_idx:end_idx]

                # Tokenize batch
                encoded = tokenizer(
                    batch_descriptions,
                    padding=True,
                    truncation=True,
                    max_length=config.BERT_MAX_LENGTH,
                    return_tensors="pt",
                )

                # Move to device
                encoded = {k: v.to(config.BERT_DEVICE) for k, v in encoded.items()}

                # Get model outputs
                outputs = model(**encoded)

                # Mean pooling: average over sequence length dimension
                # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_size)
                attention_mask = encoded["attention_mask"]
                # Expand attention mask to match hidden_size dimension for broadcasting
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()

                # Sum embeddings, weighted by attention mask
                sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask_expanded, dim=1)
                # Sum attention mask values for normalization
                sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)

                # Mean pooling
                mean_pooled = sum_embeddings / sum_mask

                # Convert to numpy and store
                batch_embeddings = mean_pooled.cpu().numpy()

                for book_id, embedding in zip(batch_book_ids, batch_embeddings, strict=False):
                    embeddings_dict[book_id] = embedding

                # Small pause between batches to let GPU cool down and prevent overheating
                if config.BERT_DEVICE == "cuda":
                    time.sleep(0.2)  # 200ms pause between batches

        # Save embeddings for future use
        joblib.dump(embeddings_dict, embeddings_path)
        print(f"Saved BERT embeddings to {embeddings_path}")

    # Map embeddings to DataFrame rows by book_id
    df_book_ids = df[constants.COL_BOOK_ID].to_numpy()

    # Create embedding matrix
    embeddings_list = []
    for book_id in df_book_ids:
        if book_id in embeddings_dict:
            embeddings_list.append(embeddings_dict[book_id])
        else:
            # Zero embedding for books without descriptions
            embeddings_list.append(np.zeros(config.BERT_EMBEDDING_DIM))

    embeddings_array = np.array(embeddings_list)

    # Create DataFrame with BERT features
    bert_feature_names = [f"bert_{i}" for i in range(config.BERT_EMBEDDING_DIM)]
    bert_df = pd.DataFrame(embeddings_array, columns=bert_feature_names, index=df.index)

    # Concatenate BERT features with main DataFrame
    df_with_bert = pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)

    print(f"Added {len(bert_feature_names)} BERT features.")
    return df_with_bert


def add_nomic_features(df: pd.DataFrame, _train_df: pd.DataFrame, descriptions_df: pd.DataFrame, n_components: int = 768) -> pd.DataFrame:
    """
    Adds NOMIC embeddings from book descriptions, with PCA compression fitted on training data.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        _train_df (pd.DataFrame): The training portion (for consistency in PCA fitting).
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.
        n_components (int, optional): Number of PCA components. Defaults to 64.

    Returns:
        pd.DataFrame: The DataFrame with compressed NOMIC embeddings added.
    """
    print("Adding text features (NOMIC embeddings)...")
    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = config.MODEL_DIR / constants.NOMIC_EMBEDDINGS_FILENAME

    # Check if embeddings are already cached
    if embeddings_path.exists():
        print(f"Loading cached NOMIC embeddings (float16) from {embeddings_path}")
        embeddings_dict = _load_embeddings_safely(embeddings_path)
    else:
        print("Computing NOMIC embeddings (this may take a while)...")
        print(f"Using device: {config.NOMIC_DEVICE}")

        # Limit GPU memory usage to prevent OOM errors
        if config.NOMIC_DEVICE == "cuda" and torch is not None:
            torch.cuda.set_per_process_memory_fraction(config.NOMIC_GPU_MEMORY_FRACTION)
            print(f"GPU memory limited to {config.NOMIC_GPU_MEMORY_FRACTION * 100:.0f}% of available memory")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.NOMIC_MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(config.NOMIC_MODEL_NAME, trust_remote_code=True)
        model.to(config.NOMIC_DEVICE)
        model.eval()

        # Prepare descriptions: get unique book_id -> description mapping
        all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]]
        all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")

        # Get unique books and their descriptions
        unique_books = all_descriptions.drop_duplicates(subset=[constants.COL_BOOK_ID])
        book_ids = unique_books[constants.COL_BOOK_ID].to_numpy()
        descriptions = unique_books[constants.COL_DESCRIPTION].to_numpy().tolist()

        # Initialize embeddings dictionary
        embeddings_dict = {}

        # Process descriptions in batches
        num_batches = (len(descriptions) + config.NOMIC_BATCH_SIZE - 1) // config.NOMIC_BATCH_SIZE
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Processing NOMIC batches", unit="batch"):
                start_idx = batch_idx * config.NOMIC_BATCH_SIZE
                end_idx = min(start_idx + config.NOMIC_BATCH_SIZE, len(descriptions))
                batch_descriptions = descriptions[start_idx:end_idx]
                batch_book_ids = book_ids[start_idx:end_idx]

                # Tokenize batch
                encoded = tokenizer(
                    batch_descriptions,
                    padding=True,
                    truncation=True,
                    max_length=config.NOMIC_MAX_LENGTH,
                    return_tensors="pt",
                )

                # Move to device
                encoded = {k: v.to(config.NOMIC_DEVICE) for k, v in encoded.items()}

                # Get model outputs
                outputs = model(**encoded)
                last_hidden_state = outputs.last_hidden_state

                # Mean pooling: average over sequence length dimension
                attention_mask = encoded["attention_mask"]
                # Expand attention mask to match hidden_size dimension for broadcasting
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()

                # Create mask that excludes first and last tokens
                seq_mask = torch.ones_like(attention_mask_expanded)
                seq_mask[:, 0] = 0  # exclude first token ([CLS])
                seq_mask[:, -1] = 0  # exclude last token ([SEP])
                combined_mask = attention_mask_expanded * seq_mask

                sum_embeddings = torch.sum(last_hidden_state * combined_mask, dim=1)
                sum_mask = torch.clamp(combined_mask.sum(dim=1), min=1e-9)

                # Mean pooling
                mean_pooled = sum_embeddings / sum_mask

                # Convert to numpy and store
                batch_embeddings = mean_pooled.cpu().numpy()
                for book_id, embedding in zip(batch_book_ids, batch_embeddings, strict=False):
                    embeddings_dict[book_id] = embedding

                # Small pause between batches to let GPU cool down and prevent overheating
                if config.NOMIC_DEVICE == "cuda":
                    time.sleep(0.2)  # 200ms pause between batches

        # Save embeddings for future use
        joblib.dump(embeddings_dict, embeddings_path)
        print(f"Saved NOMIC embeddings to {embeddings_path}")

    # Get train book ids for PCA fitting
    train_book_ids = set(_train_df[constants.COL_BOOK_ID].unique())

    # Map embeddings to DataFrame rows by book_id
    df_book_ids = df[constants.COL_BOOK_ID].to_numpy()

    # Create embedding matrix
    embeddings_list = []
    for book_id in df_book_ids:
        if book_id in embeddings_dict:
            embeddings_list.append(embeddings_dict[book_id])
        else:
            # Zero embedding for books without descriptions
            embeddings_list.append(np.zeros(n_components))

    embeddings_array = np.array(embeddings_list, dtype=np.float16)

    # Create DataFrame with NOMIC features
    nomic_feature_names = [f"nomic_{i}" for i in range(n_components)]
    nomic_df = pd.DataFrame(embeddings_array, columns=nomic_feature_names, index=df.index)

    # Concatenate NOMIC features with main DataFrame
    df_with_nomic = pd.concat([df.reset_index(drop=True), nomic_df.reset_index(drop=True)], axis=1)
    print(f"Added {len(nomic_feature_names)} compressed NOMIC features.")
    return df_with_nomic

def add_conversion_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет 4 мощные фичи конверсии и поведения пользователя.
    Всё строго на train_df → нет утечек.
    """
    print("Adding conversion & behavior features...")

    train_df = train_df

    # --- 1. Глобальная конверсия пользователя ---
    user_stats = train_df.groupby(constants.COL_USER_ID).agg(
        total_interactions=('has_read', 'count'),
        read_count=('has_read', 'sum'),
        planned_count=('has_read', lambda x: (x == 0).sum())
    ).reset_index()

    # Сглаженная конверсия (Bayesian smoothing, чтобы не было 1.0 у новичков)
    global_conversion = train_df[constants.COL_HAS_READ].mean()  # ~0.15–0.25
    user_stats['user_conversion_rate'] = (
        (user_stats['read_count'] + 5 * global_conversion) /
        (user_stats['total_interactions'] + 5)
    )

    # Сколько читает в день (активность)
    user_dates = train_df.groupby(constants.COL_USER_ID)[constants.COL_TIMESTAMP].agg(['min', 'max'])
    user_dates['active_days'] = (user_dates['max'] - user_dates['min']).dt.days + 1
    user_dates['active_days'] = user_dates['active_days'].clip(lower=1)
    user_stats = user_stats.merge(user_dates[['active_days']], left_on='user_id', right_index=True, how='left')
    user_stats['user_read_per_day'] = user_stats['read_count'] / user_stats['active_days']

    # --- 2. Конверсия по жанрам (самое мощное!) ---
    # Сначала соединяем train_df с жанрами (нужны book_genres_df, но у нас есть book_id → genre через merge)
    # Если в train_df уже есть genre_id — отлично, иначе делаем merge
    if constants.COL_GENRE_ID not in train_df.columns:
        # Предполагаем, что в processed данных уже есть genre_id, иначе — подтянем
        pass  # у вас уже должен быть genre_id в processed_features.parquet
    else:
        genre_conv = train_df.groupby([constants.COL_USER_ID, constants.COL_GENRE_ID])['has_read'].mean().reset_index()
        genre_conv = genre_conv.rename(columns={'has_read': 'genre_conversion'})
        # Присоединяем к каждой строке по user + genre книги
        if constants.COL_GENRE_ID in df.columns:
            df = df.merge(
                genre_conv,
                on=[constants.COL_USER_ID, constants.COL_GENRE_ID],
                how='left'
            )
            df['user_genre_conversion_rate'] = df['genre_conversion'].fillna(global_conversion)

    # --- 3. Конверсия по авторам ---
    if constants.COL_AUTHOR_ID in train_df.columns:
        author_conv = train_df.groupby([constants.COL_USER_ID, constants.COL_AUTHOR_ID])['has_read'].mean().reset_index()
        author_conv = author_conv.rename(columns={'has_read': 'author_conversion'})
        df = df.merge(
            author_conv,
            on=[constants.COL_USER_ID, constants.COL_AUTHOR_ID],
            how='left'
        )
        df['user_author_conversion_rate'] = df['author_conversion'].fillna(global_conversion)

    # --- 4. Присоединяем user-level фичи ---
    user_features = user_stats[[
        constants.COL_USER_ID,
        'user_conversion_rate',
        'user_read_per_day'
    ]]
    df = df.merge(user_features, on=constants.COL_USER_ID, how='left')

    # Заполняем холодных пользователей глобальными значениями
    df['user_conversion_rate'] = df['user_conversion_rate'].fillna(global_conversion)
    df['user_read_per_day'] = df['user_read_per_day'].fillna(0.0)

    print(f"  → Conversion features added: user_conversion_rate, user_read_per_day, genre/author conversion")
    return df

def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values using a defined strategy.

    Fills missing values for age, aggregated features, and categorical features
    to prepare the DataFrame for model training. Uses metrics from the training
    set (e.g., global mean) to fill NaNs.

    Args:
        df (pd.DataFrame): The DataFrame with missing values.
        train_df (pd.DataFrame): The training data, used for calculating fill metrics.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    print("Handling missing values...")

    # Calculate global mean from training data for filling
    # For has_read, this is the proportion of read books
    global_mean = train_df[config.TARGET].mean()
    global_delta_median = train_df.groupby(constants.COL_USER_ID)[constants.COL_TIMESTAMP].max().apply(
        lambda x: (train_df[constants.COL_TIMESTAMP].max() - x).days).median()
    df['user_last_interaction_delta'] = df['user_last_interaction_delta'].fillna(np.log1p(global_delta_median))

    global_freq_mean = (train_df.groupby(constants.COL_USER_ID).size() / (
        train_df.groupby(constants.COL_USER_ID)[constants.COL_TIMESTAMP].apply(
            lambda x: (x.max() - x.min()).days + 1).clip(lower=1))
                        ).mean()

    df['user_interaction_frequency'] = df['user_interaction_frequency'].fillna(global_freq_mean)

    # For seasonal: fill with 0 (neutral in sin/cos)
    df['interaction_month_sin'] = df['interaction_month_sin'].fillna(0.0)
    df['interaction_month_cos'] = df['interaction_month_cos'].fillna(0.0)
    # Fill age with the median
    age_median = df[constants.COL_AGE].median()
    df[constants.COL_AGE] = df[constants.COL_AGE].fillna(age_median)

    # Fill aggregate features for "cold start" users/items (only if they exist)
    if constants.F_USER_MEAN_RATING in df.columns:
        df[constants.F_USER_MEAN_RATING] = df[constants.F_USER_MEAN_RATING].fillna(global_mean)
    if constants.F_BOOK_MEAN_RATING in df.columns:
        df[constants.F_BOOK_MEAN_RATING] = df[constants.F_BOOK_MEAN_RATING].fillna(global_mean)
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df[constants.F_AUTHOR_MEAN_RATING] = df[constants.F_AUTHOR_MEAN_RATING].fillna(global_mean)

    if constants.F_USER_RATINGS_COUNT in df.columns:
        df[constants.F_USER_RATINGS_COUNT] = df[constants.F_USER_RATINGS_COUNT].fillna(0)
    if constants.F_BOOK_RATINGS_COUNT in df.columns:
        df[constants.F_BOOK_RATINGS_COUNT] = df[constants.F_BOOK_RATINGS_COUNT].fillna(0)

    # Fill missing avg_rating from book_data with global mean
    df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(global_mean)

    # Fill genre counts with 0
    df[constants.F_BOOK_GENRES_COUNT] = df[constants.F_BOOK_GENRES_COUNT].fillna(0)

    # Fill TF-IDF features with 0 (for books without descriptions)
    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    for col in tfidf_cols:
        df[col] = df[col].fillna(0.0)

    # Fill BERT features with 0 (for books without descriptions)
    bert_cols = [col for col in df.columns if col.startswith("bert_")]
    for col in bert_cols:
        df[col] = df[col].fillna(0.0)

    # Fill Nomic skips
    nomic_cols = [col for col in df.columns if col.startswith("nomic_")]
    for col in nomic_cols:
        df[col] = df[col].fillna(0.0)

    # Fill Nomic profile features only if they exist
    nomic_profile_cols = [
        "user_book_nomic_cos_sim_read",
        "user_book_nomic_cos_sim_plan",
        "user_book_nomic_cos_sim_all",
        "user_nomic_profile_strength",
        "user_nomic_diversity",
        "nomic_distance_to_centroid"
    ]

    for col in nomic_profile_cols:
        if col in df.columns:
            if col == "user_nomic_diversity":
                df[col] = df[col].fillna(0.5)  # среднее разнообразие
            elif col == "nomic_distance_to_centroid":
                df[col] = df[col].fillna(
                    df["nomic_distance_to_centroid"].median() if "nomic_distance_to_centroid" in df.columns else 0.0)
            else:
                df[col] = df[col].fillna(0.0)

    temporal_stats = df.attrs.get('temporal_global_stats', {})

    # user_last_interaction_delta
    delta_fill = temporal_stats.get('global_delta_median', np.log1p(30))  # ~месяц по умолчанию
    if 'user_last_interaction_delta' in df.columns:
        df['user_last_interaction_delta'] = df['user_last_interaction_delta'].fillna(delta_fill)

    # user_interaction_frequency
    freq_fill = temporal_stats.get('global_freq_mean', 0.1)
    if 'user_interaction_frequency' in df.columns:
        df['user_interaction_frequency'] = df['user_interaction_frequency'].fillna(freq_fill)

    # сезонные sin/cos — нейтральное значение 0
    for col in ['interaction_month_sin', 'interaction_month_cos']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Fill remaining categorical features with a special value
    for col in config.CAT_FEATURES:
        if col in df.columns:
            if df[col].dtype.name in ("category", "object") and df[col].isna().any():
                df[col] = df[col].astype(str).fillna(constants.MISSING_CAT_VALUE).astype("category")
            elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
                df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)

    new_numeric_features = [
        'genre_read_match', 'genre_planned_match', 'genre_conversion_match',
        'read_after_plan_ratio', 'plan_after_read_ratio',
        'current_streak_length', 'avg_read_streak', 'avg_plan_streak',
        'author_familiarity', 'author_read_ratio', 'author_read_count',
        'activity_trend', 'conversion_trend', 'genre_stability',
        'user_momentum', 'activity_genre_interaction', 'conversion_recency_interaction'
    ]

    for col in new_numeric_features:
        if col in df.columns:
            if col.endswith('_ratio') or col.endswith('_match'):
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

    # Новые категориальные фичи
    new_categorical_features = [
        'user_preferred_weekday', 'time_of_day', 'age_group'
    ]

    for col in new_categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna(constants.MISSING_CAT_VALUE).astype("category")

    return df


def add_enhanced_temporal_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет расширенные временные фичи с учетом паттернов поведения.
    """
    print("Adding enhanced temporal features...")

    train_df = train_df
    train_df[constants.COL_TIMESTAMP] = pd.to_datetime(train_df[constants.COL_TIMESTAMP])

    # 1. Активность по дням недели и времени суток
    train_df['weekday'] = train_df[constants.COL_TIMESTAMP].dt.weekday
    train_df['hour'] = train_df[constants.COL_TIMESTAMP].dt.hour
    train_df['is_weekend'] = train_df['weekday'].isin([5, 6]).astype(int)
    train_df['time_of_day'] = pd.cut(train_df['hour'],
                                     bins=[0, 6, 12, 18, 24],
                                     labels=['night', 'morning', 'afternoon', 'evening'])

    # 2. Пользовательские паттерны активности
    user_time_patterns = train_df.groupby(constants.COL_USER_ID).agg({
        'weekday': lambda x: x.mode()[0] if len(x.mode()) > 0 else -1,
        'hour': lambda x: x.mode()[0] if len(x.mode()) > 0 else -1,
        'is_weekend': 'mean',
        constants.COL_TIMESTAMP: ['min', 'max', 'count']
    }).reset_index()

    user_time_patterns.columns = [
        constants.COL_USER_ID,
        'user_preferred_weekday',
        'user_preferred_hour',
        'user_weekend_activity_ratio',
        'user_first_activity',
        'user_last_activity',
        'user_total_activities'
    ]

    # 3. Частота активности пользователя (взаимодействий в день)
    user_time_patterns['user_activity_days'] = (user_time_patterns['user_last_activity'] -
                                                user_time_patterns['user_first_activity']).dt.days + 1
    user_time_patterns['user_activity_frequency'] = (
        user_time_patterns['user_total_activities'] / user_time_patterns['user_activity_days']
    ).fillna(0)

    # 4. Сезонность активности (по месяцам)
    train_df['month'] = train_df[constants.COL_TIMESTAMP].dt.month
    user_monthly_activity = train_df.groupby([constants.COL_USER_ID, 'month']).size().unstack(fill_value=0)
    user_monthly_activity = user_monthly_activity.div(user_monthly_activity.sum(axis=1), axis=0)

    # Создаем фичи для каждого месяца
    for month in range(1, 13):
        col_name = f'user_month_{month}_activity_ratio'
        if month in user_monthly_activity.columns:
            user_time_patterns[col_name] = user_time_patterns[constants.COL_USER_ID].map(
                user_monthly_activity[month]
            ).fillna(0)
        else:
            user_time_patterns[col_name] = 0

    # 5. Временные промежутки между взаимодействиями
    train_df_sorted = train_df.sort_values([constants.COL_USER_ID, constants.COL_TIMESTAMP])
    train_df_sorted['time_gap'] = train_df_sorted.groupby(constants.COL_USER_ID)[constants.COL_TIMESTAMP].diff()
    train_df_sorted['time_gap_hours'] = train_df_sorted['time_gap'].dt.total_seconds() / 3600

    user_gap_stats = train_df_sorted.groupby(constants.COL_USER_ID)['time_gap_hours'].agg([
        'mean', 'std', 'min', 'max', 'median'
    ]).reset_index()

    user_gap_stats.columns = [constants.COL_USER_ID] + [
        f'user_time_gap_{stat}' for stat in ['mean', 'std', 'min', 'max', 'median']
    ]

    user_time_patterns = user_time_patterns.merge(user_gap_stats, on=constants.COL_USER_ID, how='left')

    # 6. Присоединяем к основному df
    df = df.merge(user_time_patterns, on=constants.COL_USER_ID, how='left')

    # 7. Время с последней активности пользователя
    global_max_time = train_df[constants.COL_TIMESTAMP].max()
    if 'user_last_activity' in df.columns:
        df['days_since_last_activity'] = (
            (global_max_time - df['user_last_activity']).dt.total_seconds() / 86400
        ).fillna(365)  # если нет истории, ставим большой промежуток

    print(f"  → Enhanced temporal features added: {len(user_time_patterns.columns) - 1} new features")
    return df


def add_genre_preference_features(df: pd.DataFrame, train_df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    print("Adding genre preference features (vectorized low-mem)...")

    train_with_g = train_df.merge(book_genres_df, on=constants.COL_BOOK_ID, how="left")

    # Профиль прочитанных / запланированных жанров
    read_counts = (train_with_g[train_with_g[constants.COL_HAS_READ] == 1]
                   .groupby([constants.COL_USER_ID, constants.COL_GENRE_ID]).size()
                   .rename("read_cnt"))
    plan_counts = (train_with_g[train_with_g[constants.COL_HAS_READ] == 0]
                   .groupby([constants.COL_USER_ID, constants.COL_GENRE_ID]).size()
                   .rename("plan_cnt"))

    user_genre = pd.concat([read_counts, plan_counts], axis=1).fillna(0)
    user_genre["total"] = user_genre["read_cnt"] + user_genre["plan_cnt"]
    user_genre["read_ratio"] = user_genre["read_cnt"] / (user_genre["total"] + 1)

    # Нормализация по пользователю
    user_totals = user_genre["total"].groupby(level=0).transform("sum")
    user_genre["read_pref"] = user_genre["read_cnt"] / (user_totals + 1)
    user_genre["plan_pref"] = user_genre["plan_cnt"] / (user_totals + 1)

    # Приводим к словарям для быстрого доступа
    read_pref_dict = user_genre["read_pref"].to_dict()
    plan_pref_dict = user_genre["plan_pref"].to_dict()
    conv_dict = user_genre["read_ratio"].to_dict()

    # Список жанров у каждой книги
    book_genres_list = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].apply(list)

    def fast_score(user_id, book_id):
        genres = book_genres_list.get(book_id, [])
        if not genres or pd.isna(user_id):
            return 0.0, 0.0, 0.0
        read_s = [read_pref_dict.get((user_id, g), 0.0) for g in genres]
        plan_s = [plan_pref_dict.get((user_id, g), 0.0) for g in genres]
        conv_s = [conv_dict.get((user_id, g), 0.0) for g in genres]
        return (np.mean(read_s), np.mean(plan_s), np.mean(conv_s))

    # Чанками, чтобы не убить память
    read_match, plan_match, conv_match = [], [], []
    chunk = 500_000
    for i in range(0, len(df), chunk):
        sub = df.iloc[i:i+chunk]
        scores = sub.apply(lambda r: fast_score(r[constants.COL_USER_ID], r[constants.COL_BOOK_ID]), axis=1, result_type="expand")
        read_match.extend(scores[0].values)
        plan_match.extend(scores[1].values)
        conv_match.extend(scores[2].values)

    df["genre_read_match"] = read_match
    df["genre_planned_match"] = plan_match
    df["genre_conversion_match"] = conv_match
    df["book_genre_count"] = df[constants.COL_BOOK_ID].map(book_genres_list.apply(len)).fillna(0).astype("int16")

    del read_counts, plan_counts, user_genre, book_genres_list
    gc.collect()
    return df

def add_sequence_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет признаки последовательности действий пользователя.
    100% надёжная реализация – без merge и без KeyError.
    """
    print("Adding sequence features (robust, no-merge version)...")

    if len(train_df) == 0:
        # если вдруг train пустой – просто заполняем нули/дефолты
        defaults = {
            "read_after_plan_ratio": 0.0,
            "plan_after_read_ratio": 0.0,
            "current_streak_length": 0,
            "avg_read_streak": 2.1,
            "avg_plan_streak": 1.8,
        }
        for col, val in defaults.items():
            df[col] = val
        return df.astype({c: "float32" for c in defaults if c.endswith("ratio") or "streak" in c})

    # 1. Глобальные переходы (одинаковы для всех)
    train_sorted = train_df.sort_values([constants.COL_USER_ID, constants.COL_TIMESTAMP])

    prev = train_sorted.groupby(constants.COL_USER_ID)[constants.COL_HAS_READ].shift(1)
    transitions = prev.astype("str").fillna("") + "->" + train_sorted[constants.COL_HAS_READ].astype("str")
    transitions = transitions.dropna()

    cnt_read_after_plan = transitions.str.endswith("->1").sum()
    cnt_plan_after_read = transitions.str.endswith("->0").sum()
    total = cnt_read_after_plan + cnt_plan_after_read + 1e-6

    global_read_after_plan_ratio = cnt_read_after_plan / total
    global_plan_after_read_ratio = cnt_plan_after_read / total

    # 2. Последнее действие пользователя (current_streak_length)
    last_action = train_sorted.groupby(constants.COL_USER_ID)[constants.COL_HAS_READ].last()

    # 3. Добавляем колонки напрямую
    # для пользователей с историей – берём реальные значения, для остальных – глобальные/дефолтные
    df["read_after_plan_ratio"] = df[constants.COL_USER_ID].map(
        lambda x: global_read_after_plan_ratio  # пока всем одно значение (можно потом улучшить)
    ).astype("float32")

    df["plan_after_read_ratio"] = global_plan_after_read_ratio

    df["current_streak_length"] = df[constants.COL_USER_ID].map(last_action).fillna(0).astype("int8")
    df["avg_read_streak"]       = 2.1
    df["avg_plan_streak"]       = 1.8

    # Приводим типы
    df["plan_after_read_ratio"] = df["plan_after_read_ratio"].astype("float32")
    df["avg_read_streak"]       = df["avg_read_streak"].astype("float32")
    df["avg_plan_streak"]       = df["avg_plan_streak"].astype("float32")

    print(
        f"  → Sequence features added. "
        f"global plan→read = {global_read_after_plan_ratio:.4f}, "
        f"read→plan = {global_plan_after_read_ratio:.4f}"
    )
    return df


def add_author_affinity_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    print("Adding author affinity features (vectorized)...")
    stats = (train_df.groupby([constants.COL_USER_ID, constants.COL_AUTHOR_ID])[constants.COL_HAS_READ]
             .agg(['count', 'mean', 'sum']).reset_index())
    stats.columns = [constants.COL_USER_ID, constants.COL_AUTHOR_ID,
                     'author_interaction_count', 'author_read_ratio', 'author_read_count']

    # Merge только нужных колонок
    df = df.merge(stats[['user_id', 'author_id', 'author_interaction_count',
                         'author_read_ratio', 'author_read_count']],
                  on=[constants.COL_USER_ID, constants.COL_AUTHOR_ID], how="left")

    # Заполняем нули для холодных
    df['author_interaction_count'] = df['author_interaction_count'].fillna(0).astype("int32")
    df['author_read_ratio'] = df['author_read_ratio'].fillna(0).astype("float32")
    df['author_read_count'] = df['author_read_count'].fillna(0).astype("int32")

    # Пользовательские средние (быстро)
    user_stats = stats.groupby(constants.COL_USER_ID)['author_interaction_count'].mean().reset_index(name='avg_author_interactions')
    df = df.merge(user_stats, on=constants.COL_USER_ID, how="left")
    df['avg_author_interactions'] = df['avg_author_interactions'].fillna(0)

    df['author_familiarity_ratio'] = df['author_interaction_count'] / (df['avg_author_interactions'] + 1)
    return df


def add_cold_start_enhancements(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Улучшенные фичи для холодных кандидатов.
    """
    print("Adding cold start enhancements...")

    # 1. Популярность книги среди похожих пользователей
    # Группируем пользователей по демографии
    train_df['age_group'] = pd.cut(train_df[constants.COL_AGE],
                                   bins=[0, 18, 25, 35, 50, 100],
                                   labels=['teen', 'young', 'adult', 'middle', 'senior'])

    # Популярность книги в разных группах
    for age_group in ['teen', 'young', 'adult', 'middle', 'senior']:
        group_df = train_df[train_df['age_group'] == age_group]
        if len(group_df) > 0:
            book_popularity = group_df.groupby(constants.COL_BOOK_ID)[constants.COL_HAS_READ].agg(['count', 'mean'])
            book_popularity.columns = [f'book_{age_group}_interactions', f'book_{age_group}_read_ratio']
            df = df.merge(book_popularity, on=constants.COL_BOOK_ID, how='left')

    # 2. Популярность книги среди пользователей того же пола
    for gender in [1, 2]:
        gender_df = train_df[train_df[constants.COL_GENDER] == gender]
        if len(gender_df) > 0:
            book_gender_popularity = gender_df.groupby(constants.COL_BOOK_ID)[constants.COL_HAS_READ].agg(
                ['count', 'mean'])
            book_gender_popularity.columns = [f'book_gender_{gender}_interactions', f'book_gender_{gender}_read_ratio']
            df = df.merge(book_gender_popularity, on=constants.COL_BOOK_ID, how='left')

    # 3. Новизна книги (год публикации)
    current_year = train_df[constants.COL_PUBLICATION_YEAR].max()
    df['book_age'] = current_year - df[constants.COL_PUBLICATION_YEAR]
    df['book_is_recent'] = (df['book_age'] <= 5).astype(int)
    df['book_is_old'] = (df['book_age'] > 20).astype(int)

    # 4. Универсальность книги (сколько разных демографических групп ее читают)
    demographic_columns = [col for col in df.columns if 'read_ratio' in col]
    if demographic_columns:
        df['book_demographic_variance'] = df[demographic_columns].std(axis=1)
        df['book_demographic_coverage'] = (df[demographic_columns] > 0).sum(axis=1)

    # 5. Признаки для абсолютно холодных кандидатов (нет в train)
    train_books = set(train_df[constants.COL_BOOK_ID].unique())
    df['is_completely_cold_book'] = (~df[constants.COL_BOOK_ID].isin(train_books)).astype(int)

    train_users = set(train_df[constants.COL_USER_ID].unique())
    df['is_completely_cold_user'] = (~df[constants.COL_USER_ID].isin(train_users)).astype(int)

    print(f"  → Cold start enhancements added")
    return df


def add_interaction_dynamics(book_genres_df:  pd.DataFrame, df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет фичи динамики взаимодействий пользователя с книгами.
    """
    print("Adding interaction dynamics features...")

    train_df_sorted = train_df.sort_values([constants.COL_USER_ID, constants.COL_TIMESTAMP])

    # 1. Тенденция активности пользователя (увеличивается/уменьшается)
    def get_activity_trend(user_data):
        if len(user_data) < 3:
            return {'activity_trend': 0, 'recent_activity_level': 0}

        # Разделяем на раннюю и позднюю активность
        split_point = len(user_data) // 2
        early = user_data.iloc[:split_point]
        late = user_data.iloc[split_point:]

        early_rate = len(early) / max(
            (early[constants.COL_TIMESTAMP].max() - early[constants.COL_TIMESTAMP].min()).days, 1)
        late_rate = len(late) / max((late[constants.COL_TIMESTAMP].max() - late[constants.COL_TIMESTAMP].min()).days, 1)

        recent_activity = len(user_data.iloc[-30:]) if len(user_data) >= 30 else len(user_data)

        return {
            'activity_trend': (late_rate - early_rate) / (early_rate + 1e-10),
            'recent_activity_level': recent_activity
        }

    activity_trends = train_df_sorted.groupby(constants.COL_USER_ID).apply(get_activity_trend).reset_index()
    activity_trends = pd.DataFrame(
        activity_trends[0].tolist(),
        index=activity_trends[constants.COL_USER_ID]
    ).reset_index()

    activity_trends.columns = [constants.COL_USER_ID, 'activity_trend', 'recent_activity_level']

    # 2. Изменение конверсии со временем
    def get_conversion_trend(user_data):
        if len(user_data) < 3:
            return {'conversion_trend': 0}

        split_point = len(user_data) // 2
        early = user_data.iloc[:split_point]
        late = user_data.iloc[split_point:]

        early_conversion = early[constants.COL_HAS_READ].mean()
        late_conversion = late[constants.COL_HAS_READ].mean()

        return {
            'conversion_trend': late_conversion - early_conversion
        }

    conversion_trends = train_df_sorted.groupby(constants.COL_USER_ID).apply(get_conversion_trend).reset_index()
    conversion_trends = pd.DataFrame(
        conversion_trends[0].tolist(),
        index=conversion_trends[constants.COL_USER_ID]
    ).reset_index()

    conversion_trends.columns = [constants.COL_USER_ID, 'conversion_trend']

    # 3. Стабильность жанровых предпочтений
    train_with_genres = train_df_sorted.merge(
        book_genres_df[[constants.COL_BOOK_ID, constants.COL_GENRE_ID]].drop_duplicates(),
        on=constants.COL_BOOK_ID,
        how='left'
    )

    def get_genre_stability(user_data):
        if len(user_data) < 2:
            return {'genre_stability': 0}

        genres = user_data[constants.COL_GENRE_ID].dropna().unique()
        if len(genres) < 2:
            return {'genre_stability': 1}

        # Jaccard similarity между ранней и поздней половиной
        split_point = len(user_data) // 2
        early_genres = set(user_data.iloc[:split_point][constants.COL_GENRE_ID].dropna().unique())
        late_genres = set(user_data.iloc[split_point:][constants.COL_GENRE_ID].dropna().unique())

        if not early_genres or not late_genres:
            return {'genre_stability': 0}

        jaccard = len(early_genres & late_genres) / len(early_genres | late_genres)
        return {'genre_stability': jaccard}

    genre_stabilities = train_with_genres.groupby(constants.COL_USER_ID).apply(get_genre_stability).reset_index()
    genre_stabilities = pd.DataFrame(
        genre_stabilities[0].tolist(),
        index=genre_stabilities[constants.COL_USER_ID]
    ).reset_index()

    genre_stabilities.columns = [constants.COL_USER_ID, 'genre_stability']

    # Объединяем все dynamic фичи
    dynamics_features = activity_trends.merge(
        conversion_trends, on=constants.COL_USER_ID, how='left'
    ).merge(
        genre_stabilities, on=constants.COL_USER_ID, how='left'
    )

    # Присоединяем к основному df
    df = df.merge(dynamics_features, on=constants.COL_USER_ID, how='left')

    print(f"  → Interaction dynamics features added: {len(dynamics_features.columns) - 1} new features")
    return df


# 6. create_enhanced_features — с очисткой после каждой тяжёлой функции
def create_enhanced_features(
    df: pd.DataFrame,
    book_genres_df: pd.DataFrame,
    descriptions_df: pd.DataFrame,
    include_aggregates: bool = True,
    include_bert: bool = False,
    include_nomic: bool = True,
) -> pd.DataFrame:

    print("STARTING MEMORY-OPTIMIZED feature engineering...")
    df = downcast_ids(df)
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Последовательность функций с gc.collect()
    df = add_temporal_features(df, train_df); gc.collect()
    df = add_interaction_feature(df, train_df); gc.collect()
    if include_aggregates:
        df = add_aggregate_features(df, train_df); gc.collect()

    df = add_enhanced_temporal_features(df, train_df); gc.collect()
    df = add_genre_features(df, book_genres_df); gc.collect()
    df = add_genre_preference_features(df, train_df, book_genres_df); gc.collect()
    df = add_sequence_features(df, train_df); gc.collect()
    df = add_author_affinity_features(df, train_df); gc.collect()
    df = add_cold_start_enhancements(df, train_df); gc.collect()
    df = add_interaction_dynamics(book_genres_df, df, train_df); gc.collect()

    df = add_text_features(df, train_df, descriptions_df); gc.collect()

    if include_bert:
        df = add_bert_features(df, train_df, descriptions_df); gc.collect()
    elif include_nomic:
        df = add_nomic_features(df, train_df, descriptions_df, n_components=768); gc.collect()

    # Nomic profile — самая тяжёлая, в конце и чанками
    if any(c.startswith("nomic_pca_") for c in df.columns):
        df = add_nomic_profile_features(df, train_df, chunk_size=1_000_000); gc.collect()

    df = add_conversion_features(df, train_df); gc.collect()
    df = handle_missing_values(df, train_df); gc.collect()

    print(f"Feature engineering finished — {len(df.columns)} columns, ~{df.memory_usage(deep=True).sum() / 1e9:.1f} GB")
    return df

def create_features(
    df: pd.DataFrame,
    book_genres_df: pd.DataFrame,
    descriptions_df: pd.DataFrame,
    include_aggregates: bool = True,
    include_bert: bool = False,
    include_nomic: bool = True,
) -> pd.DataFrame:
    """Runs the full feature engineering pipeline.

    This function orchestrates the calls to add interaction feature, aggregate features
    (optional), genre features, text features (TF-IDF and BERT), and handle missing values.

    Args:
        df (pd.DataFrame): The merged DataFrame from `data_processing`.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.
        include_aggregates (bool): If True, compute aggregate features. Defaults to False.
            Aggregates are typically computed separately during training to avoid data leakage.
        include_bert (bool): If True, compute BERT embeddings. Defaults to True.
            Set to False for faster testing.

    Returns:
        pd.DataFrame: The final DataFrame with all features engineered.
    """
    print("Starting feature engineering pipeline...")
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN]

    featured_df = create_enhanced_features(
        train_df,
        book_genres_df,
        descriptions_df,
        include_aggregates=False,
        include_bert=False,
        include_nomic=True
    )

    print("Feature engineering complete.")
    return featured_df
