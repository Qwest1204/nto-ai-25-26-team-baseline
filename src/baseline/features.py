"""
Feature engineering script.
"""

import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config, constants


def add_nomic_profile_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет мощные персональные фичи на основе Nomic эмбеддингов.
    Работает только с train_df → нет утечек!
    """
    print("Adding Nomic profile features (this is the big one)...")

    # Определяем колонки эмбеддингов (должны быть в processed_features.parquet)
    nomic_cols = [col for col in df.columns if col.startswith("nomic_pca_")]
    if len(nomic_cols) == 0:
        print("  Warning: No Nomic columns found! Skipping profile features.")
        return df
    if len(nomic_cols) != 100:
        print(f"  Warning: Expected 100 Nomic dims, got {len(nomic_cols)}")

    embed_dim = len(nomic_cols)
    embed_array = df[nomic_cols].values  # (N, 100)

    # --- 1. Создаём профили пользователей из train_df ---
    train_df = train_df.copy()

    # Разделяем на прочитанные и запланированные
    read_df = train_df[train_df[constants.COL_HAS_READ] == 1]
    plan_df = train_df[train_df[constants.COL_HAS_READ] == 0]

    # Профиль прочитанных (вес 2.0)
    user_read_profile = read_df.groupby(constants.COL_USER_ID)[nomic_cols].mean()
    user_read_profile = user_read_profile * 2.0  # усиливаем прочитанные

    # Профиль запланированных (вес 1.0)
    user_plan_profile = plan_df.groupby(constants.COL_USER_ID)[nomic_cols].mean()

    # Общий профиль: взвешенная сумма
    user_all_profile = pd.concat([user_read_profile, user_plan_profile]).groupby(level=0).sum()

    # Нормализация профиля (чтобы косинус был в [-1,1])
    user_all_profile_norm = np.linalg.norm(user_all_profile.values, axis=1, keepdims=True)
    user_all_profile_norm[user_all_profile_norm == 0] = 1.0
    user_all_profile_normalized = user_all_profile.values / user_all_profile_norm

    # --- 2. Глобальный центр (для фичи популярности) ---
    global_centroid = embed_array.mean(axis=0, keepdims=True)  # (1, 100)

    # --- 3. Присоединяем профили к df ---
    df = df.merge(user_read_profile.add_prefix("profile_read_"),
                  left_on=constants.COL_USER_ID, right_index=True, how="left")
    df = df.merge(user_plan_profile.add_prefix("profile_plan_"),
                  left_on=constants.COL_USER_ID, right_index=True, how="left")
    df = df.merge(user_all_profile,
                  left_on=constants.COL_USER_ID, right_index=True, how="left", suffixes=("", "_all"))

    # --- 4. Вычисляем косинусные сходства ---
    book_embeddings = df[nomic_cols].values.astype(np.float32)

    # Косинус с прочитанными
    read_profiles = df[[f"profile_read_nomic_{i}" for i in range(embed_dim)]].fillna(0).values
    read_norms = np.linalg.norm(read_profiles, axis=1, keepdims=True)
    read_norms[read_norms == 0] = 1.0
    cos_read = (book_embeddings * read_profiles).sum(axis=1) / (read_norms.squeeze() * np.linalg.norm(book_embeddings, axis=1))

    # Косинус с запланированными
    plan_profiles = df[[f"profile_plan_nomic_{i}" for i in range(embed_dim)]].fillna(0).values
    plan_norms = np.linalg.norm(plan_profiles, axis=1, keepdims=True)
    plan_norms[plan_norms == 0] = 1.0
    cos_plan = (book_embeddings * plan_profiles).sum(axis=1) / (plan_norms.squeeze() * np.linalg.norm(book_embeddings, axis=1))

    # Косинус с общим профилем
    cos_all = cosine_similarity(book_embeddings, user_all_profile_normalized[
        df[constants.COL_USER_ID].map({uid: idx for idx, uid in enumerate(user_all_profile.index)})
    ]).diagonal()

    df["user_book_nomic_cos_sim_read"] = np.nan_to_num(cos_read, nan=0.0)
    df["user_book_nomic_cos_sim_plan"] = np.nan_to_num(cos_plan, nan=0.0)
    df["user_book_nomic_cos_sim_all"]  = np.nan_to_num(cos_all,  nan=0.0)

    # --- 5. Сила профиля (норма вектора) ---
    profile_strength = np.linalg.norm(user_all_profile.values, axis=1)
    df["user_nomic_profile_strength"] = df[constants.COL_USER_ID].map(
        dict(zip(user_all_profile.index, profile_strength))
    ).fillna(0.0)

    # --- 6. Разнообразие вкусов (среднее расстояние между прочитанными) ---
    diversity = read_df.groupby(constants.COL_USER_ID).apply(
        lambda g: np.mean(cosine_similarity(g[nomic_cols])) if len(g) > 1 else 0.0
    )
    df["user_nomic_diversity"] = df[constants.COL_USER_ID].map(diversity).fillna(0.0)

    # --- 7. Расстояние до глобального центра ---
    df["nomic_distance_to_centroid"] = np.linalg.norm(book_embeddings - global_centroid, axis=1)

    # Очистка временных колонок
    df = df.drop(columns=[col for col in df.columns if col.startswith("profile_")], errors="ignore")

    print(f"  → Nomic profile features added: 6 new powerful features")
    return df


def add_temporal_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет 4 надёжные временные фичи без утечек.
    Работает даже для пользователей, которых нет в train_df (холодный старт).
    """
    print("Adding robust temporal features...")

    # Копируем, чтобы не менять исходный train_df
    train_df = train_df.copy()
    train_df[constants.COL_TIMESTAMP] = pd.to_datetime(train_df[constants.COL_TIMESTAMP])

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
    train_descriptions = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)].copy()
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
    all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
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
        all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
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
        print(f"Loading cached NOMIC embeddings from {embeddings_path}")
        embeddings_dict = joblib.load(embeddings_path)
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
        all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
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

    embeddings_array = np.array(embeddings_list)

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

    train_df = train_df.copy()

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
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Add temporal features (new addition to create required columns)
    df = add_temporal_features(df, train_df)

    # Add interaction feature first (must be computed before temporal split)
    # This feature helps distinguish cold candidates from interacted books
    df = add_interaction_feature(df, train_df)

    # Aggregate features are computed separately during training to ensure
    # no data leakage from validation set timestamps
    if include_aggregates:
        df = add_aggregate_features(df, train_df)

    df = add_genre_features(df, book_genres_df)
    df = add_text_features(df, train_df, descriptions_df)
    if include_bert:
        print("USING BERT FEATURES")
        df = add_bert_features(df, train_df, descriptions_df)
    elif include_nomic:
        print("USING NOMIC FEATURES")
        df = add_nomic_features(df, train_df, descriptions_df)

    # Добавляем Nomic профильные фичи ТОЛЬКО если есть Nomic столбцы
    nomic_cols = [col for col in df.columns if col.startswith("nomic_pca_")]
    if len(nomic_cols) > 0:
        df = add_nomic_profile_features(df, train_df)
    else:
        print("Skipping Nomic profile features - no Nomic columns found")
        # Создаем пустые столбцы для избежания ошибки в handle_missing_values
        df["user_book_nomic_cos_sim_read"] = 0.0
        df["user_book_nomic_cos_sim_plan"] = 0.0
        df["user_book_nomic_cos_sim_all"] = 0.0
        df["user_nomic_profile_strength"] = 0.0
        df["user_nomic_diversity"] = 0.5
        df["nomic_distance_to_centroid"] = 0.0

    df = handle_missing_values(df, train_df)

    # Convert categorical columns to pandas 'category' dtype for LightGBM
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Feature engineering complete.")
    return df
