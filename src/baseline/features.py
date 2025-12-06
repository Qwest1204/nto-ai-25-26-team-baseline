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

    train_df = train_df.copy()
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


def add_genre_preference_features(df: pd.DataFrame, train_df: pd.DataFrame,
                                  book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет фичи жанровых предпочтений пользователей.
    """
    print("Adding genre preference features...")

    # Объединяем train с жанрами
    train_with_genres = train_df.merge(
        book_genres_df,
        on=constants.COL_BOOK_ID,
        how='left'
    )

    # 1. Топ жанры пользователя (по прочитанным)
    user_genre_read = train_with_genres[train_with_genres[constants.COL_HAS_READ] == 1] \
        .groupby([constants.COL_USER_ID, constants.COL_GENRE_ID]).size().unstack(fill_value=0)
    user_genre_read_norm = user_genre_read.div(user_genre_read.sum(axis=1), axis=0).fillna(0)

    # 2. Топ жанры пользователя (по планам)
    user_genre_planned = train_with_genres[train_with_genres[constants.COL_HAS_READ] == 0] \
        .groupby([constants.COL_USER_ID, constants.COL_GENRE_ID]).size().unstack(fill_value=0)
    user_genre_planned_norm = user_genre_planned.div(user_genre_planned.sum(axis=1), axis=0).fillna(0)

    # 3. Конверсия по жанрам
    user_genre_total = train_with_genres.groupby([constants.COL_USER_ID, constants.COL_GENRE_ID]).size().unstack(fill_value=0)
    user_genre_read_count = train_with_genres[train_with_genres[constants.COL_HAS_READ] == 1] \
        .groupby([constants.COL_USER_ID, constants.COL_GENRE_ID]).size().unstack()
    user_genre_conversion = (user_genre_read_count / (user_genre_total.stack() + 1e-10)).unstack(fill_value=0)

    # mapping book_id → список жанров
    book_to_genres = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].apply(list)

    def calculate_genre_match(row):
        user_id = row[constants.COL_USER_ID]
        book_id = row[constants.COL_BOOK_ID]

        # Если книги нет в маппинге или у пользователя нет истории — возвращаем нули
        if book_id not in book_to_genres:
            return {'genre_read_match': 0.0, 'genre_planned_match': 0.0,
                    'genre_conversion_match': 0.0, 'genre_count': 0}

        genres = book_to_genres[book_id]

        # Если у пользователя нет истории — возвращаем нули
        if user_id not in user_genre_read_norm.index:
            return {'genre_read_match': 0.0, 'genre_planned_match': 0.0,
                    'genre_conversion_match': 0.0, 'genre_count': len(genres)}

        read_scores = []
        planned_scores = []
        conv_scores = []

        for genre in genres:
            # безопасное получение значения (0.0 если столбца/строки нет)
            read_scores.append(user_genre_read_norm.get((user_id, genre), 0.0))
            planned_scores.append(user_genre_planned_norm.get((user_id, genre), 0.0))
            conv_scores.append(user_genre_conversion.get((user_id, genre), 0.0))

        return {
            'genre_read_match': np.mean(read_scores) if read_scores else 0.0,
            'genre_planned_match': np.mean(planned_scores) if planned_scores else 0.0,
            'genre_conversion_match': np.mean(conv_scores) if conv_scores else 0.0,
            'genre_count': len(genres)
        }

    print("  Calculating genre matches...")
    genre_matches = df.apply(calculate_genre_match, axis=1, result_type='expand')

    df[['genre_read_match',
        'genre_planned_match',
        'genre_conversion_match',
        'book_genre_count']] = genre_matches

    print(f"  → Genre preference features added: 4 new features")
    return df


def add_sequence_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет фичи на основе последовательностей взаимодействий.
    """
    print("Adding sequence features...")

    train_df = train_df.sort_values([constants.COL_USER_ID, constants.COL_TIMESTAMP])

    # 1. Паттерны чередования read/plan
    def get_sequence_pattern(user_data):
        if len(user_data) < 2:
            return {'read_after_plan_ratio': 0, 'plan_after_read_ratio': 0}

        transitions = []
        for i in range(1, len(user_data)):
            prev = user_data.iloc[i - 1][constants.COL_HAS_READ]
            curr = user_data.iloc[i][constants.COL_HAS_READ]
            transitions.append(f"{prev}->{curr}")

        read_after_plan = transitions.count("0->1")
        plan_after_read = transitions.count("1->0")
        total_transitions = len(transitions)

        return {
            'read_after_plan_ratio': read_after_plan / total_transitions if total_transitions > 0 else 0,
            'plan_after_read_ratio': plan_after_read / total_transitions if total_transitions > 0 else 0
        }

    user_sequence_patterns = train_df.groupby(constants.COL_USER_ID).apply(get_sequence_pattern).reset_index()
    user_sequence_patterns = pd.DataFrame(
        user_sequence_patterns[0].tolist(),
        index=user_sequence_patterns[constants.COL_USER_ID]
    ).reset_index()

    user_sequence_patterns.columns = [constants.COL_USER_ID, 'read_after_plan_ratio', 'plan_after_read_ratio']

    # 2. Длина текущей серии (сколько книг подряд в планах/прочитано)
    def get_current_streak(user_data):
        if len(user_data) == 0:
            return {'current_streak_type': 0, 'current_streak_length': 0}

        last_action = user_data.iloc[-1][constants.COL_HAS_READ]
        streak_length = 1

        for i in range(len(user_data) - 2, -1, -1):
            if user_data.iloc[i][constants.COL_HAS_READ] == last_action:
                streak_length += 1
            else:
                break

        return {
            'current_streak_type': last_action,
            'current_streak_length': streak_length
        }

    user_streaks = train_df.groupby(constants.COL_USER_ID).apply(get_current_streak).reset_index()
    user_streaks = pd.DataFrame(
        user_streaks[0].tolist(),
        index=user_streaks[constants.COL_USER_ID]
    ).reset_index()

    user_streaks.columns = [constants.COL_USER_ID, 'current_streak_type', 'current_streak_length']

    # 3. Средняя длина серий
    def get_avg_streak_length(user_data):
        if len(user_data) == 0:
            return {'avg_read_streak': 0, 'avg_plan_streak': 0}

        streaks = []
        current_type = user_data.iloc[0][constants.COL_HAS_READ]
        current_length = 1

        for i in range(1, len(user_data)):
            if user_data.iloc[i][constants.COL_HAS_READ] == current_type:
                current_length += 1
            else:
                streaks.append((current_type, current_length))
                current_type = user_data.iloc[i][constants.COL_HAS_READ]
                current_length = 1

        streaks.append((current_type, current_length))

        read_streaks = [length for type_, length in streaks if type_ == 1]
        plan_streaks = [length for type_, length in streaks if type_ == 0]

        return {
            'avg_read_streak': np.mean(read_streaks) if read_streaks else 0,
            'avg_plan_streak': np.mean(plan_streaks) if plan_streaks else 0
        }

    user_avg_streaks = train_df.groupby(constants.COL_USER_ID).apply(get_avg_streak_length).reset_index()
    user_avg_streaks = pd.DataFrame(
        user_avg_streaks[0].tolist(),
        index=user_avg_streaks[constants.COL_USER_ID]
    ).reset_index()

    user_avg_streaks.columns = [constants.COL_USER_ID, 'avg_read_streak', 'avg_plan_streak']

    # Объединяем все sequence фичи
    sequence_features = user_sequence_patterns.merge(
        user_streaks, on=constants.COL_USER_ID, how='left'
    ).merge(
        user_avg_streaks, on=constants.COL_USER_ID, how='left'
    )

    # Присоединяем к основному df
    df = df.merge(sequence_features, on=constants.COL_USER_ID, how='left')

    print(f"  → Sequence features added: {len(sequence_features.columns) - 1} new features")
    return df


def add_author_affinity_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет фичи аффинити к авторам.
    """
    print("Adding author affinity features...")

    # 1. Статистика по авторам в истории пользователя
    author_stats = train_df.groupby([constants.COL_USER_ID, constants.COL_AUTHOR_ID]).agg({
        constants.COL_HAS_READ: ['count', 'mean', 'sum']
    }).reset_index()

    author_stats.columns = [
        constants.COL_USER_ID, constants.COL_AUTHOR_ID,
        'author_interaction_count', 'author_read_ratio', 'author_read_count'
    ]

    # 2. Для каждого пользователя: топ авторы
    user_author_stats = author_stats.groupby(constants.COL_USER_ID).agg({
        'author_interaction_count': ['max', 'mean', 'sum'],
        'author_read_ratio': ['max', 'mean']
    }).reset_index()

    user_author_stats.columns = [
        constants.COL_USER_ID,
        'top_author_interactions', 'avg_author_interactions', 'total_author_interactions',
        'top_author_read_ratio', 'avg_author_read_ratio'
    ]

    # 3. Для каждой пары (user, book) определяем, знаком ли автор
    df = df.merge(user_author_stats, on=constants.COL_USER_ID, how='left')

    # 4. Аффинити к автору текущей книги
    author_affinity = author_stats.set_index([constants.COL_USER_ID, constants.COL_AUTHOR_ID])

    def get_author_affinity(row):
        user_id = row[constants.COL_USER_ID]
        author_id = row[constants.COL_AUTHOR_ID]

        if (user_id, author_id) in author_affinity.index:
            stats = author_affinity.loc[(user_id, author_id)]
            return {
                'author_familiarity': stats['author_interaction_count'],
                'author_read_ratio': stats['author_read_ratio'],
                'author_read_count': stats['author_read_count']
            }
        else:
            return {
                'author_familiarity': 0,
                'author_read_ratio': 0,
                'author_read_count': 0
            }

    print("  Calculating author affinities...")
    affinities = df.apply(get_author_affinity, axis=1)

    df['author_familiarity'] = [x['author_familiarity'] for x in affinities]
    df['author_read_ratio'] = [x['author_read_ratio'] for x in affinities]
    df['author_read_count'] = [x['author_read_count'] for x in affinities]

    # 5. Отношение к среднему по пользователю
    df['author_familiarity_ratio'] = df['author_familiarity'] / (df['avg_author_interactions'] + 1e-10)
    df['author_read_ratio_diff'] = df['author_read_ratio'] - df['avg_author_read_ratio']

    print(f"  → Author affinity features added: {len(user_author_stats.columns) - 1 + 5} new features")
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


def create_enhanced_features(
    df: pd.DataFrame,
    book_genres_df: pd.DataFrame,
    descriptions_df: pd.DataFrame,
    include_aggregates: bool = True,
    include_bert: bool = False,
    include_nomic: bool = True,
) -> pd.DataFrame:
    """Улучшенный пайплайн создания фичей."""
    print("Starting ENHANCED feature engineering pipeline...")

    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # 1. Базовые фичи
    df = add_temporal_features(df, train_df)
    df = add_interaction_feature(df, train_df)

    # 2. Агрегатные фичи
    if include_aggregates:
        df = add_aggregate_features(df, train_df)

    # 3. Улучшенные временные фичи
    df = add_enhanced_temporal_features(df, train_df)

    # 4. Жанровые предпочтения
    df = add_genre_features(df, book_genres_df)
    df = add_genre_preference_features(df, train_df, book_genres_df)

    # 5. Фичи последовательностей
    df = add_sequence_features(df, train_df)

    # 6. Аффинити к авторам
    df = add_author_affinity_features(df, train_df)

    # 7. Улучшения для холодных кандидатов
    df = add_cold_start_enhancements(df, train_df)

    # 8. Динамика взаимодействий
    df = add_interaction_dynamics(book_genres_df, df, train_df)

    # 9. Текстовые фичи (опционально)
    df = add_text_features(df, train_df, descriptions_df)

    if include_bert:
        print("USING BERT FEATURES")
        df = add_bert_features(df, train_df, descriptions_df)
    elif include_nomic:
        print("USING NOMIC FEATURES")
        df = add_nomic_features(df, train_df, descriptions_df)

    # 10. Nomic профильные фичи
    nomic_cols = [col for col in df.columns if col.startswith("nomic_pca_")]
    if len(nomic_cols) > 0:
        df = add_nomic_profile_features(df, train_df)
    else:
        print("Skipping Nomic profile features - no Nomic columns found")
        # Создаем пустые столбцы
        nomic_profile_cols = [
            "user_book_nomic_cos_sim_read", "user_book_nomic_cos_sim_plan",
            "user_book_nomic_cos_sim_all", "user_nomic_profile_strength",
            "user_nomic_diversity", "nomic_distance_to_centroid"
        ]
        for col in nomic_profile_cols:
            df[col] = 0.0

    # 11. Фичи конверсии
    df = add_conversion_features(df, train_df)

    # 12. Обработка пропусков
    df = handle_missing_values(df, train_df)

    # 13. Создаем комбинированные фичи
    print("Creating combined features...")

    # Взаимодействие временных и жанровых фич
    if 'user_activity_frequency' in df.columns and 'genre_read_match' in df.columns:
        df['activity_genre_interaction'] = df['user_activity_frequency'] * df['genre_read_match']

    # Комбинация конверсии и давности
    if 'user_conversion_rate' in df.columns and 'days_since_last_activity' in df.columns:
        df['conversion_recency_interaction'] = df['user_conversion_rate'] / (df['days_since_last_activity'] + 1)

    # Сигнал "настроения" пользователя
    if 'activity_trend' in df.columns and 'conversion_trend' in df.columns:
        df['user_momentum'] = df['activity_trend'] * df['conversion_trend']

    # Конвертируем категориальные колонки
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("ENHANCED feature engineering complete.")
    print(f"Total features: {len(df.columns)}")

    # Выводим информацию о новых фичах
    new_feature_categories = {
        'temporal': ['user_preferred_weekday', 'user_activity_frequency', 'days_since_last_activity'],
        'genre': ['genre_read_match', 'genre_planned_match', 'genre_conversion_match'],
        'sequence': ['read_after_plan_ratio', 'current_streak_length', 'avg_read_streak'],
        'author': ['author_familiarity', 'author_read_ratio', 'author_familiarity_ratio'],
        'cold_start': ['book_demographic_variance', 'is_completely_cold_book'],
        'dynamics': ['activity_trend', 'conversion_trend', 'genre_stability']
    }

    for category, features in new_feature_categories.items():
        existing_features = [f for f in features if f in df.columns]
        if existing_features:
            print(f"  {category.upper()} features: {len(existing_features)}")

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

    featured_df = create_enhanced_features(
        train_df,
        book_genres_df,
        descriptions_df,
        include_aggregates=False,
        include_bert=False,
        include_nomic=True
    )

    print("Feature engineering complete.")
    return df
