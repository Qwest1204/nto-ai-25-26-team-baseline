"""
Feature engineering script.
"""

import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from . import config, constants


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

    if constants.COL_TIMESTAMP not in train_df.columns:
        raise ValueError(f"Timestamp column '{constants.COL_TIMESTAMP}' not found in train data.")
    if not pd.api.types.is_datetime64_any_dtype(train_df[constants.COL_TIMESTAMP]):
        train_df = train_df.copy()
        train_df[constants.COL_TIMESTAMP] = pd.to_datetime(train_df[constants.COL_TIMESTAMP])

    max_ts = train_df[constants.COL_TIMESTAMP].max()

    def _add_prior(mean_series: pd.Series, count_series: pd.Series, prior_count: float, prior_mean: float) -> pd.Series:
        return (mean_series * count_series + prior_mean * prior_count) / (count_series + prior_count)

    def _window(df_in: pd.DataFrame, days: int) -> pd.DataFrame:
        cutoff = max_ts - pd.Timedelta(days=days)
        return df_in[df_in[constants.COL_TIMESTAMP] >= cutoff]

    prior_mean = train_df[config.TARGET].mean()
    prior_count = 10.0

    # User-based aggregates
    user_agg = train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT,
    ]
    user_last_ts = (
        train_df.groupby(constants.COL_USER_ID)[constants.COL_TIMESTAMP].max().reset_index()
    )
    user_last_ts[constants.F_USER_LAST_INTERACTION_DAYS] = (
        (max_ts - user_last_ts[constants.COL_TIMESTAMP]).dt.total_seconds() / 86400.0
    )
    user_last_ts = user_last_ts.drop(columns=[constants.COL_TIMESTAMP])

    # User top genre (mode by count)
    user_top_genre = None
    if constants.COL_GENRE_ID in train_df.columns:
        user_top_genre = (
            train_df.groupby(constants.COL_USER_ID)[constants.COL_GENRE_ID]
            .agg(lambda s: s.value_counts().idxmax())
            .reset_index()
        )
        user_top_genre.columns = [constants.COL_USER_ID, constants.F_USER_TOP_GENRE]

    # Time-windowed user aggregates
    user_30d = _window(train_df, 30).groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    user_30d.columns = [constants.COL_USER_ID, constants.F_USER_MEAN_30D, constants.F_USER_COUNT_30D]
    user_90d = _window(train_df, 90).groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    user_90d.columns = [constants.COL_USER_ID, constants.F_USER_MEAN_90D, constants.F_USER_COUNT_90D]

    # Apply Bayesian smoothing
    for df_win, mean_col, count_col in [
        (user_30d, constants.F_USER_MEAN_30D, constants.F_USER_COUNT_30D),
        (user_90d, constants.F_USER_MEAN_90D, constants.F_USER_COUNT_90D),
    ]:
        if not df_win.empty:
            df_win[mean_col] = _add_prior(df_win[mean_col], df_win[count_col], prior_count, prior_mean)

    # Book-based aggregates
    book_agg = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT,
    ]
    book_last_ts = (
        train_df.groupby(constants.COL_BOOK_ID)[constants.COL_TIMESTAMP].max().reset_index()
    )
    book_last_ts[constants.F_BOOK_LAST_INTERACTION_DAYS] = (
        (max_ts - book_last_ts[constants.COL_TIMESTAMP]).dt.total_seconds() / 86400.0
    )
    book_last_ts = book_last_ts.drop(columns=[constants.COL_TIMESTAMP])

    # Time-windowed book aggregates
    book_30d = _window(train_df, 30).groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    book_30d.columns = [constants.COL_BOOK_ID, constants.F_BOOK_MEAN_30D, constants.F_BOOK_COUNT_30D]
    book_90d = _window(train_df, 90).groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    book_90d.columns = [constants.COL_BOOK_ID, constants.F_BOOK_MEAN_90D, constants.F_BOOK_COUNT_90D]
    for df_win, mean_col, count_col in [
        (book_30d, constants.F_BOOK_MEAN_30D, constants.F_BOOK_COUNT_30D),
        (book_90d, constants.F_BOOK_MEAN_90D, constants.F_BOOK_COUNT_90D),
    ]:
        if not df_win.empty:
            df_win[mean_col] = _add_prior(df_win[mean_col], df_win[count_col], prior_count, prior_mean)

    # Popularity split by relevance level
    book_relevance = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(
        planned_rate=lambda s: (s == 1).mean(),
        read_rate=lambda s: (s == 2).mean(),
    ).reset_index()
    book_relevance.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_PLANNED_RATE,
        constants.F_BOOK_READ_RATE,
    ]

    genre_relevance = None
    if constants.COL_GENRE_ID in train_df.columns:
        genre_relevance = train_df.groupby(constants.COL_GENRE_ID)[config.TARGET].agg(
            planned_rate=lambda s: (s == 1).mean(),
            read_rate=lambda s: (s == 2).mean(),
        ).reset_index()
        genre_relevance.columns = [
            constants.COL_GENRE_ID,
            constants.F_GENRE_PLANNED_RATE,
            constants.F_GENRE_READ_RATE,
        ]

    # Author-based aggregates
    author_agg = train_df.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(["mean"]).reset_index()
    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING]

    # Merge aggregates into the main dataframe
    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(user_last_ts, on=constants.COL_USER_ID, how="left")
    if user_top_genre is not None:
        df = df.merge(user_top_genre, on=constants.COL_USER_ID, how="left")

    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    df = df.merge(book_last_ts, on=constants.COL_BOOK_ID, how="left")
    df = df.merge(book_relevance, on=constants.COL_BOOK_ID, how="left")

    if genre_relevance is not None and constants.COL_GENRE_ID in df.columns:
        df = df.merge(genre_relevance, on=constants.COL_GENRE_ID, how="left")
        if user_top_genre is not None:
            df[constants.F_BOOK_MATCH_TOP_GENRE] = (
                df[constants.COL_GENRE_ID] == df[constants.F_USER_TOP_GENRE]
            ).astype("int8")
    else:
        if user_top_genre is not None:
            df[constants.F_BOOK_MATCH_TOP_GENRE] = 0

    # Merge time-windowed aggregates
    df = df.merge(user_30d, on=constants.COL_USER_ID, how="left")
    df = df.merge(user_90d, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_30d, on=constants.COL_BOOK_ID, how="left")
    df = df.merge(book_90d, on=constants.COL_BOOK_ID, how="left")

    # Author aggregates last to avoid chain copies
    df = df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")

    # User-text similarity (TF-IDF)
    tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
    if tfidf_cols:
        tfidf_cols_sorted = sorted(tfidf_cols)
        # user profiles computed on train_df to avoid leakage
        user_profiles = (
            train_df.groupby(constants.COL_USER_ID)[tfidf_cols_sorted].mean().astype("float32")
        ).copy()
        user_profiles["_norm"] = np.linalg.norm(user_profiles.values, axis=1)

        # Align profiles to df rows
        user_profile_mat = user_profiles[tfidf_cols_sorted].reindex(df[constants.COL_USER_ID]).to_numpy(dtype=np.float32)
        user_profile_norm = user_profiles["_norm"].reindex(df[constants.COL_USER_ID]).to_numpy(dtype=np.float32) + 1e-9

        cand_mat = df[tfidf_cols_sorted].to_numpy(dtype=np.float32)
        cand_norm = np.linalg.norm(cand_mat, axis=1) + 1e-9
        dots = np.sum(user_profile_mat * cand_mat, axis=1)
        df[constants.F_USER_TFIDF_SIM] = dots / (user_profile_norm * cand_norm + 1e-9)

    # User-text similarity (NOMIC)
    nomic_cols = [c for c in df.columns if c.startswith("nomic_")]
    if nomic_cols:
        nomic_cols_sorted = sorted(nomic_cols)
        user_profiles = (
            train_df.groupby(constants.COL_USER_ID)[nomic_cols_sorted].mean().astype("float32")
        ).copy()
        user_profiles["_norm"] = np.linalg.norm(user_profiles.values, axis=1)

        user_profile_mat = user_profiles[nomic_cols_sorted].reindex(df[constants.COL_USER_ID]).to_numpy(dtype=np.float32)
        user_profile_norm = user_profiles["_norm"].reindex(df[constants.COL_USER_ID]).to_numpy(dtype=np.float32) + 1e-9

        cand_mat = df[nomic_cols_sorted].to_numpy(dtype=np.float32)
        cand_norm = np.linalg.norm(cand_mat, axis=1) + 1e-9
        dots = np.sum(user_profile_mat * cand_mat, axis=1)
        df[constants.F_USER_NOMIC_SIM] = dots / (user_profile_norm * cand_norm + 1e-9)

    return df


def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds the count of genres for each book.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.

    Returns:
        pd.DataFrame: The DataFrame with the new 'book_genres_count' column.
    """
    print("Adding genre features...")
    genre_counts = (
        book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID]
        .count()
        .reset_index()
    )
    genre_counts.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_GENRES_COUNT,
    ]

    # Main genre per book (mode; fallback to first)
    main_genre = (
        book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .reset_index()
    )
    main_genre.columns = [constants.COL_BOOK_ID, constants.COL_GENRE_ID]

    df = df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")
    df = df.merge(main_genre, on=constants.COL_BOOK_ID, how="left")
    return df


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


def add_nomic_features(df: pd.DataFrame, _train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds NOMIC embeddings from book descriptions.
    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        _train_df (pd.DataFrame): The training portion (for consistency).
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.

    Returns:
        pd.DataFrame: The DataFrame with NOMIC embeddings added.
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
                # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_size)
                attention_mask = encoded["attention_mask"]
                # Expand attention mask to match hidden_size dimension for broadcasting


                #Pooling изменен по сравнению с BERT из за специфики nomic
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()

                # Создаем маску, которая исключает первые и последние токены
                seq_mask = torch.ones_like(attention_mask_expanded)
                seq_mask[:, 0] = 0  # исключаем первый токен ([CLS])
                seq_mask[:, -1] = 0  # исключаем последний токен ([SEP])

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

    # Map embeddings to DataFrame rows by book_id
    df_book_ids = df[constants.COL_BOOK_ID].to_numpy()

    # Create embedding matrix
    embeddings_list = []
    for book_id in df_book_ids:
        if book_id in embeddings_dict:
            embeddings_list.append(embeddings_dict[book_id])
        else:
            # Zero embedding for books without descriptions
            embeddings_list.append(np.zeros(config.NOMIC_EMBEDDING_DIM))

    embeddings_array = np.array(embeddings_list)

    # Create DataFrame with NOMIC features
    nomic_feature_names = [f"nomic_{i}" for i in range(config.NOMIC_EMBEDDING_DIM)]
    nomic_df = pd.DataFrame(embeddings_array, columns=nomic_feature_names, index=df.index)

    # Dimensionality reduction for NOMIC
    svd_dim = min(config.NOMIC_SVD_DIM, config.NOMIC_EMBEDDING_DIM)
    svd = TruncatedSVD(n_components=svd_dim, random_state=config.RANDOM_STATE)
    nomic_svd = svd.fit_transform(embeddings_array)
    nomic_svd_names = [f"nomic_svd_{i}" for i in range(svd_dim)]
    nomic_svd_df = pd.DataFrame(nomic_svd, columns=nomic_svd_names, index=df.index)

    # Concatenate NOMIC features with main DataFrame
    df_with_nomic = pd.concat(
        [df.reset_index(drop=True), nomic_df.reset_index(drop=True), nomic_svd_df.reset_index(drop=True)],
        axis=1,
    )

    print(f"Added {len(nomic_feature_names)} NOMIC features and {svd_dim} SVD components.")
    return df_with_nomic


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
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

    # Make a fresh copy to reduce fragmentation
    df = df.copy()

    # Calculate global mean from training data for filling
    # For has_read, this is the proportion of read books
    global_mean = train_df[config.TARGET].mean()

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
    if constants.F_USER_LAST_INTERACTION_DAYS in df.columns:
        df[constants.F_USER_LAST_INTERACTION_DAYS] = df[constants.F_USER_LAST_INTERACTION_DAYS].fillna(
            constants.MISSING_NUM_VALUE
        )
    if constants.F_BOOK_LAST_INTERACTION_DAYS in df.columns:
        df[constants.F_BOOK_LAST_INTERACTION_DAYS] = df[constants.F_BOOK_LAST_INTERACTION_DAYS].fillna(
            constants.MISSING_NUM_VALUE
        )
    if constants.F_BOOK_PLANNED_RATE in df.columns:
        df[constants.F_BOOK_PLANNED_RATE] = df[constants.F_BOOK_PLANNED_RATE].fillna(global_mean)
    if constants.F_BOOK_READ_RATE in df.columns:
        df[constants.F_BOOK_READ_RATE] = df[constants.F_BOOK_READ_RATE].fillna(global_mean)
    for col in [
        constants.F_USER_MEAN_30D,
        constants.F_USER_MEAN_90D,
        constants.F_BOOK_MEAN_30D,
        constants.F_BOOK_MEAN_90D,
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(global_mean)
    for col in [
        constants.F_USER_COUNT_30D,
        constants.F_USER_COUNT_90D,
        constants.F_BOOK_COUNT_30D,
        constants.F_BOOK_COUNT_90D,
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if constants.F_GENRE_PLANNED_RATE in df.columns:
        df[constants.F_GENRE_PLANNED_RATE] = df[constants.F_GENRE_PLANNED_RATE].fillna(global_mean)
    if constants.F_GENRE_READ_RATE in df.columns:
        df[constants.F_GENRE_READ_RATE] = df[constants.F_GENRE_READ_RATE].fillna(global_mean)
    if constants.F_BOOK_MATCH_TOP_GENRE in df.columns:
        df[constants.F_BOOK_MATCH_TOP_GENRE] = df[constants.F_BOOK_MATCH_TOP_GENRE].fillna(0).astype("int8")
    if constants.F_USER_TOP_GENRE in df.columns:
        df[constants.F_USER_TOP_GENRE] = df[constants.F_USER_TOP_GENRE].astype("category")
        if constants.MISSING_CAT_VALUE not in df[constants.F_USER_TOP_GENRE].cat.categories:
            df[constants.F_USER_TOP_GENRE] = df[constants.F_USER_TOP_GENRE].cat.add_categories([constants.MISSING_CAT_VALUE])
        df[constants.F_USER_TOP_GENRE] = df[constants.F_USER_TOP_GENRE].fillna(constants.MISSING_CAT_VALUE)
    if constants.F_USER_TFIDF_SIM in df.columns:
        df[constants.F_USER_TFIDF_SIM] = df[constants.F_USER_TFIDF_SIM].fillna(0.0)
    if constants.F_USER_NOMIC_SIM in df.columns:
        df[constants.F_USER_NOMIC_SIM] = df[constants.F_USER_NOMIC_SIM].fillna(0.0)

    # Bucket rare categories for selected high-cardinality columns based on train_df frequencies
    rare_cols = [constants.COL_AUTHOR_ID, constants.COL_PUBLISHER, constants.COL_LANGUAGE]
    for col in rare_cols:
        if col in df.columns and col in train_df.columns:
            vc = train_df[col].value_counts()
            rare_values = set(vc[vc < config.RARE_CATEGORY_MIN_COUNT].index)
            if rare_values:
                # Add missing category to both df and train_df if categorical
                if df[col].dtype.name == "category":
                    if constants.MISSING_CAT_VALUE not in df[col].cat.categories:
                        df[col] = df[col].cat.add_categories([constants.MISSING_CAT_VALUE])
                if train_df[col].dtype.name == "category":
                    if constants.MISSING_CAT_VALUE not in train_df[col].cat.categories:
                        train_df[col] = train_df[col].cat.add_categories([constants.MISSING_CAT_VALUE])
                # Cast to object to safely replace and avoid category validation issues
                df[col] = df[col].astype("object").where(~df[col].isin(rare_values), other=constants.MISSING_CAT_VALUE)
                train_df[col] = train_df[col].astype("object").where(~train_df[col].isin(rare_values), other=constants.MISSING_CAT_VALUE)
                # Restore to category
                df[col] = df[col].astype("category")
                train_df[col] = train_df[col].astype("category")

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

    #Fill Nomic skips
    nomic_cols = [col for col in df.columns if col.startswith("nomic_")]
    for col in nomic_cols:
        df[col] = df[col].fillna(0.0)

    # Fill remaining categorical features with a special value
    for col in config.CAT_FEATURES + [constants.COL_GENRE_ID]:
        if col in df.columns and df[col].isna().any():
            if df[col].dtype.name == "category":
                if constants.MISSING_CAT_VALUE not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories([constants.MISSING_CAT_VALUE])
                df[col] = df[col].fillna(constants.MISSING_CAT_VALUE)
            elif df[col].dtype.name == "object":
                df[col] = df[col].fillna(constants.MISSING_CAT_VALUE).astype("category")
            elif pd.api.types.is_numeric_dtype(df[col].dtype):
                df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)

    # Log-transforms for skewed counts
    log_map = {
        constants.F_USER_RATINGS_COUNT: constants.F_USER_RATINGS_COUNT_LOG,
        constants.F_BOOK_RATINGS_COUNT: constants.F_BOOK_RATINGS_COUNT_LOG,
        constants.F_BOOK_GENRES_COUNT: constants.F_BOOK_GENRES_COUNT_LOG,
        constants.F_USER_COUNT_30D: constants.F_USER_COUNT_30D_LOG,
        constants.F_USER_COUNT_90D: constants.F_USER_COUNT_90D_LOG,
        constants.F_BOOK_COUNT_30D: constants.F_BOOK_COUNT_30D_LOG,
        constants.F_BOOK_COUNT_90D: constants.F_BOOK_COUNT_90D_LOG,
    }
    log_feats = {}
    for src, dst in log_map.items():
        if src in df.columns:
            log_feats[dst] = np.log1p(df[src].astype(float))
    if log_feats:
        df = df.assign(**log_feats)

    return df


def create_features(
    df: pd.DataFrame,
    book_genres_df: pd.DataFrame,
    descriptions_df: pd.DataFrame,
    include_aggregates: bool = False,
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
    df = handle_missing_values(df, train_df)

    # Convert categorical columns to pandas 'category' dtype for LightGBM
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    if constants.COL_GENRE_ID in df.columns:
        df[constants.COL_GENRE_ID] = df[constants.COL_GENRE_ID].astype("category")

    print("Feature engineering complete.")
    return df

