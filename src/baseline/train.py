"""
Main training script for the LightGBM model.

Uses temporal split with absolute date threshold to ensure methodologically
correct validation without data leakage from future timestamps.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from . import config, constants
from .evaluate import ndcg_at_k
from .features import add_aggregate_features, handle_missing_values
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date


def add_negative_samples(
    train_df: pd.DataFrame,
    base_df: pd.DataFrame,
    n_neg: int,
    rng: np.random.Generator,
    max_total: int,
) -> pd.DataFrame:
    """Augment train_df with synthetic cold items (label=0) per user using existing books."""
    all_books = base_df[constants.COL_BOOK_ID].unique()
    book_templates = base_df.drop_duplicates(subset=[constants.COL_BOOK_ID]).set_index(constants.COL_BOOK_ID)
    min_ts = base_df[constants.COL_TIMESTAMP].min()
    ts_neg = min_ts - pd.Timedelta(seconds=1)

    new_rows = []
    total_added = 0
    for user_id, group in train_df.groupby(constants.COL_USER_ID):
        if total_added >= max_total:
            break
        user_books = set(group[constants.COL_BOOK_ID].unique())
        candidate_books = np.setdiff1d(all_books, list(user_books))
        if candidate_books.size == 0:
            continue
        k = min(n_neg, len(candidate_books), max_total - total_added)
        sampled = rng.choice(candidate_books, size=k, replace=False)
        for book_id in sampled:
            tpl = book_templates.loc[book_id].copy()
            tpl[constants.COL_USER_ID] = user_id
            tpl[config.TARGET] = 0
            tpl[constants.COL_SOURCE] = constants.VAL_SOURCE_TRAIN
            tpl[constants.COL_TIMESTAMP] = ts_neg
            new_rows.append(tpl)
            total_added += 1
            if total_added >= max_total:
                break

    if not new_rows:
        return train_df

    neg_df = pd.DataFrame(new_rows)
    return pd.concat([train_df, neg_df], ignore_index=True)



def train() -> None:
    """Runs the model training pipeline with temporal split.

    Loads prepared data from data/processed/, performs temporal split based on
    absolute date threshold, computes aggregate features on train split only,
    and trains a single LightGBM model for multiclass classification (relevance).
    Relevance classes: 0=cold candidates, 1=planned books, 2=read books.
    This ensures methodologically correct validation without data leakage from
    future timestamps.

    Note: Data must be prepared first using prepare_data.py
    """
    # Load prepared data
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    print(f"Loading prepared data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")
    print(f"Loaded {len(featured_df):,} rows with {len(featured_df.columns)} features")

    # Separate train set
    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Check for timestamp column
    if constants.COL_TIMESTAMP not in train_set.columns:
        raise ValueError(
            f"Timestamp column '{constants.COL_TIMESTAMP}' not found in train set. "
            "Make sure data was prepared with timestamp preserved."
        )

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(train_set[constants.COL_TIMESTAMP]):
        train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])

    # Perform temporal split
    print(f"\nPerforming temporal split with ratio {config.TEMPORAL_SPLIT_RATIO}...")
    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    print(f"Split date: {split_date}")

    train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)

    # Split data
    train_split = train_set[train_mask].copy()
    val_split = train_set[val_mask].copy()

    print(f"Train split: {len(train_split):,} rows")
    print(f"Validation split: {len(val_split):,} rows")

    # Verify temporal correctness
    max_train_timestamp = train_split[constants.COL_TIMESTAMP].max()
    min_val_timestamp = val_split[constants.COL_TIMESTAMP].min()
    print(f"Max train timestamp: {max_train_timestamp}")
    print(f"Min validation timestamp: {min_val_timestamp}")

    if min_val_timestamp <= max_train_timestamp:
        raise ValueError(
            f"Temporal split validation failed: min validation timestamp ({min_val_timestamp}) "
            f"is not greater than max train timestamp ({max_train_timestamp})."
        )
    print("✅ Temporal split validation passed: all validation timestamps are after train timestamps")

    # Augment train split with synthetic cold negatives to teach ranking of cold items
    rng = np.random.default_rng(config.RANDOM_STATE)
    train_split_original = train_split.copy()
    train_split = add_negative_samples(
        train_split,
        base_df=train_split_original,
        n_neg=config.NEGATIVE_SAMPLES_PER_USER,
        rng=rng,
        max_total=config.NEGATIVE_MAX_SAMPLES,
    )
    if len(train_split) > len(train_split_original):
        print(f"Added {len(train_split) - len(train_split_original):,} synthetic cold samples")

    # Compute aggregate features on train split only (to prevent data leakage)
    print("\nComputing aggregate features on train split only...")
    train_split_with_agg = add_aggregate_features(train_split.copy(), train_split_original)
    val_split_with_agg = add_aggregate_features(val_split.copy(), train_split_original)  # Use original train_split for aggregates!

    # Handle missing values (use original train split for fill values)
    print("Handling missing values...")
    train_split_final = handle_missing_values(train_split_with_agg, train_split_original)
    val_split_final = handle_missing_values(val_split_with_agg, train_split_original)

    # For ranking we need grouped data; keep samples of each user together
    train_split_final = train_split_final.sort_values(constants.COL_USER_ID).reset_index(drop=True)
    val_split_final = val_split_final.sort_values(constants.COL_USER_ID).reset_index(drop=True)

    # Define features (X) and target (y)
    # Exclude timestamp, source, target, prediction columns
    exclude_cols = [
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
    ]
    base_features = [col for col in train_split_final.columns if col not in exclude_cols]

    # Exclude any remaining object columns that are not model features
    non_feature_object_cols = train_split_final[base_features].select_dtypes(include=["object"]).columns.tolist()
    model_features = [f for f in base_features if f not in non_feature_object_cols]
    # Drop only explicit target duplicates; keep interaction signal
    model_features = [f for f in model_features if f not in ["has_read"]]
    # Drop heavy raw nomic embeddings to fit GPU memory; keep similarity features
    model_features = [f for f in model_features if not f.startswith("nomic_")]

    X_train = train_split_final[model_features].copy()
    y_train = train_split_final[config.TARGET]
    X_val = val_split_final[model_features].copy()
    y_val = val_split_final[config.TARGET]
    # Optimize memory usage: convert float64 to float32 (reduces memory by ~50%)
    print("Optimizing data types for memory efficiency...")
    float64_cols = X_train.select_dtypes(include=["float64"]).columns
    if len(float64_cols) > 0:
        print(f"  Converting {len(float64_cols)} float64 columns to float32...")
        X_train[float64_cols] = X_train[float64_cols].astype("float32")
        X_val[float64_cols] = X_val[float64_cols].astype("float32")
        print(f"  Memory saved: ~{X_train[float64_cols].memory_usage(deep=True).sum() / 1024**2 / 2:.1f} MB")

    # Identify categorical features for LightGBM
    categorical_features = [
        f for f in model_features if train_split_final[f].dtype.name == "category"
    ]
    if categorical_features:
        print(f"  Categorical features: {len(categorical_features)} ({categorical_features[:5]}...)")
        # CatBoost требует строковые категории; кастуем явно
        for col in categorical_features:
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)

    print(f"Training features: {len(model_features)}")
    print(f"  Training data shape: {X_train.shape}, Memory: {X_train.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Train model
    print("\nTraining CatBoostRanker (YetiRankPairwise)...")
    print("  Relevance: 0=cold candidates, 1=planned books, 2=read books")
    params = config.CATBOOST_RANKER_PARAMS.copy()
    model = CatBoostRanker(**params)

    # group_id is user_id for listwise ranking
    group_id_train = train_split_final[constants.COL_USER_ID].to_numpy()
    group_id_val = val_split_final[constants.COL_USER_ID].to_numpy()

    train_pool = Pool(X_train, y_train, group_id=group_id_train, cat_features=categorical_features)
    val_pool = Pool(X_val, y_val, group_id=group_id_val, cat_features=categorical_features)

    model.fit(
        train_pool,
        eval_set=val_pool,
        **config.CATBOOST_RANKER_FIT_KWARGS,
    )

    # Evaluate the model
    val_scores = model.predict(val_pool)
    val_eval_df = pd.DataFrame(
        {
            constants.COL_USER_ID: val_split_final[constants.COL_USER_ID].to_numpy(),
            "target": y_val.to_numpy(),
            "score": val_scores,
        }
    )

    ndcg_k = getattr(constants, "MAX_RANKING_LENGTH", 20)
    user_ndcg = []
    for _, group in val_eval_df.groupby(constants.COL_USER_ID):
        ranked = group.sort_values("score", ascending=False)
        relevance = ranked["target"].tolist()
        user_ndcg.append(ndcg_at_k(relevance_scores=relevance, k=ndcg_k))

    mean_ndcg = float(np.mean(user_ndcg)) if user_ndcg else 0.0
    print(f"\nValidation metrics:")
    print(f"  NDCG@{ndcg_k}: {mean_ndcg:.4f}")

    # Save the trained model
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    model.save_model(str(model_path))
    print(f"Model saved → {model_path}")

    with open(config.MODEL_DIR / "features_list.json", "w") as f:
        json.dump(model_features, f)
    print(f"Feature list saved → {config.MODEL_DIR / 'features_list.json'}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    train()
