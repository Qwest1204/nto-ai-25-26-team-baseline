"""
Main training script for the CatBoost model.

Uses temporal split with absolute date threshold to ensure methodologically
correct validation without data leakage from future timestamps.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score

from . import config, constants
from .features import add_aggregate_features, handle_missing_values
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date


def train() -> None:
    """Runs the model training pipeline with temporal split and CatBoost."""

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

    # Use only train interactions for splitting and training
    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    if constants.COL_TIMESTAMP not in train_set.columns:
        raise ValueError(f"Timestamp column '{constants.COL_TIMESTAMP}' not found in data.")

    train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])

    # Temporal split
    print(f"\nPerforming temporal split with ratio {config.TEMPORAL_SPLIT_RATIO}...")
    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    print(f"Split date: {split_date}")

    train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)
    train_split = train_set[train_mask].copy()
    val_split = train_set[val_mask].copy()

    print(f"Train split: {len(train_split):,} rows")
    print(f"Validation split: {len(val_split):,} rows")

    # Temporal correctness check
    if val_split[constants.COL_TIMESTAMP].min() <= train_split[constants.COL_TIMESTAMP].max():
        raise ValueError("Temporal split failed: validation contains past/future overlap with train.")
    print("Temporal split validation passed")

    # Aggregate features — computed ONLY on train_split
    print("\nComputing aggregate features (leak-proof)...")
    train_split = add_aggregate_features(train_split.copy(), train_split)
    val_split   = add_aggregate_features(val_split.copy(),   train_split)

    # Handle missing values
    print("Handling missing values...")
    train_split = handle_missing_values(train_split, train_split)
    val_split   = handle_missing_values(val_split,   train_split)

    # Exclude non-feature columns
    exclude_cols = [
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
    ]
    features = [col for col in train_split.columns if col not in exclude_cols]
    features = [f for f in features if train_split[f].dtype.name != "object"]

    X_train = train_split[features]
    y_train = train_split[config.TARGET]
    X_val   = val_split[features]
    y_val   = val_split[config.TARGET]

    # Convert float64 → float32 for memory
    float64_cols = X_train.select_dtypes("float64").columns
    X_train[float64_cols] = X_train[float64_cols].astype("float32")
    X_val[float64_cols]   = X_val[float64_cols].astype("float32")

    # Categorical features (CatBoost loves native categories)
    cat_features = [f for f in features if train_split[f].dtype.name == "category"]
    print(f"Training on {len(features)} features ({len(cat_features)} categorical)")

    # Prepare pools
    train_pool = Pool(X_train, y_train, cat_features=cat_features, feature_names=features)
    val_pool   = Pool(X_val,   y_val,   cat_features=cat_features, feature_names=features)

    # Model directory
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # CatBoost model
    print("\nTraining CatBoost model (3-class relevance)...")
    model = CatBoostClassifier(**config.CATBOOST_PARAMS)

    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        verbose=100,
        plot=False,
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
    )

    # Validation metrics
    val_preds = model.predict(X_val).flatten()
    val_proba = model.predict_proba(X_val)

    accuracy  = accuracy_score(y_val, val_preds)
    precision = precision_score(y_val, val_preds, average="weighted", zero_division=0)
    recall    = recall_score(y_val, val_preds, average="weighted", zero_division=0)

    print(f"\nValidation metrics:")
    print(f"  Accuracy:           {accuracy:.4f}")
    print(f"  Precision (w.):     {precision:.4f}")
    print(f"  Recall (w.):        {recall:.4f}")
    print(f"  Best iteration:     {model.best_iteration_}")

    # Save model and feature list
    model_path = config.MODEL_DIR / config.CATBOOST_MODEL_FILENAME
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")

    features_path = config.MODEL_DIR / "features_list.json"
    json.dump(features, open(features_path, "w"))
    print(f"Feature list saved to {features_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()
