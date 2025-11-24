"""
Main training script for the XGBoost model.

Uses temporal split with absolute date threshold to ensure methodologically
correct validation without data leakage from future timestamps.
"""

import json
from pathlib import Path

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from . import config, constants
from .evaluate import dcg_at_k, ndcg_at_k
from .features import add_aggregate_features, handle_missing_values
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date


def safe_label_encode(train_series: pd.Series, val_series: pd.Series):
    """
    Safely encode categorical feature handling unseen categories in validation.
    Unseen categories are mapped to -1 (unknown).

    Returns encoded train array, encoded val array, and the fitted LabelEncoder (fitted only on train).
    """
    le = LabelEncoder()

    # Fit only on training data
    train_encoded = le.fit_transform(train_series.astype(str))

    # Transform validation, but replace unseen labels with -1
    val_str = val_series.astype(str)
    try:
        val_encoded = le.transform(val_str)
    except ValueError as e:
        if "y contains previously unseen labels" in str(e):
            # Get all unique values in validation
            val_unique = val_str.unique()
            train_classes_set = set(le.classes_)
            unseen = [v for v in val_unique if v not in train_classes_set]

            if unseen:
                print(
                    f"  Warning: {len(unseen)} unseen categories in validation: {unseen[:10]}{'...' if len(unseen) > 10 else ''}")
                print("    They will be encoded as -1 (unknown).")

            # Map known values using the encoder, unknown → -1
            def encode_safe(x):
                try:
                    return le.transform([x])[0]
                except ValueError:
                    return -1

            val_encoded = np.array([encode_safe(x) for x in val_str])
        else:
            raise e

    # Optionally: ensure -1 is treated as a valid category in XGBoost by shifting all labels if needed
    # But since XGBoost handles negative labels fine in recent versions, this is usually not necessary.

    return train_encoded, val_encoded, le


def train() -> None:
    """Runs the model training pipeline with temporal split.

    Loads prepared data from data/processed/, performs temporal split based on
    absolute date threshold, computes aggregate features on train split only,
    and trains a single XGBoost model for multiclass classification (relevance).
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

    # Compute aggregate features on train split only (to prevent data leakage)
    print("\nComputing aggregate features on train split only...")
    train_split_with_agg = add_aggregate_features(train_split.copy(), train_split)
    val_split_with_agg = add_aggregate_features(val_split.copy(), train_split)  # Use train_split for aggregates!

    # Handle missing values (use train_split for fill values)
    print("Handling missing values...")
    train_split_final = handle_missing_values(train_split_with_agg, train_split)
    val_split_final = handle_missing_values(val_split_with_agg, train_split)

    # Define features (X) and target (y)
    # Exclude timestamp, source, target, prediction columns
    exclude_cols = [
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
    ]
    features = [col for col in train_split_final.columns if col not in exclude_cols]

    # Exclude any remaining object columns that are not model features
    non_feature_object_cols = train_split_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    X_train = train_split_final[features].copy()
    y_train = train_split_final[config.TARGET]
    X_val = val_split_final[features].copy()
    y_val = val_split_final[config.TARGET]
    # Ensure labels are 0, 1, 2 and force num_class=3
    target_col = config.TARGET

    # Safety remap (in case any unexpected label sneaks in)
    label_map = {0: 0, 1: 1, 2: 2}
    y_train = y_train.map(label_map).astype(int)
    y_val = y_val.map(label_map).astype(int)

    # Verify presence of all classes in training (critical!)
    train_classes = sorted(y_train.unique())
    val_classes = sorted(y_val.unique())

    print(f"Classes in training set:   {train_classes}")
    print(f"Classes in validation set: {val_classes}")

    if set(train_classes) != {0, 1, 2}:
        missing = {0, 1, 2} - set(train_classes)
        print(f"WARNING: Training split is missing classes {missing}. "
              "Adding dummy rows to enforce all 3 classes (they will be ignored during training).")

        # Create one dummy row for each missing class with a very small weight (weight=1e-8)
        dummy_X = pd.DataFrame(0, index=range(len(missing)), columns=X_train.columns)
        dummy_y = pd.Series(list(missing))
        dummy_w = np.full(len(missing), 1e-8)

        X_train = pd.concat([X_train, dummy_X], ignore_index=True)
        y_train = pd.concat([y_train, dummy_y], ignore_index=True)

        # Also pass sample_weight to fit()
        train_weights = np.ones(len(y_train))
        train_weights[-len(missing):] = 1e-8
    else:
        train_weights = None

    # Force XGBoost parameters to expect exactly 3 classes
    xgb_params = config.XGB_PARAMS.copy()
    xgb_params.update({
        "objective": "multi:softprob",  # or "multi:softmax"
        "num_class": 3,
        "enable_categorical": False,  # already manually encoded
    })
    # ---------------------------------------------------------
    # Optimize memory usage: convert float64 to float32 (reduces memory by ~50%)
    print("Optimizing data types for memory efficiency...")
    float64_cols = X_train.select_dtypes(include=["float64"]).columns
    if len(float64_cols) > 0:
        print(f"  Converting {len(float64_cols)} float64 columns to float32...")
        X_train[float64_cols] = X_train[float64_cols].astype("float32")
        X_val[float64_cols] = X_val[float64_cols].astype("float32")
        print(f"  Memory saved: ~{X_train[float64_cols].memory_usage(deep=True).sum() / 1024**2 / 2:.1f} MB")

    # Encode categorical features for XGBoost with safe handling of unseen categories
    categorical_features = [
        f for f in features if train_split_final[f].dtype.name == "category"
    ]

    label_encoders = {}
    if categorical_features:
        print(f"Encoding {len(categorical_features)} categorical features with safe encoding...")

        for col in categorical_features:
            print(f"  Encoding {col}...")
            X_train[col], X_val[col], le = safe_label_encode(X_train[col], X_val[col])
            label_encoders[col] = le

    print(f"Training features: {len(features)}")
    print(f"  Training data shape: {X_train.shape}, Memory: {X_train.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Create checkpoint directory for intermediate model saves
    checkpoint_dir = config.MODEL_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Train model
    print("\nTraining XGBoost model (multiclass classification: 3 classes)...")
    print("  Classes: 0=cold candidates, 1=planned books, 2=read books")

    # For XGBoost with encoded categorical features, we don't need enable_categorical
    xgb_params = config.XGB_PARAMS.copy()
    xgb_params["enable_categorical"] = False  # Disable since we manually encoded

    model = xgb.XGBClassifier(**xgb_params)

    # Train with early stopping
    model.fit(
        X_train,
        y_train,
        sample_weight=train_weights,
        eval_set=[(X_val, y_val)],
        #early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        verbose=True,
    )

    # Evaluate the model
    val_preds = model.predict(X_val)
    val_proba = model.predict_proba(X_val)  # Shape: (n_samples, 3) for 3 classes

    accuracy = accuracy_score(y_val, val_preds)
    # For multiclass, use average='weighted' or 'macro'
    precision = precision_score(y_val, val_preds, average="weighted", zero_division=0)
    recall = recall_score(y_val, val_preds, average="weighted", zero_division=0)

    # Class distribution
    class_dist = pd.Series(val_preds).value_counts().sort_index()
    class_proba_mean = val_proba.mean(axis=0)

    print(f"\nValidation metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted): {recall:.4f}")
    print(f"  Predicted class distribution:")
    for class_idx in range(3):
        count = class_dist.get(class_idx, 0)
        proba_mean = class_proba_mean[class_idx]
        print(f"    Class {class_idx}: {count} samples ({100*count/len(val_preds):.1f}%), mean proba: {proba_mean:.4f}")

    # Save the trained model
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    model.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Save feature list and label encoders for prediction
    features_path = config.MODEL_DIR / "features_list.json"
    with open(features_path, "w") as f:
        json.dump(features, f)
    print(f"Feature list saved to {features_path}")

    # Save label encoders if any categorical features were encoded
    if categorical_features:
        encoders_path = config.MODEL_DIR / "label_encoders.pkl"
        import pickle
        with open(encoders_path, "wb") as f:
            pickle.dump(label_encoders, f)
        print(f"Label encoders saved to {encoders_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()
