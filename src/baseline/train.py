"""
Main training script for the LightGBM model.

Uses temporal split with absolute date threshold to ensure methodologically
correct validation without data leakage from future timestamps.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier, Pool
import optuna
import torch

from . import config, constants
from .evaluate import dcg_at_k, ndcg_at_k
from .features import add_aggregate_features, handle_missing_values
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date



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

    X_train = train_split_final[features].copy().drop(["f_user_book_interaction", "has_read"], axis=1)
    y_train = train_split_final[config.TARGET].replace({1:0, 2:1})
    X_val = val_split_final[features].copy().drop(["f_user_book_interaction", "has_read"], axis=1)
    y_val = val_split_final[config.TARGET].replace({1:0, 2:1})
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
        f for f in features if train_split_final[f].dtype.name == "category"
    ]
    if categorical_features:
        print(f"  Categorical features: {len(categorical_features)} ({categorical_features[:5]}...)")

    print(f"Training features: {len(features)}")
    print(f"  Training data shape: {X_train.shape}, Memory: {X_train.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare pools for CatBoost
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    val_pool = Pool(X_val, y_val, cat_features=categorical_features)

    best_params = config.CATBOOST_PARAMS.copy()

    # Define Optuna objective
    def objective(trial):
        params = {
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1",
            "iterations": trial.suggest_int("iterations", 1000, 3000, step=100),  # Around default 2000
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),  # Around default 0.01
            "depth": trial.suggest_int("depth", 4, 10),  # Around default 6
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),  # Around default 10.0
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.5, 1.5),  # Around default 1.0
            "random_strength": trial.suggest_float("random_strength", 0.5, 2.0, log=True),  # Around default 1.0
            "max_bin": trial.suggest_int("max_bin", 100, 300, step=50),  # Around default 254
            "rsm": trial.suggest_float("rsm", 0.5, 1.0),  # Around default 0.7
            "random_seed": config.RANDOM_STATE,
            "thread_count": -1,
            "auto_class_weights": "Balanced",
            "task_type": "GPU" if torch and torch.cuda.is_available() else "CPU",
            "devices": "0" if torch and torch.cuda.is_available() else None,
        }

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            verbose=False,
        )

        val_preds = model.predict(val_pool)
        return f1_score(y_val, val_preds, average="weighted")

    if config.OPTIM_WITH_OPTUNA:
        # Run Optuna optimization
        print("\nStarting Optuna hyperparameter optimization...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)  # Adjust n_trials as needed

        print("\nBest hyperparameters found:")
        print(study.best_params)
        print(f"Best F1 score: {study.best_value:.4f}")

        # Update parameters with best found
        best_params = config.CATBOOST_PARAMS.copy()
        best_params.update(study.best_params)


    # Train final model with best parameters
    print("\nTraining final CatBoost model with final hyperparameters...")
    print("  Classes: 0=cold candidates, 1=planned books, 2=read books")
    if 'num_class' in best_params:
        del best_params['num_class']
    if 'objective' in best_params:
        best_params['loss_function'] = 'MultiClass'
    model = CatBoostClassifier(**best_params)

    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        verbose=True,
    )

    # Evaluate the model
    val_preds = model.predict(X_val).ravel()
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
    for class_idx in range(val_proba.shape[1]):
        count = class_dist.get(class_idx, 0)
        proba_mean = class_proba_mean[class_idx]
        print(f"    Class {class_idx}: {count} samples ({100*count/len(val_preds):.1f}%), mean proba: {proba_mean:.4f}")

    # Save the trained model
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    model.save_model(str(model_path))
    print(f"Model saved → {model_path}")

    with open(config.MODEL_DIR / "features_list.json", "w") as f:
        json.dump(features, f)
    print(f"Feature list saved → {config.MODEL_DIR / 'features_list.json'}")

    # Save best parameters
    with open(config.MODEL_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f)
    print(f"Best parameters saved → {config.MODEL_DIR / 'best_params.json'}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    train()
