"""
Inference script to generate predictions for the test set using 2-stage model.

Stage 1: CatBoostRanker
Stage 2: LightGBM with Stage 1 predictions as feature
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRanker, Pool

from . import config, constants
from .data_processing import expand_candidates, load_and_merge_data
from .features import add_aggregate_features, handle_missing_values, add_temporal_features


def predict() -> None:
    """Generates and saves ranked predictions for the test set using 2-stage model.

    This script:
    1. Loads targets.csv and candidates.csv
    2. Expands candidates into (user_id, book_id) pairs
    3. Computes features on all train data
    4. Generates Stage 1 predictions using CatBoostRanker
    5. Adds Stage 1 predictions as feature for Stage 2
    6. Generates final predictions using LightGBM
    7. Ranks candidates for each user and selects top-K (K = min(20, num_candidates))
    8. Saves submission.csv in format: user_id,book_id_list

    Note: Data must be prepared first using prepare_data.py, and model must be trained
    using train.py (train_2stage function)
    """
    # Load targets and candidates
    print("Loading targets and candidates...")
    targets_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.TARGETS_FILENAME,
        dtype={constants.COL_USER_ID: "int32"},
    )
    candidates_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.CANDIDATES_FILENAME,
        dtype={constants.COL_USER_ID: "int32"},
    )

    print(f"Targets: {len(targets_df):,} users")
    print(f"Candidates: {len(candidates_df):,} users")

    # Expand candidates into pairs
    print("\nExpanding candidates...")
    candidates_pairs_df = expand_candidates(candidates_df)

    # Load prepared data for base features
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    print(f"Loading prepared data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")

    # Get train data for computing aggregates
    train_df = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Load metadata for candidates
    print("Loading metadata...")
    _, _, _, book_genres_df, descriptions_df = load_and_merge_data()
    # We need users and books data separately
    user_data_df = pd.read_csv(config.RAW_DATA_DIR / constants.USER_DATA_FILENAME)
    book_data_df = pd.read_csv(config.RAW_DATA_DIR / constants.BOOK_DATA_FILENAME)

    # Merge metadata with candidates
    print("Merging metadata with candidates...")
    candidates_with_meta = candidates_pairs_df.merge(user_data_df, on=constants.COL_USER_ID, how="left")
    book_data_df = book_data_df.drop_duplicates(subset=[constants.COL_BOOK_ID])
    candidates_with_meta = candidates_with_meta.merge(book_data_df, on=constants.COL_BOOK_ID, how="left")

    # Сначала добавим жанровые фичи
    print("Adding genre features...")
    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].count().reset_index()
    genre_counts.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_GENRES_COUNT,
    ]
    candidates_with_meta = candidates_with_meta.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")

    # Добавим базовые временные фичи
    candidates_with_meta = add_temporal_features(candidates_with_meta, train_df)

    # Заполним отсутствующие значения в жанрах
    candidates_with_meta[constants.F_BOOK_GENRES_COUNT] = candidates_with_meta[constants.F_BOOK_GENRES_COUNT].fillna(0)

    # Add base features from prepared data (genres, text features)
    # We'll match by book_id to get TF-IDF and BERT features
    book_features = featured_df[[constants.COL_BOOK_ID]].drop_duplicates()
    # Get all feature columns except metadata and source columns
    feature_cols = [
        col
        for col in featured_df.columns
        if col
        not in [
            constants.COL_USER_ID,
            constants.COL_BOOK_ID,
            constants.COL_SOURCE,
            constants.COL_TIMESTAMP,
            constants.COL_HAS_READ,
            constants.COL_TARGET,
            constants.COL_PREDICTION,
            constants.COL_GENDER,
            constants.COL_AGE,
            constants.COL_AUTHOR_ID,
            constants.COL_PUBLICATION_YEAR,
            constants.COL_LANGUAGE,
            constants.COL_PUBLISHER,
            constants.COL_AVG_RATING,
        ]
        and not col.startswith("tfidf_")
        and not col.startswith("bert_")
    ]

    # Add genre count and text features
    # Get a representative row for each book (just take first occurrence)
    book_features_df = featured_df[[constants.COL_BOOK_ID] + feature_cols].drop_duplicates(
        subset=[constants.COL_BOOK_ID]
    )

    # Merge book features - drop duplicate columns before merge
    # Remove columns that will be merged from candidates_with_meta if they exist
    cols_to_drop = [col for col in feature_cols if col in candidates_with_meta.columns]
    if cols_to_drop:
        candidates_with_meta = candidates_with_meta.drop(columns=cols_to_drop)

    candidates_with_meta = candidates_with_meta.merge(
        book_features_df, on=constants.COL_BOOK_ID, how="left"
    )

    # Get TF-IDF and BERT features from prepared data
    tfidf_cols = [col for col in featured_df.columns if col.startswith("tfidf_")]
    bert_cols = [col for col in featured_df.columns if col.startswith("bert_")]
    text_feature_cols = tfidf_cols + bert_cols

    if text_feature_cols:
        book_text_features = featured_df[[constants.COL_BOOK_ID] + text_feature_cols].drop_duplicates(
            subset=[constants.COL_BOOK_ID]
        )
        candidates_with_meta = candidates_with_meta.merge(
            book_text_features, on=constants.COL_BOOK_ID, how="left"
        )

    # Compute aggregate features on ALL train data
    print("\nComputing aggregate features on all train data...")
    candidates_with_agg = add_aggregate_features(candidates_with_meta.copy(), train_df)

    # Обрабатываем отсутствующие значения - только один раз после всех добавлений
    print("Handling missing values...")
    candidates_final = handle_missing_values(candidates_with_agg, train_df)

    # Load feature list saved during training
    features_path = config.MODEL_DIR / "features_list.json"
    if features_path.exists():
        print("Loading feature list from training...")
        with open(features_path, "r") as f:
            features2 = json.load(f)  # features with score_stage1
        print(f"Loaded {len(features2)} features from training")

        # For Stage 1 (CatBoostRanker), use features without score_stage1
        features = [f for f in features2 if f != 'score_stage1']
        print(f"Features for Stage 1 (CatBoostRanker): {len(features)}")
        print(f"Features for Stage 2 (LightGBM): {len(features2)}")
    else:
        raise FileNotFoundError(
            f"Feature list not found at {features_path}. "
            "Please run 'poetry run python -m src.baseline.train' first."
        )

    # Prepare data for prediction
    # Keep only needed columns
    all_needed_cols = list(set(features2 + [constants.COL_USER_ID, constants.COL_BOOK_ID]))
    candidates_final = candidates_final[[col for col in all_needed_cols if col in candidates_final.columns]]

    # Add missing features with default values
    missing_features = [f for f in features2 if f not in candidates_final.columns]
    if missing_features:
        print(f"Warning: Missing {len(missing_features)} features in candidates, adding defaults")
        for feat in missing_features:
            if feat in train_df.columns:
                if train_df[feat].dtype.name == "category":
                    default_val = train_df[feat].cat.categories[0] if len(train_df[feat].cat.categories) > 0 else 0
                    candidates_final[feat] = pd.Categorical([default_val] * len(candidates_final),
                                                            categories=train_df[feat].cat.categories, ordered=False)
                else:
                    candidates_final[feat] = train_df[feat].iloc[0] if len(train_df) > 0 else 0
            else:
                candidates_final[feat] = 0

    # Ensure all features exist in candidates_final
    for feat in features2:
        if feat not in candidates_final.columns:
            candidates_final[feat] = 0

    # === STAGE 1: CatBoostRanker ===
    print("\n=== Stage 1: CatBoostRanker ===")

    # Prepare data for CatBoost
    X_test_stage1 = candidates_final[features].copy()

    # Identify categorical features for CatBoost
    categorical_features = [f for f in features if X_test_stage1[f].dtype.name == "category"]

    # Convert categorical features to strings for CatBoost
    for col in categorical_features:
        X_test_stage1[col] = X_test_stage1[col].astype(str).fillna('nan')

    # Load Stage 1 model
    stage1_model_path = config.MODEL_DIR / "stage1_catboost.cbm"
    if not stage1_model_path.exists():
        raise FileNotFoundError(
            f"Stage 1 model not found at {stage1_model_path}. "
            "Please run 'poetry run python -m src.baseline.train' first."
        )

    print(f"Loading Stage 1 model from {stage1_model_path}...")
    stage1_model = CatBoostRanker()
    stage1_model.load_model(str(stage1_model_path))

    # Generate Stage 1 predictions
    print("Generating Stage 1 predictions...")
    # Note: CatBoostRanker doesn't need group_id for prediction
    stage1_predictions = stage1_model.predict(X_test_stage1)

    # Add Stage 1 predictions as feature for Stage 2
    candidates_final['score_stage1'] = stage1_predictions

    # === STAGE 2: LightGBM ===
    print("\n=== Stage 2: LightGBM ===")

    # Prepare data for LightGBM
    X_test_stage2 = candidates_final[features2].copy()

    # Convert categorical features to codes for LightGBM
    for col in features2:
        if X_test_stage2[col].dtype.name == "category":
            # Use categories from training data
            if col in train_df.columns and train_df[col].dtype.name == "category":
                train_categories = train_df[col].cat.categories
                X_test_stage2[col] = pd.Categorical(X_test_stage2[col], categories=train_categories).codes
            else:
                # If not in training data, convert to codes directly
                X_test_stage2[col] = X_test_stage2[col].cat.codes

    # Convert all to float
    X_test_stage2 = X_test_stage2.astype(float)

    # Load Stage 2 model
    stage2_model_path = config.MODEL_DIR / "stage2_lightgbm.txt"
    if not stage2_model_path.exists():
        raise FileNotFoundError(
            f"Stage 2 model not found at {stage2_model_path}. "
            "Please run 'poetry run python -m src.baseline.train' first."
        )

    print(f"Loading Stage 2 model from {stage2_model_path}...")
    stage2_model = lgb.Booster(model_file=str(stage2_model_path))

    # Generate final predictions
    print("Generating final predictions...")
    final_predictions = stage2_model.predict(X_test_stage2)

    # Add final predictions to candidates dataframe
    candidates_final["prediction"] = final_predictions

    # Rank candidates for each user and select top-K
    print("\nRanking candidates for each user...")
    submission_rows = []

    # Sort candidates by user_id for consistency
    candidates_final = candidates_final.sort_values(by=constants.COL_USER_ID).reset_index(drop=True)

    for user_id in targets_df[constants.COL_USER_ID]:
        user_candidates = candidates_final[candidates_final[constants.COL_USER_ID] == user_id].copy()

        if len(user_candidates) == 0:
            # No candidates for this user - empty list
            book_id_list = ""
        else:
            # Sort by prediction score (descending)
            user_candidates = user_candidates.sort_values("prediction", ascending=False)

            # Select top-K, where K = min(20, num_candidates)
            k = min(constants.MAX_RANKING_LENGTH, len(user_candidates))
            top_books = user_candidates.head(k)

            # Create comma-separated string of book_ids
            book_id_list = ",".join([str(int(book_id)) for book_id in top_books[constants.COL_BOOK_ID]])

        submission_rows.append({constants.COL_USER_ID: user_id, constants.COL_BOOK_ID_LIST: book_id_list})

    # Create submission DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Ensure submission directory exists
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME

    # Save submission
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission file created at: {submission_path}")
    print(f"Submission shape: {submission_df.shape}")

    # Print statistics
    non_empty = submission_df[submission_df[constants.COL_BOOK_ID_LIST] != ""].shape[0]
    print(f"Users with recommendations: {non_empty}/{len(submission_df)}")

    # Print prediction statistics
    print(f"\nPrediction statistics:")
    print(f"  Stage 1 predictions range: [{stage1_predictions.min():.4f}, {stage1_predictions.max():.4f}]")
    print(f"  Stage 2 predictions range: [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")
    print(f"  Average prediction score: {final_predictions.mean():.4f}")


if __name__ == "__main__":
    predict()
