"""
Inference script to generate predictions for the test set.

Computes aggregate features on all train data and applies them to test set,
then generates predictions using the trained model and ranks candidates for each user.
"""

import json
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool

from . import config, constants
from .data_processing import expand_candidates, load_and_merge_data
from .features import add_aggregate_features, handle_missing_values


def predict() -> None:
    """Generates and saves ranked predictions for the test set.

    This script:
    1. Loads targets.csv and candidates.csv
    2. Expands candidates into (user_id, book_id) pairs
    3. Computes aggregate features on all train data
    4. Generates probabilities for 3 classes using the trained multiclass model
       (class 0=cold, 1=planned, 2=read)
    5. Calculates ranking score: p1*1 + p2*2 (weighted sum based on relevance)
    6. Ranks candidates for each user and selects top-K (K = min(20, num_candidates))
    7. Saves submission.csv in format: user_id,book_id_list

    Note: Data must be prepared first using prepare_data.py, and model must be trained
    using train.py
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

    # Handle missing values
    print("Handling missing values...")
    candidates_final = handle_missing_values(candidates_with_agg, train_df)

    # Load feature list saved during training
    features_path = config.MODEL_DIR / "features_list.json"
    if features_path.exists():
        print("Loading feature list from training...")
        with open(features_path, "r") as f:
            features = json.load(f)
        print(f"Loaded {len(features)} features from training")
    else:
        # Fallback: use same logic as training
        print("Warning: Feature list not found, using fallback logic")
        exclude_cols = [
            constants.COL_SOURCE,
            config.TARGET,
            constants.COL_PREDICTION,
            constants.COL_TIMESTAMP,
        ]
        train_features = [col for col in train_df.columns if col not in exclude_cols]
        train_non_feature_object_cols = train_df[train_features].select_dtypes(include=["object"]).columns.tolist()
        features = [f for f in train_features if f not in train_non_feature_object_cols]

    # Remove any columns that shouldn't be features (like title, author_name, etc.)
    exclude_cols = [
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
        constants.COL_USER_ID,
        constants.COL_BOOK_ID,
    ]
    candidates_final = candidates_final.drop(
        columns=[col for col in candidates_final.columns if col not in features and col not in exclude_cols + [constants.COL_USER_ID, constants.COL_BOOK_ID]],
        errors="ignore"
    )

    # Add missing features with default values
    missing_features = [f for f in features if f not in candidates_final.columns]
    if missing_features:
        print(f"Warning: Missing {len(missing_features)} features in candidates, adding defaults")
        for feat in missing_features:
            if feat in train_df.columns:
                if train_df[feat].dtype.name == "category":
                    default_val = train_df[feat].cat.categories[0] if len(train_df[feat].cat.categories) > 0 else 0
                    candidates_final[feat] = pd.Categorical([default_val] * len(candidates_final), categories=train_df[feat].cat.categories, ordered=False)
                else:
                    candidates_final[feat] = train_df[feat].iloc[0] if len(train_df) > 0 else 0
            else:
                candidates_final[feat] = 0

    # Ensure all features exist in candidates_final (add missing ones)
    # features list is from training, so we need all of them
    for feat in features:
        if feat not in candidates_final.columns:
            # Add with default value
            if feat in train_df.columns:
                if train_df[feat].dtype.name == "category":
                    default_val = train_df[feat].cat.categories[0] if len(train_df[feat].cat.categories) > 0 else 0
                    candidates_final[feat] = pd.Categorical([default_val] * len(candidates_final), categories=train_df[feat].cat.categories, ordered=False)
                else:
                    candidates_final[feat] = train_df[feat].iloc[0] if len(train_df) > 0 else 0
            else:
                candidates_final[feat] = 0

    # Final check: ensure we have all features in the exact order
    features = [f for f in features if f in candidates_final.columns]

    # Convert categorical columns to pandas 'category' dtype for CatBoost (same as in train.py)
    # Use categories from featured_df (processed data) to ensure they match training data
    # This is critical: CatBoost requires exact match of categorical features
    # Only process columns that were actually categorical in training data
    for col in features:
        if col in featured_df.columns and featured_df[col].dtype.name == "category":
            # Get categories from processed training data
            train_categories = list(featured_df[col].cat.categories)

            # Convert to string first to handle any type mismatches
            candidates_final[col] = candidates_final[col].astype(str)

            # Replace any values not in train categories with first train category
            valid_mask = candidates_final[col].isin([str(cat) for cat in train_categories])
            if not valid_mask.all():
                invalid_count = (~valid_mask).sum()
                print(f"Warning: {invalid_count} values in {col} not in training categories, replacing with first category")
                candidates_final.loc[~valid_mask, col] = str(train_categories[0]) if len(train_categories) > 0 else "0"

            # Convert categories back to original type and create categorical
            # Convert train_categories to same type as in training
            train_cat_values = [str(cat) for cat in train_categories]
            candidates_final[col] = pd.Categorical(candidates_final[col], categories=train_cat_values, ordered=False)

            # Convert categorical codes back to original type if needed
            # This ensures the internal representation matches training
            if len(train_categories) > 0:
                # Re-map to original category values
                candidates_final[col] = candidates_final[col].astype(str).map(
                    {str(cat): cat for cat in train_categories}
                ).fillna(train_categories[0])
                candidates_final[col] = pd.Categorical(candidates_final[col], categories=train_categories, ordered=False)

    X_test = candidates_final[features].copy().drop(["f_user_book_interaction", "has_read"], axis=1, errors="ignore")
    print(f"Prediction features: {len(X_test.columns)}")  # Updated to use X_test.columns

    # Identify categorical features for CatBoost, similar to training
    categorical_features = [f for f in X_test.columns if X_test[f].dtype.name == "category"]

    # Load trained model
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. " "Please run 'poetry run python -m src.baseline.train' first."
        )

    print(f"\nLoading model from {model_path}...")
    model = CatBoostClassifier().load_model(str(model_path))

    # Generate probabilities for multiclass (3 classes)
    # For multiclass, model.predict_proba() returns probabilities for all classes
    # Shape: (n_samples, 3) with [p0, p1, p2] for each sample
    # p0 = probability of class 0 (cold candidates)
    # p1 = probability of class 1 (planned books)
    # p2 = probability of class 2 (read books)
    print("Generating predictions...")
    test_pool = Pool(
        data=X_test,
        group_id=candidates_final[constants.COL_USER_ID],  # обязательно!
        cat_features=categorical_features
    )
    rank_scores = model.predict(test_pool)  # Returns probabilities for all classes
    # Convert to numpy array if needed and ensure it's 2D array: (n_samples, num_classes)
    # Add predictions to candidates dataframe
    candidates_final["prediction"] = rank_scores

    # Rank candidates for each user and select top-K
    print("\nRanking candidates for each user...")
    submission_rows = []

    for user_id in targets_df[constants.COL_USER_ID]:
        user_candidates = candidates_final[candidates_final[constants.COL_USER_ID] == user_id].copy()

        if len(user_candidates) == 0:
            # No candidates for this user - empty list
            book_id_list = ""
        else:
            # Sort by prediction probability (descending)
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


if __name__ == "__main__":
    predict()
