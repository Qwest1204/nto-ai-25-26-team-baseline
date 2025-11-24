"""
Data preparation script that processes raw data and saves it to processed directory.

This script loads raw data, performs feature engineering (without aggregates),
and saves the processed data to data/processed/ for use in training and prediction.
"""

from . import config, constants
from .data_processing import load_and_merge_data
from .features import create_features


def prepare_data() -> None:
    """Processes raw data and saves prepared features to processed directory.

    This function:
    1. Loads raw data from data/raw/
    2. Uses all records from train.csv (both has_read=0 and has_read=1)
    3. Applies feature engineering (genres, TF-IDF, BERT) - NO aggregates to avoid data leakage
    4. Saves processed data to data/processed/processed_features.parquet
    5. Preserves timestamp for temporal splitting

    Note: Aggregate features are computed separately during training to ensure
    temporal correctness (no data leakage from validation set).

    The processed data can then be used by train.py and predict.py without
    re-running the expensive feature engineering steps.
    """
    print("=" * 60)
    print("Data Preparation Pipeline")
    print("=" * 60)

    # Load and merge raw data
    merged_df, targets_df, candidates_df, book_genres_df, descriptions_df = load_and_merge_data()

    # Apply feature engineering WITHOUT aggregates
    # Aggregates will be computed during training on train split only
    # BERT disabled for faster testing (can be enabled by setting include_bert=True)
    featured_df = create_features(merged_df, book_genres_df, descriptions_df, include_aggregates=False, include_bert=False, include_nomic=True)

    # Ensure processed directory exists
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Define the output path
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    # Save processed data as parquet for efficiency
    print(f"\nSaving processed data to {processed_path}...")
    featured_df.to_parquet(processed_path, index=False, engine="pyarrow", compression="snappy")
    print("Processed data saved successfully!")

    # Print statistics
    train_rows = len(featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN])
    total_features = len(featured_df.columns)

    print("\nData preparation complete!")
    print(f"  - Train rows: {train_rows:,}")
    print(f"  - Total features: {total_features}")
    print(f"  - Output file: {processed_path}")


if __name__ == "__main__":
    prepare_data()

