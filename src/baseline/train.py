import json
from pathlib import Path
import gc
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from . import config, constants
from .features import (
    add_aggregate_features, add_temporal_features,
    add_nomic_profile_features, add_conversion_features, handle_missing_values
)
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date

BOOK_GENRES_PATH = config.RAW_DATA_DIR / "book_genres.csv"          # или как у вас называется файл
BOOK_DESCRIPTIONS_PATH = config.RAW_DATA_DIR / "book_descriptions.csv"

def train_2stage() -> None:
    """
    Обучает двухэтапную модель ранжирования:
        Stage 1 – CatBoostRanker (YetiRank)
        Stage 2 – LightGBM LambdaRank с предсказаниями Stage 1 в качестве дополнительной фичи
    Все признаки генерируются через единый пайплайн create_enhanced_features().
    """
    from .features import create_enhanced_features
    from .evaluate import ndcg_at_k
    from .temporal_split import get_split_date_from_ratio, temporal_split_by_date

    # ------------------------------------------------------------------
    # 1. Загрузка обработанных данных
    # ------------------------------------------------------------------
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    df_full = pd.read_parquet(processed_path, engine="pyarrow")

    # Оставляем только тренировочную часть (источник = train)
    train_raw = df_full[df_full[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # ------------------------------------------------------------------
    # 2. Временной сплит (без утечек)
    # ------------------------------------------------------------------
    split_date = get_split_date_from_ratio(train_raw, config.TEMPORAL_SPLIT_RATIO)
    train_mask, val_mask = temporal_split_by_date(train_raw, split_date)

    train_df_raw = train_raw[train_mask].copy()
    val_df_raw   = train_raw[val_mask].copy()

    print(f"Temporal split: train {len(train_df_raw):,} rows, val {len(val_df_raw):,} rows "
          f"(split date {split_date.date()})")

    # ------------------------------------------------------------------
    # 3. Единая генерация признаков (включает все функции из features.py)
    # ------------------------------------------------------------------
    print("=== Feature engineering (enhanced pipeline) ===")
    # Для валидации используем только тренировочные данные как базу статистики
    df_with_features = create_enhanced_features(
        df=pd.concat([train_df_raw, val_df_raw], ignore_index=True),
        book_genres_df=pd.read_csv(BOOK_GENRES_PATH),      # предполагается, что путь задан в config
        descriptions_df=pd.read_csv(BOOK_DESCRIPTIONS_PATH),
        include_aggregates=True,
        include_bert=False,       # BERT слишком тяжёлый, оставляем только Nomic
        include_nomic=True,
    )

    # Разделяем обратно на train / val после генерации признаков
    train_df = df_with_features.iloc[:len(train_df_raw)].copy()
    val_df   = df_with_features.iloc[len(train_df_raw):].reset_index(drop=True)

    del df_with_features, train_df_raw, val_df_raw, train_raw, df_full
    gc.collect()

    # ------------------------------------------------------------------
    # 4. Подготовка признаков и меток
    # ------------------------------------------------------------------
    exclude_cols = {
        constants.COL_USER_ID, constants.COL_BOOK_ID, constants.COL_SOURCE,
        constants.COL_TIMESTAMP, constants.COL_HAS_READ, constants.COL_RELEVANCE,
        'prediction', 'group_id'
    }

    features = [c for c in train_df.columns if c not in exclude_cols and
                train_df[c].dtype.name in {'int8', 'int16', 'int32', 'int64',
                                           'uint8', 'uint16', 'uint32', 'uint64',
                                           'float16', 'float32', 'float64', 'bool', 'category'}]

    print(f"Selected {len(features)} features for modelling")

    # Приведение категориальных колонок к типу category (CatBoost требует)
    for col in features:
        if train_df[col].dtype.name == "object":
            train_df[col] = train_df[col].astype("category")
            val_df[col]   = val_df[col].astype("category")

    # Сортировка по user_id – обязательно для обоих ранкеров
    train_df = train_df.sort_values(constants.COL_USER_ID).reset_index(drop=True)
    val_df   = val_df.sort_values(constants.COL_USER_ID).reset_index(drop=True)

    X_train = train_df[features]
    X_val   = val_df[features]
    y_train = train_df[constants.COL_RELEVANCE].values
    y_val   = val_df[constants.COL_RELEVANCE].values

    group_train = train_df.groupby(constants.COL_USER_ID, sort=False).size().values
    group_val   = val_df.groupby(constants.COL_USER_ID, sort=False).size().values

    cat_indices = [i for i, col in enumerate(features) if X_train[col].dtype.name == "category"]
    print(f"Categorical feature indices ({len(cat_indices)}): {cat_indices}")

    # ------------------------------------------------------------------
    # 5. Stage 1 – CatBoostRanker
    # ------------------------------------------------------------------
    print("\n=== Training Stage 1: CatBoostRanker (YetiRank) ===")
    X_train_cb = X_train.copy()
    X_val_cb   = X_val.copy()

    for idx in cat_indices:
        col = features[idx]
        X_train_cb[col] = X_train_cb[col].astype(str).fillna("missing")
        X_val_cb[col]   = X_val_cb[col].astype(str).fillna("missing")

    train_pool = Pool(data=X_train_cb, label=y_train,
                      group_id=train_df[constants.COL_USER_ID].values,
                      cat_features=cat_indices)
    val_pool   = Pool(data=X_val_cb, label=y_val,
                      group_id=val_df[constants.COL_USER_ID].values,
                      cat_features=cat_indices)

    cb_model = CatBoostRanker(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=10,
        loss_function="YetiRank",
        eval_metric="NDCG:top=20",
        random_seed=42,
        verbose=200,
        early_stopping_rounds=150,
        #task_type="GPU" if config.USE_GPU else "CPU",
        #devices="0" if config.USE_GPU else None,
    )

    cb_model.fit(train_pool, eval_set=val_pool)
    cb_model.save_model(str(config.MODEL_DIR / "stage1_catboost.cbm"))
    print("Stage 1 model saved")

    score1_train = cb_model.predict(train_pool)
    score1_val   = cb_model.predict(val_pool)

    # ------------------------------------------------------------------
    # 6. Stage 2 – LightGBM LambdaRank + score_stage1
    # ------------------------------------------------------------------
    print("\n=== Training Stage 2: LightGBM LambdaRank ===")
    X_train2 = X_train.copy()
    X_val2   = X_val.copy()
    X_train2["score_stage1"] = score1_train
    X_val2["score_stage1"]   = score1_val

    features2 = features + ["score_stage1"]

    # Приведение всех колонок к float32 (LightGBM требует числовые признаки)
    X_train2 = X_train2.astype("float32")
    X_val2   = X_val2.astype("float32")

    lgb_train = lgb.Dataset(X_train2, label=y_train.astype(int), group=group_train)
    lgb_val   = lgb.Dataset(X_val2,   label=y_val.astype(int),   group=group_val, reference=lgb_train)

    lgb_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_at": [10, 20],
        "learning_rate": 0.02,
        "num_leaves": 128,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l2": 10,
        "verbose": -1,
        "seed": 42,
        #"device": "gpu" if config.USE_GPU else "cpu",
    }

    lgb_model = lgb.train(
        params=lgb_params,
        train_set=lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=True)],
    )

    lgb_model.save_model(str(config.MODEL_DIR / "stage2_lightgbm.txt"))
    print("Stage 2 model saved")

    # ------------------------------------------------------------------
    # 7. Финальная валидация NDCG@20
    # ------------------------------------------------------------------
    final_pred_val = lgb_model.predict(X_val2)

    ndcgs = []
    for user_id, grp in val_df.groupby(constants.COL_USER_ID, sort=False):
        idx = grp.index
        scores = final_pred_val[val_df.index.get_indexer(idx)]
        relevances = grp[constants.COL_RELEVANCE].values
        top_k_rel = sorted(zip(scores, relevances), reverse=True)[:20]
        ndcgs.append(ndcg_at_k([r for _, r in top_k_rel], k=20))

    final_ndcg = np.mean(ndcgs)
    print(f"\nFinal Validation NDCG@20 = {final_ndcg:.6f}")

    # Сохраняем список использованных признаков (важно для инференса)
    with open(config.MODEL_DIR / "features_list.json", "w") as f:
        json.dump(features2, f, indent=2)

    print("Two-stage training completed successfully!")


# Запуск
if __name__ == "__main__":
    train_2stage()
