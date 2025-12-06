import json
from pathlib import Path
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


def train_2stage():
    # === 1. Загрузка данных (как раньше) ===
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    df = pd.read_parquet(processed_path, engine="pyarrow")
    train_set = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Temporal split
    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO)
    train_mask, val_mask = temporal_split_by_date(train_set, split_date)

    train_split = train_set[train_mask].copy()
    val_split = train_set[val_mask].copy()

    # === 2. Полный feature engineering ===
    for split, base in [(train_split, train_split), (val_split, train_split)]:
        split = add_aggregate_features(split, base)
        split = add_temporal_features(split, base)
        split = add_nomic_profile_features(split, base)
        split = add_conversion_features(split, base)
        split = handle_missing_values(split, base)

    # Финальные сеты
    train_df = train_split
    val_df = val_split

    # === 3. Подготовка фич ===
    exclude_cols = [constants.COL_USER_ID, constants.COL_BOOK_ID, constants.COL_SOURCE,
                    constants.COL_TIMESTAMP, constants.COL_HAS_READ, constants.COL_RELEVANCE,
                    'prediction', 'group_id']

    # Сначала получим все колонки, кроме исключенных
    all_features = [c for c in train_df.columns if c not in exclude_cols]

    # Отфильтруем только числовые и категориальные колонки
    features = []
    for c in all_features:
        dtype = train_df[c].dtype.name
        # Включаем числовые типы и категории
        if dtype in ['int8', 'int16', 'int32', 'int64',
                     'float16', 'float32', 'float64', 'bool',
                     'category']:
            features.append(c)
        else:
            print(f"Warning: Excluding column {c} with dtype {dtype} from features")

    print(f"Selected {len(features)} features")

    # Убедимся, что все категориальные колонки имеют тип 'category'
    for col in features:
        if train_df[col].dtype.name == 'object':
            print(f"Converting {col} from object to category")
            # Преобразуем object в category
            train_df[col] = train_df[col].astype('category')
            val_df[col] = val_df[col].astype('category')

    X_train = train_df[features]
    X_val = val_df[features]
    y_train = train_df[constants.COL_RELEVANCE]
    y_val = val_df[constants.COL_RELEVANCE]

    # Проверим, что нет строковых значений в данных
    print("Checking for non-numeric values in features...")
    for col in features:
        if train_df[col].dtype.name not in ['category', 'bool']:
            # Проверим на наличие нечисловых значений
            try:
                # Пробуем преобразовать в float
                _ = train_df[col].astype(float)
            except Exception as e:
                print(f"Error in column {col}: {e}")
                print(f"Unique values in {col}: {train_df[col].unique()[:10]}")

    # Преобразуем bool в int для CatBoost
    for col in features:
        if train_df[col].dtype.name == 'bool':
            train_df[col] = train_df[col].astype(int)
            val_df[col] = val_df[col].astype(int)

    # Обновляем X_train и X_val после преобразований
    X_train = train_df[features]
    X_val = val_df[features]

    groups_train = train_df.groupby(constants.COL_USER_ID).size().values
    groups_val = val_df.groupby(constants.COL_USER_ID).size().values

    cat_features = [i for i, c in enumerate(features) if X_train[c].dtype.name == 'category']

    print(f"Number of categorical features: {len(cat_features)}")
    print(f"Categorical feature indices: {cat_features}")

    # === 4. Stage 1: CatBoostRanker ===
    print("Training Stage 1: CatBoostRanker...")

    # Создадим копии данных для CatBoost
    X_train_cb = X_train.copy()
    X_val_cb = X_val.copy()

    # Преобразуем категориальные колонки в строки для CatBoost
    for idx in cat_features:
        col = features[idx]
        X_train_cb[col] = X_train_cb[col].astype(str).fillna('nan')
        X_val_cb[col] = X_val_cb[col].astype(str).fillna('nan')

    cb_pool_train = Pool(X_train_cb, y_train, group_id=train_df[constants.COL_USER_ID], cat_features=cat_features)
    cb_pool_val = Pool(X_val_cb, y_val, group_id=val_df[constants.COL_USER_ID], cat_features=cat_features)

    cb_model = CatBoostRanker(
        iterations=2500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=10,
        loss_function='YetiRank',
        eval_metric='NDCG:top=20',
        random_seed=42,
        verbose=200,
    )

    try:
        cb_model.fit(cb_pool_train, eval_set=cb_pool_val, early_stopping_rounds=100)
    except Exception as e:
        print(f"Error fitting CatBoost: {e}")
        # Выведем больше информации о данных
        print("Data types in X_train_cb:")
        print(X_train_cb.dtypes.value_counts())
        print("\nSample of X_train_cb:")
        print(X_train_cb.head())
        raise

    score1_train = cb_model.predict(cb_pool_train)
    score1_val = cb_model.predict(cb_pool_val)

    # Сохраняем Stage 1
    cb_model.save_model(str(config.MODEL_DIR / "stage1_catboost.cbm"))

    # === 5. Stage 2: LightGBM на остатках ===
    print("Training Stage 2: LightGBM on residuals...")

    # Остатки: relevance - λ × score1
    lambda_residual = 0.8
    residual_train = y_train.values - lambda_residual * score1_train
    residual_val = y_val.values - lambda_residual * score1_val

    # Добавляем score1 как главную фичу
    X_train2 = X_train.copy()
    X_val2 = X_val.copy()
    X_train2['score_stage1'] = score1_train
    X_val2['score_stage1'] = score1_val
    features2 = features + ['score_stage1']

    # Для LightGBM преобразуем категории в целые числа
    for col in features2:
        if X_train2[col].dtype.name == 'category':
            X_train2[col] = X_train2[col].cat.codes
            X_val2[col] = X_val2[col].cat.codes

    lgb_train = lgb.Dataset(X_train2, label=residual_train, group=groups_train)
    lgb_val = lgb.Dataset(X_val2, label=residual_val, group=groups_val, reference=lgb_train)

    lgb_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_at': [20],
        'learning_rate': 0.02,
        'num_leaves': 128,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'lambda_l2': 10,
    }

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(150)],
        verbose_eval=100,
    )

    lgb_model.save_model(str(config.MODEL_DIR / "stage2_lightgbm.txt"))

    # === 6. Финальный скор ===
    final_pred_val = score1_val + lgb_model.predict(X_val2)

    # Оценка
    from .evaluate import ndcg_at_k
    ndcgs = []
    for user_id, group in val_df.groupby(constants.COL_USER_ID):
        idx = group.index
        scores = final_pred_val[val_df.index.get_indexer(idx)]
        relevance = group[constants.COL_RELEVANCE].values
        # реальный ndcg
        ranked_rel = [r for _, r in sorted(zip(scores, relevance), reverse=True)][:20]
        ndcgs.append(ndcg_at_k(ranked_rel, k=20))
    print(f"Final Validation NDCG@20: {np.mean(ndcgs):.6f}")

    # Сохраняем список фич
    features_path = config.MODEL_DIR / "features_list.json"
    with open(features_path, "w") as f:
        json.dump(features2, f)
    print(f"Features list saved to {features_path}")

    print("2-stage model trained and saved!")


train_2stage()
