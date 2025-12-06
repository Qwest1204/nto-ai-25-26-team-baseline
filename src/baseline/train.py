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

    # Сортируем данные по user_id для CatBoost Ranker
    print("Sorting data by user_id for CatBoost Ranker...")
    train_df = train_df.sort_values(by=constants.COL_USER_ID).reset_index(drop=True)
    val_df = val_df.sort_values(by=constants.COL_USER_ID).reset_index(drop=True)

    X_train = train_df[features]
    X_val = val_df[features]
    y_train = train_df[constants.COL_RELEVANCE]
    y_val = val_df[constants.COL_RELEVANCE]

    # Для CatBoost Ranker нам нужен массив group_id, где для каждой строки указан group_id
    train_group_ids = train_df[constants.COL_USER_ID].values
    val_group_ids = val_df[constants.COL_USER_ID].values

    cat_features = [i for i, c in enumerate(features) if X_train[c].dtype.name == 'category']

    print(f"Number of categorical features: {len(cat_features)}")
    print(f"Categorical feature indices: {cat_features}")
    print(f"Categorical feature names: {[features[i] for i in cat_features]}")

    # Проверим размеры групп
    print(f"Train groups: {len(np.unique(train_group_ids))} users, {len(train_df)} rows")
    print(f"Val groups: {len(np.unique(val_group_ids))} users, {len(val_df)} rows")

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

    # Создаем Pool для CatBoost Ranker
    print("Creating CatBoost Pool...")

    try:
        cb_pool_train = Pool(
            data=X_train_cb,
            label=y_train,
            group_id=train_group_ids,
            cat_features=cat_features
        )

        cb_pool_val = Pool(
            data=X_val_cb,
            label=y_val,
            group_id=val_group_ids,
            cat_features=cat_features
        )
    except Exception as e:
        print(f"Error creating CatBoost Pool: {e}")
        raise

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

    print("Fitting CatBoost Ranker...")
    cb_model.fit(cb_pool_train, eval_set=cb_pool_val, early_stopping_rounds=100)

    score1_train = cb_model.predict(cb_pool_train)
    score1_val = cb_model.predict(cb_pool_val)

    # Сохраняем Stage 1
    cb_model.save_model(str(config.MODEL_DIR / "stage1_catboost.cbm"))
    print("Stage 1 model saved.")

    # === 5. Stage 2: LightGBM с фичей из Stage 1 ===
    print("Training Stage 2: LightGBM with Stage 1 predictions as feature...")

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

    # Для LightGBM LambdaRank метки должны быть целыми числами
    # У нас метки relevance: 0, 1, 2 - это уже целые числа
    lgb_y_train = y_train.values
    lgb_y_val = y_val.values

    # Для LightGBM нам также нужны группы
    # LightGBM использует параметр group, который содержит размеры групп
    train_group_sizes = train_df.groupby(constants.COL_USER_ID).size().values
    val_group_sizes = val_df.groupby(constants.COL_USER_ID).size().values

    # Проверяем типы данных
    print(f"X_train2 dtypes: {X_train2.dtypes.unique()}")
    print(f"y_train dtype: {lgb_y_train.dtype}")

    # Убедимся, что все данные имеют правильный тип
    X_train2 = X_train2.astype(float)
    X_val2 = X_val2.astype(float)
    lgb_y_train = lgb_y_train.astype(int)
    lgb_y_val = lgb_y_val.astype(int)

    lgb_train = lgb.Dataset(
        X_train2,
        label=lgb_y_train,
        group=train_group_sizes
    )
    lgb_val = lgb.Dataset(
        X_val2,
        label=lgb_y_val,
        group=val_group_sizes,
        reference=lgb_train
    )

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

    print("Training LightGBM model...")
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(150)],
        #verbose_eval=100,
    )

    lgb_model.save_model(str(config.MODEL_DIR / "stage2_lightgbm.txt"))
    print("Stage 2 model saved.")

    # === 6. Финальный скор ===
    final_pred_val = lgb_model.predict(X_val2)

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
