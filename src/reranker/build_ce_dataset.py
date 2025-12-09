"""
Подготовка датасета для кросс-энкодера (user-book reranker).

Идея:
- История пользователя берется из train_split (до split_date) и агрегируется в
  компактный текстовый профиль (последние N взаимодействий).
- Пары (user, book) для обучения:
    * Позитивы: прочитанные (relevance=2) и запланированные (relevance=1)
      из train_split.
    * Негативы: синтетические "холодные" книги (relevance=0), которых
      пользователь не видел до split_date.
- Валидация: записи из val_split, профиль построен только из train_split
  (корректное временное разделение).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from src.baseline import config, constants
from src.baseline.temporal_split import get_split_date_from_ratio, temporal_split_by_date

MAX_HISTORY = 12  # сколько последних взаимодействий включаем в профиль
NEG_PER_USER = 2  # сколько холодных книг добавляем на пользователя
RANDOM_STATE = 56


def _read_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Читает raw-данные с метаданными без тяжелых фич."""
    train_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.TRAIN_FILENAME,
        dtype={constants.COL_USER_ID: "int32", constants.COL_BOOK_ID: "int32", constants.COL_HAS_READ: "int8"},
        parse_dates=[constants.COL_TIMESTAMP],
    )
    train_df[constants.COL_RELEVANCE] = train_df[constants.COL_HAS_READ].map({1: 2, 0: 1}).astype("int8")

    targets_df = pd.read_csv(config.RAW_DATA_DIR / constants.TARGETS_FILENAME, dtype={constants.COL_USER_ID: "int32"})
    candidates_df = pd.read_csv(config.RAW_DATA_DIR / constants.CANDIDATES_FILENAME, dtype={constants.COL_USER_ID: "int32"})

    books_df = pd.read_csv(config.RAW_DATA_DIR / constants.BOOK_DATA_FILENAME)
    books_df = books_df.drop_duplicates(subset=[constants.COL_BOOK_ID]).set_index(constants.COL_BOOK_ID)

    book_genres_df = pd.read_csv(config.RAW_DATA_DIR / constants.BOOK_GENRES_FILENAME)
    genres_df = pd.read_csv(config.RAW_DATA_DIR / constants.GENRES_FILENAME)
    genres_map = dict(zip(genres_df["genre_id"], genres_df["genre_name"]))

    descriptions_df = pd.read_csv(config.RAW_DATA_DIR / constants.BOOK_DESCRIPTIONS_FILENAME)
    desc_map = dict(zip(descriptions_df[constants.COL_BOOK_ID], descriptions_df[constants.COL_DESCRIPTION]))

    return train_df, targets_df, candidates_df, book_genres_df, books_df, genres_map, desc_map


def _book_genres_lookup(book_genres_df: pd.DataFrame, genres_map: dict[int, str]) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for _, row in book_genres_df.iterrows():
        book_id = int(row[constants.COL_BOOK_ID])
        genre_id = row[constants.COL_GENRE_ID]
        name = genres_map.get(genre_id)
        if name:
            grouped[book_id].append(name)
    return grouped


def _format_book(book_id: int, books_df: pd.DataFrame, genres_lookup: dict[int, list[str]], desc_map: dict[int, str]) -> str:
    """Собирает компактный текст о книге."""
    row = books_df.loc[book_id] if book_id in books_df.index else None
    parts: list[str] = []
    if row is not None:
        title = row.get("title")
        author = row.get("author_name")
        year = row.get(constants.COL_PUBLICATION_YEAR)
        lang = row.get(constants.COL_LANGUAGE)
        publisher = row.get(constants.COL_PUBLISHER)
        avg_rating = row.get(constants.COL_AVG_RATING)
        if pd.notna(title):
            parts.append(f"Название: {title}")
        if pd.notna(author):
            parts.append(f"Автор: {author}")
        if pd.notna(year):
            parts.append(f"Год: {int(year)}")
        if pd.notna(lang):
            parts.append(f"Язык: {lang}")
        if pd.notna(publisher):
            parts.append(f"Издательство: {publisher}")
        if pd.notna(avg_rating):
            parts.append(f"Средняя_оценка: {avg_rating:.2f}")
    genres = genres_lookup.get(book_id, [])
    if genres:
        parts.append("Жанры: " + "; ".join(genres[:5]))
    desc = desc_map.get(book_id)
    if isinstance(desc, str) and desc.strip():
        parts.append("Описание: " + desc.strip())
    return " | ".join(parts) if parts else f"Книга {book_id}"


def _build_user_profiles(history_df: pd.DataFrame, books_df: pd.DataFrame, genres_lookup: dict[int, list[str]], desc_map: dict[int, str]) -> dict[int, str]:
    """Строит текстовый профиль пользователя на основе последних взаимодействий."""
    history_df = history_df.sort_values(constants.COL_TIMESTAMP)
    profiles: dict[int, str] = {}
    for user_id, group in history_df.groupby(constants.COL_USER_ID):
        tail = group.tail(MAX_HISTORY)
        snippets: list[str] = []
        for _, row in tail.iterrows():
            bid = int(row[constants.COL_BOOK_ID])
            rel = int(row[constants.COL_RELEVANCE])
            rel_txt = "прочитал" if rel == 2 else "добавил"
            snippets.append(f"{rel_txt}: " + _format_book(bid, books_df, genres_lookup, desc_map))
        profiles[user_id] = " || ".join(snippets) if snippets else "нет истории"
    return profiles


def _sample_cold_books(all_books: np.ndarray, user_books: set[int], k: int, rng: np.random.Generator) -> Iterable[int]:
    candidates = np.setdiff1d(all_books, list(user_books))
    if len(candidates) == 0 or k <= 0:
        return []
    k = min(k, len(candidates))
    return rng.choice(candidates, size=k, replace=False)


def _make_pairs(
    split_df: pd.DataFrame,
    profiles: dict[int, str],
    books_df: pd.DataFrame,
    genres_lookup: dict[int, list[str]],
    desc_map: dict[int, str],
    all_books: np.ndarray,
    add_cold: bool,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    user_seen = {uid: set(grp[constants.COL_BOOK_ID].unique()) for uid, grp in split_df.groupby(constants.COL_USER_ID)}
    for _, row in split_df.iterrows():
        uid = int(row[constants.COL_USER_ID])
        bid = int(row[constants.COL_BOOK_ID])
        rel = int(row[constants.COL_RELEVANCE])
        rows.append(
            {
                constants.COL_USER_ID: uid,
                constants.COL_BOOK_ID: bid,
                "label": rel,
                "user_profile": profiles.get(uid, "нет истории"),
                "book_text": _format_book(bid, books_df, genres_lookup, desc_map),
            }
        )
    if add_cold:
        for uid, seen in user_seen.items():
            for neg_bid in _sample_cold_books(all_books, seen, NEG_PER_USER, rng):
                rows.append(
                    {
                        constants.COL_USER_ID: uid,
                        constants.COL_BOOK_ID: int(neg_bid),
                        "label": 0,
                        "user_profile": profiles.get(uid, "нет истории"),
                        "book_text": _format_book(int(neg_bid), books_df, genres_lookup, desc_map),
                    }
                )
    return rows


def build_ce_dataset() -> None:
    """Основной пайплайн подготовки train/val для кросс-энкодера."""
    train_df, _, _, book_genres_df, books_df, genres_map, desc_map = _read_raw()
    print(f"train_df: {len(train_df):,}")

    split_date = get_split_date_from_ratio(train_df, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    train_mask, val_mask = temporal_split_by_date(train_df, split_date, constants.COL_TIMESTAMP)
    train_split = train_df[train_mask].copy()
    val_split = train_df[val_mask].copy()
    print(f"split_date={split_date}, train={len(train_split):,}, val={len(val_split):,}")

    genres_lookup = _book_genres_lookup(book_genres_df, genres_map)
    all_books = train_df[constants.COL_BOOK_ID].unique()
    rng = np.random.default_rng(RANDOM_STATE)

    # Профили строим только по train_split (до split_date)
    profiles = _build_user_profiles(train_split, books_df, genres_lookup, desc_map)

    train_rows = _make_pairs(train_split, profiles, books_df, genres_lookup, desc_map, all_books, add_cold=True, rng=rng)
    val_rows = _make_pairs(val_split, profiles, books_df, genres_lookup, desc_map, all_books, add_cold=False, rng=rng)

    out_dir = config.PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "ce_train.parquet"
    val_path = out_dir / "ce_val.parquet"

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    print(f"Saved CE train: {train_path} ({len(train_rows):,} rows)")
    print(f"Saved CE val:   {val_path} ({len(val_rows):,} rows)")


if __name__ == "__main__":
    build_ce_dataset()


