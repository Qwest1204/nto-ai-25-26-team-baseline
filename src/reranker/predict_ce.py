"""
Инференс кросс-энкодера для переранкинга кандидатов.

Шаги:
- Строим профиль пользователя из всей истории train.csv (последние MAX_HISTORY).
- Формируем пары (user, candidate book) из candidates.csv.
- Считаем score = p1*1 + p2*2 и ранжируем до 20.
- Сохраняем submission_ce.csv.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.baseline import config, constants
from src.baseline.data_processing import expand_candidates

MAX_HISTORY = 12
MAX_LEN = 320
BATCH_SIZE = 8
MODEL_OUT_DIR = config.MODEL_DIR / "ce_reranker"
MAX_RANK_K = getattr(constants, "MAX_RANKING_LENGTH", 20)
# TODO: добавить взвешенный фьюжн с LGBM/CatBoost при наличии обоих скоринг-потоков


class CeInferDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]


def _read_raw():
    train_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.TRAIN_FILENAME,
        dtype={constants.COL_USER_ID: "int32", constants.COL_BOOK_ID: "int32", constants.COL_HAS_READ: "int8"},
        parse_dates=[constants.COL_TIMESTAMP],
    )
    train_df[constants.COL_RELEVANCE] = train_df[constants.COL_HAS_READ].map({1: 2, 0: 1}).astype("int8")
    candidates_df = pd.read_csv(config.RAW_DATA_DIR / constants.CANDIDATES_FILENAME, dtype={constants.COL_USER_ID: "int32"})
    targets_df = pd.read_csv(config.RAW_DATA_DIR / constants.TARGETS_FILENAME, dtype={constants.COL_USER_ID: "int32"})
    books_df = pd.read_csv(config.RAW_DATA_DIR / constants.BOOK_DATA_FILENAME)
    books_df = books_df.drop_duplicates(subset=[constants.COL_BOOK_ID]).set_index(constants.COL_BOOK_ID)
    book_genres_df = pd.read_csv(config.RAW_DATA_DIR / constants.BOOK_GENRES_FILENAME)
    genres_df = pd.read_csv(config.RAW_DATA_DIR / constants.GENRES_FILENAME)
    genres_map = dict(zip(genres_df["genre_id"], genres_df["genre_name"]))
    descriptions_df = pd.read_csv(config.RAW_DATA_DIR / constants.BOOK_DESCRIPTIONS_FILENAME)
    desc_map = dict(zip(descriptions_df[constants.COL_BOOK_ID], descriptions_df[constants.COL_DESCRIPTION]))
    return train_df, candidates_df, targets_df, book_genres_df, books_df, genres_map, desc_map


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


def _collate(batch, tokenizer):
    texts = batch["texts"]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc["user_ids"] = torch.tensor(batch["user_ids"], dtype=torch.long)
    enc["book_ids"] = torch.tensor(batch["book_ids"], dtype=torch.long)
    return enc


def _ranking_scores(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    p1 = probs[:, 1]
    p2 = probs[:, 2]
    return p1 * 1.0 + p2 * 2.0


def predict_ce() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not MODEL_OUT_DIR.exists():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_OUT_DIR}. Сначала обучите train_ce.py")

    train_df, candidates_df, targets_df, book_genres_df, books_df, genres_map, desc_map = _read_raw()
    genres_lookup = _book_genres_lookup(book_genres_df, genres_map)
    profiles = _build_user_profiles(train_df, books_df, genres_lookup, desc_map)

    expanded_candidates = expand_candidates(candidates_df)
    expanded_candidates = expanded_candidates.merge(targets_df, on=constants.COL_USER_ID, how="inner")

    rows = []
    for _, row in expanded_candidates.iterrows():
        uid = int(row[constants.COL_USER_ID])
        bid = int(row[constants.COL_BOOK_ID])
        text = f"User: {profiles.get(uid, 'нет истории')} [SEP] Book: {_format_book(bid, books_df, genres_lookup, desc_map)}"
        rows.append({"user_id": uid, "book_id": bid, "text": text})

    dataset = CeInferDataset(rows)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_OUT_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_OUT_DIR)
    model.to(device)
    model.eval()

    def collate_fn(batch_list):
        texts = [b["text"] for b in batch_list]
        user_ids = [b["user_id"] for b in batch_list]
        book_ids = [b["book_id"] for b in batch_list]
        return _collate({"texts": texts, "user_ids": user_ids, "book_ids": book_ids}, tokenizer)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    user_scores = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            user_ids = batch.pop("user_ids")
            book_ids = batch.pop("book_ids")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            scores = _ranking_scores(outputs.logits).cpu().numpy()
            for uid, bid, sc in zip(user_ids.numpy(), book_ids.numpy(), scores):
                user_scores[uid].append((sc, bid))

    submission_rows = []
    for uid in targets_df[constants.COL_USER_ID]:
        candidates = user_scores.get(uid, [])
        ranked = sorted(candidates, key=lambda x: x[0], reverse=True)
        top = ranked[: min(MAX_RANK_K, len(ranked))]
        book_id_list = ",".join(str(int(bid)) for _, bid in top)
        submission_rows.append({constants.COL_USER_ID: uid, constants.COL_BOOK_ID_LIST: book_id_list})

    submission_df = pd.DataFrame(submission_rows)
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.SUBMISSION_DIR / "submission_ce.csv"
    submission_df.to_csv(out_path, index=False)
    print(f"Saved CE submission → {out_path} shape={submission_df.shape}")


if __name__ == "__main__":
    predict_ce()


