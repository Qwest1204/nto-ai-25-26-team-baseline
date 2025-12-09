"""
Обучение кросс-энкодера (user-book reranker) на данных ce_train/ce_val.

Текстовая формулировка для токенизатора:
    "User: {user_profile} [SEP] Book: {book_text}"
Метка: 0/1/2 (cold / planned / read).

Сохранение модели: output/models/ce_reranker
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.baseline import config, constants
from src.baseline.evaluate import ndcg_at_k

MODEL_NAME = constants.BERT_MODEL_NAME
MAX_LEN = 320
BATCH_SIZE = 8  # ускоряем обучение (при OOM уменьшите)
EPOCHS = 1
LR = 3e-5
GRAD_ACCUM = 2
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
MAX_RANK_K = getattr(constants, "MAX_RANKING_LENGTH", 20)
MODEL_OUT_DIR = config.MODEL_DIR / "ce_reranker"
EVAL_EVERY = 10000  # батчей между промежуточными оценками
SMOOTH_WINDOW = 500  # окно скользящего среднего лосса


@dataclass
class CeSample:
    user_id: int
    book_id: int
    text: str
    label: int


class CeDataset(Dataset):
    def __init__(self, rows: Iterable[CeSample]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> CeSample:
        return self.rows[idx]


def _load_ce_split(path: Path) -> list[CeSample]:
    df = pd.read_parquet(path)
    samples: list[CeSample] = []
    for _, row in df.iterrows():
        text = f"User: {row['user_profile']} [SEP] Book: {row['book_text']}"
        samples.append(
            CeSample(
                user_id=int(row[constants.COL_USER_ID]),
                book_id=int(row[constants.COL_BOOK_ID]),
                text=text,
                label=int(row["label"]),
            )
        )
    return samples


def _collate(batch: list[CeSample], tokenizer: AutoTokenizer):
    texts = [b.text for b in batch]
    labels = torch.tensor([b.label for b in batch], dtype=torch.long)
    user_ids = torch.tensor([b.user_id for b in batch], dtype=torch.long)
    book_ids = torch.tensor([b.book_id for b in batch], dtype=torch.long)

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc["labels"] = labels
    enc["user_ids"] = user_ids
    enc["book_ids"] = book_ids
    return enc


def _to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _ranking_scores(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    p1 = probs[:, 1]
    p2 = probs[:, 2]
    return p1 * 1.0 + p2 * 2.0


def evaluate(model, dataloader, device) -> float:
    model.eval()
    user_scores = {}
    user_labels = {}
    with torch.no_grad():
        for batch in dataloader:
            batch = _to_device(batch, device)
            labels = batch.pop("labels")
            user_ids = batch.pop("user_ids").cpu().numpy()
            book_ids = batch.pop("book_ids").cpu().numpy()

            outputs = model(**batch)
            logits = outputs.logits
            scores = _ranking_scores(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for uid, bid, sc, lb in zip(user_ids, book_ids, scores, labels_np):
                user_scores.setdefault(uid, []).append((sc, lb))
                user_labels.setdefault(uid, []).append(lb)

    # NDCG@K по пользователям
    ndcgs = []
    for uid, items in user_scores.items():
        ranked = sorted(items, key=lambda x: x[0], reverse=True)
        rels = [lb for _, lb in ranked]
        ndcgs.append(ndcg_at_k(relevance_scores=rels, k=MAX_RANK_K))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def train_ce() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = config.PROCESSED_DATA_DIR / "ce_train.parquet"
    val_path = config.PROCESSED_DATA_DIR / "ce_val.parquet"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("CE датасеты не найдены. Сначала запустите build_ce_dataset.py")

    train_samples = _load_ce_split(train_path)
    val_samples = _load_ce_split(val_path)
    print(f"Train samples: {len(train_samples):,}, Val samples: {len(val_samples):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device)

    train_loader = DataLoader(
        CeDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: _collate(b, tokenizer),
    )
    val_loader = DataLoader(
        CeDataset(val_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: _collate(b, tokenizer),
    )

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader) * EPOCHS) // GRAD_ACCUM
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    ce_loss_fn = nn.CrossEntropyLoss()

    global_step = 0
    best_ndcg = -1.0
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    loss_window = deque(maxlen=SMOOTH_WINDOW)

    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(train_loader):
            batch = _to_device(batch, device)
            labels = batch.pop("labels")
            user_ids = batch.pop("user_ids")
            book_ids = batch.pop("book_ids")

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(**batch)
                loss = ce_loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            loss_window.append(loss.item())

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            if (step + 1) % 200 == 0:
                avg_loss = sum(loss_window) / len(loss_window) if loss_window else loss.item()
                print(f"Epoch {epoch+1} step {step+1}: loss={loss.item():.4f} (avg{SMOOTH_WINDOW}={avg_loss:.4f})")

            if (step + 1) % EVAL_EVERY == 0:
                model.eval()
                val_ndcg = evaluate(model, val_loader, device)
                print(f"[INTERIM] Epoch {epoch+1} step {step+1}: val NDCG@{MAX_RANK_K}={val_ndcg:.4f}")
                if val_ndcg > best_ndcg:
                    best_ndcg = val_ndcg
                    model.save_pretrained(MODEL_OUT_DIR)
                    tokenizer.save_pretrained(MODEL_OUT_DIR)
                    print(f"[INTERIM] Saved best model → {MODEL_OUT_DIR}")
                model.train()

        val_ndcg = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: val NDCG@{MAX_RANK_K}={val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            model.save_pretrained(MODEL_OUT_DIR)
            tokenizer.save_pretrained(MODEL_OUT_DIR)
            print(f"Saved best model → {MODEL_OUT_DIR}")

    print(f"Training finished. Best val NDCG@{MAX_RANK_K}={best_ndcg:.4f}")


if __name__ == "__main__":
    train_ce()


