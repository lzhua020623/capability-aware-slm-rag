"""SQuAD/NQ-style exact match and token-level F1."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(overlap.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def gold_answers_from_record(record: dict) -> list[str]:
    if record.get("gold_answers"):
        return [str(item) for item in record["gold_answers"] if str(item).strip()]
    for key in ("short_answer", "gold_answer"):
        value = record.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value]
    return []


def best_em_f1(prediction: str, golds: list[str]) -> tuple[int, float]:
    if not golds:
        return 0, 0.0
    return max(exact_match(prediction, gold) for gold in golds), max(
        token_f1(prediction, gold) for gold in golds
    )
