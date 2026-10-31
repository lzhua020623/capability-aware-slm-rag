"""FEVER label scoring for baselines (SUPPORTS / REFUTES only).

Primary metrics are label accuracy and macro-F1 — not NQ EM/Token F1.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

LEGAL = ("SUPPORTS", "REFUTES")
_LABEL_TOKEN = re.compile(r"\b(SUPPORTS|REFUTES)\b", re.IGNORECASE)
_EDGE_PUNCT = re.compile(r'^[\s\'"`“”‘’(\[{<]+|[\s\'"`“”‘’)\]}>.,;:!?]+$')


def normalize_fever_prediction(text: str | None) -> str:
    """Map raw generator text → SUPPORTS | REFUTES | INVALID.

    Allows only light cleanup (whitespace / surrounding quotes & punctuation)
    plus a single unambiguous SUPPORTS/REFUTES token. Never defaults to a class.
    """
    if text is None:
        return "INVALID"
    raw = str(text).strip()
    if not raw:
        return "INVALID"
    cleaned = raw
    # Peel surrounding quotes/brackets/punctuation a few times.
    for _ in range(3):
        nxt = _EDGE_PUNCT.sub("", cleaned).strip()
        if nxt == cleaned:
            break
        cleaned = nxt
    if not cleaned:
        return "INVALID"
    upper = cleaned.upper()
    if upper in LEGAL:
        return upper
    matches = [m.upper() for m in _LABEL_TOKEN.findall(cleaned)]
    uniq = list(dict.fromkeys(matches))
    if len(uniq) == 1:
        return uniq[0]
    return "INVALID"


def gold_fever_label(gold: Any) -> str:
    """Extract a single gold label from string or list-shaped gold."""
    if isinstance(gold, list):
        if not gold:
            return "INVALID"
        return gold_fever_label(gold[0])
    label = normalize_fever_prediction(str(gold) if gold is not None else "")
    return label if label in LEGAL else "INVALID"


def score_fever_prediction(prediction: str, gold: Any) -> dict[str, Any]:
    norm = normalize_fever_prediction(prediction)
    gold_label = gold_fever_label(gold)
    invalid = norm == "INVALID"
    correct = (not invalid) and gold_label in LEGAL and norm == gold_label
    return {
        "normalized_prediction": norm,
        "correct": bool(correct),
        "invalid": bool(invalid),
        "gold_label": gold_label,
    }


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def gold_evidence_hit_at_k(
    hits: list[dict] | None,
    gold_page_ids: list[str] | None,
    *,
    k: int = 5,
) -> bool | None:
    """True if any gold page_id equals a retrieved passage_id in Top-k.

    FEVER corpus used here has passage_id == page_id (one passage per page),
    so page-level gold maps reliably to passage_id. Returns None if gold pages
    unavailable (do not invent a hit).
    """
    if not gold_page_ids:
        return None
    hits = hits or []
    retrieved = {str(h.get("passage_id")) for h in hits[:k] if h.get("passage_id")}
    gold = {str(x) for x in gold_page_ids if str(x).strip()}
    if not gold:
        return None
    return bool(retrieved & gold)


def aggregate_fever_metrics(rows: list[dict], *, mode: str = "fever_qwen_only") -> dict:
    n = len(rows)
    if n == 0:
        out = {
            "n": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "supports_precision": 0.0,
            "supports_recall": 0.0,
            "supports_f1": 0.0,
            "refutes_precision": 0.0,
            "refutes_recall": 0.0,
            "refutes_f1": 0.0,
            "invalid_n": 0,
            "mode": mode,
            "dataset": "fever",
        }
        if "rag" in mode:
            out["evidence_hit_at_5"] = None
            out["evidence_hit_at_5_n"] = 0
            out["evidence_hit_at_5_scored_n"] = 0
        return out

    correct_n = 0
    invalid_n = 0
    # confusion vs gold SUPPORTS/REFUTES; INVALID preds count as neither class
    tp = Counter()
    fp = Counter()
    fn = Counter()
    hit_yes = 0
    hit_scored = 0
    for row in rows:
        gold = gold_fever_label(row.get("gold_label") or row.get("gold"))
        if "normalized_prediction" in row:
            pred = str(row["normalized_prediction"])
            if pred not in LEGAL and pred != "INVALID":
                pred = normalize_fever_prediction(pred)
        else:
            pred = normalize_fever_prediction(row.get("prediction"))
        if pred == "INVALID":
            invalid_n += 1
            if gold in LEGAL:
                fn[gold] += 1
        elif gold in LEGAL:
            if pred == gold:
                correct_n += 1
                tp[gold] += 1
            else:
                fp[pred] += 1
                fn[gold] += 1

        if "gold_evidence_hit_at_5" in row and row["gold_evidence_hit_at_5"] is not None:
            hit_scored += 1
            if row["gold_evidence_hit_at_5"]:
                hit_yes += 1

    s_p, s_r, s_f = _prf(tp["SUPPORTS"], fp["SUPPORTS"], fn["SUPPORTS"])
    r_p, r_r, r_f = _prf(tp["REFUTES"], fp["REFUTES"], fn["REFUTES"])
    out = {
        "n": n,
        "accuracy": correct_n / n,
        "macro_f1": 0.5 * (s_f + r_f),
        "supports_precision": s_p,
        "supports_recall": s_r,
        "supports_f1": s_f,
        "refutes_precision": r_p,
        "refutes_recall": r_r,
        "refutes_f1": r_f,
        "invalid_n": invalid_n,
        "mode": mode,
        "dataset": "fever",
    }
    if "rag" in mode or hit_scored:
        out["evidence_hit_at_5"] = (hit_yes / hit_scored) if hit_scored else None
        out["evidence_hit_at_5_n"] = hit_yes
        out["evidence_hit_at_5_scored_n"] = hit_scored
    return out


def build_fever_output_row(
    *,
    sample_id: str,
    claim: str,
    prediction: str,
    gold: str | list[str],
    mode: str = "fever_qwen_only",
    hits: list[dict] | None = None,
    gold_page_ids: list[str] | None = None,
) -> dict:
    scores = score_fever_prediction(prediction, gold)
    gold_label = scores["gold_label"]
    row = {
        "sample_id": str(sample_id),
        "dataset": "fever",
        "mode": mode,
        "claim": claim,
        "query": claim,  # harness compatibility: query == claim
        "prediction": prediction,
        "gold": gold_label if gold_label in LEGAL else gold,
        "normalized_prediction": scores["normalized_prediction"],
        "correct": scores["correct"],
        "invalid": scores["invalid"],
    }
    if hits is not None:
        row["retrieved_passage_ids"] = [h.get("passage_id") for h in hits]
        row["retrieved_ranks"] = [int(h.get("rank", i + 1)) for i, h in enumerate(hits)]
        row["retrieved_scores"] = [float(h.get("score", 0.0)) for h in hits]
        hit = gold_evidence_hit_at_k(hits, gold_page_ids, k=5)
        row["gold_evidence_hit_at_5"] = hit
        if gold_page_ids is not None:
            row["gold_page_ids"] = list(gold_page_ids)
    return row
