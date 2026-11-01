"""Unified conservative automatic answer scoring for baseline evaluation.

EM / token F1 delegate to ``src.evaluation.qa`` (unchanged regression).
``normalized_match`` allows only answer *form* differences (Unicode, case,
punctuation, articles, whitespace, obvious separators) — never synonyms,
embeddings, fuzzy semantic thresholds, or partial semantic containment.
"""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Any

from src.evaluation.qa import best_em_f1, normalize_answer, token_f1

# Obvious separators → space (hyphen / unicode dash / underscore / slash / dots).
_SEPARATORS = re.compile(
    r"["
    r"\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D"
    r"\u00B7\u2022\u2027\u2219"
    r"_/"
    r"\-"
    r"]+"
)

_SCORE_KEYS = (
    "em",
    "f1",
    "normalized_match",
    "normalized_match_type",
)

_WS_ONLY = re.compile(r"\s+")


def normalize_spacing(text: str) -> str:
    """Leading/trailing/multiple whitespace only (no case/punct changes)."""
    return _WS_ONLY.sub(" ", str(text).strip())


def normalize_standard(text: str) -> str:
    """NFKC + lowercase + punct removal + a/an/the + whitespace (no separator rewrite)."""
    text = unicodedata.normalize("NFKC", str(text))
    return normalize_answer(text)


def normalize_form(text: str) -> str:
    """Full form normalize used for ``normalized_match`` (includes separators)."""
    text = unicodedata.normalize("NFKC", str(text))
    text = _SEPARATORS.sub(" ", text)
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


# Back-compat aliases used by older internal callers/tests.
normalize_strict = normalize_standard


def _as_gold_list(gold: Any) -> list[str]:
    if gold is None:
        return []
    if isinstance(gold, str):
        return [gold] if gold.strip() else []
    if isinstance(gold, list):
        return [str(x) for x in gold if str(x).strip()]
    return [str(gold)] if str(gold).strip() else []


def _match_type_against_one(prediction: str, gold: str) -> str | None:
    """Return the strongest (most specific) form-match type, or None."""
    if prediction == gold:
        return "exact"
    if normalize_spacing(prediction) == normalize_spacing(gold):
        return "spacing_normalization"
    if normalize_standard(prediction) == normalize_standard(gold):
        return "standard_normalization"
    if normalize_form(prediction) == normalize_form(gold):
        return "separator_normalization"
    return None


_TYPE_RANK = {
    "exact": 0,
    "spacing_normalization": 1,
    "standard_normalization": 2,
    "separator_normalization": 3,
}


def _best_match_type(prediction: str, golds: list[str]) -> str:
    best: str | None = None
    for g in golds:
        t = _match_type_against_one(prediction, g)
        if t is None:
            continue
        if best is None or _TYPE_RANK[t] < _TYPE_RANK[best]:
            best = t
    return best if best is not None else "none"


def score_answer(prediction: str, gold: Any) -> dict[str, Any]:
    """Score one prediction against one or more golds (scoring fields only)."""
    golds = _as_gold_list(gold)
    pred = "" if prediction is None else str(prediction)
    em, f1 = best_em_f1(pred, golds) if golds else (0, 0.0)
    match_type = _best_match_type(pred, golds) if golds else "none"
    normalized_ok = int(match_type != "none")
    return {
        "em": int(em),
        "f1": float(f1),
        "normalized_match": normalized_ok,
        "normalized_match_type": match_type,
    }


def answer_in_context(gold: Any, passages: list[dict] | None) -> int:
    """1 if any gold string appears in Top passages (form-normalized containment)."""
    golds = _as_gold_list(gold)
    if not golds or not passages:
        return 0
    blob = "\n".join(str(p.get("text") or "") for p in passages)
    blob_l = blob.lower()
    blob_norm = normalize_form(blob)
    blob_sp = re.sub(r"\s+", " ", blob_l).strip()
    for g in golds:
        g0 = str(g)
        if g0.lower() in blob_l:
            return 1
        gn = normalize_form(g0)
        if gn and gn in blob_norm:
            return 1
        g_sp = re.sub(r"\s+", " ", g0.lower()).strip()
        if g_sp and g_sp in blob_sp:
            return 1
    return 0


def apply_scores_to_row(row: dict) -> dict:
    """Refresh scoring fields; preserve identity fields; drop deprecated relaxed_*."""
    sample_id = row["sample_id"]
    query = row["query"]
    prediction = row["prediction"]
    gold = row["gold"]
    mode = row.get("mode", "")
    scores = score_answer(prediction, gold)
    out = {
        "sample_id": sample_id,
        "query": query,
        "prediction": prediction,
        "gold": gold,
        "mode": mode,
        **scores,
    }
    for key in ("retrieved_passage_ids", "retrieved_ranks", "answer_in_context"):
        if key in row:
            out[key] = row[key]
    return out


def aggregate_answer_metrics(rows: list[dict], *, mode: str) -> dict:
    n = len(rows)
    empty = {
        "n": 0,
        "em": 0.0,
        "f1": 0.0,
        "mean_f1": 0.0,
        "normalized_accuracy": 0.0,
        "normalized_additional_correct": 0,
        "mode": mode,
    }
    if n == 0:
        return empty
    em_sum = f1_sum = norm_sum = 0.0
    norm_extra = 0
    aic_yes = aic_no = 0
    aic_present = 0
    for row in rows:
        if all(k in row for k in _SCORE_KEYS):
            em = int(row["em"])
            f1 = float(row["f1"])
            nm = int(row["normalized_match"])
        else:
            scored = score_answer(row.get("prediction", ""), row.get("gold"))
            em, f1 = scored["em"], scored["f1"]
            nm = scored["normalized_match"]
        em_sum += em
        f1_sum += f1
        norm_sum += nm
        if nm == 1 and em == 0:
            norm_extra += 1
        if "answer_in_context" in row:
            aic_present += 1
            if int(row["answer_in_context"]):
                aic_yes += 1
            else:
                aic_no += 1
    mean_f1 = f1_sum / n
    out = {
        "n": n,
        "em": em_sum / n,
        "f1": mean_f1,
        "mean_f1": mean_f1,
        "normalized_accuracy": norm_sum / n,
        "normalized_additional_correct": norm_extra,
        "mode": mode,
    }
    if aic_present:
        out["answer_in_context_at_5"] = aic_yes / n
        out["answer_in_context_yes_n"] = aic_yes
        out["answer_in_context_no_n"] = aic_no
    return out


__all__ = [
    "normalize_spacing",
    "normalize_standard",
    "normalize_form",
    "normalize_strict",
    "score_answer",
    "answer_in_context",
    "apply_scores_to_row",
    "aggregate_answer_metrics",
    "best_em_f1",
    "token_f1",
]
