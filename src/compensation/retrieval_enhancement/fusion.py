"""Conservative dataset-scoped deduplication and reciprocal rank fusion.

Only canonical Unicode (NFC) and whitespace are normalized for comparison.
Case, punctuation, numbers and negation remain significant. Original text and
the first occurrence's retrieval score are retained. No Gold/target input is
accepted or used. All returned evidence is independent of the caller's data.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Real

from .types import Evidence, FusionResult


def _sequence(value, name: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of evidence lists/items")


def _nonempty_string(value, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be blank")


def _finite_score(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{name} must be finite")
    return score


def _normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _validate(item, dataset: str | None, location: str) -> Evidence:
    if not isinstance(item, Mapping):
        raise TypeError(f"{location} must be an evidence dictionary")
    for key in ("passage_id", "text", "score"):
        if key not in item:
            raise ValueError(f"{location} missing required field {key!r}")
    _nonempty_string(item["passage_id"], f"{location}.passage_id")
    _nonempty_string(item["text"], f"{location}.text")
    score = _finite_score(item["score"], f"{location}.score")
    item_dataset = item.get("dataset", dataset)
    if item_dataset is not None:
        _nonempty_string(item_dataset, f"{location}.dataset")
    elif "dataset" in item:
        raise ValueError(f"{location}.dataset must not be None when supplied")
    if dataset is not None and item_dataset != dataset:
        raise ValueError(f"{location}.dataset conflicts with dataset={dataset!r}")
    aliases = item.get("alias_passage_ids", [])
    if not isinstance(aliases, list):
        raise TypeError(f"{location}.alias_passage_ids must be a list")
    for alias in aliases:
        _nonempty_string(alias, f"{location}.alias_passage_ids entry")
    provenance = item.get("provenance", [])
    if not isinstance(provenance, list) or any(not isinstance(p, dict) for p in provenance):
        raise TypeError(f"{location}.provenance must be a list of dictionaries")
    if "rank" in item:
        rank = item["rank"]
        if type(rank) is not int or rank < 1:
            raise ValueError(f"{location}.rank must be a positive integer")
    if "fusion_score" in item:
        _finite_score(item["fusion_score"], f"{location}.fusion_score")
    result = deepcopy(dict(item))
    result["score"] = score
    if item_dataset is not None:
        result["dataset"] = item_dataset
    return result


def _collect(lists: Sequence[Sequence[Evidence]], dataset: str | None):
    if dataset is not None:
        _nonempty_string(dataset, "dataset")
    groups = []
    by_id = {}
    by_text = {}
    for list_index, items in enumerate(lists):
        _sequence(items, f"retrieval list {list_index}")
        source = "base" if list_index == 0 else f"additional:{list_index - 1}"
        for rank, raw in enumerate(items, start=1):
            item = _validate(raw, dataset, f"{source}[{rank - 1}]")
            scope = item.get("dataset")
            text = _normalized_text(item["text"])
            ids = list(dict.fromkeys([item["passage_id"], *item.get("alias_passage_ids", [])]))
            for passage_id in ids:
                existing = by_id.get((scope, passage_id))
                if existing is not None and existing["text"] != text:
                    raise ValueError(
                        f"Conflicting text for passage_id={passage_id!r} in dataset={scope!r}"
                    )
            group = by_text.get((scope, text))
            if group is None:
                evidence = deepcopy(item)
                evidence["provenance"] = []
                evidence["alias_passage_ids"] = []
                group = {"evidence": evidence, "text": text, "ranks": {}}
                groups.append(group)
                by_text[(scope, text)] = group
            evidence = group["evidence"]
            for passage_id in ids:
                by_id[(scope, passage_id)] = group
                if passage_id != evidence["passage_id"] and passage_id not in evidence["alias_passage_ids"]:
                    evidence["alias_passage_ids"].append(passage_id)
            for entry in item.get("provenance", []):
                if entry not in evidence["provenance"]:
                    evidence["provenance"].append(deepcopy(entry))
            occurrence = {
                "source": source, "rank": rank,
                "passage_id": item["passage_id"], "score": item["score"],
            }
            if scope is not None:
                occurrence["dataset"] = scope
            if occurrence not in evidence["provenance"]:
                evidence["provenance"].append(occurrence)
            # One vote per retrieval list, at its earliest original position.
            group["ranks"].setdefault(list_index, rank)
    return groups


def deduplicate_evidence(
    evidence: Sequence[Evidence], *, dataset: str | None = None,
) -> list[Evidence]:
    """Return unique evidence in first-seen order, with aliases and provenance.

    Omitted dataset values share an unscoped namespace; pass dataset explicitly
    for Base Retriever lists, which currently omit it. Conflicting ID/text pairs
    raise ValueError. No semantic similarity or retrieval-score threshold is used.
    """
    result = []
    for rank, group in enumerate(_collect([evidence], dataset), start=1):
        item = deepcopy(group["evidence"])
        item["rank"] = rank
        result.append(item)
    return result


def rrf_fuse(
    base_evidence: Sequence[Evidence],
    additional_results: Sequence[Sequence[Evidence]] = (),
    *, dataset: str | None = None, k: int = 60, top_k: int = 5,
) -> FusionResult:
    """Fuse ordered retrieval lists with sum(1 / (k + original_rank)).

    k must be a nonnegative integer; top_k is restricted to 1..5. List order,
    not an input item's optional rank field, defines retrieval rank. Ties use
    first encounter (Base before additional lists, in supplied list order).
    Without a genuinely new deduplicated candidate, Base order is preserved,
    even when extra duplicate-only lists would otherwise reorder its RRF scores.
    Candidates expose the full pool; evidence is capped and independently copied.
    """
    if type(k) is not int or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if type(top_k) is not int or not 1 <= top_k <= 5:
        raise ValueError("top_k must be an integer between 1 and 5")
    _sequence(additional_results, "additional_results")
    groups = _collect([base_evidence, *additional_results], dataset)
    new_count = sum(0 not in group["ranks"] for group in groups)
    scores = [math.fsum(1.0 / (k + rank) for rank in group["ranks"].values()) for group in groups]
    order = list(range(len(groups)))
    if new_count:
        order.sort(key=lambda i: (-scores[i], i))
    candidates = []
    for rank, i in enumerate(order, start=1):
        item = deepcopy(groups[i]["evidence"])
        item["fusion_score"] = scores[i]
        item["rank"] = rank
        candidates.append(item)
    return {
        "evidence": deepcopy(candidates[:top_k]),
        "candidates": candidates,
        "new_evidence_count": new_count,
    }
