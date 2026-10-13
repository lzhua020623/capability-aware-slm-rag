"""Offline paired NQ retrieval metrics. Never imported by the Enhancer.

NQ's prepared retrieval data has one gold_passage_id per query. Strict ID
Recall@5 matches the Base retriever metric. Alias-aware metrics are reported
separately so conservative text deduplication is not mistaken for evidence loss.
Alias-aware matching expands Base/selected IDs through candidate dedup groups;
it does not count an identical-text alias of existing Base evidence as recovery.
These are retrieval metrics, not QA accuracy or preliminary Recoverable Failure
Rate. An undefined denominator is reported as None (JSON null), never as zero.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence


def sample_id(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("sample id must be a nonempty string or integer")
    return str(value)


def _ids(evidence, name, *, aliases=False, top5=False):
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        raise ValueError(f"{name} must be an evidence list")
    if top5 and len(evidence) > 5:
        raise ValueError(f"{name} exceeds Top-5")
    result = set()
    for item in evidence:
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} contains a non-dictionary evidence item")
        pid = item.get("passage_id")
        if not isinstance(pid, str) or not pid.strip():
            raise ValueError(f"{name} contains an invalid passage_id")
        extra = item.get("alias_passage_ids", [])
        if not isinstance(extra, list) or any(not isinstance(x, str) or not x.strip() for x in extra):
            raise ValueError(f"{name} contains invalid alias_passage_ids")
        result.add(pid)
        if aliases:
            result.update(extra)
    return result


def _nonnegative(value, name, *, integer=False):
    if integer:
        valid = type(value) is int and value >= 0
    else:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    if not valid:
        raise ValueError(f"{name} must be a finite nonnegative {'integer' if integer else 'number'}")
    return value


def _alias_ids(evidence, candidates):
    result = _ids(evidence, "evidence", aliases=True)
    for candidate in candidates:
        group = _ids([candidate], "candidate", aliases=True)
        if group & result:
            result.update(group)
    return result


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate_nq(records: Sequence[dict], annotations: Sequence[dict]) -> dict:
    """Join saved runtime records to labels by ID, after all retrieval decisions.

    Required runtime fields: sample_id, base_evidence, enhanced_evidence,
    candidates, base_latency and trace (retrieval_calls, llm_calls, latency,
    status). Annotations need only id and gold_passage_id; answers/text are never
    read. Missing labels, duplicate IDs and malformed runs fail explicitly rather
    than silently changing the denominator. Extra unselected annotations are OK.
    """
    gold_by_id = {}
    for annotation in annotations:
        sid = sample_id(annotation["id"])
        if sid in gold_by_id:
            raise ValueError(f"duplicate annotation id: {sid}")
        gold_by_id[sid] = annotation.get("gold_passage_id")
    per_sample, seen = [], set()
    totals = Counter()
    statuses = Counter()
    for record in records:
        if record.get("dataset", "nq") not in ("nq", "natural_questions"):
            raise ValueError("offline NQ metrics cannot score another dataset")
        sid = sample_id(record["sample_id"])
        if sid in seen:
            raise ValueError(f"duplicate runtime sample_id: {sid}")
        seen.add(sid)
        gold = gold_by_id.get(sid)
        if not isinstance(gold, str) or not gold.strip():
            raise ValueError(f"missing/invalid gold_passage_id for sample {sid}")
        all_candidates = _ids(record["candidates"], "candidates", aliases=True)
        for field in ("base_evidence", "enhanced_evidence"):
            if not _ids(record[field], field, aliases=True, top5=True) <= all_candidates:
                raise ValueError(f"{field} contains evidence absent from candidates")
        row = {"sample_id": sid, "gold_passage_id": gold}
        for mode, aliases in (("strict_id", False), ("alias_aware", True)):
            base_ids = (_alias_ids(record["base_evidence"], record["candidates"]) if aliases
                        else _ids(record["base_evidence"], "base_evidence"))
            selected_ids = (_alias_ids(record["enhanced_evidence"], record["candidates"]) if aliases
                            else _ids(record["enhanced_evidence"], "enhanced_evidence"))
            base_hit = gold in base_ids
            candidate_hit = gold in _ids(record["candidates"], "candidates", aliases=aliases)
            enhanced_hit = gold in selected_ids
            row[mode] = {
                "base_hit": base_hit, "candidate_hit": candidate_hit,
                "enhanced_hit": enhanced_hit,
                "candidate_recovered": not base_hit and candidate_hit,
                "selected_recovered": not base_hit and enhanced_hit,
                "evidence_lost": base_hit and not enhanced_hit,
            }
        trace = record["trace"]
        status = trace["status"]
        if not isinstance(status, str) or not status.strip():
            raise ValueError("trace.status must be a nonempty string")
        statuses[status] += 1
        for field in ("retrieval_calls", "llm_calls"):
            totals[field] += _nonnegative(trace[field], field, integer=True)
        base_latency = _nonnegative(record["base_latency"], "base_latency")
        extra_latency = _nonnegative(trace["latency"], "enhancement latency")
        totals["base_latency"] += base_latency
        totals["enhancement_latency"] += extra_latency
        per_sample.append(row)
    n = len(per_sample)
    metrics = {"n": n}
    for mode in ("strict_id", "alias_aware"):
        counts = {field: sum(row[mode][field] for row in per_sample)
                  for field in ("base_hit", "candidate_hit", "enhanced_hit",
                                "candidate_recovered", "selected_recovered", "evidence_lost")}
        base_hits = counts["base_hit"]
        base_misses = n - base_hits
        metrics[mode] = {
            **counts, "base_misses": base_misses,
            "base_recall_at5": _rate(base_hits, n),
            "enhanced_recall_at5": _rate(counts["enhanced_hit"], n),
            "recall_at5_delta": _rate(counts["enhanced_hit"] - base_hits, n),
            "candidate_recovery_rate": _rate(counts["candidate_recovered"], base_misses),
            "selected_recovery_rate": _rate(counts["selected_recovered"], base_misses),
            "evidence_loss_rate": _rate(counts["evidence_lost"], base_hits),
        }
    metrics["cost"] = {
        "base_retrieval_calls": n, "additional_retrieval_calls": totals["retrieval_calls"],
        "total_retrieval_calls": n + totals["retrieval_calls"], "llm_calls": totals["llm_calls"],
        "mean_additional_retrieval_calls": _rate(totals["retrieval_calls"], n),
        "mean_base_latency_seconds": _rate(totals["base_latency"], n),
        "mean_enhancement_latency_seconds": _rate(totals["enhancement_latency"], n),
        "mean_total_latency_seconds": _rate(totals["base_latency"] + totals["enhancement_latency"], n),
    }
    metrics["statuses"] = dict(sorted(statuses.items()))
    return {"metrics": metrics, "per_sample": per_sample}
