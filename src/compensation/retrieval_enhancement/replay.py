"""Strict C1 Replay input validation and query-only offline coverage analysis.

No model imports or Unicode-unescape. Labels are optional offline sidecar data,
joined only after query eligibility has been computed; they never select queries.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import math
import re
import unicodedata

from .rewriting import RuleQueryRewriter

_FIELDS = {"sample_id", "dataset", "question", "base_evidence"}
_EVIDENCE_FIELDS = {"passage_id", "text", "score"}


def validate_replay(records):
    """Return independent original records and cross-sample text conflict audit."""
    if not isinstance(records, list) or not records:
        raise ValueError("Replay must be a nonempty list of records")
    seen = set()
    variants = defaultdict(lambda: defaultdict(list))
    for position, record in enumerate(records, 1):
        if not isinstance(record, dict) or set(record) != _FIELDS:
            raise ValueError(f"Replay row {position} must contain only {sorted(_FIELDS)}")
        sid = record["sample_id"]
        if not isinstance(sid, str) or not sid.strip():
            raise ValueError(f"Replay row {position}: sample_id must be a nonempty string")
        if sid in seen:
            raise ValueError(f"duplicate Replay sample_id: {sid}")
        seen.add(sid)
        if record["dataset"] not in ("nq", "natural_questions"):
            raise ValueError(f"Replay row {position}: dataset must be nq/natural_questions")
        if not isinstance(record["question"], str) or not record["question"].strip():
            raise ValueError(f"Replay row {position}: question must be nonempty")
        evidence = record["base_evidence"]
        if not isinstance(evidence, list) or len(evidence) != 5:
            raise ValueError(f"Replay row {position}: C1 base_evidence must contain exactly 5 items")
        passage_ids = set()
        previous_score = math.inf
        for rank, item in enumerate(evidence, 1):
            if not isinstance(item, dict) or set(item) != _EVIDENCE_FIELDS:
                raise ValueError(f"Replay row {position} rank {rank}: Evidence must contain only passage_id/text/score")
            for name in ("passage_id", "text"):
                if not isinstance(item[name], str) or not item[name].strip():
                    raise ValueError(f"Replay row {position} rank {rank}: invalid {name}")
            pid, score = item["passage_id"], item["score"]
            if pid in passage_ids:
                raise ValueError(f"Replay row {position}: duplicate passage_id {pid}")
            passage_ids.add(pid)
            try:
                finite = type(score) in (int, float) and math.isfinite(score)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"Replay row {position} rank {rank}: score must be finite numeric")
            if score > previous_score:
                raise ValueError(f"Replay row {position}: scores are not in descending order; will not reorder")
            previous_score = score
            variants[pid][item["text"]].append({"sample_id": sid, "input_row": position, "rank": rank})
    conflicts = []
    for pid, bodies in variants.items():
        normalized = {" ".join(unicodedata.normalize("NFC", text).split()) for text in bodies}
        if len(normalized) > 1:
            conflicts.append({"passage_id": pid, "variants": [
                {"text": text, "occurrences": occurrences} for text, occurrences in bodies.items()]})
    return deepcopy(records), {"input_records": len(records), "conflicts": conflicts,
        "warnings": (["Cross-sample passage_id text conflicts detected; preserve text and verify against original Metadata."] if conflicts else []),
        "provenance_status": "unverified", "text_policy": "preserve exact text, including literal Unicode escapes"}


def manifest_ids(manifest):
    ids = manifest.get("ids")
    if (manifest.get("dataset") != "natural_questions" or not isinstance(ids, list) or not ids
            or any(not isinstance(sid, str) or not sid.strip() for sid in ids)
            or len(set(ids)) != len(ids) or type(manifest.get("n")) is not int or manifest["n"] != len(ids)):
        raise ValueError("NQ Manifest requires n matching unique string IDs")
    return ids


def select_replay(records, manifest, *, scope="frozen", limit=None):
    """Frozen IDs must be the exact input prefix; no reordering or resampling.

    A full C1 file may append Development records. Development-only files are
    allowed in development scope, but must contain no frozen IDs. Any partial or
    misordered frozen overlap is rejected, even for a short smoke prefix.
    Caller must first validate all Replay records (including unselected rows).
    """
    ids = manifest_ids(manifest)
    frozen = set(ids)
    overlap = [item["sample_id"] for item in records if item["sample_id"] in frozen]
    if overlap and (overlap != ids or [item["sample_id"] for item in records[:len(ids)]] != ids):
        raise ValueError("Frozen Replay IDs/order must strictly match Manifest as the complete input prefix")
    if scope == "frozen":
        if overlap != ids:
            raise ValueError("Frozen Replay is missing Manifest IDs")
        selected = records[:len(ids)]
    elif scope == "development":
        selected = [item for item in records if item["sample_id"] not in frozen]
    else:
        raise ValueError("Replay scope must be frozen or development")
    if not selected:
        raise ValueError(f"Replay has no {scope} samples")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= len(selected)):
        raise ValueError("Replay limit must be within the selected scope")
    return deepcopy(selected[:limit])


def rewrite_coverage(records, manifest, annotations=()):
    """Describe eligibility without inference or retrieval; never change Rewriter.

    Reason categories describe the existing four rules, not semantic failures.
    Development means all supplied records outside the frozen manifest.
    AI initial labels are descriptive strata only, not Gold or routing signals.
    """
    ids = manifest_ids(manifest)
    frozen = set(ids)
    if [item["sample_id"] for item in records[:len(ids)]] != ids:
        raise ValueError("Coverage input must start with the complete frozen Manifest")
    rewriter = RuleQueryRewriter()
    eligibility, reasons = {}, {}
    patterns = [pattern for pattern, _ in rewriter._RULES]
    for item in records:
        sid, query = item["sample_id"], item["question"]
        output = rewriter.rewrite(query, "nq")  # Query only, before label access.
        eligibility[sid] = bool(output)
        cleaned = " ".join(unicodedata.normalize("NFC", query).split())
        reasons[sid] = ("eligible" if output else "matched_template_but_rejected_conservatively"
                        if any(re.fullmatch(p, cleaned, re.I) for p in patterns) else "no_supported_template")
    label_by_id = {}
    for item in annotations:
        sid = item["sample_id"]
        if not isinstance(sid, str) or sid in label_by_id:
            raise ValueError("Offline coverage labels require unique string sample_id")
        label_by_id[sid] = item.get("primary_label")

    def summary(sample_ids):
        n = len(sample_ids)
        eligible = sum(eligibility[sid] for sid in sample_ids)
        return {"n": n, "eligible": eligible, "coverage": eligible / n if n else None,
                "reasons": dict(Counter(reasons[sid] for sid in sample_ids))}

    all_ids = [item["sample_id"] for item in records]
    result = {"frozen": summary(ids), "development": summary([sid for sid in all_ids if sid not in frozen]),
              "all": summary(all_ids), "label_use": "post-hoc descriptive analysis only; not Gold or routing"}
    if annotations:
        if set(all_ids) != set(label_by_id):
            raise ValueError("Offline coverage labels must match all supplied Replay IDs")
        insuff = [sid for sid in all_ids if label_by_id[sid] == "Evidence Insufficiency"]
        result["ai_initial_evidence_insufficiency"] = {
            "all": summary(insuff), "frozen": summary([sid for sid in insuff if sid in frozen]),
            "development": summary([sid for sid in insuff if sid not in frozen])}
    return result
