"""Load query/gold by sample_id from diagnosis JSONL (no split regeneration).

Also loads flat FEVER fever_dev rows (id / claim / label) for FEVER baselines.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

FEVER_LEGAL_LABELS = ("SUPPORTS", "REFUTES")


def _dig(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def normalize_gold(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    return [str(value)] if str(value).strip() else []


def load_records_by_ids(
    records_path: Path,
    sample_ids: list[str],
    *,
    id_field: str = "input.sample_id",
    query_field: str = "input.query",
    gold_field: str = "metadata.annotation_reference.gold_answer",
) -> list[dict]:
    """Return rows in the same order as sample_ids. Raises if any ID missing."""
    wanted = set(sample_ids)
    found: dict[str, dict] = {}
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = _dig(rec, id_field)
            if sid is None:
                continue
            sid = str(sid)
            if sid not in wanted:
                continue
            query = _dig(rec, query_field)
            gold = normalize_gold(_dig(rec, gold_field))
            found[sid] = {
                "sample_id": sid,
                "query": str(query or ""),
                "gold": gold,
            }
            if len(found) == len(wanted):
                break
    missing = [sid for sid in sample_ids if sid not in found]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} sample_ids missing from {records_path} "
            f"(e.g. {missing[:5]})"
        )
    return [found[sid] for sid in sample_ids]


def load_fever_dev_records(
    records_path: Path,
    *,
    id_field: str = "id",
    claim_field: str = "claim",
    gold_field: str = "label",
    expected_n: int | None = 13332,
    legal_labels: tuple[str, ...] | list[str] = FEVER_LEGAL_LABELS,
) -> list[dict]:
    """Load full flat fever_dev JSONL; validate n / unique ids / legal labels."""
    legal = {str(x).upper() for x in legal_labels}
    rows: list[dict] = []
    seen: set[str] = set()
    with records_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = _dig(rec, id_field) if "." in id_field else rec.get(id_field)
            claim = _dig(rec, claim_field) if "." in claim_field else rec.get(claim_field)
            label_raw = _dig(rec, gold_field) if "." in gold_field else rec.get(gold_field)
            if sid is None:
                raise ValueError(f"Missing {id_field} at line {line_no} in {records_path}")
            sid = str(sid)
            if sid in seen:
                raise ValueError(f"Duplicate sample_id {sid!r} in {records_path}")
            seen.add(sid)
            label = str(label_raw or "").strip().upper()
            if label not in legal:
                raise ValueError(
                    f"Illegal FEVER label {label_raw!r} for id={sid} "
                    f"(expected one of {sorted(legal)})"
                )
            claim_s = str(claim or "")
            query_raw = rec.get("query")
            query_s = str(query_raw) if query_raw is not None else claim_s
            rows.append(
                {
                    "sample_id": sid,
                    "claim": claim_s,
                    "query": query_s,
                    "gold": label,
                    "dataset": "fever",
                }
            )
    if expected_n is not None and len(rows) != int(expected_n):
        raise ValueError(
            f"fever_dev n={len(rows)} != expected_n={expected_n} ({records_path})"
        )
    return rows


def sample_fever_smoke_records(
    records: list[dict],
    *,
    n: int = 30,
    seed: int = 42,
    balanced: bool = True,
) -> list[dict]:
    """Fixed-seed subsample; when balanced, take n//2 per SUPPORTS/REFUTES."""
    if n < 1:
        raise ValueError("smoke n must be >= 1")
    if n > len(records):
        raise ValueError(f"smoke n={n} exceeds dataset size {len(records)}")
    rng = random.Random(seed)
    if not balanced:
        picked = list(records)
        rng.shuffle(picked)
        return picked[:n]

    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_label[str(row["gold"]).upper()].append(row)
    for label in FEVER_LEGAL_LABELS:
        if label not in by_label:
            raise ValueError(f"Cannot balance smoke: missing label {label}")
    # Prefer even split; if n odd, SUPPORTS gets the extra slot.
    n_supports = (n + 1) // 2
    n_refutes = n // 2
    if len(by_label["SUPPORTS"]) < n_supports or len(by_label["REFUTES"]) < n_refutes:
        raise ValueError(
            f"Not enough rows to balance smoke n={n} "
            f"(SUPPORTS={len(by_label['SUPPORTS'])}, REFUTES={len(by_label['REFUTES'])})"
        )
    supports = list(by_label["SUPPORTS"])
    refutes = list(by_label["REFUTES"])
    rng.shuffle(supports)
    rng.shuffle(refutes)
    out = supports[:n_supports] + refutes[:n_refutes]
    rng.shuffle(out)
    return out


def synthetic_smoke_records(n: int = 3) -> list[dict]:
    """Tiny non-Test fixtures for CLI / schema / resume smoke tests."""
    fixtures = [
        {
            "sample_id": "smoke_001",
            "query": "What color is the sky on a clear day?",
            "gold": ["blue"],
        },
        {
            "sample_id": "smoke_002",
            "query": "How many days are in a leap year?",
            "gold": ["366", "three hundred sixty-six"],
        },
        {
            "sample_id": "smoke_003",
            "query": "What is 2 + 2?",
            "gold": ["4", "four"],
        },
    ]
    return fixtures[: max(1, n)]
