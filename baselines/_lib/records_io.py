"""Load query/gold by sample_id from diagnosis JSONL (no split regeneration)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
