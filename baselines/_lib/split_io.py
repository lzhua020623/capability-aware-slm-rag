"""Frozen split loader / schema checks. Never regenerates Test IDs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SplitError(ValueError):
    """Invalid or missing frozen split."""


def _dig(obj: Any, keys: list[str]) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            raise SplitError(f"Missing split path key {keys!r} at {key!r}")
        cur = cur[key]
    return cur


def load_split_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SplitError("Split root must be a JSON object")
    return data


def extract_split_ids(
    split: dict,
    *,
    id_key: list[str],
    expected_n: int,
    version_prefix: str | None = None,
    split_name: str = "split",
) -> list[str]:
    """Return ordered sample_id strings for any split key; reject dupes / wrong count."""
    if version_prefix:
        version = str(split.get("version") or "")
        if version_prefix not in version and not version.startswith(version_prefix):
            if not version:
                raise SplitError("Split missing version field")
    raw = _dig(split, id_key)
    if not isinstance(raw, list) or not raw:
        raise SplitError(f"{split_name} ID list empty or missing at {id_key}")
    ids = [str(x) for x in raw]
    if len(ids) != expected_n:
        raise SplitError(
            f"{split_name} ID count {len(ids)} != expected_n {expected_n}"
        )
    if len(set(ids)) != len(ids):
        seen: set[str] = set()
        dupes = []
        for sid in ids:
            if sid in seen:
                dupes.append(sid)
            seen.add(sid)
        raise SplitError(
            f"Duplicate sample_ids in {split_name} split (e.g. {dupes[:5]})"
        )
    return ids


def extract_test_ids(
    split: dict,
    *,
    test_key: list[str],
    expected_test_n: int,
    version_prefix: str | None = None,
) -> list[str]:
    """Return ordered Test sample_id strings; reject duplicates / wrong count."""
    return extract_split_ids(
        split,
        id_key=test_key,
        expected_n=expected_test_n,
        version_prefix=version_prefix,
        split_name="Test",
    )


def validate_split_file(
    path: Path,
    *,
    test_key: list[str],
    expected_test_n: int,
    version_prefix: str | None = None,
) -> dict:
    """Load + validate Test; returns summary dict (does not load record content)."""
    split = load_split_json(path)
    ids = extract_test_ids(
        split,
        test_key=test_key,
        expected_test_n=expected_test_n,
        version_prefix=version_prefix,
    )
    return {
        "path": str(path),
        "version": split.get("version"),
        "test_n": len(ids),
        "expected_test_n": expected_test_n,
        "ok": True,
    }


def validate_eval_split_file(
    path: Path,
    *,
    id_key: list[str],
    expected_n: int,
    split_name: str,
    version_prefix: str | None = None,
) -> dict:
    """Validate a named eval split (validation|test) without loading record content."""
    split = load_split_json(path)
    ids = extract_split_ids(
        split,
        id_key=id_key,
        expected_n=expected_n,
        version_prefix=version_prefix,
        split_name=split_name,
    )
    return {
        "path": str(path),
        "version": split.get("version"),
        "split_name": split_name,
        "n": len(ids),
        "expected_n": expected_n,
        "ok": True,
    }
