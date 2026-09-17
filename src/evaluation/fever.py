"""FEVER SUPPORTS/REFUTES parsing and retrieval-hit checks."""

from __future__ import annotations

import re

LABEL_RE = re.compile(r"\b(SUPPORTS|REFUTES)\b", re.IGNORECASE)


def parse_fever_prediction(text: str) -> str:
    matches = LABEL_RE.findall(text or "")
    if not matches:
        return "UNKNOWN"
    return matches[0].upper()


def gold_evidence_in_top5(hits: list[dict], gold_page_ids: list[str] | None) -> bool:
    if not gold_page_ids:
        return False
    retrieved = {hit.get("passage_id") for hit in hits}
    return any(page_id in retrieved for page_id in gold_page_ids)
