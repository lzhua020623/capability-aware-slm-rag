"""Prediction I/O, resume, aggregate metrics, and atomic reevaluate writes."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from baselines._lib.answer_scoring import aggregate_answer_metrics, score_answer


def load_done_ids(predictions_path: Path) -> set[str]:
    if not predictions_path.is_file():
        return set()
    done: set[str] = set()
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("sample_id")
            if sid is not None:
                done.add(str(sid))
    return done


def append_prediction(predictions_path: Path, row: dict) -> None:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_predictions(predictions_path: Path) -> list[dict]:
    if not predictions_path.is_file():
        return []
    rows = []
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_row(prediction: str, gold: list[str]) -> dict:
    """Full automatic score dict (em/f1/normalized_match)."""
    return score_answer(prediction, gold)


def aggregate_metrics(rows: list[dict], *, mode: str) -> dict:
    return aggregate_answer_metrics(rows, mode=mode)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def backup_file(path: Path, backup_dir: Path) -> Path | None:
    """Copy ``path`` into backup_dir with timestamp; return backup path or None."""
    if not path.is_file():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = backup_dir / f"{path.name}.{stamp}.bak"
    shutil.copy2(path, dest)
    return dest


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL via temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
