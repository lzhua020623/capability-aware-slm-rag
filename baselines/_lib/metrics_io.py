"""Prediction I/O, resume, and aggregate EM/F1."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.qa import best_em_f1


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


def score_row(prediction: str, gold: list[str]) -> tuple[int, float]:
    return best_em_f1(prediction, gold)


def aggregate_metrics(rows: list[dict], *, mode: str) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "em": 0.0, "f1": 0.0, "mode": mode}
    em_sum = 0.0
    f1_sum = 0.0
    for row in rows:
        gold = row.get("gold") or []
        if isinstance(gold, str):
            gold = [gold]
        pred = str(row.get("prediction") or "")
        if "em" in row and "f1" in row:
            em_sum += float(row["em"])
            f1_sum += float(row["f1"])
        else:
            em, f1 = best_em_f1(pred, list(gold))
            em_sum += em
            f1_sum += f1
    return {
        "n": n,
        "em": em_sum / n,
        "f1": f1_sum / n,
        "mode": mode,
    }


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
