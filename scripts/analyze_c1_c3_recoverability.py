"""Pair frozen C1/C3 results and save recoverability cohorts for handoff.

This script performs no inference. It validates a complete one-to-one pairing
by sample_id, prints the Recoverable Failure Rate, and writes both C1-failure
cohorts for later analysis or Person D's larger-model routing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_preliminary_config
from src.evaluation.recoverability import recoverable_failure_rate

DATASETS = {
    "nq": "natural_questions",
    "fever": "fever",
}


def load_unique_results(path: Path, dataset: str, condition: str) -> tuple[list[dict], dict[str, dict]]:
    if not path.is_file():
        raise RuntimeError(f"Missing {condition} result file: {path}")
    ordered: list[dict] = []
    by_id: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{number}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"Expected an object at {path}:{number}")
            if record.get("dataset") != dataset or record.get("condition") != condition:
                raise RuntimeError(
                    f"Unexpected dataset/condition at {path}:{number}: "
                    f"{record.get('dataset')}/{record.get('condition')}"
                )
            sample_id = str(record.get("sample_id"))
            if sample_id in by_id:
                raise RuntimeError(f"Duplicate sample ID {sample_id} in {path}")
            if type(record.get("correct")) is not bool:
                raise RuntimeError(f"Missing boolean correct value for {sample_id} in {path}")
            ordered.append(record)
            by_id[sample_id] = record
    return ordered, by_id


def paired_record(c1: dict, c3: dict, outcome: str) -> dict:
    return {
        "sample_id": c1["sample_id"],
        "dataset": c1["dataset"],
        "outcome": outcome,
        "model_name": c3.get("model_name"),
        "question_or_claim": c1["question_or_claim"],
        "gold_answer_or_label": c1["gold_answer_or_label"],
        "oracle_evidence": c3.get("evidence", []),
        "c1_prediction": c1.get("prediction"),
        "c1_correct": c1["correct"],
        "c1_em": c1.get("em"),
        "c1_f1": c1.get("f1"),
        "c3_prediction": c3.get("prediction"),
        "c3_correct": c3["correct"],
        "c3_em": c3.get("em"),
        "c3_f1": c3.get("f1"),
    }


def pair_results(c1_path: Path, c3_path: Path, dataset: str) -> tuple[dict, list[dict], list[dict]]:
    c1_ordered, c1_by_id = load_unique_results(c1_path, dataset, "C1")
    c3_ordered, c3_by_id = load_unique_results(c3_path, dataset, "C3")
    c1_ids = set(c1_by_id)
    c3_ids = set(c3_by_id)
    if c1_ids != c3_ids:
        raise RuntimeError(
            "C1/C3 sample IDs differ: "
            f"missing_from_c3={len(c1_ids - c3_ids)}, "
            f"missing_from_c1={len(c3_ids - c1_ids)}"
        )
    if len(c1_ids) != 500:
        raise RuntimeError(f"Expected 500 paired frozen samples, found {len(c1_ids)}")

    recoverable: list[dict] = []
    persistent: list[dict] = []
    for c1 in c1_ordered:
        sample_id = str(c1["sample_id"])
        c3 = c3_by_id[sample_id]
        for key in ("question_or_claim", "gold_answer_or_label"):
            if c1.get(key) != c3.get(key):
                raise RuntimeError(f"Paired {key} mismatch for sample {sample_id}")
        if c1.get("model_name") != c3.get("model_name"):
            raise RuntimeError(f"Paired model mismatch for sample {sample_id}")
        if c1["correct"]:
            continue
        if c3["correct"]:
            recoverable.append(paired_record(c1, c3, "c1_wrong_c3_correct"))
        else:
            persistent.append(paired_record(c1, c3, "c1_wrong_c3_wrong"))

    c1_wrong_ids = {str(row["sample_id"]) for row in recoverable + persistent}
    c3_correct_ids = {str(row["sample_id"]) for row in recoverable}
    rate = recoverable_failure_rate(c1_wrong_ids, c3_correct_ids)
    summary = {
        "dataset": dataset,
        "paired_samples": len(c1_ordered),
        "total_c1_wrong": len(c1_wrong_ids),
        "c1_wrong_c3_correct": len(recoverable),
        "c1_wrong_c3_wrong": len(persistent),
        "recoverable_failure_rate": rate,
    }
    # Preserve C1 manifest/result order in both reusable cohorts.
    assert len(c3_ordered) == len(c1_ordered)
    return summary, recoverable, persistent


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def analyze_dataset(
    dataset_key: str,
    *,
    config: dict | None = None,
    root: Path = ROOT,
    c1_path: Path | None = None,
    c3_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    if dataset_key not in DATASETS:
        raise RuntimeError(f"Unsupported dataset: {dataset_key}")
    config = config or load_preliminary_config()
    paths = config["result_paths"]
    c1_path = c1_path or root / paths[f"{dataset_key}_c1_7b"]
    c3_path = c3_path or root / paths[f"{dataset_key}_c3_7b"]
    output_dir = output_dir or root / "results/preliminary"
    summary, recoverable, persistent = pair_results(
        c1_path, c3_path, DATASETS[dataset_key]
    )

    recoverable_path = output_dir / f"{dataset_key}_c1_c3_recoverable.jsonl"
    persistent_path = output_dir / f"{dataset_key}_c1_c3_persistent_failures.jsonl"
    summary_path = output_dir / f"{dataset_key}_c1_c3_recoverability_summary.json"
    summary.update(
        {
            "c1_path": str(c1_path),
            "c3_path": str(c3_path),
            "recoverable_set_path": str(recoverable_path),
            "persistent_failure_set_path": str(persistent_path),
        }
    )
    atomic_write_jsonl(recoverable_path, recoverable)
    atomic_write_jsonl(persistent_path, persistent)
    atomic_write_json(summary_path, summary)
    print(
        f"{dataset_key}: paired={summary['paired_samples']}, "
        f"C1 wrong={summary['total_c1_wrong']}, "
        f"C1 wrong/C3 correct={summary['c1_wrong_c3_correct']}, "
        f"C1 wrong/C3 wrong={summary['c1_wrong_c3_wrong']}, "
        f"Recoverable Failure Rate={summary['recoverable_failure_rate']:.4f}"
    )
    print(f"Recoverable set: {recoverable_path}")
    print(f"Persistent-failure set: {persistent_path}")
    print(f"Summary: {summary_path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pair frozen C1/C3 results and save recoverability cohorts"
    )
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--c1", type=Path, help="override the configured C1 JSONL")
    parser.add_argument("--c3", type=Path, help="override the configured C3 JSONL")
    parser.add_argument(
        "--output-dir", type=Path, help="output directory (default: results/preliminary)"
    )
    args = parser.parse_args()
    analyze_dataset(
        args.dataset,
        c1_path=args.c1,
        c3_path=args.c3,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
