"""Normalize the returned NQ Train annotation workbooks (the NQ_15 delivery) into JSONL.

One Excel pass. The 15 workbooks are opened read-only and never modified; the
raw drop itself stays out of git (see .gitignore). Two outputs:

* `data/annotations/normalized/nq15_train_labels.jsonl` -- sample_id, dataset
  and the frozen primary label, nothing else. Small enough to commit.
* `data/processed/diagnosis/nq15_train_records.jsonl` -- the full record
  (query, Top-5 evidence, initial answer, label and the offline
  `annotation_reference`) for the downstream builders. Large, gitignored.

`annotation_reference` holds gold answers and gold passage IDs. They are
**offline supervision only**: they may be used to build training data and to
evaluate, and must never reach a runtime `DiagnosisInput`, a Probe prompt or a
Controller input. That is why they live in a separate field here instead of
being merged into the record body.

Usage:
    python scripts/normalize_nq15_annotations.py
    python scripts/normalize_nq15_annotations.py --source data/NQ_15/NQ_15批 --limit 2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnosis.labels import DiagnosisLabel  # noqa: E402
from src.diagnosis.schema import DatasetName, normalize_sample_id  # noqa: E402

DEFAULT_SOURCE = ROOT / "data" / "NQ_15" / "NQ_15批"
DEFAULT_LABELS_OUT = ROOT / "data" / "annotations" / "normalized" / "nq15_train_labels.jsonl"
DEFAULT_RECORDS_OUT = ROOT / "data" / "processed" / "diagnosis" / "nq15_train_records.jsonl"
DEFAULT_STATS_OUT = ROOT / "results" / "compensation" / "nq15_train_label_stats.json"

# The returned workbooks use the display names; "Correct / No Failure" is the
# spelling used by the earlier Dev workbooks and is accepted too.
PRIMARY_LABEL_MAP: dict[str, DiagnosisLabel] = {
    "Correct": DiagnosisLabel.CORRECT,
    "Correct / No Failure": DiagnosisLabel.CORRECT,
    "Evidence Insufficiency": DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
    "Evidence Accessibility Failure": DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
    "Evidence Utilisation Failure": DiagnosisLabel.EVIDENCE_UTILISATION,
}

REQUIRED_COLUMNS = ("sample_id", "dataset", "query", "initial_answer", "evidence", "annotation_reference", "label")
REFERENCE_FIELDS = ("gold_answer", "gold_evidence_text", "gold_passage_id", "gold_in_top1", "gold_in_top3", "gold_in_top5")


def parse_primary_label(value: object) -> DiagnosisLabel:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing primary label: {value!r}")
    text = value.strip()
    if text not in PRIMARY_LABEL_MAP:
        raise ValueError(f"Unknown primary label {text!r}; expected one of: {', '.join(PRIMARY_LABEL_MAP)}")
    return PRIMARY_LABEL_MAP[text]


def parse_json_cell(name: str, value: object) -> object:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty JSON string")
    return json.loads(value)


def parse_evidence(value: object) -> list[dict]:
    evidence = parse_json_cell("evidence", value)
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence must be a non-empty list")
    passages = []
    for index, passage in enumerate(evidence):
        if not isinstance(passage, dict):
            raise ValueError(f"evidence[{index}] must be an object")
        passages.append(
            {
                "rank": int(passage["rank"]),
                "passage_id": str(passage["passage_id"]),
                "text": str(passage["text"]),
                "score": float(passage["score"]),
            }
        )
    ranks = [passage["rank"] for passage in passages]
    if sorted(ranks) != list(range(1, len(ranks) + 1)):
        raise ValueError(f"evidence ranks must be 1..n, got {ranks}")
    return passages


def parse_reference(value: object) -> dict:
    reference = parse_json_cell("annotation_reference", value)
    if not isinstance(reference, dict):
        raise ValueError("annotation_reference must be an object")
    return {field: reference.get(field) for field in REFERENCE_FIELDS}


def convert(source: Path, limit: int | None) -> tuple[list[dict], list[dict], dict]:
    workbooks = sorted(source.glob("*.xlsx"))
    if limit is not None:
        workbooks = workbooks[:limit]
    if not workbooks:
        raise SystemExit(f"No .xlsx files found in {source}")

    labels: list[dict] = []
    records: list[dict] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    per_file: list[dict] = []

    for workbook in workbooks:
        frame = pd.read_excel(workbook)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise SystemExit(f"{workbook.name} is missing columns: {missing}")
        file_counts: Counter[str] = Counter()
        for position, row in frame.iterrows():
            try:
                sample_id = normalize_sample_id(row["sample_id"])
                label = parse_primary_label(row["label"])
                dataset = DatasetName(str(row["dataset"]).strip())
                evidence = parse_evidence(row["evidence"])
                reference = parse_reference(row["annotation_reference"])
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise SystemExit(f"{workbook.name} row {position + 2}: {exc}") from exc
            if sample_id in seen:
                raise SystemExit(f"{workbook.name} row {position + 2}: duplicate sample_id {sample_id}")
            seen.add(sample_id)
            counts[label.value] += 1
            file_counts[label.value] += 1
            labels.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset.value,
                    "primary_label": label.value,
                    "annotation_source": workbook.name,
                }
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset.value,
                    "query": str(row["query"]),
                    "initial_answer": "" if pd.isna(row["initial_answer"]) else str(row["initial_answer"]),
                    "evidence": evidence,
                    "primary_label": label.value,
                    "annotation_reference": reference,
                }
            )
        per_file.append({"file": workbook.name, "rows": int(len(frame)), "labels": dict(file_counts)})
        print(f"{workbook.name:45s} {len(frame):6d} rows")

    stats = {
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "workbooks": len(workbooks),
        "rows": len(labels),
        "unique_sample_ids": len(seen),
        "label_distribution": {label: counts[label] for label in sorted(counts, key=lambda key: -counts[key])},
        "label_share": {label: round(counts[label] / len(labels), 4) for label in sorted(counts, key=lambda key: -counts[key])},
        "per_file": per_file,
        "note": "annotation_reference (gold answer / gold passage) is offline supervision only and is kept "
        "out of the committed label file.",
    }
    return labels, records, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--labels-out", type=Path, default=DEFAULT_LABELS_OUT)
    parser.add_argument("--records-out", type=Path, default=DEFAULT_RECORDS_OUT)
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_STATS_OUT)
    parser.add_argument("--limit", type=int, default=None, help="only the first N workbooks (smoke runs)")
    args = parser.parse_args()

    labels, records, stats = convert(args.source, args.limit)
    write_jsonl(args.labels_out, labels)
    write_jsonl(args.records_out, records)
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    with args.stats_out.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    print()
    print(json.dumps({key: stats[key] for key in ("rows", "unique_sample_ids", "label_distribution", "label_share")}, indent=2))
    print(f"wrote {args.labels_out}")
    print(f"wrote {args.records_out}")
    print(f"wrote {args.stats_out}")


if __name__ == "__main__":
    main()
