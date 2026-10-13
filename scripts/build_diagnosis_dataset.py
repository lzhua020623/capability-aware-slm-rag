"""Build the Diagnosis development datasets (NQ 500, FEVER 500).

Joins canonical C1 records with normalized annotations one-to-one by
sample_id. No samples are dropped or resampled; needs_manual_review is kept as
metadata so trusted-only vs all-data can be decided at training time.
Output contains no gold, correctness, EM/F1, C0/C3 or larger-model fields.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnosis.annotations import load_annotations
from src.diagnosis.dev_dataset import join_development_records, load_jsonl
from src.diagnosis.labels import RUNTIME_LABELS

EXPECTED_SIZE = 500
C1_DIR = ROOT / "results" / "preliminary"
ANNOTATION_DIR = ROOT / "data" / "annotations" / "normalized"
OUTPUT_DIR = ROOT / "data" / "processed" / "diagnosis"

DATASETS = {
    "nq": (C1_DIR / "nq_c1_7b.jsonl", ANNOTATION_DIR / "nq_500_labels.jsonl", OUTPUT_DIR / "nq_500.jsonl"),
    "fever": (C1_DIR / "fever_c1_7b.jsonl", ANNOTATION_DIR / "fever_500_labels.jsonl", OUTPUT_DIR / "fever_500.jsonl"),
}


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    for name, (c1_path, annotation_path, output_path) in DATASETS.items():
        records = join_development_records(load_jsonl(c1_path), load_annotations(annotation_path))
        if len(records) != EXPECTED_SIZE:
            raise ValueError(f"{name}: expected {EXPECTED_SIZE} records, got {len(records)}")
        write_jsonl(output_path, [record.to_dict() for record in records])

        labels = Counter(record.target_label for record in records)
        review = sum(record.metadata.needs_manual_review for record in records)
        print(f"{name}: {len(records)} records -> {output_path.relative_to(ROOT)}")
        for label in RUNTIME_LABELS:
            print(f"  {label.value}: {labels[label]}")
        print(f"  needs_manual_review: {review}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
