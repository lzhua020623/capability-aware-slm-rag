"""Rebuild data/processed/diagnosis/nq_2698_diagnosis_trusted.jsonl deterministically.

Inputs (both committed, plain-text JSONL; no workbook is read):
- data/annotations/normalized/nq_3216_diagnosis_inputs.jsonl: the frozen
  runtime inputs (sample_id, question, c1_prediction, top5_evidence), exported
  verbatim by scripts/export_nq_3216_diagnosis_inputs.py. It contains no gold
  answer, label or evaluation field.
- data/annotations/normalized/nq_3216_labels_normalized.jsonl: the only label
  source (primary_label, secondary_label, needs_manual_review, review_reason,
  annotation_source). Labels are copied unchanged; nothing is re-annotated.

Records are Development Records ({"input", "target_label", "metadata"}) in
source order; the trusted file keeps records with needs_manual_review ==
false. The original conversion read the annotation workbook with pandas
default NA parsing, so predictions such as the literal model output "None"
(four NQ samples) were stored as the string "nan". This artifact is
reproduced verbatim so the frozen v2 split, model and results stay valid.

If the output file already exists it is never overwritten: the rebuilt
records are compared line by line with it and the script fails on any
difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_diagnosis_classifier import input_from_dict

INPUTS_PATH = ROOT / "data" / "annotations" / "normalized" / "nq_3216_diagnosis_inputs.jsonl"
LABELS_PATH = ROOT / "data" / "annotations" / "normalized" / "nq_3216_labels_normalized.jsonl"
SUMMARY_PATH = ROOT / "data" / "annotations" / "normalized" / "nq_3216_conversion_summary.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "diagnosis" / "nq_2698_diagnosis_trusted.jsonl"
EXPECTED_ALL = 3216
EXPECTED_TRUSTED = 2698
EMPTY_PREDICTION = "nan"
# pandas.read_excel default NA strings, applied by the original conversion to c1_prediction.
PANDAS_NA_STRINGS = frozenset({
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
})


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_input(row: dict) -> dict:
    evidence = [
        {"rank": e["rank"], "passage_id": e["passage_id"], "score": e["score"], "text": e["text"]}
        for e in row["top5_evidence"]
    ]
    if [e["rank"] for e in evidence] != [1, 2, 3, 4, 5]:
        raise ValueError(f"{row['sample_id']}: expected evidence ranks 1..5, got {[e['rank'] for e in evidence]}")
    prediction = row["c1_prediction"]
    return {
        "sample_id": row["sample_id"],
        "dataset": "nq",
        "query": row["question"],
        "evidence": evidence,
        "initial_answer": EMPTY_PREDICTION if prediction is None or prediction in PANDAS_NA_STRINGS
        else prediction,
    }


def build_records() -> list[dict]:
    rows = load_jsonl(INPUTS_PATH)
    labels = load_jsonl(LABELS_PATH)
    if len(rows) != EXPECTED_ALL or len(labels) != EXPECTED_ALL:
        raise ValueError(f"Expected {EXPECTED_ALL} rows, got inputs={len(rows)} labels={len(labels)}")
    inputs = [build_input(row) for row in rows]
    ids = [item["sample_id"] for item in inputs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample_id in inputs")
    if ids != [label["sample_id"] for label in labels]:
        raise ValueError("Input and normalized label files differ in sample IDs or order")
    records = []
    for item, label in zip(inputs, labels):
        if label["dataset"] != "nq":
            raise ValueError(f"Unexpected dataset in labels: {label['dataset']!r}")
        input_from_dict(item)
        records.append({
            "input": item,
            "target_label": label["primary_label"],
            "metadata": {
                "secondary_label": label["secondary_label"],
                "needs_manual_review": label["needs_manual_review"],
                "review_reason": label["review_reason"],
                "annotation_source": label["annotation_source"],
            },
        })
    return records


def check_summary(all_records: list[dict], trusted: list[dict]) -> dict:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    observed = {
        "all_records": len(all_records),
        "trusted_records": len(trusted),
        "all_label_distribution": dict(Counter(r["target_label"] for r in all_records)),
        "trusted_label_distribution": dict(Counter(r["target_label"] for r in trusted)),
    }
    for key, value in observed.items():
        if summary[key] != value:
            raise ValueError(f"{key} differs from {SUMMARY_PATH.name}: {value} != {summary[key]}")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    records = build_records()
    trusted = [r for r in records if not r["metadata"]["needs_manual_review"]]
    if len(trusted) != EXPECTED_TRUSTED:
        raise ValueError(f"Expected {EXPECTED_TRUSTED} trusted records, got {len(trusted)}")
    observed = check_summary(records, trusted)
    lines = [json.dumps(record, ensure_ascii=False) for record in trusted]

    if args.output.exists():
        existing = args.output.read_text(encoding="utf-8").splitlines()
        diffs = [i for i, (a, b) in enumerate(zip(lines, existing)) if a != b]
        if len(existing) != len(lines) or diffs:
            raise SystemExit(f"Rebuilt records differ from {args.output}: lines {len(lines)} vs {len(existing)}, "
                             f"first differing record index {diffs[:5]}")
        print(f"Verified: {len(lines)} rebuilt records are identical to {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"Wrote {len(lines)} records to {args.output}")
    print(json.dumps(observed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
