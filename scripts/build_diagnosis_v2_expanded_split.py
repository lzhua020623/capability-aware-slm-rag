"""Build the fixed Diagnosis v2-expanded train/validation/test split.

Same method as v1 (scripts/build_diagnosis_splits.py): trusted records only,
query near-duplicate groups never cross splits, per dataset x label
stratification, seed 42, 70/15/15. Only the inputs change: the expanded NQ
trusted set (2698) plus the trusted FEVER records from fever_500. The v1
split file is never written.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_diagnosis_splits import (
    SEED,
    SPLITS,
    build_split,
    query_groups,
    trusted_records,
    validate_split,
)
from src.diagnosis.dev_dataset import load_jsonl

VERSION = "diagnosis_v2_expanded"
DATASET_PATHS = {
    "nq": ROOT / "data" / "processed" / "diagnosis" / "nq_2698_diagnosis_trusted.jsonl",
    "fever": ROOT / "data" / "processed" / "diagnosis" / "fever_500.jsonl",
}
OUTPUT_PATH = ROOT / "configs" / "splits" / "diagnosis_v2_expanded_split.json"
V1_SPLIT_PATH = ROOT / "configs" / "splits" / "diagnosis_v1_split.json"
PROMPT_DEVELOPMENT_PATH = ROOT / "configs" / "probe" / "prompt_development_samples.json"


def main() -> int:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"{OUTPUT_PATH.relative_to(ROOT)} already exists; refusing to overwrite")
    records_by_dataset = {name: load_jsonl(path) for name, path in DATASET_PATHS.items()}
    split = build_split(records_by_dataset, SEED) | {
        "version": VERSION,
        "sources": {name: str(path.relative_to(ROOT).as_posix()) for name, path in DATASET_PATHS.items()},
    }
    summary = validate_split(split, records_by_dataset)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")

    for dataset, records in records_by_dataset.items():
        trusted = trusted_records(records)
        multi = [g for g in query_groups(trusted) if len(g) > 1]
        print(f"{dataset}: total={len(records)} trusted={len(trusted)} excluded={len(records) - len(trusted)} "
              f"multi-sample groups={len(multi)} covering {sum(len(g) for g in multi)} samples")
        for name in SPLITS:
            labels = {label: n for (d, s, label), n in sorted(summary["labels"].items()) if d == dataset and s == name}
            print(f"  {name}: {summary['counts'][(dataset, name)]} {labels}")
    for name in SPLITS:
        labels = Counter()
        for (_, s, label), n in summary["labels"].items():
            if s == name:
                labels[label] += n
        print(f"all {name}: {sum(labels.values())} {dict(sorted(labels.items()))}")
    print(f"Validation passed; {summary['groups_checked']} multi-sample groups kept within one split.")

    # Where earlier-used IDs land; informational only (IDs, no records or results are read).
    v1 = json.loads(V1_SPLIT_PATH.read_text(encoding="utf-8"))["splits"]
    v2_of = {(d, i): s for d, by in split["splits"].items() for s, ids in by.items() for i in ids}
    moves = Counter((d, s1, v2_of.get((d, i), "absent")) for d, by in v1.items() for s1, ids in by.items() for i in ids)
    print("v1 split -> v2 split:", {f"{d}:{a}->{b}": n for (d, a, b), n in sorted(moves.items())})
    dev = json.loads(PROMPT_DEVELOPMENT_PATH.read_text(encoding="utf-8"))
    dev_ids = [(s["dataset"], s["sample_id"]) for s in dev["samples"]] if "samples" in dev else []
    print("probe prompt-development samples in v2:", Counter(v2_of.get(key, "absent") for key in dev_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
