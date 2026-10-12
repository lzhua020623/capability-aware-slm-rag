"""Train and select the hierarchical Diagnosis v2-expanded classifier on TRAIN + VALIDATION only.

Same features, scaling, class weighting, architecture, candidate grids and
selection rules as scripts/train_diagnosis_hierarchical.py; only the split
and datasets change. The test split is never loaded. The v1 hierarchical
validation summary is read for a trend comparison only (different
validation sets, so no paired comparison is possible). No v1 file is written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_diagnosis_v2_expanded_split import DATASET_PATHS, OUTPUT_PATH as SPLIT_PATH
from scripts.train_diagnosis_hierarchical import load_records, train_and_evaluate

V1_HIERARCHICAL_PATH = ROOT / "results" / "diagnosis" / "diagnosis_hierarchical_validation.json"
OUTPUT_PATH = ROOT / "results" / "diagnosis" / "diagnosis_v2_expanded_validation.json"
MODEL_PATH = ROOT / "models" / "diagnosis" / "diagnosis_v2_expanded_hierarchical.json"
NOTE = (
    "Diagnosis v2-expanded (NQ 2698 trusted + FEVER 473 trusted). Validation-only; the test split was not "
    "loaded. FEVER has no trusted evidence_accessibility samples (not computable). The v1-small comparison "
    "uses a different, smaller validation set and is a trend comparison only."
)


def v1_small_summary() -> dict:
    data = json.loads(V1_HIERARCHICAL_PATH.read_text(encoding="utf-8"))
    return data["comparison"]["hierarchical"] | {
        "configuration": "v1-small hierarchical (different validation set)",
        "validation_n": data["split_sizes"]["validation"],
    }


def main() -> int:
    for path in (OUTPUT_PATH, MODEL_PATH):
        if path.exists():
            raise FileExistsError(f"{path.relative_to(ROOT)} already exists; refusing to overwrite")
    splits = load_records(SPLIT_PATH, DATASET_PATHS)
    train_and_evaluate(splits, OUTPUT_PATH, MODEL_PATH, {"v1_small_hierarchical": v1_small_summary()}, NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
