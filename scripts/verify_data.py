"""Read-only validation of downloaded Natural Questions and FEVER data."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
NQ_DEV_DIR = ROOT / "data" / "raw" / "natural_questions" / "dev"
FEVER_TRAIN = ROOT / "data" / "raw" / "fever" / "train.jsonl"
FEVER_DEV = ROOT / "data" / "raw" / "fever" / "shared_task_dev.jsonl"

NQ_REQUIRED_FIELDS = ("question", "document", "annotations", "long_answer_candidates")
FEVER_REQUIRED_FIELDS = ("claim", "label", "evidence")
FEVER_LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


def sequence_len(value) -> int:
    if isinstance(value, dict):
        if not value:
            return 0
        first = next(iter(value.values()))
        try:
            return len(first)
        except TypeError:
            return 0
    try:
        return len(value)
    except TypeError:
        return 0


def question_text(question) -> str:
    if isinstance(question, dict) and "text" in question:
        return str(question["text"])
    return str(question)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def check_nq() -> bool:
    print("=== Natural Questions ===")
    if not NQ_DEV_DIR.exists():
        print(f"ERROR: missing {NQ_DEV_DIR}")
        return False

    dataset = load_from_disk(str(NQ_DEV_DIR))
    count = len(dataset)
    first = dataset[0]
    missing = [field for field in NQ_REQUIRED_FIELDS if field not in first]
    candidates = first.get("long_answer_candidates")

    print(f"样本数量: {count}")
    print(f"第一条 question: {question_text(first.get('question'))}")
    print(f"第一条 annotations: {first.get('annotations')}")
    print(f"第一条 long_answer_candidates 数量: {sequence_len(candidates)}")

    if missing:
        print(f"ERROR: missing fields: {', '.join(missing)}")
        return False

    print("字段检查通过: question, document, annotations, long_answer_candidates")
    return True


def check_fever() -> bool:
    print("=== FEVER ===")
    ok = True
    for path in (FEVER_TRAIN, FEVER_DEV):
        if not path.exists():
            print(f"ERROR: missing {path}")
            ok = False
    if not ok:
        return False

    train = load_jsonl(FEVER_TRAIN)
    dev = load_jsonl(FEVER_DEV)
    print(f"train 数量: {len(train)}")
    print(f"dev 数量: {len(dev)}")

    if not train:
        print("ERROR: FEVER train is empty")
        return False

    first = train[0]
    print(f"第一条 claim: {first.get('claim')}")
    print(f"第一条 label: {first.get('label')}")
    print(f"第一条 evidence: {first.get('evidence')}")

    missing = [field for field in FEVER_REQUIRED_FIELDS if field not in first]
    if missing:
        print(f"ERROR: first train sample missing fields: {', '.join(missing)}")
        ok = False

    label_counts = Counter(record.get("label") for record in train)
    print("FEVER train label counts:")
    for label in FEVER_LABELS:
        print(f"  {label}: {label_counts.get(label, 0)}")

    if not dev:
        print("ERROR: FEVER dev is empty")
        ok = False
    return ok


def main() -> int:
    nq_ok = False
    fever_ok = False
    try:
        nq_ok = check_nq()
    except Exception as exc:
        print(f"ERROR: Natural Questions check failed: {type(exc).__name__}: {exc}")
    print()
    try:
        fever_ok = check_fever()
    except Exception as exc:
        print(f"ERROR: FEVER check failed: {type(exc).__name__}: {exc}")

    print()
    if nq_ok and fever_ok:
        print("DATA VALIDATION PASSED")
        return 0
    print("DATA VALIDATION FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
