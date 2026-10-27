"""Build training data for an answer selector: is this answer right, given this evidence?

The measured bottleneck is not presenting the evidence but choosing between
candidate answers: across the end-to-end arms at least one candidate is correct
for 23.9% of the held-out samples, while the shipped grounding gate delivers
~11%. A selector scores (query, evidence, answer) and the refiner keeps the
best-scoring candidate.

Two traps shape how the examples are built.

*Between-sample shortcut.* Almost every EM-correct stored answer belongs to a
`correct`-labelled sample and almost none to a failure-labelled one, so a
classifier trained on "sample -> label" could score the sample's *type* and
never look at the answer. Examples are therefore built **within a sample**: one
query and one evidence set carry both a right and a wrong answer, so anything
constant per sample cancels out, which is also exactly the comparison made at
inference time.

*Writing-style shortcut.* Gold answers are human-written and model answers are
not, so "looks like a gold string" would separate the classes on its own. Each
sample therefore contributes a gold answer as a positive **and** a gold answer
borrowed from another sample as a negative, putting the same style on both
sides; the model's own wrong answer is the second negative.

The current CLI partitions by the frozen team train/validation roles BEFORE
building examples or borrowing answers. Donors stay in their own role and are
recorded by ID. Old reserve/selection restrictions also apply; test is sealed.
The output is a checksum-bound bundle, not a new random split. These remain
weak supervision labels, not independently verified semantic gold.

Usage:
    python scripts/build_selector_training_data.py
    python scripts/build_selector_training_data.py --max-records 20000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_rerank_training_data import held_out_filter, tokens  # noqa: E402
from src.evaluation.qa import exact_match  # noqa: E402

DEFAULT_RECORDS = ROOT / "data" / "processed" / "external" / "nq_train_delivery_20261017"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "team_nq_v1" / "selector"
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gold_answer(record: dict) -> str:
    return str((record.get("annotation_reference") or {}).get("gold_answer") or "").strip()


def example(record: dict, answer: str, label: int, source: str) -> dict:
    return {
        "sample_id": record["sample_id"],
        "query": record["query"],
        "evidence": [passage["text"] for passage in record["evidence"]],
        "answer": answer,
        "label": label,
        "source": source,
        "primary_label": record["primary_label"],
    }


def build_examples(records: list[dict], seed: int = SEED) -> tuple[list[dict], Counter]:
    """Per record: the gold answer as a positive, plus wrong-answer negatives."""
    reasons: Counter[str] = Counter()
    rng = random.Random(seed)
    golds = [gold_answer(record) for record in records]
    pool = [(record["sample_id"], text) for record, text in zip(records, golds) if text]
    examples: list[dict] = []

    for record, gold in zip(records, golds):
        if not gold:
            reasons["skipped_no_gold"] += 1
            continue
        reasons["records"] += 1
        stored = str(record["initial_answer"]).strip()

        # Positive: the model's own answer when it is right, else the gold string.
        if stored and exact_match(stored, gold):
            examples.append(example(record, stored, 1, "model_answer_correct"))
            reasons["positive_model_answer"] += 1
        else:
            examples.append(example(record, gold, 1, "gold_answer"))
            reasons["positive_gold"] += 1
            if stored:
                examples.append(example(record, stored, 0, "model_answer_wrong"))
                reasons["negative_model_answer"] += 1

        # Hard negative in the positives' own writing style: another sample's gold
        # answer, unless it happens to be right here too.
        for _ in range(8):
            donor_id, borrowed = rng.choice(pool)
            if borrowed != gold and not exact_match(borrowed, gold):
                borrowed_example = example(record, borrowed, 0, "borrowed_gold")
                borrowed_example["donor_sample_id"] = donor_id
                examples.append(borrowed_example)
                reasons["negative_borrowed_gold"] += 1
                break
        else:
            reasons["skipped_no_borrowed_negative"] += 1

    return examples, reasons


def main(argv=None) -> None:
    from src.data.team_nq import TeamNQBoundary, prepare_training_bundle
    parser = argparse.ArgumentParser(description="Build selector data separately within team train/validation")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-records", type=int, default=None, help="cap per role for engineering smoke")
    args = parser.parse_args(argv)
    if args.max_records is not None and args.max_records < 1:
        parser.error("--max-records must be positive")
    boundary = TeamNQBoundary(ROOT)
    def build(rows):
        return build_examples(rows[:args.max_records] if args.max_records is not None else rows)
    report = prepare_training_bundle(boundary, args.records, args.out_dir, build, component="answer_selector")
    print(json.dumps({"boundary": report["boundary"], "role_examples":
        {r: spec["examples"] for r, spec in report["role_files"].items()}}, indent=2))


if __name__ == "__main__":
    main()
