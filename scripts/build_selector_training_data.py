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

Leak control matches the reranker's: nothing from the accessibility validation
or test split, and nothing whose question is close to a frozen Dev/Test
question.

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

DEFAULT_RECORDS = ROOT / "data" / "processed" / "diagnosis" / "nq15_train_records.jsonl"
DEFAULT_SPLIT_DIR = ROOT / "data" / "processed" / "rerank"
DEFAULT_HELD_OUT = ROOT / "data" / "annotations" / "normalized" / "nq_3216_diagnosis_inputs.jsonl"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "selector"
DEFAULT_STATS = ROOT / "results" / "compensation" / "selector_training_data_stats.json"
SEED = 42
VALIDATION_FRACTION = 0.1


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
    pool = [text for text in golds if text]
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
            borrowed = rng.choice(pool)
            if borrowed != gold and not exact_match(borrowed, gold):
                examples.append(example(record, borrowed, 0, "borrowed_gold"))
                reasons["negative_borrowed_gold"] += 1
                break
        else:
            reasons["skipped_no_borrowed_negative"] += 1

    return examples, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--held-out", type=Path, default=DEFAULT_HELD_OUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--max-records", type=int, default=None, help="cap the records used (smoke runs)")
    parser.add_argument("--jaccard", type=float, default=0.6)
    args = parser.parse_args()

    records = read_jsonl(args.records)
    print(f"{len(records)} annotated records")

    evaluated = set()
    for split in ("validation", "test"):
        path = args.split_dir / f"{split}.jsonl"
        if path.exists():
            evaluated |= {row["sample_id"] for row in read_jsonl(path)}
    records = [record for record in records if record["sample_id"] not in evaluated]
    print(f"{len(records)} left after removing the {len(evaluated)} accessibility validation/test samples")

    too_close, held_out_size = held_out_filter(args.held_out, args.jaccard)
    records = [record for record in records if not too_close(tokens(record["query"]))]
    print(f"{len(records)} left after removing questions close to the {held_out_size} frozen Dev/Test ones")
    if args.max_records:
        records = records[: args.max_records]

    examples, reasons = build_examples(records)
    rng = random.Random(SEED)
    by_sample: dict[str, list[dict]] = {}
    for item in examples:
        by_sample.setdefault(item["sample_id"], []).append(item)
    sample_ids = sorted(by_sample)
    rng.shuffle(sample_ids)
    cut = int(len(sample_ids) * (1 - VALIDATION_FRACTION))
    splits = {
        "train": [item for sample_id in sample_ids[:cut] for item in by_sample[sample_id]],
        "validation": [item for sample_id in sample_ids[cut:] for item in by_sample[sample_id]],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rows in splits.items():
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        positives = sum(row["label"] for row in rows)
        counts[name] = {
            "examples": len(rows),
            "positives": positives,
            "negatives": len(rows) - positives,
            "samples": len({row["sample_id"] for row in rows}),
            "path": str(path.relative_to(ROOT)),
        }
        print(f"  {name:11s} {len(rows):7d} examples ({positives} positive) over {counts[name]['samples']} samples")

    stats = {
        "source_records": str(args.records.relative_to(ROOT)),
        "records_used": reasons["records"],
        "examples": len(examples),
        "example_sources": {key: value for key, value in reasons.items() if key.startswith(("positive", "negative"))},
        "skipped": {key: value for key, value in reasons.items() if key.startswith("skipped")},
        "splits": counts,
        "seed": SEED,
        "note": "Examples are built within a sample and put gold-style strings on both sides, so neither "
        "the sample's annotation label nor the answer's writing style separates the classes.",
    }
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("examples", "example_sources", "skipped")}, indent=2, ensure_ascii=False))
    print(f"wrote {args.stats_out}")


if __name__ == "__main__":
    main()
