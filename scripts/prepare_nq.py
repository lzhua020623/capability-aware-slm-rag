"""Build a controlled single-passage Natural Questions subset.

Reads data/raw/natural_questions/dev and writes processed JSONL + stats.
Does not modify the original dataset and does not run retrieval.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
NQ_DEV_DIR = ROOT / "data" / "raw" / "natural_questions" / "dev"
OUT_DIR = ROOT / "data" / "processed" / "nq_controlled"
OUT_JSONL = OUT_DIR / "dev.jsonl"
OUT_STATS = OUT_DIR / "stats.json"

SEED = 42
N_PREVIEW = 5

STAGE_NAMES = [
    "original samples",
    "has short answer",
    "exactly one short-answer span",
    "non-yes/no",
    "short answer inside gold long answer",
    "paragraph long answer",
]

OPEN_P = re.compile(r"<P>", re.IGNORECASE)
CLOSE_P = re.compile(r"</P>", re.IGNORECASE)


def is_none_yes_no(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().upper() in {"NONE", "-1"}
    return int(value) == -1


def short_spans(short_answers) -> list[dict]:
    if isinstance(short_answers, dict) and "text" in short_answers:
        n = len(short_answers["text"])
        return [
            {
                "start_token": int(short_answers["start_token"][i]),
                "end_token": int(short_answers["end_token"][i]),
                "start_byte": int(short_answers["start_byte"][i]),
                "end_byte": int(short_answers["end_byte"][i]),
                "text": short_answers["text"][i],
            }
            for i in range(n)
        ]
    if isinstance(short_answers, list):
        return list(short_answers)
    raise TypeError(f"Unexpected short_answers type: {type(short_answers)}")


def iter_annotations(example: dict):
    annotations = example["annotations"]
    if isinstance(annotations, dict) and "id" in annotations:
        n = len(annotations["id"])
        for i in range(n):
            yield {
                "id": annotations["id"][i],
                "long_answer": annotations["long_answer"][i],
                "short_answers": annotations["short_answers"][i],
                "yes_no_answer": annotations["yes_no_answer"][i],
            }
        return
    if isinstance(annotations, list):
        yield from annotations
        return
    raise TypeError(f"Unexpected annotations type: {type(annotations)}")


def valid_short_spans(spans: list[dict]) -> list[dict]:
    kept = []
    for span in spans:
        text = (span.get("text") or "").strip()
        start = int(span["start_token"])
        end = int(span["end_token"])
        if text and start >= 0 and end > start:
            kept.append(span)
    return kept


def extract_text(tokens: list[str], is_html: list[bool], start: int, end: int) -> str:
    pieces = [
        token
        for token, html in zip(tokens[start:end], is_html[start:end])
        if not html
    ]
    return " ".join(pieces).strip()


def is_paragraph_long_answer(
    tokens: list[str], is_html: list[bool], start: int, end: int
) -> bool:
    if start < 0 or end > len(tokens) or end <= start:
        return False
    if not is_html[start] or not is_html[end - 1]:
        return False
    return bool(OPEN_P.fullmatch(tokens[start]) and CLOSE_P.fullmatch(tokens[end - 1]))


def question_text(question) -> str:
    if isinstance(question, dict) and "text" in question:
        return question["text"]
    return str(question)


def evaluate_annotation(example: dict, annotation: dict) -> tuple[int, dict | None]:
    tokens = example["document"]["tokens"]["token"]
    is_html = example["document"]["tokens"]["is_html"]
    spans = valid_short_spans(short_spans(annotation["short_answers"]))
    if not spans:
        return 0, None
    if len(spans) != 1:
        return 1, None
    if not is_none_yes_no(annotation["yes_no_answer"]):
        return 2, None

    long_answer = annotation["long_answer"]
    sa = spans[0]
    la_start = int(long_answer["start_token"])
    la_end = int(long_answer["end_token"])
    sa_start = int(sa["start_token"])
    sa_end = int(sa["end_token"])
    if la_start < 0 or la_end <= la_start:
        return 3, None
    if not (la_start <= sa_start and sa_end <= la_end):
        return 3, None
    if not is_paragraph_long_answer(tokens, is_html, la_start, la_end):
        return 4, None

    short_text = (sa.get("text") or "").strip() or extract_text(
        tokens, is_html, sa_start, sa_end
    )
    record = {
        "id": example["id"],
        "question": question_text(example["question"]),
        "short_answer": short_text,
        "gold_long_answer_text": extract_text(tokens, is_html, la_start, la_end),
        "document_title": example["document"]["title"],
        "document_url": example["document"]["url"],
        "short_answer_start_token": sa_start,
        "short_answer_end_token": sa_end,
        "long_answer_start_token": la_start,
        "long_answer_end_token": la_end,
        "annotation_id": annotation["id"],
        "long_answer_candidate_index": int(long_answer["candidate_index"]),
    }
    return 5, record


def process_example(example: dict) -> tuple[int, dict | None]:
    best_stage = 0
    chosen = None
    for annotation in iter_annotations(example):
        stage, record = evaluate_annotation(example, annotation)
        if stage > best_stage:
            best_stage = stage
            chosen = record
        elif stage == best_stage and chosen is None and record is not None:
            chosen = record
        if best_stage == 5 and chosen is not None:
            break
    return best_stage, chosen


def main() -> int:
    if not NQ_DEV_DIR.exists():
        print(f"ERROR: missing {NQ_DEV_DIR}")
        return 1

    dataset = load_from_disk(str(NQ_DEV_DIR))
    stage_counts = [0] * len(STAGE_NAMES)
    kept: list[dict] = []

    print(f"Reading {NQ_DEV_DIR}")
    for example in dataset:
        stage_counts[0] += 1
        best_stage, record = process_example(example)
        for level in range(1, best_stage + 1):
            stage_counts[level] += 1
        if best_stage == 5 and record is not None:
            kept.append(record)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = {
        "source": str(NQ_DEV_DIR),
        "output": str(OUT_JSONL),
        "seed": SEED,
        "original samples": stage_counts[0],
        "has short answer": stage_counts[1],
        "exactly one short-answer span": stage_counts[2],
        "non-yes/no": stage_counts[3],
        "short answer inside gold long answer": stage_counts[4],
        "paragraph long answer": stage_counts[5],
        "final samples": len(kept),
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Filter statistics:")
    for name in STAGE_NAMES:
        print(f"  {name}: {stats[name]}")
    print(f"  final samples: {stats['final samples']}")
    print(f"Wrote {OUT_JSONL}")
    print(f"Wrote {OUT_STATS}")

    if kept:
        rng = random.Random(SEED)
        preview = rng.sample(kept, k=min(N_PREVIEW, len(kept)))
        print()
        print(f"Random preview ({len(preview)} samples, seed={SEED}):")
        for i, record in enumerate(preview, start=1):
            print(f"\n[{i}] id={record['id']}")
            print(f"  question: {record['question']}")
            print(f"  short_answer: {record['short_answer']}")
            print(f"  gold paragraph: {record['gold_long_answer_text']}")
    else:
        print("No samples retained.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
