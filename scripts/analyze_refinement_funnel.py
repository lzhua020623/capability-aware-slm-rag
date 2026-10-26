"""Why does reranking repair so few answers? A stage-by-stage funnel.

Reordering evidence can only repair an answer when a chain of conditions holds,
and each link loses samples:

1. an answer-bearing passage must exist in the frozen Top-5 at all — if the
   information is absent, no ordering can help (that is the retrieval owner's
   lane, not this one);
2. reranking must actually change what the generator sees — if rank 1 is the
   same passage as before, the context is nearly identical and the answer
   rarely moves;
3. the answer-bearing passage should end up at rank 1 — the intended effect;
4. and the generator must then use it — if it still fails on a clean,
   top-ranked passage, the failure was never about accessibility.

Each stage reports how many already-wrong answers were repaired inside it, so
the loss is attributable rather than guessed at. Gold answers are read only to
score, after the answers existed.

Usage:
    python scripts/analyze_refinement_funnel.py \
        --results results/compensation/refinement_end_to_end_accessibility.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compensation.rerank import CrossEncoderScorer, rerank_evidence  # noqa: E402
from src.compensation.refinement_schema import RefinementConfig  # noqa: E402
from src.diagnosis.schema import EvidenceItem  # noqa: E402
from src.evaluation.qa import normalize_answer  # noqa: E402

DEFAULT_RECORDS = ROOT / "data" / "processed" / "diagnosis" / "nq15_train_records.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def answer_bearing_ids(gold: str, evidence) -> set[str]:
    """Passages containing the gold answer as a contiguous normalised token run."""
    needle = normalize_answer(gold).split()
    if not needle:
        return set()
    span = len(needle)
    found = set()
    for passage in evidence:
        hay = normalize_answer(passage.text).split()
        if any(hay[start : start + span] == needle for start in range(len(hay) - span + 1)):
            found.add(passage.passage_id)
    return found


def classify(row: dict) -> dict:
    """Which funnel stage one sample reached, given its before/after evidence."""
    bearing = row["answer_bearing_ids"]
    before, after = row["before_ids"], row["after_ids"]
    return {
        "has_answer_passage": bool(bearing),
        "rank1_changed": bool(before) and bool(after) and before[0] != after[0],
        "answer_at_rank1_before": bool(before) and before[0] in bearing,
        "answer_at_rank1_after": bool(after) and after[0] in bearing,
    }


def rate(repaired: int, total: int) -> float | None:
    return round(repaired / total, 4) if total else None


def bucket(rows: list[dict], name: str) -> dict:
    wrong = [row for row in rows if row["em_before"] == 0]
    repaired = sum(row["em_after"] == 1 for row in wrong)
    return {
        "stage": name,
        "samples": len(rows),
        "already_wrong": len(wrong),
        "repaired": repaired,
        "repair_rate": rate(repaired, len(wrong)),
    }


def build_funnel(rows: list[dict]) -> list[dict]:
    no_passage = [row for row in rows if not row["has_answer_passage"]]
    has_passage = [row for row in rows if row["has_answer_passage"]]
    already_first = [row for row in has_passage if row["answer_at_rank1_before"]]
    held = [row for row in already_first if row["answer_at_rank1_after"]]
    demoted = [row for row in already_first if not row["answer_at_rank1_after"]]
    promotable = [row for row in has_passage if not row["answer_at_rank1_before"]]
    promoted = [row for row in promotable if row["answer_at_rank1_after"]]
    not_promoted = [row for row in promotable if not row["answer_at_rank1_after"]]
    return [
        bucket(rows, "all"),
        bucket(no_passage, "1. no answer-bearing passage in Top-5 (out of scope)"),
        bucket(already_first, "2. answer already at rank 1 before reranking (little to do)"),
        bucket(held, "2a.  ... and reranking kept it there"),
        bucket(demoted, "2b.  ... and reranking pushed it off the top"),
        bucket(promoted, "3. answer promoted to rank 1 by reranking (the intended effect)"),
        bucket(not_promoted, "4. answer present but reranking did not surface it"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="an end-to-end result file saved with --save-rows")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not rows:
        raise SystemExit(f"{args.results} has no rows; re-run the evaluation with --save-rows")
    config = RefinementConfig(**payload["config"])
    records = {record["sample_id"]: record for record in read_jsonl(args.records)}
    scorer = CrossEncoderScorer.from_pretrained(config.reranker_model)
    print(f"{len(rows)} rows | reranker {config.reranker_model}")

    for index, row in enumerate(rows, start=1):
        record = records[row["sample_id"]]
        evidence = [
            EvidenceItem(rank=p["rank"], passage_id=p["passage_id"], text=p["text"], score=p["score"])
            for p in record["evidence"]
        ]
        kept = rerank_evidence(record["query"], evidence, scorer, config).evidence
        row["answer_bearing_ids"] = answer_bearing_ids(row["gold_answer"], evidence)
        row["before_ids"] = [passage.passage_id for passage in evidence]
        row["after_ids"] = [passage.passage_id for passage in kept]
        row.update(classify(row))
        if index % 500 == 0 or index == len(rows):
            print(f"  rebuilt {index}/{len(rows)}")

    funnel = build_funnel(rows)
    width = max(len(stage["stage"]) for stage in funnel)
    for stage in funnel:
        print(
            f"  {stage['stage']:<{width}}  n={stage['samples']:5d}  wrong={stage['already_wrong']:5d}  "
            f"repaired={stage['repaired']:4d}  rate={stage['repair_rate']}"
        )
    moved = sum(row["rank1_changed"] for row in rows)
    print(f"  rank 1 changed on {moved}/{len(rows)} samples ({moved / len(rows):.1%})")

    out = args.out or args.results.with_name(args.results.stem + "_funnel.json")
    out.write_text(
        json.dumps(
            {
                "note": "Stage-by-stage account of where reranking can and cannot repair an answer. "
                "Gold answers are used only for scoring, after the answers existed.",
                "source": str(args.results.relative_to(ROOT)) if args.results.is_relative_to(ROOT) else str(args.results),
                "label": payload.get("label"),
                "config": payload["config"],
                "rank1_changed": moved,
                "funnel": funnel,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
