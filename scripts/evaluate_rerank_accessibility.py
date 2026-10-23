"""Offline check of evidence reranking on the annotated C1 Reference RAG records.

This is an **offline evaluation**: it reads gold answers and the annotation
labels, which the runtime refiner may never see. It answers one question only:
after reranking the frozen Top-5, is an answer-bearing passage closer to the
top (and still kept) than it was before? It does not load Qwen and therefore
says nothing about the final answer; that needs a generator run on a machine
that holds the model and the NQ index.

Groups are the annotation labels, so the `correct` group doubles as the harm
check: pruning must not throw away the answer-bearing passage of samples that
were already right.

Usage:
    python scripts/evaluate_rerank_accessibility.py --keep-k 3
    python scripts/evaluate_rerank_accessibility.py --labels evidence_accessibility --out results/compensation/acc.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compensation.rerank import CrossEncoderScorer, rerank_evidence  # noqa: E402
from src.compensation.schema import DEFAULT_REFINEMENT_CONFIG, RefinementConfig  # noqa: E402
from src.diagnosis.adapter import c1_record_to_input  # noqa: E402
from src.diagnosis.schema import DiagnosisInput, EvidenceItem  # noqa: E402
from src.evaluation.qa import normalize_answer  # noqa: E402

DEFAULT_LABELS = ROOT / "data/annotations/normalized/nq_3216_labels_normalized.jsonl"
DEFAULT_C1 = ROOT / "results/preliminary/nq_c1_7b.jsonl"
DEFAULT_OUT = ROOT / "results/compensation/rerank_accessibility_offline.json"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def answer_bearing_ranks(gold_answer: str, evidence: tuple[EvidenceItem, ...]) -> list[int]:
    """Ranks of the passages that contain the gold answer as a contiguous token run (offline only)."""
    tokens = normalize_answer(gold_answer).split()
    if not tokens:
        return []
    span = len(tokens)
    ranks = []
    for passage in evidence:
        passage_tokens = normalize_answer(passage.text).split()
        if any(passage_tokens[start : start + span] == tokens for start in range(len(passage_tokens) - span + 1)):
            ranks.append(passage.rank)
    return ranks


def summarise(rows: list[dict]) -> dict:
    with_answer = [row for row in rows if row["best_rank_before"] is not None]
    kept = [row for row in with_answer if row["best_rank_after"] is not None]
    return {
        "n": len(rows),
        "with_answer_bearing_passage_in_top5": len(with_answer),
        "answer_bearing_at_rank1_before": sum(row["best_rank_before"] == 1 for row in with_answer),
        "answer_bearing_at_rank1_after": sum(row["best_rank_after"] == 1 for row in kept),
        "mean_best_rank_before": round(statistics.mean([row["best_rank_before"] for row in with_answer]), 4)
        if with_answer
        else None,
        "mean_best_rank_after": round(statistics.mean([row["best_rank_after"] for row in kept]), 4) if kept else None,
        "answer_bearing_passage_kept": len(kept),
        "answer_bearing_passage_lost_by_pruning": len(with_answer) - len(kept),
        "promoted": sum(
            row["best_rank_after"] is not None and row["best_rank_after"] < row["best_rank_before"] for row in with_answer
        ),
        "demoted": sum(
            row["best_rank_after"] is not None and row["best_rank_after"] > row["best_rank_before"] for row in with_answer
        ),
        "order_changed": sum(row["order_changed"] for row in rows),
    }


def evaluate(
    records: list[dict],
    labels: dict[str, dict],
    scorer: CrossEncoderScorer,
    config: RefinementConfig,
    wanted_labels: set[str] | None,
    trusted_only: bool,
    limit: int | None,
) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    for record in records:
        sample_id = str(record["sample_id"])
        annotation = labels.get(sample_id)
        if annotation is None:
            continue
        label = annotation["primary_label"]
        if wanted_labels and label not in wanted_labels:
            continue
        if trusted_only and annotation.get("needs_manual_review"):
            continue
        item: DiagnosisInput = c1_record_to_input(record)
        if not item.evidence:
            continue
        before = tuple(item.evidence)
        outcome = rerank_evidence(item.query, before, scorer, config)
        gold = str(record["gold_answer_or_label"])  # offline-only field
        ranks_before = answer_bearing_ranks(gold, before)
        answer_bearing_ids = {passage.passage_id for passage in before if passage.rank in ranks_before}
        kept_ids = {passage.passage_id for passage in outcome.evidence}
        ranks_after = [passage.rank for passage in outcome.evidence if passage.passage_id in answer_bearing_ids]
        rows.append(
            {
                "sample_id": sample_id,
                "primary_label": label,
                "needs_manual_review": bool(annotation.get("needs_manual_review")),
                "best_rank_before": min(ranks_before) if ranks_before else None,
                "best_rank_after": min(ranks_after) if ranks_after else None,
                "order_changed": tuple(passage.passage_id for passage in outcome.evidence)
                != tuple(passage.passage_id for passage in before)[: len(outcome.evidence)],
                "kept_passage_ids": [passage.passage_id for passage in outcome.evidence],
                "dropped_passage_ids": list(outcome.dropped_passage_ids),
                "rerank_scores": [round(score, 4) for score in outcome.scores],
                "answer_bearing_kept": bool(ranks_after),
                "answer_bearing_dropped_ids": sorted(answer_bearing_ids - kept_ids),
            }
        )
        if limit is not None and len(rows) >= limit:
            break

    by_label: dict[str, dict] = {}
    for label in sorted({row["primary_label"] for row in rows}):
        by_label[label] = summarise([row for row in rows if row["primary_label"] == label])
    return {"all": summarise(rows), "by_label": by_label}, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--c1", type=Path, default=DEFAULT_C1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--keep-k", type=int, default=DEFAULT_REFINEMENT_CONFIG.keep_k)
    parser.add_argument("--model", default=DEFAULT_REFINEMENT_CONFIG.reranker_model)
    parser.add_argument("--labels", nargs="*", default=None, help="only these primary labels")
    parser.add_argument("--trusted-only", action="store_true", help="skip needs_manual_review annotations")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-rows", action="store_true", help="write only the summary")
    args = parser.parse_args()

    config = RefinementConfig(keep_k=args.keep_k, reranker_model=args.model)
    labels = {str(row["sample_id"]): row for row in read_jsonl(args.labels_file)}
    records = read_jsonl(args.c1)
    print(f"Loading reranker {config.reranker_model}")
    scorer = CrossEncoderScorer.from_pretrained(config.reranker_model)

    start = time.perf_counter()
    summary, rows = evaluate(
        records,
        labels,
        scorer,
        config,
        set(args.labels) if args.labels else None,
        args.trusted_only,
        args.limit,
    )
    elapsed = time.perf_counter() - start

    payload = {
        "note": "Offline evaluation. Gold answers and annotation labels are used here only to score "
        "reranking after the fact; the runtime refiner never sees them. No generator was run, so this "
        "says nothing about final answer accuracy.",
        "config": config.to_dict(),
        "inputs": {
            "c1_records": str(args.c1.relative_to(ROOT)) if args.c1.is_relative_to(ROOT) else str(args.c1),
            "labels": str(args.labels_file.relative_to(ROOT)) if args.labels_file.is_relative_to(ROOT) else str(args.labels_file),
            "scored_samples": summary["all"]["n"],
            "trusted_only": args.trusted_only,
            "label_filter": sorted(args.labels) if args.labels else None,
        },
        "metric_definitions": {
            "answer_bearing_passage": "a passage containing the gold answer as a contiguous normalised token run",
            "best_rank_before": "best answer-bearing rank in the frozen Top-5",
            "best_rank_after": "best answer-bearing rank after reranking and pruning, None if none survived",
            "answer_bearing_passage_lost_by_pruning": "had one in the Top-5, none left after pruning (harm)",
        },
        "elapsed_seconds": round(elapsed, 2),
        "summary": summary,
    }
    if not args.no_rows:
        payload["rows"] = rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
