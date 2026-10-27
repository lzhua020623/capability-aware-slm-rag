"""Can a score threshold drop distractors without dropping the answer?

Selecting a fixed `keep_k` hands the generator exactly five passages whether or
not five are relevant, so on a question with two good candidates the other
three are distractors the module chose to include. A threshold lets the count
vary per question, which is step 4 of the usual reranking recipe and the one
part of it this module has never used (`RefinementConfig.min_score` defaults to
`None`).

It is only worth enabling if the scores separate. This scores each sample's
candidate pool once, records every selected passage as (score, carries the gold
answer), and sweeps thresholds over the saved scores, so the sweep costs
nothing to repeat.

Reported per threshold: how much of the answer survives, how many distractors
go, how many passages are left per sample, and -- the number that matters --
how many samples end up with no answer-bearing passage at all, because those
can no longer be repaired by anything downstream.

Scores come from the cross-encoder as sentence-transformers returns them, which
for a single-output model is a sigmoid in [0, 1], not a raw logit.

Usage:
    python scripts/analyze_rerank_threshold.py --limit 400
    python scripts/analyze_rerank_threshold.py --scores-from results/compensation/rerank_threshold.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.qa import normalize_answer  # noqa: E402

DEFAULT_RECORDS = ROOT / "data" / "processed" / "diagnosis" / "nq15_train_records.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "processed" / "rerank" / "test.jsonl"
DEFAULT_OUT = ROOT / "results" / "compensation" / "rerank_threshold.json"
DEFAULT_THRESHOLDS = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
DEFAULT_RATIOS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def carries_answer(answer: str, text: str) -> bool:
    needle = normalize_answer(answer).split()
    if not needle:
        return False
    hay = normalize_answer(text).split()
    span = len(needle)
    return any(hay[start : start + span] == needle for start in range(len(hay) - span + 1))


def sweep(samples: list[dict], thresholds=DEFAULT_THRESHOLDS) -> list[dict]:
    """What each threshold keeps. `samples` is a list of [score, has_answer] lists."""
    rows = []
    total_passages = sum(len(sample) for sample in samples)
    positives = sum(1 for sample in samples for _, has in sample if has)
    negatives = total_passages - positives
    for threshold in thresholds:
        kept_pos = kept_neg = 0
        emptied = 0
        lost_answer = 0
        kept_counts = []
        for sample in samples:
            kept = [(score, has) for score, has in sample if score >= threshold]
            kept_counts.append(len(kept))
            kept_pos += sum(1 for _, has in kept if has)
            kept_neg += sum(1 for _, has in kept if not has)
            if not kept:
                emptied += 1
            had = any(has for _, has in sample)
            if had and not any(has for _, has in kept):
                lost_answer += 1
        rows.append(
            {
                "threshold": threshold,
                "answer_passages_kept": round(kept_pos / positives, 4) if positives else None,
                "distractors_kept": round(kept_neg / negatives, 4) if negatives else None,
                "mean_passages_per_sample": round(statistics.mean(kept_counts), 2) if kept_counts else 0.0,
                "samples_emptied": emptied,
                "samples_that_lost_their_only_answer": lost_answer,
            }
        )
    return rows


def sweep_relative(samples: list[dict], ratios=DEFAULT_RATIOS) -> list[dict]:
    """Keep a passage when its score is at least `ratio` of that sample's own best.

    An absolute cut has to work across questions whose score scales differ; a
    relative one adapts to each question, and because the top passage always
    clears its own ratio the context can never be emptied -- which an absolute
    threshold can do.
    """
    rows = []
    positives = sum(1 for sample in samples for _, has in sample if has)
    negatives = sum(1 for sample in samples for _, has in sample if not has)
    for ratio in ratios:
        kept_pos = kept_neg = lost_answer = 0
        kept_counts = []
        for sample in samples:
            if not sample:
                kept_counts.append(0)
                continue
            best = max(score for score, _ in sample)
            kept = [(score, has) for score, has in sample if score >= ratio * best]
            kept_counts.append(len(kept))
            kept_pos += sum(1 for _, has in kept if has)
            kept_neg += sum(1 for _, has in kept if not has)
            if any(has for _, has in sample) and not any(has for _, has in kept):
                lost_answer += 1
        rows.append(
            {
                "ratio": ratio,
                "answer_passages_kept": round(kept_pos / positives, 4) if positives else None,
                "distractors_kept": round(kept_neg / negatives, 4) if negatives else None,
                "mean_passages_per_sample": round(statistics.mean(kept_counts), 2) if kept_counts else 0.0,
                "samples_that_lost_their_only_answer": lost_answer,
            }
        )
    return rows


def collect(records: list[dict], provider, scorer, config, limit: int | None) -> list[dict]:
    from src.compensation.rerank import build_candidate_pool, rerank_evidence
    from src.diagnosis.schema import DatasetName, DiagnosisInput, EvidenceItem

    samples = []
    for index, record in enumerate(records, start=1):
        item = DiagnosisInput(
            sample_id=record["sample_id"],
            dataset=DatasetName.NQ,
            query=record["query"],
            evidence=[
                EvidenceItem(rank=p["rank"], passage_id=p["passage_id"], text=p["text"], score=p["score"])
                for p in record["evidence"]
            ],
            initial_answer=record["initial_answer"],
        )
        outcome = rerank_evidence(item.query, build_candidate_pool(item, provider), scorer, config)
        gold = str((record.get("annotation_reference") or {}).get("gold_answer") or "")
        samples.append([[float(score), carries_answer(gold, passage.text)] for passage, score in zip(outcome.evidence, outcome.scores)])
        if index % 100 == 0 or index == len(records):
            print(f"  scored {index}/{len(records)}")
        if limit and index >= limit:
            break
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scores-from", type=Path, default=None, help="re-sweep saved scores, no GPU")
    parser.add_argument("--model", default="models/reranker/accessibility_v1")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--keep-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()

    if args.scores_from:
        payload = json.loads(args.scores_from.read_text(encoding="utf-8"))
        samples = payload["samples"]
        meta = payload["config"]
    else:
        from src.compensation.candidates import IndexCandidateProvider
        from src.compensation.plugin import POOL_CORPUS, POOL_IDS, POOL_INDEX
        from src.compensation.refinement_schema import RefinementConfig
        from src.compensation.rerank import scorer_from_name

        wanted = {row["sample_id"] for row in read_jsonl(args.split)}
        records = [row for row in read_jsonl(args.records) if row["sample_id"] in wanted]
        config = RefinementConfig(keep_k=args.keep_k, reranker_model=args.model)
        provider = IndexCandidateProvider.from_frozen_artifacts(
            corpus_path=POOL_CORPUS, index_path=POOL_INDEX, ids_path=POOL_IDS, top_n=args.top_n
        )
        samples = collect(records, provider, scorer_from_name(args.model), config, args.limit)
        meta = {"model": args.model, "top_n": args.top_n, "keep_k": args.keep_k, "samples": len(samples)}

    flat = [score for sample in samples for score, _ in sample]
    pos = [score for sample in samples for score, has in sample if has]
    neg = [score for sample in samples for score, has in sample if not has]
    quant = lambda values, p: round(sorted(values)[max(0, int(p * len(values)) - 1)], 4) if values else None
    distribution = {
        "selected_passages": len(flat),
        "answer_bearing": len(pos),
        "answer_bearing_median": round(statistics.median(pos), 4) if pos else None,
        "answer_bearing_p10": quant(pos, 0.10),
        "answer_bearing_p25": quant(pos, 0.25),
        "distractor_median": round(statistics.median(neg), 4) if neg else None,
        "distractor_p75": quant(neg, 0.75),
        "distractor_p90": quant(neg, 0.90),
    }
    rows = sweep(samples)
    relative = sweep_relative(samples)

    payload = {
        "note": "Scores are sentence-transformers CrossEncoder outputs, a sigmoid in [0, 1] for a "
        "single-output model. The sweep is offline over saved scores.",
        "config": meta,
        "distribution": distribution,
        "sweep_absolute": rows,
        "sweep_relative": relative,
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(distribution, indent=2, ensure_ascii=False))
    width = max(len(str(row["threshold"])) for row in rows)
    print(f"\n{'thr':>{width}}  {'answer kept':>12}{'distractors kept':>18}{'passages/sample':>17}{'lost the answer':>17}")
    for row in rows:
        print(
            f"{row['threshold']:>{width}}  {row['answer_passages_kept']:>12.1%}{row['distractors_kept']:>18.1%}"
            f"{row['mean_passages_per_sample']:>17.2f}{row['samples_that_lost_their_only_answer']:>17}"
        )
    print()
    print(f"{'ratio':>6}  {'answer kept':>12}{'distractors kept':>18}{'passages/sample':>17}{'lost the answer':>17}")
    for row in relative:
        print(
            f"{row['ratio']:>6}  {row['answer_passages_kept']:>12.1%}{row['distractors_kept']:>18.1%}"
            f"{row['mean_passages_per_sample']:>17.2f}{row['samples_that_lost_their_only_answer']:>17}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
