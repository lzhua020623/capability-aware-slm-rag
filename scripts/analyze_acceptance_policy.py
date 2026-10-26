"""What would a different acceptance rule have done? Counterfactual, no new generation.

The end-to-end runs record both answers for every sample: the one the generator
gave on the original order and the one it gave on the reranked evidence. The
acceptance rule decides which of the two is reported. Because reranking is
deterministic, the kept evidence can be rebuilt locally, so alternative rules
can be scored on answers that already exist -- no API calls, no GPU generation.

Policies compared:

* `always_keep`   -- never replace. The floor: whatever the Reference RAG said.
* `always_accept` -- replace whenever a candidate exists. The ceiling of the
  reranking signal, with no safety rule at all.
* `grounded`      -- the shipped rule: accept unless the candidate is a refusal
  or is not grounded in the kept evidence.
* `protect_grounded_original` -- `grounded`, plus: if the original answer is
  itself grounded in the kept evidence, keep it. The intent is to stop the
  module from overwriting answers that the evidence already supported.

Correctness is normalised exact match against the gold answer, which is offline
supervision read after the answers existed.

Usage:
    python scripts/analyze_acceptance_policy.py \
        --results results/compensation/refinement_end_to_end_harm.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compensation.evidence_refinement import is_grounded, is_refusal  # noqa: E402
from src.compensation.rerank import CrossEncoderScorer, rerank_evidence  # noqa: E402
from src.compensation.refinement_schema import RefinementConfig  # noqa: E402
from src.diagnosis.schema import EvidenceItem  # noqa: E402
from src.evaluation.qa import exact_match  # noqa: E402

DEFAULT_RECORDS = ROOT / "data" / "processed" / "diagnosis" / "nq15_train_records.jsonl"
POLICIES = ("always_keep", "always_accept", "grounded", "protect_grounded_original")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def decide(policy: str, row: dict) -> str:
    """Which answer the policy reports for one sample."""
    initial, candidate = row["initial_answer"], row["candidate_answer"]
    if policy == "always_keep" or candidate is None:
        return initial
    if policy == "always_accept":
        return candidate
    if is_refusal(candidate) or not row["candidate_grounded"]:
        return initial
    if policy == "protect_grounded_original" and row["initial_grounded"]:
        return initial
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="an end-to-end result file saved with --save-rows")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--out", type=Path, default=None, help="default: <results>_policies.json")
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not rows:
        raise SystemExit(f"{args.results} has no rows; re-run the evaluation with --save-rows")
    config = RefinementConfig(**{key: value for key, value in payload["config"].items()})
    records = {row["sample_id"]: row for row in read_jsonl(args.records)}
    print(f"{len(rows)} rows | reranker {config.reranker_model} | keep_k {config.keep_k}")

    scorer = CrossEncoderScorer.from_pretrained(config.reranker_model)
    for index, row in enumerate(rows, start=1):
        record = records[row["sample_id"]]
        evidence = [
            EvidenceItem(rank=p["rank"], passage_id=p["passage_id"], text=p["text"], score=p["score"])
            for p in record["evidence"]
        ]
        kept = rerank_evidence(record["query"], evidence, scorer, config).evidence
        row["initial_grounded"] = is_grounded(row["initial_answer"], kept)
        row["candidate_grounded"] = bool(row["candidate_answer"]) and is_grounded(row["candidate_answer"], kept)
        if index % 200 == 0 or index == len(rows):
            print(f"  rebuilt {index}/{len(rows)}")

    total = len(rows)
    summary = {}
    for policy in POLICIES:
        correct = fixed = broken = changed = 0
        for row in rows:
            answer = decide(policy, row)
            score = exact_match(answer, row["gold_answer"])
            correct += score
            changed += answer != row["initial_answer"]
            fixed += score > row["em_before"]
            broken += score < row["em_before"]
        summary[policy] = {
            "em": round(correct / total, 4),
            "fixed": fixed,
            "broken": broken,
            "net_fixed": fixed - broken,
            "answers_changed": changed,
        }
        print(f"  {policy:28s} EM={summary[policy]['em']:.4f} fixed={fixed} broken={broken} net={fixed - broken:+d}")

    grounded_counts = {
        "initial_grounded": sum(row["initial_grounded"] for row in rows),
        "candidate_grounded": sum(row["candidate_grounded"] for row in rows),
        "both_grounded": sum(row["initial_grounded"] and row["candidate_grounded"] for row in rows),
    }
    out = args.out or args.results.with_name(args.results.stem + "_policies.json")
    out.write_text(
        json.dumps(
            {
                "note": "Counterfactual over answers that already existed; no new generation. "
                "EM is normalised exact match against the offline gold answer.",
                "source": str(args.results.relative_to(ROOT)) if args.results.is_relative_to(ROOT) else str(args.results),
                "label": payload.get("label"),
                "samples": total,
                "config": payload["config"],
                "grounding": grounded_counts,
                "policies": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
