"""Is the bottleneck presenting the evidence, or picking the answer?

Every end-to-end arm leaves one answer per sample, and the arms share their
"before" answer, so several runs together are several perturbed attempts at the
same question. This script scores what could be had from those attempts:

* each arm on its own;
* majority vote over the arms, the cheap non-oracle way to combine them;
* an oracle that always picks the best available candidate -- not achievable,
  but it bounds what any selector could reach;
* how often at least one candidate is correct at all.

The gap between the achieved number and the oracle is headroom for a better
acceptance rule; the gap between the oracle and 1.0 is what no selector can fix
and has to come from retrieval or the generator.

Gold answers are read only to score, after every answer already existed.

Usage:
    python scripts/analyze_answer_selection.py \
        --arms trained=results/compensation/refinement_end_to_end_accessibility.json \
               shuffle=results/compensation/refinement_end_to_end_shuffle_control.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.qa import exact_match, normalize_answer  # noqa: E402


def load_arms(specs: list[str]) -> dict[str, dict[str, dict]]:
    arms = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--arms wants name=path entries, got {spec!r}")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = payload.get("rows")
        if not rows:
            raise SystemExit(f"{path} has no rows; re-run that evaluation with --save-rows")
        arms[name] = {row["sample_id"]: row for row in rows}
    return arms


def majority_vote(candidates: list[str], fallback: str) -> str:
    """The most repeated answer; the fallback when every candidate differs.

    Answers are counted after normalisation so spelling and punctuation do not
    split a majority, but the original string is returned.
    """
    if not candidates:
        return fallback
    counts = Counter(normalize_answer(candidate) for candidate in candidates)
    top, count = counts.most_common(1)[0]
    if count == 1:
        return fallback
    return next(candidate for candidate in candidates if normalize_answer(candidate) == top)


def best_candidate(candidates: list[str], gold: str) -> str:
    """Oracle: a correct candidate when one exists, else the first."""
    for candidate in candidates:
        if exact_match(candidate, gold):
            return candidate
    return candidates[0]


def score(answers: dict[str, str], gold: dict[str, str]) -> float:
    return round(sum(exact_match(answers[key], gold[key]) for key in answers) / max(len(answers), 1), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", required=True, help="name=path entries, saved with --save-rows")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "compensation" / "answer_selection.json")
    args = parser.parse_args()

    arms = load_arms(args.arms)
    ids = sorted(set.intersection(*(set(rows) for rows in arms.values())))
    if not ids:
        raise SystemExit("the arms share no sample ids")
    reference = next(iter(arms.values()))
    gold = {key: reference[key]["gold_answer"] for key in ids}
    before = {key: reference[key]["initial_answer"] for key in ids}
    for name, rows in arms.items():
        mismatched = [key for key in ids if rows[key]["initial_answer"] != before[key]]
        if mismatched:
            raise SystemExit(f"arm {name} has different before answers on {len(mismatched)} samples; not comparable")

    per_arm = {name: score({key: rows[key]["final_answer"] for key in ids}, gold) for name, rows in arms.items()}
    candidates = {key: [rows[key]["final_answer"] for rows in arms.values()] for key in ids}

    vote_arms = {key: majority_vote(candidates[key], before[key]) for key in ids}
    vote_with_before = {key: majority_vote(candidates[key] + [before[key]], before[key]) for key in ids}
    oracle_pair = {
        key: best_candidate([before[key], next(iter(arms.values()))[key]["final_answer"]], gold[key]) for key in ids
    }
    oracle_all = {key: best_candidate([before[key]] + candidates[key], gold[key]) for key in ids}
    any_correct = sum(any(exact_match(answer, gold[key]) for answer in [before[key]] + candidates[key]) for key in ids)

    summary = {
        "samples": len(ids),
        "baseline_before": score(before, gold),
        "per_arm": per_arm,
        "majority_vote_arms": score(vote_arms, gold),
        "majority_vote_with_before": score(vote_with_before, gold),
        "oracle_before_or_first_arm": score(oracle_pair, gold),
        "oracle_any_candidate": score(oracle_all, gold),
        "at_least_one_candidate_correct": any_correct,
        "at_least_one_candidate_correct_rate": round(any_correct / len(ids), 4),
    }
    payload = {
        "note": "Offline re-scoring of answers that already existed; no generation. The oracle rows are "
        "upper bounds for a selector, not achievable results.",
        "arms": {name: spec.split("=", 1)[1] for name, spec in zip(arms, args.arms)},
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
