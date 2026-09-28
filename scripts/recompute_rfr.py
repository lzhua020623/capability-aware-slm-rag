"""Recompute the Recoverable Failure Rate using judge verdicts instead of exact match.

    RFR = |C1 wrong AND C3 correct| / |C1 wrong|

Both terms come from the same correctness rule, so swapping exact match for
judge verdicts has to be done on both -- judging only the persistent-failure set
shrinks the numerator's complement without touching the denominator and yields a
rate that is not comparable to anything.

Correctness per sample is the judge verdict where one exists, falling back to the
frozen exact-match result where the judge was not run (exact match never marks a
wrong answer correct, so unjudged samples keep their original label).

Reports the frozen and recomputed rates side by side; the frozen number stays the
headline unless the team agrees to change the metric.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson score interval -- honest at the small n these subsets produce."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def rate_line(label: str, num: int, den: int) -> str:
    if den == 0:
        return f"  {label:28s}      n/a (empty denominator)"
    lo, hi = wilson(num, den)
    return f"  {label:28s} {num:4d}/{den:4d} = {num / den:6.1%}   95% CI [{lo:.1%}, {hi:.1%}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--c1", type=Path, required=True)
    parser.add_argument("--c3", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        help="Also write the judge-corrected recoverable and persistent-failure "
        "cohorts here, mirroring the frozen cohort schema with a _judged suffix. "
        "The frozen cohort files are never touched.",
    )
    args = parser.parse_args()

    c1 = {str(r["sample_id"]): r for r in load_jsonl(args.c1)}
    c3 = {str(r["sample_id"]): r for r in load_jsonl(args.c3)}
    verdicts = load_jsonl(args.verdicts)

    judged: dict[tuple[str, str], bool] = {}
    unparsed = 0
    for v in verdicts:
        if v["judge_verdict"] == "UNPARSED":
            unparsed += 1
            continue
        judged[(v["condition"], str(v["sample_id"]))] = v["judge_verdict"] == "CORRECT"

    shared = sorted(set(c1) & set(c3))
    if len(shared) != len(c1) or len(shared) != len(c3):
        print(f"WARNING: C1 has {len(c1)}, C3 has {len(c3)}, {len(shared)} in both\n")

    def correct(condition: str, sid: str, frozen: dict) -> bool:
        return judged.get((condition, sid), bool(frozen[sid]["correct"]))

    frozen_c1w = [s for s in shared if not c1[s]["correct"]]
    frozen_num = sum(1 for s in frozen_c1w if c3[s]["correct"])
    judged_c1w = [s for s in shared if not correct("C1", s, c1)]
    judged_num = sum(1 for s in judged_c1w if correct("C3", s, c3))

    flips_c1 = sum(1 for s in shared if not c1[s]["correct"] and correct("C1", s, c1))
    flips_c3 = sum(1 for s in shared if not c3[s]["correct"] and correct("C3", s, c3))

    print(f"paired samples: {len(shared)}")
    print(f"judge verdicts applied: {len(judged)}" + (f"  ({unparsed} unparsed, ignored)" if unparsed else ""))
    print(f"exact-match-wrong answers the judge accepted:  C1 {flips_c1}   C3 {flips_c3}\n")

    print("Recoverable Failure Rate")
    print(rate_line("frozen (exact match)", frozen_num, len(frozen_c1w)))
    print(rate_line("judge-corrected", judged_num, len(judged_c1w)))

    print("\nCondition accuracy")
    for name, records, condition in (("C1", c1, "C1"), ("C3", c3, "C3")):
        frozen_acc = sum(1 for s in shared if records[s]["correct"])
        judged_acc = sum(1 for s in shared if correct(condition, s, records))
        print(f"  {name}: exact match {frozen_acc / len(shared):6.1%}   judge {judged_acc / len(shared):6.1%}")

    persistent = [s for s in judged_c1w if not correct("C3", s, c3)]
    recoverable = [s for s in judged_c1w if correct("C3", s, c3)]
    print("\nPersistent failures (C1 wrong AND C3 wrong)")
    print(f"  frozen : {sum(1 for s in frozen_c1w if not c3[s]['correct'])}")
    print(f"  judged : {len(persistent)}    <- the real target set for a larger-model control")

    if args.cohort_dir:
        args.cohort_dir.mkdir(parents=True, exist_ok=True)
        for name, ids, outcome in (
            ("nq_c1_c3_recoverable_judged", recoverable, "c1_wrong_c3_correct"),
            ("nq_c1_c3_persistent_failures_judged", persistent, "c1_wrong_c3_wrong"),
        ):
            path = args.cohort_dir / f"{name}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for sid in ids:
                    handle.write(
                        json.dumps(
                            {
                                "sample_id": sid,
                                "dataset": c3[sid].get("dataset"),
                                "outcome": outcome,
                                "correctness_rule": "llm_judge",
                                "model_name": c3[sid].get("model_name"),
                                "question_or_claim": c3[sid].get("question_or_claim"),
                                "gold_answer_or_label": c3[sid].get("gold_answer_or_label"),
                                "oracle_evidence": c3[sid].get("evidence"),
                                "c1_prediction": c1[sid].get("prediction"),
                                "c1_correct": correct("C1", sid, c1),
                                "c1_exact_match": bool(c1[sid]["correct"]),
                                "c1_f1": c1[sid].get("f1"),
                                "c3_prediction": c3[sid].get("prediction"),
                                "c3_correct": correct("C3", sid, c3),
                                "c3_exact_match": bool(c3[sid]["correct"]),
                                "c3_f1": c3[sid].get("f1"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            print(f"Wrote {path}  ({len(ids)} samples)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "paired_samples": len(shared),
            "judge_verdicts_applied": len(judged),
            "unparsed_verdicts": unparsed,
            "frozen": {
                "c1_wrong": len(frozen_c1w),
                "c1_wrong_c3_correct": frozen_num,
                "recoverable_failure_rate": frozen_num / len(frozen_c1w) if frozen_c1w else 0.0,
            },
            "judge_corrected": {
                "c1_wrong": len(judged_c1w),
                "c1_wrong_c3_correct": judged_num,
                "recoverable_failure_rate": judged_num / len(judged_c1w) if judged_c1w else 0.0,
                "persistent_failures": len(persistent),
                "persistent_failure_ids": persistent,
            },
        }
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
