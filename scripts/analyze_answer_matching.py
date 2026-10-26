"""How much of a "failure" is the matching rule rather than the answer?

The frozen NQ metric is normalised exact match against one gold string, and the
frozen prompt asks for "only the shortest answer span". NQ gold strings are
often whole clauses, so a model that obeys the instruction can answer correctly
and still score zero. Published error analyses report the same thing: extraction
failures are frequently aliasing or normalisation artifacts rather than failures
to find the answer (arXiv:2603.09341).

This script re-scores a finished end-to-end run under progressively laxer
matching rules and reports how many of the "wrong" answers merely differ in span
length from the gold string. It changes no protocol: the frozen exact match
stays the headline, and this is the sensitivity analysis beside it.

A containment relation does not prove an answer is right -- "Atlanta, Georgia"
may be an acceptable short form of "at Mercedes-Benz Stadium in Atlanta,
Georgia", or an incomplete answer to a question about the stadium. The counts
below bound how much the metric could be hiding; settling each case needs a
judge.

Usage:
    python scripts/analyze_answer_matching.py \
        --results results/compensation/refinement_end_to_end_accessibility.json
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

from src.evaluation.qa import exact_match, normalize_answer, token_f1  # noqa: E402

DEFAULT_RECORDS = ROOT / "data" / "processed" / "diagnosis" / "nq15_train_records.jsonl"


def contains_either_way(answer: str, gold: str) -> bool:
    """One normalised string is a substring of the other (alias or span-length difference)."""
    answer, gold = normalize_answer(answer), normalize_answer(gold)
    return bool(answer) and bool(gold) and (answer in gold or gold in answer)


def is_strict_substring(answer: str, gold: str) -> bool:
    """The answer is a shorter span of the gold string -- what the frozen prompt asks for."""
    answer, gold = normalize_answer(answer), normalize_answer(gold)
    return bool(answer) and answer in gold and len(answer) < len(gold)


MATCHERS = {
    "exact_match (frozen)": exact_match,
    "containment": lambda answer, gold: int(contains_either_way(answer, gold)),
    "token_f1>=0.8": lambda answer, gold: int(token_f1(answer, gold) >= 0.8),
    "token_f1>=0.6": lambda answer, gold: int(token_f1(answer, gold) >= 0.6),
}


def rate(count: int, total: int) -> float | None:
    return round(count / total, 4) if total else None


def score_run(rows: list[dict]) -> dict:
    """Before/after accuracy and net repairs under each matching rule."""
    out = {}
    for name, matcher in MATCHERS.items():
        before = [matcher(row["initial_answer"], row["gold_answer"]) for row in rows]
        after = [matcher(row["final_answer"], row["gold_answer"]) for row in rows]
        repaired = sum(b < a for b, a in zip(before, after))
        broken = sum(b > a for b, a in zip(before, after))
        out[name] = {
            "before": rate(sum(before), len(rows)),
            "after": rate(sum(after), len(rows)),
            "repaired": repaired,
            "broken": broken,
            "net": repaired - broken,
        }
    return out


def span_report(answers: list[str], golds: list[str]) -> dict:
    """How many exact-match failures are only a difference in span length."""
    wrong = [(answer, gold) for answer, gold in zip(answers, golds) if not exact_match(answer, gold)]
    contained = [pair for pair in wrong if contains_either_way(*pair)]
    shorter = [pair for pair in wrong if is_strict_substring(*pair)]
    gold_lengths = [len(normalize_answer(gold).split()) for gold in golds]
    answer_lengths = [len(normalize_answer(answer).split()) for answer in answers]
    return {
        "answers": len(answers),
        "exact_match_wrong": len(wrong),
        "wrong_but_contained": len(contained),
        "wrong_but_contained_rate_of_wrong": rate(len(contained), len(wrong)),
        "wrong_but_shorter_span_of_gold": len(shorter),
        "wrong_but_shorter_span_rate_of_all": rate(len(shorter), len(answers)),
        "gold_words_median": statistics.median(gold_lengths) if gold_lengths else None,
        "gold_words_p75": sorted(gold_lengths)[int(0.75 * len(gold_lengths))] if gold_lengths else None,
        "gold_words_max": max(gold_lengths) if gold_lengths else None,
        "answer_words_median": statistics.median(answer_lengths) if answer_lengths else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="an end-to-end result file saved with --save-rows")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS, help="for the originally annotated answers")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not rows:
        raise SystemExit(f"{args.results} has no rows; re-run the evaluation with --save-rows")
    golds = [row["gold_answer"] for row in rows]

    report = {
        "note": "Sensitivity analysis only. The frozen exact match remains the headline metric; "
        "containment does not prove correctness.",
        "source": str(args.results.relative_to(ROOT)) if args.results.is_relative_to(ROOT) else str(args.results),
        "label": payload.get("label"),
        "samples": len(rows),
        "by_matching_rule": score_run(rows),
        "spans_run_before_answers": span_report([row["initial_answer"] for row in rows], golds),
    }

    if args.records and args.records.exists():
        wanted = {row["sample_id"] for row in rows}
        stored = {}
        with args.records.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["sample_id"] in wanted:
                    stored[record["sample_id"]] = record["initial_answer"]
        if stored:
            ordered = [(stored[row["sample_id"]], row["gold_answer"]) for row in rows if row["sample_id"] in stored]
            report["spans_annotated_answers"] = span_report([a for a, _ in ordered], [g for _, g in ordered])

    out = args.out or args.results.with_name(args.results.stem + "_matching.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "note"}, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
