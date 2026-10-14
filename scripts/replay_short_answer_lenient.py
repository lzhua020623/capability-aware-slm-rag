"""Post-hoc replay: how would a format-tolerant parser have scored the saved v7 NQ raw outputs?

This is an analysis of the frozen v7 run, not a new strategy. It re-parses
the archived raw outputs with three tolerances (RANK written as "[n]", a
quotation-mark-wrapped QUOTE/ANSWER, and "ANSWER: INSUFFICIENT" treated as an
abstention) and rescores with the frozen EM/F1. The model, prompt and
inference are untouched; nothing here is promoted or used for tuning. It
exists so the claim "most rejections were format artefacts" can be checked
against what those outputs would have done to the answers.

Usage: python scripts/replay_short_answer_lenient.py \
    --bundle-dir artifacts/answer_repair/v7_reviewed/gpu_bundle \
    --results-dir artifacts/answer_repair/v7_reviewed/kaggle_20261014 \
    --out artifacts/answer_repair/v7_reviewed/release_audit/lenient_parser_replay.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compensation.anchors import nq_passage_anchored
from src.compensation.short_answer_repair import MAX_ANSWER_WORDS, MAX_QUOTE_WORDS, locate_excerpt
from src.evaluation.qa import best_em_f1, normalize_answer

COHORTS = {"reviewed": "nq_short_answer_quote.jsonl", "dev50": "nq_short_answer_quote_dev50.jsonl",
           "heldout": "nq_short_answer_quote_heldout.jsonl"}
WRAPS = (('"', '"'), ("“", "”"), ("'", "'"), ("``", "''"))


def strip_wrap(text: str) -> str:
    text = text.strip()
    for a, b in WRAPS:
        if text.startswith(a) and text.endswith(b) and len(text) > len(a) + len(b):
            return text[len(a):-len(b)].strip()
    return text


def lenient(raw: str, record: dict) -> tuple[str, str | None, str | None]:
    """Return (status, rejection_reason, final_answer_slice)."""
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if lines == ["INSUFFICIENT"]:
        return "insufficient", None, None
    if len(lines) != 3:
        return "invalid", "line_count", None
    values = []
    for label, line in zip(("RANK:", "QUOTE:", "ANSWER:"), lines):
        if not line.upper().startswith(label):
            return "invalid", "label", None
        values.append(line[len(label):].strip())
    rank_text, quote, answer = values
    match = re.fullmatch(r"\[?\s*(\d+)\s*\]?", rank_text)
    if not match:
        return "invalid", "rank_format", None
    rank, quote, answer = int(match.group(1)), strip_wrap(quote), strip_wrap(answer)
    if answer.upper() == "INSUFFICIENT":
        return "insufficient", None, None
    evidence = record["evidence"]
    if not 1 <= rank <= len(evidence):
        return "invalid", "rank_range", None
    passage = evidence[rank - 1]["text"]
    try:
        located_quote, _, _ = locate_excerpt(quote, passage)
    except ValueError:
        return "invalid", "quote_not_located", None
    if len(located_quote.split()) > MAX_QUOTE_WORDS:
        return "invalid", "quote_too_long", None
    try:
        located_answer, _, _ = locate_excerpt(answer, located_quote)
    except ValueError:
        return "invalid", "answer_not_in_quote", None
    if len(located_answer.split()) > MAX_ANSWER_WORDS:
        return "invalid", "answer_too_long", None
    if normalize_answer(located_answer) == normalize_answer(record["prediction"]):
        return "kept", None, None
    if not nq_passage_anchored(record["question_or_claim"], passage)["anchored"]:
        return "unanchored", None, None
    return "repaired", None, located_answer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohort-spec", action="append", default=[],
                        help="name=inputs/<c1>.jsonl=<result>.jsonl; overrides the default three cohorts")
    args = parser.parse_args(argv)
    cohorts = dict(COHORTS)
    inputs = {name: f"inputs/nq_{name}_c1.jsonl" for name in COHORTS}
    if args.cohort_spec:
        cohorts, inputs = {}, {}
        for spec in args.cohort_spec:
            name, input_path, result_name = spec.split("=")
            cohorts[name], inputs[name] = result_name, input_path
    report = {"post_hoc_analysis_only": True, "promoted": False,
              "tolerances": ["RANK may be written as [n]", "QUOTE/ANSWER may be wrapped in quotation marks",
                             "ANSWER: INSUFFICIENT counts as abstention"], "cohorts": {}}
    for cohort, result_name in cohorts.items():
        source = {json.loads(l)["sample_id"]: json.loads(l)
                  for l in (args.bundle_dir / inputs[cohort]).read_text().splitlines() if l.strip()}
        rows = [json.loads(l) for l in (args.results_dir / result_name).read_text().splitlines() if l.strip()]
        frozen = collections.Counter(r["status"] for r in rows)
        statuses, reasons = collections.Counter(), collections.Counter()
        initial = final = fixed = regressed = 0
        f1_initial = f1_final = 0.0
        changed = []
        for row in rows:
            record = source[row["sample_id"]]
            golds = record["gold_answer_or_label"] if isinstance(record["gold_answer_or_label"], list) else [record["gold_answer_or_label"]]
            status, reason, answer = lenient(row["raw_output"], record)
            statuses[status] += 1
            if reason:
                reasons[reason] += 1
            final_answer = answer if status == "repaired" else record["prediction"]
            em_i, f1_i = best_em_f1(record["prediction"], golds)
            em_f, f1_f = best_em_f1(final_answer, golds)
            initial += em_i; final += em_f; fixed += (not em_i) and em_f; regressed += em_i and not em_f
            f1_initial += f1_i; f1_final += f1_f
            if status == "repaired" and final_answer != row["final_answer"]:
                changed.append({"sample_id": row["sample_id"], "query": row["query"], "initial": record["prediction"],
                                "frozen_v7_final": row["final_answer"], "frozen_v7_status": row["status"],
                                "lenient_final": final_answer, "em_initial": em_i, "em_lenient": em_f})
        n = len(rows)
        report["cohorts"][cohort] = {
            "samples": n, "frozen_v7_status_counts": dict(frozen),
            "frozen_v7_em": {"initial": sum(r["initial_metrics"]["correct"] for r in rows),
                             "final": sum(r["final_metrics"]["correct"] for r in rows)},
            "lenient_status_counts": dict(statuses), "lenient_remaining_rejections": dict(reasons),
            "lenient_em": {"initial": initial, "final": final, "fixed": fixed, "regressed": regressed},
            "f1": {"initial": round(f1_initial / n, 4), "lenient_final": round(f1_final / n, 4)},
            "answers_newly_changed_by_lenient_parse": changed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for cohort, c in report["cohorts"].items():
        print(cohort, "frozen EM", c["frozen_v7_em"], "| lenient EM", c["lenient_em"], "| lenient statuses", c["lenient_status_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
