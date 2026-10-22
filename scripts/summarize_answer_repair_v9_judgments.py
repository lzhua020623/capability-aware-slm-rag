"""Offline original-answer judgment diagnostics; never feed these labels to E.

Explicitly marking a correct original for repair is a judgment error, not proof
that its replacement is wrong. Abstentions/invalid outputs stay separate, so
an always-uncertain model cannot appear to have solved verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_answer_repair import read_jsonl


def summarize(rows, labels, dataset):
    lookup = {}
    for label in labels:
        sid, value = label["sample_id"], label["initial_correct"]
        if sid in lookup or value is not None and type(value) is not bool:
            raise ValueError("Duplicate IDs or non-boolean human labels")
        lookup[sid] = value
    seen, counts = set(), Counter()
    for row in rows:
        sid = row["sample_id"]
        if sid in seen or sid not in lookup or row["dataset"] != dataset:
            raise ValueError("Result IDs/dataset do not match the label source")
        seen.add(sid)
        counts["samples"] += 1
        counts["calls"] += row["extra_calls"]
        counts["changed"] += bool(row["changed"])
        counts["invalid"] += row["status"].startswith("invalid_")
        correct = lookup[sid]
        if correct is None:
            counts["without_human_initial_label"] += 1
            continue
        counts["human_initial_scored"] += 1
        group = "correct" if correct else "wrong"
        counts["original_" + group] += 1
        verdict = row.get("verdict")
        if verdict in ("keep", "repair"):
            counts["classified"] += 1
            counts["classified_" + group] += 1
            counts["false_accept_wrong"] += not correct and verdict == "keep"
            counts["false_reject_correct"] += correct and verdict == "repair"
        else:
            counts["unclassified_" + group] += 1
        counts["wrong_original_retained"] += not correct and not row["changed"]
        counts["correct_original_changed"] += correct and row["changed"]
    def ratio(n, d):
        return counts[n] / counts[d] if counts[d] else None
    return {"dataset": dataset, "counts": dict(counts),
            "classified_coverage_of_human_scored": ratio("classified", "human_initial_scored"),
            "wrong_original_explicit_false_accept_rate": ratio("false_accept_wrong", "original_wrong"),
            "correct_original_explicit_false_reject_rate": ratio("false_reject_correct", "original_correct"),
            "warning": "Coverage and abstention must accompany error rates. Changed correct originals are not automatically semantic regressions. No final-answer human accuracy or faithfulness is inferred.",
            "label_basis": "Prior reference-visible chat-assisted original-answer confirmations, not independent blind labels."}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("nq", "fever"), required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--results", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.out.resolve() in {args.labels.resolve(), *(path.resolve() for path in args.results)}:
        p.error("Report cannot overwrite inputs")
    labels = read_jsonl(args.labels.read_bytes(), str(args.labels))
    report = {path.name: summarize(read_jsonl(path.read_bytes(), str(path)), labels, args.dataset)
              for path in args.results}
    if len(report) != len(args.results):
        raise ValueError("Result filenames must be unique")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
