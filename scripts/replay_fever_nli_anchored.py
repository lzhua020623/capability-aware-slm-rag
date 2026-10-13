"""Replay archived v6r1 FEVER NLI traces under the v7 anchored decision rule.

No GPU and no new classifier forward: the stored per-window probabilities are
reused, so only the decision rule changes. This is development evidence on the
same cohorts the anchor rule was designed against; it is not held-out.

Usage: python scripts/replay_fever_nli_anchored.py \
    --bundle artifacts/answer_repair/v6r1_reviewed/answer_repair_v6r1_reviewed_gpu.tar.gz \
    --results-dir artifacts/answer_repair/v6r1_reviewed/kaggle_20261014 \
    --out-dir artifacts/answer_repair/v7_design
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_answer_repair import prepare_records, read_jsonl
from src.compensation.nli_repair import NLI_ANCHORED_VERSION, NLI_THRESHOLD, repair_from_nli_trace, repair_from_nli_trace_anchored
from src.diagnosis import DatasetName
from src.evaluation.answer_repair import score_answer

COHORTS = {"reviewed": ("inputs/fever_reviewed_c1.jsonl", "fever_fever_nli.jsonl"),
           "dev50": ("inputs/fever_dev50_c1.jsonl", "fever_fever_nli_dev50.jsonl")}


def transition(initial: bool, final: bool) -> str:
    if initial == final:
        return "correct_preserved" if initial else "still_wrong"
    return "fixed" if final else "regressed"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    bundle_name = args.bundle.name.removesuffix(".tar.gz")
    report = {"rule": NLI_ANCHORED_VERSION, "threshold": NLI_THRESHOLD, "development_evidence_only": True,
              "bundle_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(), "cohorts": {}}
    rows_out = []
    with tarfile.open(args.bundle) as archive:
        for cohort, (input_name, result_name) in COHORTS.items():
            source = archive.extractfile(f"{bundle_name}/{input_name}").read()
            prepared = prepare_records(read_jsonl(source, input_name), DatasetName.FEVER)
            results = read_jsonl((args.results_dir / result_name).read_bytes(), result_name)
            if [r["sample_id"] for r in results] != [q.sample_id for _, q in prepared]:
                raise ValueError(f"Result/input ID mismatch in {cohort}")
            summary = {"samples": len(results), "initial_correct": 0, "v6_final_correct": 0, "v7_final_correct": 0,
                       "v6": {"fixed": 0, "regressed": 0}, "v7": {"fixed": 0, "regressed": 0},
                       "v6_to_v7_changed_final": 0, "v7_excluded_decisive_windows": 0,
                       "v6_status_counts": {}, "v7_status_counts": {}}
            for row, (original, request) in zip(results, prepared):
                trace = row["verifier_trace"]
                v6 = repair_from_nli_trace(request, trace)
                if (v6.final_answer, v6.status) != (row["final_answer"], row["status"]):
                    raise ValueError(f"v6 replay differs from archived result: {request.sample_id}")
                v7 = repair_from_nli_trace_anchored(request, trace)
                initial_ok = score_answer(request.initial_answer, original, DatasetName.FEVER)["correct"]
                v6_ok = score_answer(v6.final_answer, original, DatasetName.FEVER)["correct"]
                v7_ok = score_answer(v7.final_answer, original, DatasetName.FEVER)["correct"]
                excluded = v7.anchor_filter["excluded_decisive_windows"]
                windows = trace["windows"]
                rows_out.append({
                    "cohort": cohort, "sample_id": request.sample_id, "claim": request.query,
                    "initial_answer": request.initial_answer, "frozen_reference": original["gold_answer_or_label"],
                    "v6_final": v6.final_answer, "v6_status": v6.status, "v7_final": v7.final_answer, "v7_status": v7.status,
                    "initial_correct": initial_ok, "v6_correct": v6_ok, "v7_correct": v7_ok,
                    "v6_transition": transition(initial_ok, v6_ok), "v7_transition": transition(initial_ok, v7_ok),
                    "anchors": v7.anchor_filter["anchors"],
                    "v6_citation": [c.passage_id for c in v6.citations], "v7_citation": [c.passage_id for c in v7.citations],
                    "excluded_decisive_windows": [{"rank": windows[i]["rank"], "passage_id": windows[i]["passage_id"],
                                                   "entailment": round(windows[i]["probabilities"]["ENTAILMENT"], 4),
                                                   "contradiction": round(windows[i]["probabilities"]["CONTRADICTION"], 4),
                                                   "present": v7.anchor_filter["windows"][i]["present"],
                                                   "missing": v7.anchor_filter["windows"][i]["missing"]} for i in excluded],
                })
                summary["initial_correct"] += initial_ok
                summary["v6_final_correct"] += v6_ok
                summary["v7_final_correct"] += v7_ok
                for version, ok in (("v6", v6_ok), ("v7", v7_ok)):
                    t = transition(initial_ok, ok)
                    if t in ("fixed", "regressed"):
                        summary[version][t] += 1
                summary["v6_to_v7_changed_final"] += v6.final_answer != v7.final_answer
                summary["v7_excluded_decisive_windows"] += len(excluded)
                summary["v6_status_counts"][v6.status] = summary["v6_status_counts"].get(v6.status, 0) + 1
                summary["v7_status_counts"][v7.status] = summary["v7_status_counts"].get(v7.status, 0) + 1
            report["cohorts"][cohort] = summary
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "fever_anchored_replay.json").write_text(
        json.dumps({**report, "rows": rows_out}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# FEVER v7 anchored decision: offline replay of v6r1 traces", "",
             "Same MNLI windows and probabilities as v6r1; only the decision rule changes. Development evidence on the cohorts the rule was designed against, not held-out.", "",
             "| cohort | n | initial correct | v6 final | v7 final | v6 fixed/regressed | v7 fixed/regressed | finals changed v6→v7 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cohort, s in report["cohorts"].items():
        lines.append(f"| {cohort} | {s['samples']} | {s['initial_correct']} | {s['v6_final_correct']} | {s['v7_final_correct']} | "
                     f"{s['v6']['fixed']}/{s['v6']['regressed']} | {s['v7']['fixed']}/{s['v7']['regressed']} | {s['v6_to_v7_changed_final']} |")
    lines += ["", "## Samples whose final answer changed between v6 and v7", ""]
    for r in rows_out:
        if r["v6_final"] != r["v7_final"]:
            lines.append(f"- {r['cohort']} `{r['sample_id']}` {r['claim']} | initial {r['initial_answer']} | v6 {r['v6_final']} ({r['v6_status']}, {r['v6_transition']}) → v7 {r['v7_final']} ({r['v7_status']}, {r['v7_transition']}) | reference {r['frozen_reference']}")
            for w in r["excluded_decisive_windows"]:
                lines.append(f"  - excluded R{w['rank']} `{w['passage_id']}` E={w['entailment']} C={w['contradiction']} missing anchors {w['missing']}")
    lines += ["", "## Samples where v7 excluded a decisive window without changing the final answer", ""]
    for r in rows_out:
        if r["v6_final"] == r["v7_final"] and r["excluded_decisive_windows"]:
            lines.append(f"- {r['cohort']} `{r['sample_id']}` {r['claim']} | {r['v6_status']} → {r['v7_status']} | excluded " +
                         ", ".join(f"R{w['rank']} {w['passage_id']}" for w in r["excluded_decisive_windows"]))
    (args.out_dir / "fever_anchored_replay.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["cohorts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
