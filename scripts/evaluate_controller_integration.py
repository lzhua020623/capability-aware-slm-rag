"""Offline paired QA metrics; gold is joined only AFTER runtime predictions exist."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnosis.adapter import canonical_dataset
from src.diagnosis.schema import normalize_sample_id
from src.evaluation.answer_repair import score_answer, summarize_repairs
from src.controller.execution import record_action


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(predictions, references, *, registration=None):
    registered_analysis = None
    if registration is not None:
        from src.controller.evaluation import analyze
        registered_analysis = analyze(predictions, references, registration)
    gold = {}
    for row in references:
        key = (canonical_dataset(row["dataset"]), normalize_sample_id(row["sample_id"]))
        if key in gold:
            raise ValueError(f"Duplicate reference: {key}")
        if "gold_answer_or_label" not in row:
            raise ValueError(f"Reference missing gold_answer_or_label: {key}")
        gold[key] = row
    if not predictions:
        raise ValueError("No predictions")
    groups, seen, inputs = defaultdict(list), set(), {}
    for r in predictions:
        dataset = canonical_dataset(r["dataset"])
        key = (dataset, normalize_sample_id(r["sample_id"]))
        run_key = (*key, r["mode"])
        if run_key in seen:
            raise ValueError(f"Duplicate prediction: {run_key}")
        seen.add(run_key)
        if key not in gold:
            raise ValueError(f"Missing reference: {key}")
        signature = (r["initial_answer"], r["input_sha256"])
        if key in inputs and inputs[key] != signature:
            raise ValueError(f"Comparison arms used different C1 inputs: {key}")
        inputs[key] = signature
        base = {"sample_id": key[1], "initial_metrics": score_answer(r["initial_answer"], gold[key], dataset),
                "final_metrics": score_answer(r["final_answer"], gold[key], dataset),
                "extra_calls": r["cost"]["llm_calls"],
                "latency": sum(e["seconds"] for e in r["calls"] if e["kind"] == "llm"),
                "pipeline_seconds": r["total_seconds"],
                "input_tokens": r["cost"]["input_tokens"],
                "output_tokens_estimate": r["cost"]["output_tokens_estimate"],
                "changed": r["initial_answer"] != r["final_answer"], "status": r["status"],
                "action": record_action(r), "executed": r["executed"],
                "retrieval_calls": r["cost"]["retrieval_calls"],
                "scoring_calls": r["cost"]["scoring_calls"],
                "failed_calls": r["cost"]["failed_calls"], "budget_blocks": r["cost"]["budget_blocks"]}
        groups[(dataset.value, r["mode"])].append(base)
    # Do not silently compare different subsets or a partially completed arm.
    for dataset in {key[0] for key in groups}:
        id_sets = [{r["sample_id"] for r in batch} for (ds, _), batch in groups.items() if ds == dataset]
        if any(ids != id_sets[0] for ids in id_sets):
            raise ValueError(f"Unpaired sample sets across modes for {dataset}")
    result = {}
    for (dataset, mode), batch in groups.items():
        result[f"{dataset}/{mode}"] = {
            **summarize_repairs(batch), "executed": sum(r["executed"] for r in batch),
            "retrieval_calls": sum(r["retrieval_calls"] for r in batch),
            "scoring_calls": sum(r["scoring_calls"] for r in batch),
            "failed_calls": sum(r["failed_calls"] for r in batch),
            "budget_blocks": sum(r["budget_blocks"] for r in batch),
            "pipeline_seconds": sum(r["pipeline_seconds"] for r in batch),
            "action_counts": dict(Counter(r["action"] for r in batch)),
            "by_action": {action: summarize_repairs([r for r in batch if r["action"] == action])
                          for action in sorted({r["action"] for r in batch})},
        }
    return {"schema_version": "controller_offline_evaluation_v1", "groups": result,
            "registered_analysis": registered_analysis,
            "scoring": "NQ normalized EM/token F1; FEVER label accuracy; not human semantic review",
            "cost_scope": "incremental after cached C1; resource loading excluded",
            "comparison": "paired within dataset; fixed-action decision labels are not diagnosis predictions",
            "quality_claim": "No evidence-faithfulness validation is implied by span checks or these scores"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, nargs='+')
    parser.add_argument("--gold", type=Path, required=True, nargs='+')
    parser.add_argument('--execution-seal', type=Path)
    parser.add_argument('--routes', type=Path, help='Offline candidate route JSONL; never an online input')
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    registration = json.loads(args.execution_seal.read_text()) if args.execution_seal else None
    if registration is not None:
        import hashlib
        expected = sorted(d['offline_gold_sha256'] for d in registration['datasets'].values())
        actual = sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in args.gold)
        if actual != expected:
            raise ValueError('Reference file hash differs from frozen offline gold; do not change aliases after inference')
    predictions = [r for p in args.predictions for r in rows(p)]
    report = evaluate(predictions, [r for p in args.gold for r in rows(p)], registration=registration)
    if args.routes:
        from src.controller.evaluation import replay_routes
        report['counterfactual_replay'] = replay_routes(predictions, rows(args.routes))
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"groups": list(report["groups"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
