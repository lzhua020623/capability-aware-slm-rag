"""Offline paired evaluation of C1 -> E, separate from runtime repair."""

from __future__ import annotations

from collections import Counter
from math import comb

from src.diagnosis import DatasetName
from src.evaluation.fever import parse_fever_prediction
from src.evaluation.qa import best_em_f1


def score_answer(answer: str, source: dict, dataset: DatasetName) -> dict:
    gold = source["gold_answer_or_label"]
    if dataset == DatasetName.FEVER:
        if gold not in ("SUPPORTS", "REFUTES"):
            raise ValueError("Invalid FEVER reference label")
        prediction = parse_fever_prediction(answer)
        return {"correct": prediction == gold, "parsed_prediction": prediction}
    golds = gold if isinstance(gold, list) else [gold]
    if not golds or any(not isinstance(g, str) or not g.strip() for g in golds):
        raise ValueError("NQ reference answers must be non-empty strings")
    em, f1 = best_em_f1(answer, golds)
    return {"correct": bool(em), "em": em, "f1": f1}


def summarize_repairs(records: list[dict]) -> dict:
    n = len(records)
    initial_correct = sum(r["initial_metrics"]["correct"] for r in records)
    final_correct = sum(r["final_metrics"]["correct"] for r in records)
    fixed = sum(not r["initial_metrics"]["correct"] and r["final_metrics"]["correct"] for r in records)
    regressed = sum(r["initial_metrics"]["correct"] and not r["final_metrics"]["correct"] for r in records)
    calls = sum(r["extra_calls"] for r in records)
    changed = sum(r["changed"] for r in records)
    discordant = fixed + regressed
    summary = {
        "evaluated_samples": n,
        "initial_correct": initial_correct,
        "final_correct": final_correct,
        "initial_accuracy": initial_correct / n if n else None,
        "final_accuracy": final_correct / n if n else None,
        "fixed": fixed,
        "regressed": regressed,
        "net_correct_gain": fixed - regressed,
        "accuracy_gain": (fixed - regressed) / n if n else None,
        "mcnemar_exact_pvalue": min(1.0, 2 * sum(comb(discordant, k) for k in range(min(fixed, regressed) + 1))
                                     / (2 ** discordant)) if discordant else 1.0,
        "repair_success_rate": fixed / (n - initial_correct) if n > initial_correct else None,
        "regression_rate": regressed / initial_correct if initial_correct else None,
        "changed": changed,
        "verifier_calls": sum(r.get("verifier_calls", 0) for r in records),
        "verifier_latency_total": sum(r.get("verifier_latency", 0) for r in records),
        "verifier_input_tokens_total": sum(r.get("verifier_input_tokens") or 0 for r in records),
        "extra_calls": calls,
        "extra_calls_per_sample": calls / n if n else None,
        "extra_generation_latency_total": sum(r["latency"] for r in records),
        "extra_generation_latency_per_sample": sum(r["latency"] for r in records) / n if n else None,
        "status_counts": dict(Counter(r["status"] for r in records)),
        "independent_support_checked": sum(r.get("independent_support_status") != "not_checked"
                                           and r.get("independent_support_status") is not None for r in records),
        "same_model_support_check_counts": dict(Counter(
            r["support_check"]["status"] for r in records if r.get("support_check"))),
    }
    for field in ("input_tokens", "output_tokens_estimate"):
        values = [r[field] for r in records if r.get(field) is not None]
        summary[field + "_total"] = sum(values) if values else None
        summary[field + "_observed_samples"] = len(values)
        summary[field + "_observed_calls"] = sum(r["extra_calls"] for r in records if r.get(field) is not None)
    if records and "f1" in records[0]["initial_metrics"]:
        summary["initial_f1"] = sum(r["initial_metrics"]["f1"] for r in records) / n
        summary["final_f1"] = sum(r["final_metrics"]["f1"] for r in records) / n
    return summary
