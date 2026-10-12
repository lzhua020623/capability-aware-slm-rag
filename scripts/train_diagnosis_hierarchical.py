"""Train and select the hierarchical Diagnosis classifier on TRAIN + VALIDATION only.

The test split is never loaded. Selection:
- Stage 1 (C x class_weight x threshold): among settings with validation
  failure recall >= MIN_FAILURE_RECALL, maximize binary macro-F1; within
  TOLERANCE prefer fewer false positives, then smaller C, then no class weight.
- Stage 2 (C): maximize macro-F1 on validation samples whose true label is a
  failure; within TOLERANCE prefer smaller C.
The single-stage baseline result files are read, never written.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_diagnosis_classifier import (
    DATASET_PATHS,
    SPLIT_PATH,
    evaluate,
    evaluate_by_dataset,
    input_from_dict,
    write_json,
)
from src.diagnosis.classifier import RANDOM_STATE, feature_matrix, feature_names_for
from src.diagnosis.dev_dataset import load_jsonl
from src.diagnosis.hierarchical_classifier import (
    CLASS_WEIGHTS,
    FAILURE_LABELS,
    STAGE1_LABELS,
    HierarchicalDiagnosisClassifier,
    LogisticStage,
)
from src.diagnosis.labels import DiagnosisLabel, parse_label

USED_SPLITS: tuple[str, ...] = ("train", "validation")
C_CANDIDATES: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
THRESHOLDS: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
INCLUDE_REFUTES_PREDICTION = True
MIN_FAILURE_RECALL = 0.70
TOLERANCE = 0.01
BASELINE_PATH = ROOT / "results" / "diagnosis" / "diagnosis_v1_validation.json"
OUTPUT_PATH = ROOT / "results" / "diagnosis" / "diagnosis_hierarchical_validation.json"
MODEL_PATH = ROOT / "models" / "diagnosis" / "diagnosis_hierarchical_v1.json"
NOTE = (
    "Validation-only prototype; the test split was not loaded. Minority classes are tiny, so per-class "
    "metrics are unstable. FEVER has no trusted evidence_accessibility samples (not computable). "
    "is_refutes_prediction is kept; it may act as a FEVER-specific shortcut."
)


def load_records() -> dict[str, list[dict]]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    by_split: dict[str, list[dict]] = {name: [] for name in USED_SPLITS}
    for dataset, path in DATASET_PATHS.items():
        by_id = {record["input"]["sample_id"]: record for record in load_jsonl(path)}
        for name in USED_SPLITS:
            for sample_id in split["splits"][dataset][name]:
                record = by_id[sample_id]
                if record["metadata"]["needs_manual_review"]:
                    raise ValueError(f"{dataset}:{sample_id} in {name} needs manual review")
                by_split[name].append(record)
    return by_split


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float:
    if not precision or not recall:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_binary(true_failure: np.ndarray, pred_failure: np.ndarray) -> dict:
    tp = int(np.sum(true_failure & pred_failure))
    fn = int(np.sum(true_failure & ~pred_failure))
    fp = int(np.sum(~true_failure & pred_failure))
    tn = int(np.sum(~true_failure & ~pred_failure))
    failure_p, failure_r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    correct_p, correct_r = _ratio(tn, tn + fn), _ratio(tn, tn + fp)
    failure_f1, correct_f1 = _f1(failure_p, failure_r), _f1(correct_p, correct_r)
    return {
        "n": len(true_failure),
        "accuracy": _ratio(tp + tn, len(true_failure)),
        "failure_precision": failure_p,
        "failure_recall": failure_r,
        "failure_f1": failure_f1,
        "correct_precision": correct_p,
        "correct_recall": correct_r,
        "correct_f1": correct_f1,
        "binary_macro_f1": (failure_f1 + correct_f1) / 2,
        "false_positives": fp,
        "false_negatives": fn,
        "confusion_matrix": {"labels": list(STAGE1_LABELS), "rows_true_cols_pred": [[tn, fp], [fn, tp]]},
    }


def fit_quietly(*args, **kwargs) -> LogisticStage:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return LogisticStage.fit(*args, **kwargs)


def select_stage1(results: list[dict]) -> tuple[dict, dict]:
    candidates = [
        (config, sweep) for config in results for sweep in config["threshold_sweep"]
        if sweep["failure_recall"] >= MIN_FAILURE_RECALL
    ]
    if not candidates:
        raise RuntimeError("No Stage 1 setting reaches the minimum failure recall")
    best = max(sweep["binary_macro_f1"] for _, sweep in candidates)
    close = [(c, s) for c, s in candidates if best - s["binary_macro_f1"] <= TOLERANCE]
    return min(close, key=lambda cs: (cs[1]["false_positives"], cs[0]["c"], cs[0]["class_weight"] is not None))


def select_stage2(results: list[dict]) -> dict:
    best = max(r["validation_true_failures"]["macro_f1"] for r in results)
    close = [r for r in results if best - r["validation_true_failures"]["macro_f1"] <= TOLERANCE]
    return min(close, key=lambda r: r["c"])


def baseline_summary() -> dict:
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    chosen = data["chosen_configuration"]
    config = next(r for r in data["configurations"]
                  if r["configuration"] == chosen["configuration"] and r["c"] == chosen["c"])
    return summarize_four_class(config["validation"]["all"]) | {
        "configuration": f"single_stage {chosen['configuration']} C={chosen['c']}"
    }


def summarize_four_class(metrics: dict) -> dict:
    matrix = metrics["confusion_matrix"]["rows_true_cols_pred"]
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "failure_detection_recall": metrics["failure_detection_recall"],
        "correct_recall": metrics["per_class"]["correct"]["recall"],
        "correct_to_failure_false_positives": sum(matrix[0][1:]),
        "failure_to_correct_false_negatives": sum(row[0] for row in matrix[1:]),
    }


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


def main() -> int:
    splits = load_records()
    print({name: len(records) for name, records in splits.items()})
    feature_names = feature_names_for(INCLUDE_REFUTES_PREDICTION)
    matrices = {
        name: feature_matrix([input_from_dict(r["input"]) for r in records], feature_names)
        for name, records in splits.items()
    }
    labels = {name: [parse_label(r["target_label"]) for r in records] for name, records in splits.items()}
    is_failure = {name: np.asarray([label is not DiagnosisLabel.CORRECT for label in values])
                  for name, values in labels.items()}

    # Stage 1: C x class_weight, each swept over thresholds on validation.
    stage1_results, stage1_models = [], {}
    for class_weight in CLASS_WEIGHTS:
        for c in C_CANDIDATES:
            stage = fit_quietly(matrices["train"], is_failure["train"].astype(int), STAGE1_LABELS, c, class_weight)
            stage1_models[(class_weight, c)] = stage
            p_failure = stage.predict_proba(matrices["validation"])[:, 1]
            stage1_results.append({
                "class_weight": class_weight,
                "c": c,
                "n_iter": stage.model.n_iter_.tolist(),
                "roc_auc": float(roc_auc_score(is_failure["validation"], p_failure)),
                "threshold_sweep": [
                    {"threshold": t} | evaluate_binary(is_failure["validation"], p_failure >= t) for t in THRESHOLDS
                ],
            })
    print("\n== Stage 1 (validation) ==")
    for r in stage1_results:
        at_half = next(s for s in r["threshold_sweep"] if s["threshold"] == 0.5)
        print(f"weight={str(r['class_weight']):8s} C={r['c']:<5} auc={r['roc_auc']:.3f} @0.5: "
              f"macro_f1={at_half['binary_macro_f1']:.3f} fail_R={at_half['failure_recall']:.3f} "
              f"FP={at_half['false_positives']} FN={at_half['false_negatives']}")

    chosen_config, chosen_sweep = select_stage1(stage1_results)
    stage1 = stage1_models[(chosen_config["class_weight"], chosen_config["c"])]
    print(f"\nChosen Stage 1: weight={chosen_config['class_weight']} C={chosen_config['c']} "
          f"threshold={chosen_sweep['threshold']}")
    for s in chosen_config["threshold_sweep"]:
        print(f"  t={s['threshold']:.2f} fail_P={_fmt(s['failure_precision'])} fail_R={_fmt(s['failure_recall'])} "
              f"fail_F1={s['failure_f1']:.3f} corr_R={_fmt(s['correct_recall'])} macro_f1={s['binary_macro_f1']:.3f} "
              f"FP={s['false_positives']} FN={s['false_negatives']}")

    # Stage 2: trained on train failures, evaluated on validation true failures.
    train_failures = is_failure["train"]
    val_failures = is_failure["validation"]
    stage2_targets = [FAILURE_LABELS.index(label) for label in labels["train"] if label is not DiagnosisLabel.CORRECT]
    val_failure_labels = [label for label in labels["validation"] if label is not DiagnosisLabel.CORRECT]
    stage2_results, stage2_models = [], {}
    for c in C_CANDIDATES:
        stage = fit_quietly(matrices["train"][train_failures], stage2_targets,
                            [label.value for label in FAILURE_LABELS], c, "balanced")
        stage2_models[c] = stage
        proba = stage.predict_proba(matrices["validation"][val_failures])
        predicted = [FAILURE_LABELS[i] for i in proba.argmax(axis=1)]
        metrics = evaluate(val_failure_labels, predicted)
        metrics["per_class"].pop(DiagnosisLabel.CORRECT.value)
        metrics["confusion_matrix"] = {
            "labels": [label.value for label in FAILURE_LABELS],
            "rows_true_cols_pred": [row[1:] for row in metrics["confusion_matrix"]["rows_true_cols_pred"][1:]],
        }
        for key in ("failure_detection_recall", "failure_support"):
            metrics.pop(key)
        stage2_results.append({"c": c, "n_iter": stage.model.n_iter_.tolist(), "validation_true_failures": metrics})
    print("\n== Stage 2 (validation true failures) ==")
    for r in stage2_results:
        m = r["validation_true_failures"]
        print(f"C={r['c']:<5} macro_f1={m['macro_f1']:.3f} acc={m['accuracy']:.3f} "
              + " ".join(f"{k[9:14]}:P={_fmt(v['precision'])}/R={_fmt(v['recall'])}/F1={_fmt(v['f1'])}"
                         for k, v in m["per_class"].items())
              + f" cm={m['confusion_matrix']['rows_true_cols_pred']}")
    chosen_stage2 = select_stage2(stage2_results)
    print(f"Chosen Stage 2: C={chosen_stage2['c']}")

    # End-to-end on the full validation split.
    classifier = HierarchicalDiagnosisClassifier(
        stage1=stage1,
        stage2=stage2_models[chosen_stage2["c"]],
        feature_names=feature_names,
        stage1_threshold=chosen_sweep["threshold"],
        include_refutes_prediction=INCLUDE_REFUTES_PREDICTION,
        random_state=RANDOM_STATE,
    )
    diagnoses = classifier.diagnose_matrix(matrices["validation"])
    end_to_end = evaluate_by_dataset(splits["validation"], [d.final_label for d in diagnoses])
    stage1_by_dataset = {}
    for dataset in DATASET_PATHS:
        idx = np.asarray([r["input"]["dataset"] == dataset for r in splits["validation"]])
        pred = np.asarray([d.stage1_label == "failure" for d in diagnoses])
        stage1_by_dataset[dataset] = evaluate_binary(val_failures[idx], pred[idx])

    baseline = baseline_summary()
    hierarchical = summarize_four_class(end_to_end["all"]) | {"configuration": "hierarchical"}
    print("\n== End-to-end validation ==")
    for subset, m in end_to_end.items():
        print(f"[{subset}] n={m['n']} acc={_fmt(m['accuracy'])} macro_f1={_fmt(m['macro_f1'])} "
              f"failure_recall={_fmt(m['failure_detection_recall'])}")
        for label, pc in m["per_class"].items():
            print(f"   {label:24s} P={_fmt(pc['precision'])} R={_fmt(pc['recall'])} F1={_fmt(pc['f1'])} "
                  f"support={pc['support']} predicted={pc['predicted']}")
        print(f"   confusion: {m['confusion_matrix']['rows_true_cols_pred']}")
    for dataset, m in stage1_by_dataset.items():
        print(f"Stage 1 [{dataset}] fail_R={_fmt(m['failure_recall'])} corr_R={_fmt(m['correct_recall'])} "
              f"FP={m['false_positives']} FN={m['false_negatives']}")
    print(f"\nBaseline:     {baseline}\nHierarchical: {hierarchical}")

    classifier.save(MODEL_PATH)
    write_json(OUTPUT_PATH, {
        "note": NOTE,
        "split_sizes": {name: len(records) for name, records in splits.items()},
        "feature_names": list(feature_names),
        "include_refutes_prediction": INCLUDE_REFUTES_PREDICTION,
        "random_state": RANDOM_STATE,
        "stage1": {
            "selection_rule": (
                f"failure_recall >= {MIN_FAILURE_RECALL}; max binary_macro_f1; within {TOLERANCE} prefer fewer "
                "false positives, then smaller C, then class_weight=None"
            ),
            "configurations": stage1_results,
            "chosen": {"class_weight": chosen_config["class_weight"], "c": chosen_config["c"],
                       "threshold": chosen_sweep["threshold"], "validation": chosen_sweep},
            "chosen_by_dataset": stage1_by_dataset,
        },
        "stage2": {
            "selection_rule": f"max macro_f1 on validation true failures; within {TOLERANCE} prefer smaller C",
            "configurations": stage2_results,
            "chosen": {"c": chosen_stage2["c"], "class_weight": "balanced"},
        },
        "end_to_end_validation": end_to_end,
        "comparison": {"single_stage_baseline": baseline, "hierarchical": hierarchical},
        "predictions": [
            {
                "sample_id": r["input"]["sample_id"],
                "dataset": r["input"]["dataset"],
                "target_label": r["target_label"],
                "stage1_failure_probability": d.stage1_failure_probability,
                "stage1_label": d.stage1_label,
                "stage2_class_scores": None if d.stage2_class_scores is None
                else {k.value: v for k, v in d.stage2_class_scores.items()},
                "final_label": d.final_label.value,
                "confidence": d.confidence,
            }
            for r, d in zip(splits["validation"], diagnoses)
        ],
    })
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)} and {MODEL_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
