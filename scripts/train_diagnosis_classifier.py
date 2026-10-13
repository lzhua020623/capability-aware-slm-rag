"""Train and select the Diagnosis v1 Logistic Regression classifier.

Protocol:
1. Fit every configuration (C x is_refutes_prediction ablation) on TRAIN only.
2. Select one configuration by VALIDATION Macro-F1 (see select_configuration).
3. Save the chosen artifact and the validation report.
4. Only then evaluate the chosen configuration on TEST, once.

Test results must never feed back into C, features or rules.
"""

from __future__ import annotations

import json
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnosis.classifier import LABEL_ORDER, RANDOM_STATE, DiagnosisClassifier
from src.diagnosis.dev_dataset import load_jsonl
from src.diagnosis.labels import DiagnosisLabel, parse_label
from src.diagnosis.schema import DiagnosisInput, EvidenceItem

DATASET_PATHS = {
    "nq": ROOT / "data" / "processed" / "diagnosis" / "nq_500.jsonl",
    "fever": ROOT / "data" / "processed" / "diagnosis" / "fever_500.jsonl",
}
SPLIT_PATH = ROOT / "configs" / "splits" / "diagnosis_v1_split.json"
MODEL_PATH = ROOT / "models" / "diagnosis" / "diagnosis_v1.json"
RESULTS_DIR = ROOT / "results" / "diagnosis"
C_CANDIDATES: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
REFUTES_OPTIONS: tuple[bool, ...] = (True, False)
MACRO_F1_TOLERANCE = 0.01
PROTOTYPE_NOTE = (
    "Prototype: minority classes have very few samples, so per-class metrics are unstable. "
    "FEVER has no trusted evidence_accessibility samples; its accessibility recall/F1 are not computable "
    "and must not be read as good or bad performance."
)


def input_from_dict(data: dict) -> DiagnosisInput:
    return DiagnosisInput(
        sample_id=data["sample_id"],
        dataset=data["dataset"],
        query=data["query"],
        evidence=[EvidenceItem(**item) for item in data["evidence"]],
        initial_answer=data["initial_answer"],
    )


def load_split_records() -> dict[str, list[dict]]:
    """Return split -> development records, restricted to the frozen trusted split."""
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for dataset, path in DATASET_PATHS.items():
        by_id = {record["input"]["sample_id"]: record for record in load_jsonl(path)}
        for name in by_split:
            for sample_id in split["splits"][dataset][name]:
                record = by_id[sample_id]
                if record["metadata"]["needs_manual_review"]:
                    raise ValueError(f"{dataset}:{sample_id} in {name} needs manual review")
                by_split[name].append(record)
    return by_split


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate(true: Sequence[DiagnosisLabel], predicted: Sequence[DiagnosisLabel]) -> dict:
    """Metrics in LABEL_ORDER. Undefined values (zero denominators) are None.

    macro_f1 averages F1 over classes with support > 0 in this subset.
    """
    index = {label: i for i, label in enumerate(LABEL_ORDER)}
    matrix = np.zeros((len(LABEL_ORDER), len(LABEL_ORDER)), dtype=int)
    for t, p in zip(true, predicted):
        matrix[index[t], index[p]] += 1

    per_class = {}
    for i, label in enumerate(LABEL_ORDER):
        tp, support, predicted_count = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        precision, recall = _ratio(tp, predicted_count), _ratio(tp, support)
        if support == 0:
            f1 = None
        elif tp == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[label.value] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": support, "predicted": predicted_count,
        }

    f1_values = [m["f1"] for m in per_class.values() if m["f1"] is not None]
    failure_true = [p for t, p in zip(true, predicted) if t is not DiagnosisLabel.CORRECT]
    detected = sum(p is not DiagnosisLabel.CORRECT for p in failure_true)
    return {
        "n": len(true),
        "accuracy": _ratio(int(np.trace(matrix)), len(true)),
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "macro_f1_classes": [label for label, m in per_class.items() if m["f1"] is not None],
        "failure_detection_recall": _ratio(detected, len(failure_true)),
        "failure_support": len(failure_true),
        "per_class": per_class,
        "confusion_matrix": {"labels": [label.value for label in LABEL_ORDER], "rows_true_cols_pred": matrix.tolist()},
    }


def evaluate_by_dataset(records: list[dict], predicted: Sequence[DiagnosisLabel]) -> dict:
    true = [parse_label(record["target_label"]) for record in records]
    report = {"all": evaluate(true, predicted)}
    for dataset in DATASET_PATHS:
        idx = [i for i, record in enumerate(records) if record["input"]["dataset"] == dataset]
        report[dataset] = evaluate([true[i] for i in idx], [predicted[i] for i in idx])
    return report


def fit_configuration(train: list[dict], c: float, include_refutes: bool) -> DiagnosisClassifier:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return DiagnosisClassifier.fit(
            [input_from_dict(record["input"]) for record in train],
            [record["target_label"] for record in train],
            c=c,
            include_refutes_prediction=include_refutes,
            random_state=RANDOM_STATE,
        )


def predict_records(classifier: DiagnosisClassifier, records: list[dict]) -> tuple[list[DiagnosisLabel], list[dict]]:
    proba = classifier.predict_proba_batch([input_from_dict(record["input"]) for record in records])
    labels, rows = [], []
    for record, scores in zip(records, proba):
        output = classifier.output_from_proba(scores)
        labels.append(output.diagnosis_label)
        rows.append({
            "sample_id": record["input"]["sample_id"],
            "dataset": record["input"]["dataset"],
            "target_label": record["target_label"],
            "predicted_label": output.diagnosis_label.value,
            "confidence": output.confidence,
            "class_scores": {label.value: score for label, score in output.class_scores.items()},
        })
    return labels, rows


def select_configuration(results: list[dict]) -> dict:
    """Highest validation Macro-F1; within MACRO_F1_TOLERANCE prefer fewer features, then smaller C
    (stronger regularization), then higher failure detection recall."""
    best = max(r["validation"]["all"]["macro_f1"] for r in results)
    close = [r for r in results if best - r["validation"]["all"]["macro_f1"] <= MACRO_F1_TOLERANCE]
    return min(close, key=lambda r: (
        r["n_features"], r["c"], -r["validation"]["all"]["failure_detection_recall"],
    ))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


def print_report(title: str, report: dict) -> None:
    print(f"\n== {title} ==")
    for subset, m in report.items():
        print(f"[{subset}] n={m['n']} acc={_fmt(m['accuracy'])} macro_f1={_fmt(m['macro_f1'])} "
              f"failure_recall={_fmt(m['failure_detection_recall'])} (failures={m['failure_support']})")
        for label, pc in m["per_class"].items():
            print(f"   {label:24s} P={_fmt(pc['precision'])} R={_fmt(pc['recall'])} F1={_fmt(pc['f1'])} "
                  f"support={pc['support']} predicted={pc['predicted']}")
        print(f"   confusion (rows=true, cols=pred): {m['confusion_matrix']['rows_true_cols_pred']}")


def main() -> int:
    splits = load_split_records()
    print({name: len(records) for name, records in splits.items()})

    results, classifiers = [], {}
    for include_refutes in REFUTES_OPTIONS:
        for c in C_CANDIDATES:
            classifier = fit_configuration(splits["train"], c, include_refutes)
            predicted, _ = predict_records(classifier, splits["validation"])
            key = ("full" if include_refutes else "no_refutes", c)
            classifiers[key] = classifier
            results.append({
                "configuration": key[0],
                "include_refutes_prediction": include_refutes,
                "c": c,
                "n_features": len(classifier.feature_names),
                "n_iter": classifier.model.n_iter_.tolist(),
                "validation": evaluate_by_dataset(splits["validation"], predicted),
            })
            m = results[-1]["validation"]["all"]
            print(f"{key[0]:10s} C={c:<5} macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} "
                  f"failure_recall={m['failure_detection_recall']:.4f} n_iter={results[-1]['n_iter']}")

    chosen = select_configuration(results)
    classifier = classifiers[(chosen["configuration"], chosen["c"])]
    chosen_config = {
        "configuration": chosen["configuration"],
        "c": chosen["c"],
        "include_refutes_prediction": chosen["include_refutes_prediction"],
        "feature_names": list(classifier.feature_names),
        "random_state": RANDOM_STATE,
    }
    classifier.save(MODEL_PATH)
    _, validation_rows = predict_records(classifier, splits["validation"])
    write_json(RESULTS_DIR / "diagnosis_v1_validation.json", {
        "note": PROTOTYPE_NOTE,
        "split_sizes": {name: len(records) for name, records in splits.items()},
        "selection_rule": (
            f"max validation macro_f1 (all); within {MACRO_F1_TOLERANCE} prefer fewer features, "
            "then smaller C, then higher failure_detection_recall. Test not used."
        ),
        "chosen_configuration": chosen_config,
        "configurations": results,
        "chosen_predictions": validation_rows,
    })
    print(f"\nChosen configuration (fixed before test): {chosen_config['configuration']} C={chosen_config['c']}")
    print_report("Validation (chosen)", chosen["validation"])

    predicted, test_rows = predict_records(classifier, splits["test"])
    test_report = evaluate_by_dataset(splits["test"], predicted)
    write_json(RESULTS_DIR / "diagnosis_v1_test.json", {
        "note": PROTOTYPE_NOTE + " Single final evaluation; not used for any selection.",
        "chosen_configuration": chosen_config,
        "test": test_report,
        "predictions": test_rows,
    })
    print_report("Test (single prototype evaluation)", test_report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
