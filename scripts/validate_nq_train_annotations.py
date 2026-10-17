"""Validate and merge returned NQ Train Diagnosis annotations.

Annotators must only fill ``label`` (optional ``annotation_note`` allowed).
All delivery input and annotation_reference fields are frozen: any mutation
fails validation with no silent repair.

Usage:
  python scripts/validate_nq_train_annotations.py
  python scripts/validate_nq_train_annotations.py --merge

``--merge`` writes normalized outputs only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DELIVERY_DIR = ROOT / "data" / "annotations" / "nq_train_delivery"
RETURNED_DIR = ROOT / "data" / "annotations" / "nq_train_returned"
MANIFEST_PATH = DELIVERY_DIR / "manifest.json"
MASTER_PATH = ROOT / "data" / "annotations" / "nq_train_annotation_candidates.jsonl"
MERGE_OUT = ROOT / "data" / "annotations" / "normalized" / "nq_train_74097_labels.jsonl"
SUMMARY_OUT = ROOT / "data" / "annotations" / "normalized" / "nq_train_74097_labels_summary.json"

EXPECTED_TOTAL = 74_097
ALLOWED_LABELS = frozenset(
    {
        "correct",
        "evidence_insufficiency",
        "evidence_accessibility",
        "evidence_utilisation",
    }
)
FROZEN_TOP_KEYS = ("sample_id", "dataset", "query", "initial_answer", "evidence", "annotation_reference")
OPTIONAL_RETURN_KEYS = frozenset({"annotation_note"})


def _fail(errors: list[str], message: str) -> None:
    errors.append(message)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_no}: invalid JSON ({exc})") from exc
    return rows


def load_delivery() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing delivery manifest: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    by_id: dict[str, dict[str, Any]] = {}
    for shard in manifest["shards"]:
        path = DELIVERY_DIR / shard["file"]
        if not path.exists():
            raise FileNotFoundError(f"Missing delivery shard: {path}")
        rows = load_jsonl(path)
        if len(rows) != int(shard["n_samples"]):
            raise ValueError(
                f"{shard['file']}: expected {shard['n_samples']} rows, got {len(rows)}"
            )
        for row in rows:
            sample_id = str(row["sample_id"])
            if sample_id in by_id:
                raise ValueError(f"Duplicate sample_id in delivery: {sample_id}")
            by_id[sample_id] = row
    if len(by_id) != EXPECTED_TOTAL:
        raise ValueError(f"Delivery has {len(by_id)} unique IDs, expected {EXPECTED_TOTAL}")
    return by_id, manifest


def load_returned() -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """Map shard filename -> list of (source_file, record)."""
    if not RETURNED_DIR.exists():
        raise FileNotFoundError(
            f"Returned directory missing: {RETURNED_DIR}. "
            "Place annotated shards there before validating."
        )
    files = sorted(RETURNED_DIR.glob("nq_train_annotation_*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No returned shards matching nq_train_annotation_*.jsonl in {RETURNED_DIR}"
        )
    by_shard: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for path in files:
        rows = load_jsonl(path)
        by_shard[path.name] = [(path.name, row) for row in rows]
    return by_shard


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compare_frozen(delivery_row: dict[str, Any], returned_row: dict[str, Any], sample_id: str) -> list[str]:
    problems: list[str] = []
    for key in FROZEN_TOP_KEYS:
        if key not in returned_row:
            problems.append(f"{sample_id}: missing frozen field {key!r}")
            continue
        if canonical_json(delivery_row[key]) != canonical_json(returned_row[key]):
            problems.append(f"{sample_id}: frozen field {key!r} was modified")
    return problems


def validate(
    delivery: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    returned_by_shard: dict[str, list[tuple[str, dict[str, Any]]]],
) -> tuple[bool, dict[str, Any], list[str]]:
    errors: list[str] = []
    expected_shards = {shard["file"] for shard in manifest["shards"]}
    returned_names = set(returned_by_shard)

    if returned_names != expected_shards:
        missing_shards = sorted(expected_shards - returned_names)
        extra_shards = sorted(returned_names - expected_shards)
        if missing_shards:
            _fail(errors, f"Missing returned shards: {missing_shards}")
        if extra_shards:
            _fail(errors, f"Unexpected returned shards: {extra_shards}")

    returned_rows: list[tuple[str, dict[str, Any]]] = []
    for name in sorted(returned_by_shard):
        returned_rows.extend(returned_by_shard[name])

    total = len(returned_rows)
    if total != EXPECTED_TOTAL:
        _fail(errors, f"Total returned samples={total}, expected {EXPECTED_TOTAL}")

    ids = [str(row["sample_id"]) for _, row in returned_rows]
    id_counts = Counter(ids)
    duplicates = sorted(sample_id for sample_id, count in id_counts.items() if count > 1)
    if duplicates:
        _fail(errors, f"Duplicate sample_id count={len(duplicates)}; examples={duplicates[:10]}")

    returned_ids = set(id_counts)
    delivery_ids = set(delivery)
    missing = sorted(delivery_ids - returned_ids)
    extra = sorted(returned_ids - delivery_ids)
    if missing:
        _fail(errors, f"Missing sample_ids vs delivery: {len(missing)}; examples={missing[:10]}")
    if extra:
        _fail(errors, f"Unknown sample_ids not in delivery: {len(extra)}; examples={extra[:10]}")

    empty_label = 0
    invalid_label = 0
    label_counts: Counter[str] = Counter()
    per_shard_labels: dict[str, Counter[str]] = defaultdict(Counter)
    mutated = 0

    for shard_name, row in returned_rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in delivery:
            continue
        frozen_problems = compare_frozen(delivery[sample_id], row, sample_id)
        if frozen_problems:
            mutated += len(frozen_problems)
            errors.extend(frozen_problems[:3] if len(errors) < 50 else [])
            if len(frozen_problems) > 3 and len(errors) < 55:
                errors.append(f"{sample_id}: ... and {len(frozen_problems) - 3} more frozen-field diffs")

        allowed_keys = set(FROZEN_TOP_KEYS) | {"label"} | OPTIONAL_RETURN_KEYS
        unexpected = sorted(set(row) - allowed_keys)
        if unexpected:
            _fail(errors, f"{sample_id}: unexpected fields {unexpected}")

        label = row.get("label")
        if label is None or (isinstance(label, str) and label.strip() == ""):
            empty_label += 1
        elif not isinstance(label, str) or label not in ALLOWED_LABELS:
            invalid_label += 1
            if len(errors) < 80:
                _fail(errors, f"{sample_id}: invalid label {label!r}")
        else:
            label_counts[label] += 1
            per_shard_labels[shard_name][label] += 1

    if empty_label:
        _fail(errors, f"Empty/null label count={empty_label}")
    if invalid_label:
        _fail(errors, f"Invalid label count={invalid_label}")
    if mutated:
        _fail(errors, f"Frozen field mutation diffs={mutated} (FAIL; no silent repair)")

    # Expected per-shard sizes from delivery manifest
    for shard in manifest["shards"]:
        name = shard["file"]
        got = len(returned_by_shard.get(name, []))
        expected_n = int(shard["n_samples"])
        if got != expected_n:
            _fail(errors, f"Shard {name}: returned n={got}, expected {expected_n}")

    percentages = {
        label: round(100.0 * label_counts[label] / EXPECTED_TOTAL, 4) if EXPECTED_TOTAL else 0.0
        for label in sorted(ALLOWED_LABELS)
    }
    # Include zeros for missing classes in counts
    for label in ALLOWED_LABELS:
        label_counts.setdefault(label, 0)

    report = {
        "ok": not errors,
        "total": total,
        "expected_total": EXPECTED_TOTAL,
        "shard_count_returned": len(returned_by_shard),
        "shard_count_expected": len(expected_shards),
        "duplicate_sample_id_count": len(duplicates),
        "missing_sample_id_count": len(missing),
        "extra_sample_id_count": len(extra),
        "empty_label_count": empty_label,
        "invalid_label_count": invalid_label,
        "frozen_field_mutation_diff_count": mutated,
        "label_counts": dict(sorted(label_counts.items())),
        "label_percentages": percentages,
        "per_shard_distribution": {
            name: dict(sorted(per_shard_labels[name].items()))
            for name in sorted(per_shard_labels)
        },
        "allowed_labels": sorted(ALLOWED_LABELS),
        "error_count": len(errors),
        "errors_preview": errors[:40],
    }
    return not errors, report, errors


def master_order_ids() -> list[str]:
    """Canonical order = master annotation-candidate file order (not delivery sort)."""
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing master candidates: {MASTER_PATH}")
    ids: list[str] = []
    with MASTER_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(str(json.loads(line)["sample_id"]))
    if len(ids) != EXPECTED_TOTAL:
        raise ValueError(f"Master file has {len(ids)} rows, expected {EXPECTED_TOTAL}")
    if len(set(ids)) != EXPECTED_TOTAL:
        raise ValueError("Master file has duplicate sample_id values")
    return ids


def merge(
    delivery: dict[str, dict[str, Any]],
    returned_by_shard: dict[str, list[tuple[str, dict[str, Any]]]],
    report: dict[str, Any],
) -> None:
    by_id: dict[str, dict[str, Any]] = {}
    for _, row in (item for rows in returned_by_shard.values() for item in rows):
        by_id[str(row["sample_id"])] = row

    order = master_order_ids()
    MERGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    if MERGE_OUT.exists() or SUMMARY_OUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing merge outputs: {MERGE_OUT} / {SUMMARY_OUT}"
        )

    with MERGE_OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for sample_id in order:
            returned = by_id[sample_id]
            delivery_row = delivery[sample_id]
            # Preserve delivery payload; attach returned label (+ optional note).
            out = {
                "sample_id": delivery_row["sample_id"],
                "dataset": delivery_row["dataset"],
                "query": delivery_row["query"],
                "initial_answer": delivery_row["initial_answer"],
                "evidence": delivery_row["evidence"],
                "annotation_reference": delivery_row["annotation_reference"],
                "label": returned["label"],
            }
            if "annotation_note" in returned and returned["annotation_note"] is not None:
                out["annotation_note"] = returned["annotation_note"]
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")

    summary = {
        "source_delivery": str(DELIVERY_DIR.relative_to(ROOT).as_posix()),
        "source_returned": str(RETURNED_DIR.relative_to(ROOT).as_posix()),
        "master_order_file": str(MASTER_PATH.relative_to(ROOT).as_posix()),
        "output": str(MERGE_OUT.relative_to(ROOT).as_posix()),
        "total": EXPECTED_TOTAL,
        "label_counts": report["label_counts"],
        "label_percentages": report["label_percentages"],
        "shard_counts": {
            name: len(rows) for name, rows in sorted(returned_by_shard.items())
        },
        "duplicate_count": report["duplicate_sample_id_count"],
        "missing_count": report["missing_sample_id_count"],
        "invalid_label_count": report["invalid_label_count"],
        "per_shard_distribution": report["per_shard_distribution"],
        "validation_ok": True,
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {MERGE_OUT.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_OUT.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Write normalized merge outputs only if validation passes",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write the validation JSON report",
    )
    args = parser.parse_args()

    try:
        delivery, manifest = load_delivery()
        returned_by_shard = load_returned()
    except (FileNotFoundError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1

    ok, report, errors = validate(delivery, manifest, returned_by_shard)
    print(json.dumps(report, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")

    if not ok:
        print(f"VALIDATION FAILED ({len(errors)} issues). Merge refused.")
        return 1

    print("VALIDATION PASSED.")
    if args.merge:
        try:
            merge(delivery, returned_by_shard, report)
        except FileExistsError as exc:
            print(str(exc))
            return 1
    else:
        print("Merge not requested. Re-run with --merge after validation passes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
