"""Verify registered T2 files, paired IDs, gold isolation and protected reserves.

This checks structural data integrity only; it never scores an answer, starts
inference, or certifies unavailable indexes. --require-full-t2 returns 2 for
this partial release, where FEVER C1 and resource seals are still pending.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_controller_data import file_sha, read_json, rows
from src.compensation.nq_train_adapter import runtime_request
from src.controller.data_preparation import QueryGuard, ROLES, query_key

REGISTRY = "configs/splits/controller_data_v1.json"


def safe_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected a relative artifact path")
    return root / path


def verify(root=ROOT, registry_path=REGISTRY, *, check_sources=False):
    root = Path(root)
    registry = read_json(safe_path(root, registry_path))
    if registry["schema_version"] != "controller_data_registration_v1":
        raise ValueError("Unknown registration schema")
    manifest_path = safe_path(root, registry["manifest_path"])
    if file_sha(manifest_path) != registry["manifest_sha256"]:
        raise ValueError("Registered manifest hash mismatch")
    manifest, bundle = read_json(manifest_path), manifest_path.parent
    if manifest["full_T2_ready"] is not False or manifest["fever"]["paired_runtime_and_gold_ready"] is not False:
        raise ValueError("This release has no FEVER C1/resource seal; cannot claim full T2 readiness")
    for path, info in manifest["artifacts"].items():
        if file_sha(safe_path(bundle, path)) != info["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {path}")
    ids = read_json(bundle / "selection_ids.json")
    reserves = read_json(bundle / "evaluation_reserve_ids.json")
    exposure = read_json(bundle / "exposure_audit.json")
    policy = read_json(bundle / "policy.json")
    if ids != registry["active_sample_ids"]:
        raise ValueError("Registered cohort IDs disagree with exported IDs")
    if set(ids) != {"nq", "fever"} or set(reserves) != set(ids):
        raise ValueError("Both datasets require cohorts and a protected evaluation pool")
    report = {"schema_version": "controller_data_validation_v1", "data_integrity_passed": False,
              "full_T2_ready": False, "manifest_sha256": registry["manifest_sha256"],
              "source_hashes_checked": 0, "datasets": {}}
    for ds in ("nq", "fever"):
        if set(ids[ds]) != set(ROLES):
            raise ValueError("Missing cohort role")
        protected = set(reserves[ds])
        summary = manifest["evaluation_reserve"][ds]
        fraction = len(protected) / summary["population_count"]
        if len(protected) != len(reserves[ds]) or len(protected) != summary["protected_count"]:
            raise ValueError("Protected reserve count mismatch or duplicate IDs")
        limits = policy["evaluation_reserve"]
        if not limits["minimum_fraction"] <= fraction <= limits["maximum_fraction"]:
            raise ValueError("Protected reserve fraction outside frozen limits")
        if set(ids[ds]["development"] + ids[ds]["selection"]) & protected:
            raise ValueError("Protected evaluation ID used for optimization")
        if not set(ids[ds]["final_candidate"]) <= protected:
            raise ValueError("Final subset not contained in full protected reserve")
        guard, seen, counts = QueryGuard(), set(), {}
        known_ids = set(exposure["known_ids"][ds])
        known_queries = set(exposure["normalized_query_sha256"][ds])
        for role in ROLES:
            name = "runtime.jsonl" if ds == "nq" else "c1_tasks.jsonl"
            inputs = list(rows(bundle / ds / role / name))
            references = list(rows(bundle / "offline" / ds / role / "gold.jsonl"))
            expected_ids = ids[ds][role]
            if len(expected_ids) != manifest["counts_per_dataset"][role]:
                raise ValueError("Cohort count differs from frozen target")
            for batch in (inputs, references):
                if [r["sample_id"] for r in batch] != expected_ids or any(r["dataset"] != ds for r in batch):
                    raise ValueError("Input/gold order, membership or dataset mismatch")
            for record, gold in zip(inputs, references):
                if record["sample_id"] in seen or record["sample_id"] in known_ids:
                    raise ValueError("Duplicate or historically exposed selected ID")
                seen.add(record["sample_id"])
                if query_key(record["query"]) in known_queries or guard.conflicts(record["query"]):
                    raise ValueError("Exposed normalized query or duplicate/similar selected query")
                guard.add(record["query"])
                fields = {"dataset", "sample_id", "query"}
                if ds == "nq":
                    fields |= {"initial_answer", "evidence"}
                    runtime_request(record)
                    if any(set(e) != {"rank", "passage_id", "text", "score"} for e in record["evidence"]):
                        raise ValueError("Extra runtime evidence fields")
                if set(record) != fields:
                    raise ValueError("Unexpected runtime/task fields, including possible gold leakage")
                if set(gold) != {"dataset", "sample_id", "gold_answer_or_label"}:
                    raise ValueError("Unexpected offline reference fields")
                answer = gold["gold_answer_or_label"]
                if ds == "fever" and answer not in ("SUPPORTS", "REFUTES"):
                    raise ValueError("Invalid FEVER reference")
                if ds == "nq" and (not isinstance(answer, list) or not answer
                                   or any(not isinstance(a, str) or not a.strip() for a in answer)):
                    raise ValueError("Invalid NQ reference answers")
            counts[role] = len(inputs)
        report["datasets"][ds] = {"counts": counts, "protected_count": len(protected),
            "protected_fraction": fraction, "paired_runtime_ready": ds == "nq"}
    if check_sources:
        checks = {r["path"]: r["sha256"] for r in [*manifest["source_files"], *exposure["files"]]}
        checks.update(manifest["source_code_sha256"])
        checks["configs/controller/data_policy_v1.json"] = manifest["policy_sha256"]
        checks["configs/controller/experiment_protocol_v1.json"] = manifest["protocol_sha256"]
        for path, expected in checks.items():
            if file_sha(safe_path(root, path)) != expected:
                raise ValueError(f"Source drift: {path}")
        report["source_hashes_checked"] = len(checks)
    report["data_integrity_passed"] = True
    report["remaining_requirements"] = manifest["remaining_requirements"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=REGISTRY)
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument("--require-full-t2", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = verify(registry_path=args.registry, check_sources=args.check_sources)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"data_integrity_passed": False, "error": str(exc)}))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if args.require_full_t2 else 0


if __name__ == "__main__":
    raise SystemExit(main())
