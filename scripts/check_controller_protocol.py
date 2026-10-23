"""Read-only T1 design audit. This checker cannot approve an execution seal.

No model imports, downloads, inference or benchmark reads. Reference hashes are
checked against the recorded Git commit; worktree drift is reported separately
so T3 can add implementations without rewriting the historical T1 snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = "configs/controller/experiment_protocol_v1.json"
PRIMARY = {"c1", "always_retrieve", "always_rerank", "always_repair",
           "full_enhancement", "adaptive"}
DIAGNOSTIC = {"adaptive_no_probe", "human_type_routed"}
BUDGET_KEYS = {"llm_calls", "retrieval_calls", "scoring_calls", "max_output_tokens"}
GATES = {"T2_DATA", "T2_ASSETS", "T3_ARMS", "T3_EVALUATOR",
         "T3_ENVIRONMENT", "T3_EXECUTION_SEAL"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected repository-relative path: {value}")
    return str(path)


def validate_design(plan):
    if plan["schema_version"] != "controller_experiment_design_v1":
        raise ValueError("Unsupported design schema")
    if plan["state"] != "design_frozen_execution_pending" or plan["execution_ready"] is not False:
        raise ValueError("T1 is a design, not an execution approval; a separate seal is required")
    if plan["datasets"] != ["nq", "fever"]:
        raise ValueError("The baseline design requires separate NQ and FEVER cohorts")
    arms = plan["arms"]
    names = [a["id"] for a in arms]
    if len(names) != len(set(names)) or set(names) != PRIMARY | DIAGNOSTIC:
        raise ValueError("Missing, duplicate or unexpected comparison arms")
    if {a["id"] for a in arms if a["role"] == "primary"} != PRIMARY:
        raise ValueError("Wrong primary comparison family")
    if {a["id"] for a in arms if a["role"] == "diagnostic"} != DIAGNOSTIC:
        raise ValueError("Wrong diagnostic comparison family")
    operations = plan["operation_bounds"]
    if set(operations) != {"probe", "retrieve", "rerank", "repair"}:
        raise ValueError("Unknown or missing operation bounds")
    for bounds in [*operations.values(), *(a["budget"] for a in arms)]:
        if set(bounds) != BUDGET_KEYS or any(type(v) is not int or v < 0 for v in bounds.values()):
            raise ValueError("Budgets must contain exactly four non-negative integer limits")
    for arm in arms:
        stages = [operations[name] for name in arm["stages"]]
        kind = arm["execution"]
        if kind == "sequence":
            expected = {k: sum(s[k] for s in stages) for k in BUDGET_KEYS}
        elif kind in ("one_or_none", "optional_probe_then_one_or_none"):
            expected = {k: max((s[k] for s in stages), default=0) for k in BUDGET_KEYS}
            if kind == "optional_probe_then_one_or_none":
                expected = {k: v + operations["probe"][k] for k, v in expected.items()}
        else:
            raise ValueError(f"Unknown execution kind: {kind}")
        if arm["budget"] != expected:
            raise ValueError(f"Budget does not match its execution paths: {arm['id']}")
    full = next(a for a in arms if a["id"] == "full_enhancement")
    if (full["stages"] != plan["full_composition"]["order"]
            or full["stages"] != ["retrieve", "rerank", "repair"]
            or plan["full_composition"]["budget_reset"] is not False):
        raise ValueError("Full requires ordered stages and a shared, non-resetting budget")
    gate_ids = [g["id"] for g in plan["execution_gates"]]
    if len(gate_ids) != len(set(gate_ids)) or set(gate_ids) != GATES:
        raise ValueError("Missing or duplicate T2/T3 execution gates")
    if not re.fullmatch(r"[0-9a-f]{40}", plan["reference_commit"]):
        raise ValueError("Reference must be a full Git commit")


def git_reference(root, commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root,
                                   stderr=subprocess.PIPE)


def audit_protocol(root=ROOT, protocol_path=DEFAULT_PROTOCOL, *, read_reference=git_reference):
    root = Path(root)
    result = {"schema_version": "controller_protocol_audit_v1", "design_valid": False,
              "execution_ready": False, "errors": [], "worktree_drift": [],
              "reference_files_verified": 0,
              "scope": "T1 design integrity only; does not enforce or approve runner execution"}
    try:
        protocol = root / relative_path(protocol_path)
        raw = protocol.read_bytes()
        expected = protocol.with_suffix(".sha256").read_text().split()[0]
        if digest(raw) != expected:
            raise ValueError("Protocol checksum mismatch: preserve v1, register a new design revision")
        plan = json.loads(raw)
        validate_design(plan)
        result.update(protocol_id=plan["protocol_id"], protocol_sha256=digest(raw),
                      reference_commit=plan["reference_commit"],
                      execution_blockers=plan["execution_gates"])
        linked = {}
        for name in ("document", "reference_snapshot"):
            entry = plan[name]
            data = (root / relative_path(entry["path"])).read_bytes()
            if digest(data) != entry["sha256"]:
                raise ValueError(f"{name} checksum mismatch")
            linked[name] = data
        snapshot = json.loads(linked["reference_snapshot"])
        if (snapshot["schema_version"] != "controller_reference_sources_v1"
                or snapshot["git_commit"] != plan["reference_commit"]):
            raise ValueError("Reference snapshot commit/schema differs from protocol")
        required = {plan["runtime_reference_config"]["path"], plan["components"]["diagnosis"]["path"],
                    "src/controller/execution.py", "src/generation/prompts.py",
                    "src/probe/prompts.py", "src/compensation/short_answer_repair.py",
                    "src/compensation/answer_repair.py", "src/evaluation/qa.py",
                    "src/evaluation/fever.py"}
        if not required <= set(snapshot["files"]):
            raise ValueError("Reference snapshot is missing required implementation files")
        for path, expected_hash in snapshot["files"].items():
            path = relative_path(path)
            reference = read_reference(root, snapshot["git_commit"], path)
            if digest(reference) != expected_hash:
                raise ValueError(f"Reference Git blob mismatch: {path}")
            result["reference_files_verified"] += 1
            local = root / path
            actual_hash = digest(local.read_bytes()) if local.is_file() else None
            if actual_hash != expected_hash:
                result["worktree_drift"].append({"path": path, "reference_sha256": expected_hash,
                                                "worktree_sha256": actual_hash})
            if path == plan["runtime_reference_config"]["path"]:
                if json.loads(reference) != plan["runtime_reference_config"]["values"]:
                    raise ValueError("Declared runtime config differs from reference Git config")
        result["design_valid"] = True
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.CalledProcessError) as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL, help="Repository-relative design JSON")
    parser.add_argument("--require-execution-ready", action="store_true",
                        help="Exit 2 for this design-only phase; T2/T3 still require a separate execution seal")
    args = parser.parse_args(argv)
    report = audit_protocol(protocol_path=args.protocol)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if not report["design_valid"]:
        return 1
    return 2 if args.require_execution_ready else 0


if __name__ == "__main__":
    sys.exit(main())
