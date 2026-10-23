"""Frozen design integrity, executable budget arithmetic and no false readiness."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.check_controller_protocol import (
    ROOT, DEFAULT_PROTOCOL, audit_protocol, digest, main, validate_design,
)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((ROOT / DEFAULT_PROTOCOL).read_text())

    def test_real_git_snapshot_and_design_validate_without_inference(self):
        report = audit_protocol()
        self.assertTrue(report["design_valid"], report)
        self.assertGreater(report["reference_files_verified"], 0)
        self.assertFalse(report["execution_ready"])
        self.assertEqual(len(report["execution_blockers"]), 6)

    def test_full_needs_two_scorings_and_adaptive_counts_probe(self):
        for name, key, value in (("full_enhancement", "scoring_calls", 1),
                                 ("adaptive", "llm_calls", 1),
                                 ("adaptive", "max_output_tokens", 384)):
            plan = deepcopy(self.plan)
            next(a for a in plan["arms"] if a["id"] == name)["budget"][key] = value
            with self.subTest(name=name, key=key), self.assertRaisesRegex(ValueError, "Budget"):
                validate_design(plan)

    def test_extra_budget_and_boolean_budget_are_not_silent_changes(self):
        for value in (9, True):
            plan = deepcopy(self.plan)
            plan["arms"][0]["budget"]["llm_calls"] = value
            with self.assertRaises(ValueError):
                validate_design(plan)

    def test_missing_or_duplicate_arm_rejected(self):
        for arms in (self.plan["arms"][:-1], self.plan["arms"] + self.plan["arms"][:1]):
            plan = deepcopy(self.plan)
            plan["arms"] = arms
            with self.assertRaisesRegex(ValueError, "comparison arms"):
                validate_design(plan)

    def test_cannot_relabel_t1_as_execution_ready_or_drop_gate(self):
        for field, value in (("execution_ready", True), ("execution_gates", [])):
            plan = deepcopy(self.plan)
            plan[field] = value
            with self.assertRaises(ValueError):
                validate_design(plan)

    def test_full_cannot_reset_shared_budget_or_change_order(self):
        for field, value in (("budget_reset", True), ("order", ["repair", "retrieve", "rerank"])):
            plan = deepcopy(self.plan)
            plan["full_composition"][field] = value
            with self.assertRaisesRegex(ValueError, "Full requires"):
                validate_design(plan)

    def fixture(self, root):
        """Copy just protocol and its pinned files, without benchmark data or Git."""
        snapshot = json.loads((ROOT / self.plan["reference_snapshot"]["path"]).read_bytes())
        paths = [DEFAULT_PROTOCOL, str(Path(DEFAULT_PROTOCOL).with_suffix(".sha256")),
                 self.plan["document"]["path"], self.plan["reference_snapshot"]["path"],
                 *snapshot["files"]]
        reference = {}
        for name in paths:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = (ROOT / name).read_bytes()
            target.write_bytes(data)
            reference[name] = data
        # Reference content remains frozen even if a future worktree is changed.
        from scripts.check_controller_protocol import git_reference
        for name in snapshot["files"]:
            reference[name] = git_reference(ROOT, snapshot["git_commit"], name)
        return lambda _root, _commit, name: reference[name]

    def test_local_prompt_drift_reported_without_rewriting_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reader = self.fixture(root)
            prompt = root / "src/probe/prompts.py"
            prompt.write_text(prompt.read_text() + "\n# changed\n")
            report = audit_protocol(root, read_reference=reader)
            self.assertTrue(report["design_valid"], report)
            self.assertIn("src/probe/prompts.py", [r["path"] for r in report["worktree_drift"]])
            self.assertFalse(report["execution_ready"])

    def test_corrupt_git_blob_fails_reference_check(self):
        report = audit_protocol(read_reference=lambda *_: b"corrupt")
        self.assertFalse(report["design_valid"])
        self.assertIn("Git blob mismatch", report["errors"][0])

    def test_invalid_protocol_path_returns_failed_audit(self):
        for name in ("../outside.json", "/outside.json"):
            report = audit_protocol(protocol_path=name)
            self.assertFalse(report["design_valid"])
            self.assertIn("repository-relative", report["errors"][0])

    def test_unregistered_plan_or_document_edits_are_rejected(self):
        for name in (DEFAULT_PROTOCOL, self.plan["document"]["path"]):
            with self.subTest(path=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                reader = self.fixture(root)
                (root / name).write_bytes((root / name).read_bytes() + b"\n")
                report = audit_protocol(root, read_reference=reader)
                self.assertFalse(report["design_valid"])
                self.assertIn("checksum mismatch", report["errors"][0])

    def test_declared_config_must_match_reference_even_if_design_relocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reader = self.fixture(root)
            plan = deepcopy(self.plan)
            plan["runtime_reference_config"]["values"]["repair_max_new_tokens"] = 4096
            raw = json.dumps(plan).encode()
            (root / DEFAULT_PROTOCOL).write_bytes(raw)
            (root / Path(DEFAULT_PROTOCOL).with_suffix(".sha256")).write_text(digest(raw))
            report = audit_protocol(root, read_reference=reader)
            self.assertFalse(report["design_valid"])
            self.assertIn("runtime config", report["errors"][0])

    def test_cli_distinguishes_design_success_from_execution_block(self):
        report = {"design_valid": True, "execution_ready": False, "execution_blockers": ["T2/T3"]}
        with patch("scripts.check_controller_protocol.audit_protocol", return_value=report), redirect_stdout(io.StringIO()):
            self.assertEqual(main([]), 0)
            self.assertEqual(main(["--require-execution-ready"]), 2)

    def test_cli_design_failure_is_not_a_readiness_block(self):
        with patch("scripts.check_controller_protocol.audit_protocol", return_value={"design_valid": False}), redirect_stdout(io.StringIO()):
            self.assertEqual(main([]), 1)
            self.assertEqual(main(["--require-execution-ready"]), 1)


if __name__ == "__main__":
    unittest.main()
