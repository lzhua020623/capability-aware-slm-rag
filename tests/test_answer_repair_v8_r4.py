"""FACT contract, frozen r3 invariants, and predeclared stage-only selection."""
import ast
import copy
import hashlib
import json
import unittest
from pathlib import Path

from scripts.check_answer_repair_v8_smoke import evaluate_known_cases, evaluate_smoke
from src.compensation.conservative_repair import build_check_prompt, build_minimal_repair_prompt, parse_check
from src.compensation.line_contract import CHECK_LABELS, FormatError, complete_labeled_output
from tests.test_answer_repair import request
from tests.test_answer_repair_v8 import check
from tests.test_answer_repair_v8_r2 import real_request
from tests import test_answer_repair_v8_r2 as r2_tests

ROOT = Path(__file__).parents[1]


class FactAuditTests(unittest.TestCase):
    def test_fact_is_required_first_and_saved_without_repairing_output(self):
        parsed = parse_check(check(), request())
        self.assertEqual(parsed["fact"], "The capital of France is Paris.")
        for text in (check().split("\n", 1)[1], check().replace("FACT: The capital of France is Paris.", "FACT:"),
                     check() + "\nFACT: another claim", "FACT: another claim\n" + check()):
            with self.subTest(text=text), self.assertRaises(FormatError):
                parse_check(text, request())

    def test_ten_line_stop_waits_for_final_reason_and_rejects_repetition(self):
        self.assertEqual(len(CHECK_LABELS), 10)
        self.assertFalse(complete_labeled_output(check(), CHECK_LABELS))
        self.assertTrue(complete_labeled_output(check() + "\n", CHECK_LABELS))
        self.assertFalse(complete_labeled_output(check() + "\n" + check() + "\n", CHECK_LABELS))
        for prefix in ("\n".join(check().splitlines()[:n]) + "\n" for n in range(1, 10)):
            self.assertFalse(complete_labeled_output(prefix, CHECK_LABELS))

    def test_candidate_audit_receives_only_current_candidate_as_audit_target(self):
        prompt = build_check_prompt(request(answer="ORIGINAL_SENTINEL"), "CANDIDATE_SENTINEL")
        self.assertIn('Answer to audit: "CANDIDATE_SENTINEL"', prompt)
        self.assertNotIn("ORIGINAL_SENTINEL", prompt)
        self.assertIn("FACT:", prompt)

    def test_non_audit_logic_constants_and_all_minimal_prompts_match_frozen_r3(self):
        fixture = json.loads((ROOT / "tests/fixtures/answer_repair_v8r4_invariants.json").read_text())
        module = ast.parse((ROOT / "src/compensation/conservative_repair.py").read_text())
        # Frozen hashes predate Python 3.12's empty FunctionDef.type_params.
        # Remove only that absent-in-the-source field; retain all semantic AST
        # fields and the original fixture hashes (including all prompt bytes).
        def digest(n):
            rendered = ast.dump(n, include_attributes=False).replace(", type_params=[]", "")
            return hashlib.sha256(rendered.encode()).hexdigest()
        funcs = {n.name: digest(n) for n in module.body if isinstance(n, ast.FunctionDef)
                 and n.name not in ("build_check_prompt", "parse_check")}
        constants = {t.id: digest(n.value) for n in module.body if isinstance(n, ast.Assign)
                     for t in n.targets if isinstance(t, ast.Name) and t.id != "V8_VERSION"}
        self.assertEqual(funcs, fixture["function_ast_sha256"])
        self.assertEqual(constants, fixture["assignment_ast_sha256"])
        self.assertEqual(len(fixture["runtime_and_prompt_cases"]), 8)
        for case in fixture["runtime_and_prompt_cases"]:
            prompt = build_minimal_repair_prompt(real_request(case))
            self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(), case["minimal_prompt_sha256"])

    def inputs(self):
        config = json.loads((ROOT / "configs/answer_repair_v8r4_smoke8_v1.json").read_text())
        cases = copy.deepcopy(config["semantic_regressions"]["cases"])
        rows = []
        for case in cases:
            case["sample_id"] = str(case["review_number"])
            rows.append({"sample_id": case["sample_id"], "initial_answer": "original", "final_answer": "original",
                         "candidate_answer": case.get("expected_candidate_answer"),
                         "repair_trace": {"stages": [{"stage": k, "parsed": {"status": v, "fact": "synthetic fixture"},
                                                      "validation_error": None}
                                                     for k, v in case["expected_stage_status"].items()]}})
        return {"semantic_regressions": {"strategy": "conservative_repair", "cases": cases}}, {"conservative_repair": rows}

    def test_known_regressions_require_named_stages_and_candidate_identity(self):
        manifest, arms = self.inputs()
        self.assertTrue(evaluate_known_cases(manifest, arms)["passed"])
        for issue in ("wrong_status", "wrong_candidate", "missing_stage", "changed_control"):
            altered = copy.deepcopy(arms)
            if issue == "wrong_status":
                altered["conservative_repair"][0]["repair_trace"]["stages"][0]["parsed"]["status"] = "supported"
            elif issue == "wrong_candidate":
                altered["conservative_repair"][1]["candidate_answer"] = "Dhebar Lake"
            elif issue == "missing_stage":
                altered["conservative_repair"][1]["repair_trace"]["stages"].pop()
            else:
                altered["conservative_repair"][2]["final_answer"] = "unnecessary rewrite"
            self.assertFalse(evaluate_known_cases(manifest, altered)["passed"], issue)

    def test_n31_downstream_guard_failure_does_not_redefine_its_audit_regression(self):
        manifest, arms = self.inputs()
        arms["conservative_repair"][0].update(status="candidate_invalid")
        arms["conservative_repair"][0]["repair_trace"]["stages"].append(
            {"stage": "minimal_repair", "parsed": None, "validation_error": "non-contiguous quote"})
        self.assertTrue(evaluate_known_cases(manifest, arms)["passed"])

    def test_fallback_is_predeclared_and_does_not_use_em(self):
        manifest, arms = r2_tests.SmokeGateTests().inputs()
        manifest.update(failure_policy="r3 if either gate fails", semantic_regressions={
            "strategy": "conservative_repair", "cases": [{"case": "synthetic", "sample_id": "test-only",
                "expected_stage_status": {"original_check": "unsupported"}}]})
        report = evaluate_smoke(manifest, arms)
        self.assertTrue(report["engineering_gate_passed"])
        self.assertEqual(report["selected_final_revision"], "r3")
        manifest["semantic_regressions"]["cases"][0]["expected_stage_status"]["original_check"] = "supported"
        for metric in (0, 1, 999):
            arms["conservative_repair"][0]["final_metrics"] = {"em": metric}
            self.assertEqual(evaluate_smoke(manifest, arms)["selected_final_revision"], "r4")


if __name__ == "__main__":
    unittest.main()
