"""Tests for the parameterised trigger policy and the v2-expanded trigger search / evaluation helpers."""

from __future__ import annotations

import inspect
import itertools
import unittest

from scripts.evaluate_probe_v2_expanded_policies import probe_label_analysis
from scripts.search_probe_trigger_v2_expanded import (
    candidate_policies,
    distribution,
    error_category,
    evaluate_policy,
    select_policy,
)
from src.diagnosis.labels import DiagnosisLabel
from src.probe.pipeline import apply_probe
from src.probe.probe import EvidenceProbe
from src.probe.schema import ProbeLabel
from src.probe.trigger import V1_TRIGGER, V2_EXPANDED_TRIGGER, TriggerPolicy, should_probe
from tests.test_probe_mapping import RecordingGenerator, diagnosis, nq_input, probe_json
from tests.test_probe_trigger import CONFIDENT_STAGE2, UNSURE_STAGE2, stage2_with_max


def row(p: float, stage2: float | None, target: str, final: str, dataset: str = "nq") -> dict:
    scores = None if stage2 is None else {"a": stage2, "b": (1 - stage2) / 2, "c": (1 - stage2) / 2}
    return {
        "dataset": dataset,
        "stage1_failure_probability": p,
        "stage2_class_scores": scores,
        "target_label": target,
        "final_label": final,
        "category": error_category(target, final),
    }


class TriggerPolicyTests(unittest.TestCase):
    def test_v2_constants(self):
        self.assertEqual((V2_EXPANDED_TRIGGER.low, V2_EXPANDED_TRIGGER.high, V2_EXPANDED_TRIGGER.stage2_threshold),
                         (0.50, 0.65, 0.50))
        self.assertEqual((V1_TRIGGER.low, V1_TRIGGER.high, V1_TRIGGER.stage2_threshold), (0.40, 0.55, 0.50))

    def test_band_inclusive(self):
        policy = TriggerPolicy(0.50, 0.65, 0.50)
        self.assertTrue(policy.should_probe(0.50, CONFIDENT_STAGE2))
        self.assertTrue(policy.should_probe(0.65, CONFIDENT_STAGE2))
        self.assertFalse(policy.should_probe(0.6501, CONFIDENT_STAGE2))
        self.assertFalse(policy.should_probe(0.4999, None))

    def test_stage2_rule_only_for_predicted_failure(self):
        policy = TriggerPolicy(0.50, 0.65, 0.50)
        self.assertTrue(policy.should_probe(0.90, stage2_with_max(0.4999)))
        self.assertFalse(policy.should_probe(0.90, stage2_with_max(0.50)))
        self.assertFalse(policy.should_probe(0.90, None))
        self.assertFalse(policy.should_probe(0.20, None))
        # Predicted failure below the band (Stage 1 threshold < low) still uses the Stage 2 rule.
        self.assertTrue(TriggerPolicy(0.55, 0.70, 0.50).should_probe(0.52, UNSURE_STAGE2))

    def test_v1_policy_matches_v1_function(self):
        for p, scores in itertools.product((0.40, 0.45, 0.55, 0.56, 0.80, 1.0),
                                           (CONFIDENT_STAGE2, UNSURE_STAGE2, stage2_with_max(0.50))):
            self.assertEqual(V1_TRIGGER.should_probe(p, scores), should_probe(p, scores))
        for p in (0.0, 0.2, 0.3999):
            self.assertEqual(V1_TRIGGER.should_probe(p, None), should_probe(p, None))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            TriggerPolicy(0.6, 0.5, 0.5)
        with self.assertRaises(ValueError):
            TriggerPolicy(0.4, 1.2, 0.5)
        policy = TriggerPolicy(0.5, 0.65, 0.5)
        with self.assertRaises(ValueError):
            policy.should_probe(float("nan"), None)
        with self.assertRaises(TypeError):
            policy.should_probe("0.5", None)
        with self.assertRaises(ValueError):
            policy.should_probe(0.9, {})

    def test_reads_only_diagnosis_outputs(self):
        params = list(inspect.signature(TriggerPolicy.should_probe).parameters)
        self.assertEqual(params, ["self", "stage1_failure_probability", "stage2_class_scores"])

    def test_pipeline_uses_given_policy(self):
        item = nq_input()
        d = diagnosis(0.45, DiagnosisLabel.CORRECT, None)
        v1 = apply_probe(item, d, EvidenceProbe(RecordingGenerator(probe_json(answer_supported="supported"))))
        self.assertTrue(v1.probe_triggered)
        generator = RecordingGenerator(probe_json(answer_supported="supported"))
        v2 = apply_probe(item, d, EvidenceProbe(generator), V2_EXPANDED_TRIGGER)
        self.assertFalse(v2.probe_triggered)
        self.assertEqual(generator.prompts, [])


class SearchTests(unittest.TestCase):
    def test_error_category(self):
        self.assertEqual(error_category("correct", "correct"), "diagnosis_correct")
        self.assertEqual(error_category("correct", "evidence_utilisation"), "A_correct_to_failure")
        self.assertEqual(error_category("evidence_insufficiency", "correct"), "B_failure_to_correct")
        self.assertEqual(error_category("evidence_insufficiency", "evidence_utilisation"), "C_failure_type_error")

    def test_distribution(self):
        d = distribution([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertAlmostEqual(d["mean"], 0.3)
        self.assertAlmostEqual(d["median"], 0.3)
        self.assertAlmostEqual(d["p25"], 0.2)
        self.assertAlmostEqual(d["p75"], 0.4)
        self.assertEqual((d["min"], d["max"]), (0.1, 0.5))
        self.assertEqual(distribution([]), {"n": 0})

    def test_candidate_grid(self):
        policies = candidate_policies()
        self.assertEqual(len(policies), 343)
        for p in policies:
            self.assertGreaterEqual(round(p.high - p.low, 2), 0.10)
            self.assertLessEqual(p.high, 0.80)
            self.assertTrue(0.25 <= p.low <= 0.55 and 0.35 <= p.stage2_threshold <= 0.65)

    def test_evaluate_policy_counts(self):
        rows = [
            row(0.55, 0.9, "correct", "evidence_utilisation"),            # A, band
            row(0.90, 0.40, "evidence_insufficiency", "evidence_utilisation"),  # C, stage2 rule
            row(0.30, None, "evidence_insufficiency", "correct"),          # B, not triggered
            row(0.10, None, "correct", "correct", "fever"),                # correct, not triggered
            row(0.60, 0.9, "evidence_insufficiency", "evidence_insufficiency", "fever"),  # self-check
        ]
        r = evaluate_policy(TriggerPolicy(0.50, 0.65, 0.50), rows)
        self.assertEqual(r["triggered"], 3)
        self.assertEqual(r["errors_captured"], 2)
        self.assertAlmostEqual(r["error_capture_rate"], 2 / 3)
        self.assertEqual(r["capture_by_category"]["B_failure_to_correct"]["captured"], 0)
        self.assertEqual(r["a_plus_c_captured"], 2)
        self.assertEqual(r["self_checks"], 1)
        self.assertEqual(r["by_dataset"]["fever"], {"triggered": 1, "n": 2, "coverage": 0.5})

    def test_select_policy(self):
        def result(cov, cap, ac, self_checks, low=0.5, high=0.6, s=0.5):
            return {"coverage": cov, "error_capture_rate": cap, "a_plus_c_capture_rate": ac,
                    "self_checks": self_checks, "low": low, "high": high, "stage2_threshold": s}
        out_of_range = result(0.40, 0.99, 0.99, 0)
        best = result(0.34, 0.80, 0.90, 30)
        near_cheaper = result(0.30, 0.79, 0.89, 20)
        worse_ac = result(0.25, 0.79, 0.80, 5)
        chosen = select_policy([out_of_range, best, near_cheaper, worse_ac])
        self.assertIs(chosen, near_cheaper)
        with self.assertRaises(RuntimeError):
            select_policy([out_of_range])


class EvaluationTests(unittest.TestCase):
    def test_probe_label_analysis(self):
        def r(dataset, probe_label, target):
            return {"dataset": dataset, "probe_result": {"probe_label": probe_label},
                    "offline_evaluation": {"target_label": target}}
        rows = [
            r("nq", "evidence_insufficiency", "evidence_insufficiency"),
            r("nq", "evidence_insufficiency", "correct"),
            r("nq", "correct", "evidence_insufficiency"),
            r("fever", "evidence_insufficiency", "correct"),
        ]
        a = probe_label_analysis(rows, ProbeLabel.EVIDENCE_INSUFFICIENCY)
        self.assertEqual((a["all"]["predicted"], a["all"]["true"], a["all"]["true_positives"]), (3, 2, 1))
        self.assertAlmostEqual(a["all"]["precision"], 1 / 3)
        self.assertAlmostEqual(a["nq"]["recall"], 0.5)
        self.assertEqual(a["fever"]["precision"], 0.0)
        self.assertIsNone(a["fever"]["recall"])


if __name__ == "__main__":
    unittest.main()
