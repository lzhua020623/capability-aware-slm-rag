"""Compatibility and result integrity for the optional Diagnosis framework."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

from src.compensation import EnhancementExecutor, EnhancementRequest, EnhancementResult, EnhancementStatus, RefinementConfig
from src.controller.controller import decide_action
from src.controller.schema import ControllerAction, ControllerDecision
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import EvidenceItem
from src.probe.schema import ProbeResult

ROOT = Path(__file__).resolve().parents[1]


class MergeCompatibilityTests(unittest.TestCase):
    def request(self):
        decision = ControllerDecision(ControllerAction.RETRIEVAL_ENHANCEMENT,
            DiagnosisLabel.EVIDENCE_INSUFFICIENCY, .9, False, False)
        return EnhancementRequest.from_runtime(query="Who designed fictional Luma?", initial_answer="Amber",
            evidence=[EvidenceItem(1, "fiction", "Luma was designed by Amber.", .9)],
            controller_decision=decision, dataset="nq")

    def execute(self, request, result):
        class Module:
            def enhance(self, entering):
                return result
        executor = EnhancementExecutor()
        executor.register(request.controller_decision.action, Module())
        return executor.execute(request)

    def test_old_refiner_and_new_interfaces_coexist(self):
        self.assertEqual(RefinementConfig().keep_k, 5)
        self.assertEqual(EnhancementExecutor().registered_actions(), frozenset())
        from src.compensation import CrossEncoderScorer, EvidenceRefiner
        self.assertTrue(callable(CrossEncoderScorer) and callable(EvidenceRefiner))

    def test_invalid_results_cannot_publish_an_answer_or_false_action_receipt(self):
        request = self.request()
        valid = EnhancementResult(action=request.controller_decision.action, final_answer="Different",
            evidence=request.evidence, changed_answer=True, status="success")
        self.assertIs(self.execute(request, valid), valid)
        variants = [replace(valid, action=ControllerAction.GENERATION_REPAIR),
                    replace(valid, changed_answer=False)]
        for status in (EnhancementStatus.ERROR, EnhancementStatus.UNAVAILABLE, EnhancementStatus.NO_ACTION):
            variants.append(replace(valid, status=status))
            variants.append(replace(valid, status=status, final_answer=request.initial_answer,
                changed_answer=False, evidence=tuple(reversed(request.evidence)) + request.evidence))
        for result in variants:
            with self.subTest(action=result.action, status=result.status, changed=result.changed_answer):
                observed = self.execute(request, result)
                self.assertEqual(observed.status, EnhancementStatus.ERROR)
                self.assertEqual(observed.action, request.controller_decision.action)
                self.assertEqual(observed.final_answer, request.initial_answer)
                self.assertEqual(observed.evidence, request.evidence)
                self.assertFalse(observed.changed_answer)

    def test_nonfinite_latency_rejected(self):
        request = self.request()
        for value in (float("inf"), float("nan"), -1):
            with self.assertRaises(ValueError):
                EnhancementResult(action=request.controller_decision.action,
                    final_answer=request.initial_answer, evidence=request.evidence,
                    changed_answer=False, status="success", latency_ms=value)

    def test_legacy_runtime_never_implicitly_loads_or_selects_p60(self):
        from tests.test_controller_integration import pipeline
        from scripts.run_controller_integration import smoke_inputs
        with patch("src.controller.selection.FrozenOnlineSelectionAdapter.from_frozen_defaults",
                   side_effect=AssertionError("P60 must be explicit")):
            result = pipeline().run(smoke_inputs()[0])
        self.assertEqual(result["routing"]["decision"]["action"], "no_action")
        self.assertEqual(result["calls"], [])

    def test_all_800_historical_legacy_decisions_preserve_existing_fields(self):
        checked = 0
        with zipfile.ZipFile(ROOT / "artifacts/controller/t4_v1/controller_t4_development_v1_outputs.zip") as z:
            with z.open("predictions.jsonl") as source:
                for line in source:
                    r = json.loads(line)
                    if r["mode"] not in ("adaptive", "adaptive_no_probe"):
                        continue
                    d = r["routing"]["diagnosis"]
                    d["final_label"] = DiagnosisLabel(d["final_label"])
                    for name in ("stage2_class_scores", "class_scores"):
                        if d[name] is not None:
                            d[name] = {DiagnosisLabel(k): v for k, v in d[name].items()}
                    raw = r["routing"]["probe"]
                    probe = None if raw is None else ProbeResult(**raw)
                    decision = decide_action(HierarchicalDiagnosis(**d), probe is not None, probe, r["sample_id"]).to_dict()
                    self.assertEqual({k: decision[k] for k in r["routing"]["decision"]}, r["routing"]["decision"])
                    checked += 1
        self.assertEqual(checked, 800)


if __name__ == "__main__":
    unittest.main()
