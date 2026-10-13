"""Tests for Controller v1 routing and Safe Binary resolution. No model is loaded."""

from __future__ import annotations

import dataclasses
import inspect
import subprocess
import sys
import unittest
from pathlib import Path

from src.controller.controller import (
    ACTION_BY_LABEL,
    SAFE_OVERRIDE_MAX_STAGE1,
    SAFE_OVERRIDE_MAX_STAGE2,
    decide_action,
    decide_from_probed,
    resolve_final_diagnosis,
    route_controller,
    run_controller,
)
from src.controller.schema import ControllerAction, ControllerDecision
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import FORBIDDEN_RUNTIME_FIELDS
from src.probe.pipeline import apply_probe
from src.probe.probe import EvidenceProbe
from src.probe.schema import ProbeResult
from src.probe.trigger import V2_EXPANDED_TRIGGER
from tests.test_probe_mapping import RecordingGenerator, nq_input, probe_json

ROOT = Path(__file__).resolve().parents[1]
FAILURES = (
    DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
    DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
    DiagnosisLabel.EVIDENCE_UTILISATION,
)
LOW_STAGE2 = {
    DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.40,
    DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.35,
    DiagnosisLabel.EVIDENCE_UTILISATION: 0.25,
}
HIGH_STAGE2 = {
    DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.70,
    DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.20,
    DiagnosisLabel.EVIDENCE_UTILISATION: 0.10,
}


def probe(label: str) -> ProbeResult:
    judgments = {
        "correct": ("supported", "sufficient", "clear"),
        "evidence_insufficiency": ("unsupported", "insufficient", "uncertain"),
        "evidence_accessibility": ("unsupported", "sufficient", "hard_to_access"),
        "evidence_utilisation": ("unsupported", "sufficient", "clear"),
        "uncertain": ("uncertain", "uncertain", "uncertain"),
    }[label]
    return ProbeResult.from_judgments(*judgments)


def failure_diagnosis(p: float = 0.60, label: DiagnosisLabel = DiagnosisLabel.EVIDENCE_UTILISATION,
                      stage2: dict | None = None) -> HierarchicalDiagnosis:
    return HierarchicalDiagnosis(
        stage1_failure_probability=p,
        stage1_label="failure",
        stage2_class_scores=LOW_STAGE2 if stage2 is None else stage2,
        final_label=label,
        confidence=0.25,
        class_scores=None,
    )


def correct_diagnosis(p: float = 0.30) -> HierarchicalDiagnosis:
    return HierarchicalDiagnosis(p, "correct", None, DiagnosisLabel.CORRECT, 1.0 - p, None)


class RoutingTests(unittest.TestCase):
    def test_mapping(self):
        self.assertIs(route_controller(DiagnosisLabel.CORRECT), ControllerAction.NO_ACTION)
        self.assertIs(route_controller(DiagnosisLabel.EVIDENCE_INSUFFICIENCY), ControllerAction.RETRIEVAL_ENHANCEMENT)
        self.assertIs(route_controller(DiagnosisLabel.EVIDENCE_ACCESSIBILITY), ControllerAction.EVIDENCE_REFINEMENT)
        self.assertIs(route_controller(DiagnosisLabel.EVIDENCE_UTILISATION), ControllerAction.GENERATION_REPAIR)
        self.assertIs(route_controller("evidence_utilisation"), ControllerAction.GENERATION_REPAIR)

    def test_mapping_is_complete_and_one_to_one(self):
        self.assertEqual(set(ACTION_BY_LABEL), set(DiagnosisLabel))
        self.assertEqual(set(ACTION_BY_LABEL.values()), set(ControllerAction))

    def test_unknown_label_raises(self):
        for bad in ("uncertain", "Correct", "", "evidence_utilization", None, 3):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                route_controller(bad)

    def test_action_values(self):
        self.assertEqual([a.value for a in ControllerAction],
                         ["no_action", "retrieval_enhancement", "evidence_refinement", "generation_repair"])


class SafeBinaryTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual((SAFE_OVERRIDE_MAX_STAGE1, SAFE_OVERRIDE_MAX_STAGE2), (0.70, 0.50))

    def test_not_triggered_keeps_diagnosis(self):
        for label in FAILURES:
            self.assertIs(resolve_final_diagnosis(label, 0.60, LOW_STAGE2, False, None), label)
        self.assertIs(resolve_final_diagnosis(DiagnosisLabel.CORRECT, 0.30, None, False, None), DiagnosisLabel.CORRECT)

    def test_uncertain_or_parse_failure_keeps_diagnosis(self):
        unparsable = ProbeResult.unparsable("garbage", "no JSON")
        for result in (probe("uncertain"), unparsable):
            self.assertIs(resolve_final_diagnosis(DiagnosisLabel.EVIDENCE_ACCESSIBILITY, 0.60, LOW_STAGE2, True, result),
                          DiagnosisLabel.EVIDENCE_ACCESSIBILITY)

    def test_correct_diagnosis_stays_correct(self):
        for label in ("correct",) + tuple(f.value for f in FAILURES):
            self.assertIs(resolve_final_diagnosis(DiagnosisLabel.CORRECT, 0.30, None, True, probe(label)),
                          DiagnosisLabel.CORRECT)

    def test_safe_override(self):
        for label in FAILURES:
            self.assertIs(resolve_final_diagnosis(label, 0.60, LOW_STAGE2, True, probe("correct")),
                          DiagnosisLabel.CORRECT)
        boundary = {DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.50, DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.30,
                    DiagnosisLabel.EVIDENCE_UTILISATION: 0.20}
        self.assertIs(resolve_final_diagnosis(FAILURES[0], 0.70, boundary, True, probe("correct")),
                      DiagnosisLabel.CORRECT)

    def test_no_override_above_stage1_threshold(self):
        for p in (0.7001, 0.80, 1.0):
            self.assertIs(resolve_final_diagnosis(FAILURES[2], p, LOW_STAGE2, True, probe("correct")), FAILURES[2])

    def test_no_override_above_stage2_threshold(self):
        just_above = {DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.5001, DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.30,
                      DiagnosisLabel.EVIDENCE_UTILISATION: 0.1999}
        for scores in (HIGH_STAGE2, just_above):
            self.assertIs(resolve_final_diagnosis(FAILURES[0], 0.55, scores, True, probe("correct")), FAILURES[0])

    def test_no_override_without_stage2_scores(self):
        self.assertIs(resolve_final_diagnosis(FAILURES[1], 0.55, None, True, probe("correct")), FAILURES[1])

    def test_probe_never_changes_failure_type(self):
        for original in FAILURES:
            for probe_label in FAILURES:
                with self.subTest(original=original, probe=probe_label):
                    self.assertIs(
                        resolve_final_diagnosis(original, 0.55, LOW_STAGE2, True, probe(probe_label.value)), original)

    def test_inconsistent_inputs_raise(self):
        with self.assertRaises(ValueError):
            resolve_final_diagnosis(FAILURES[0], 0.6, LOW_STAGE2, True, None)
        with self.assertRaises(ValueError):
            resolve_final_diagnosis(FAILURES[0], 0.6, LOW_STAGE2, False, probe("correct"))
        with self.assertRaises(ValueError):
            resolve_final_diagnosis("unknown", 0.6, LOW_STAGE2, False, None)
        with self.assertRaises(ValueError):
            resolve_final_diagnosis(FAILURES[0], 1.5, LOW_STAGE2, False, None)
        with self.assertRaises(TypeError):
            resolve_final_diagnosis(FAILURES[0], 0.6, LOW_STAGE2, True, {"probe_label": "correct"})


class DecisionTests(unittest.TestCase):
    def test_decision_fields(self):
        d = decide_action(failure_diagnosis(), True, probe("correct"), sample_id="s1")
        self.assertIs(d.action, ControllerAction.NO_ACTION)
        self.assertIs(d.diagnosis_label, DiagnosisLabel.CORRECT)
        self.assertEqual(d.diagnosis_confidence, 0.25)
        self.assertTrue(d.used_probe and d.probe_overrode_diagnosis)
        self.assertFalse(d.needs_compensation)

        kept = decide_action(failure_diagnosis(p=0.9), True, probe("correct"))
        self.assertIs(kept.action, ControllerAction.GENERATION_REPAIR)
        self.assertTrue(kept.used_probe)
        self.assertFalse(kept.probe_overrode_diagnosis)
        self.assertTrue(kept.needs_compensation)

        plain = decide_action(correct_diagnosis(), False, None)
        self.assertIs(plain.action, ControllerAction.NO_ACTION)
        self.assertFalse(plain.used_probe or plain.probe_overrode_diagnosis or plain.needs_compensation)

    def test_needs_compensation_is_derived(self):
        for label, action in ACTION_BY_LABEL.items():
            d = ControllerDecision(action, label, 0.5, False, False)
            self.assertEqual(d.needs_compensation, action is not ControllerAction.NO_ACTION)
        with self.assertRaises(TypeError):
            ControllerDecision(ControllerAction.NO_ACTION, DiagnosisLabel.CORRECT, 0.5, False, False,
                               needs_compensation=True)
        with self.assertRaises(ValueError):
            ControllerDecision(ControllerAction.NO_ACTION, DiagnosisLabel.CORRECT, 0.5, False, True)
        with self.assertRaises(ValueError):
            ControllerDecision("repair_everything", DiagnosisLabel.CORRECT, 0.5, False, False)

    def test_sample_id_does_not_affect_decision(self):
        for sample_id in (None, "a", "-1608173516669914422", "test-123"):
            self.assertEqual(decide_action(failure_diagnosis(), True, probe("correct"), sample_id=sample_id),
                             decide_action(failure_diagnosis(), True, probe("correct")))
        a = decide_action(failure_diagnosis(p=0.9), True, probe("correct"), sample_id="x").to_dict()
        b = decide_action(failure_diagnosis(p=0.9), True, probe("correct"), sample_id="y").to_dict()
        a.pop("sample_id"), b.pop("sample_id")
        self.assertEqual(a, b)

    def test_no_gold_or_evaluation_inputs(self):
        forbidden = set(FORBIDDEN_RUNTIME_FIELDS) | {"target_label", "metadata", "needs_manual_review", "gold",
                                                     "oracle", "label", "offline_evaluation"}
        for fn in (resolve_final_diagnosis, route_controller, decide_action, decide_from_probed, run_controller):
            self.assertFalse(set(inspect.signature(fn).parameters) & forbidden, fn.__name__)
        fields = {f.name for f in dataclasses.fields(ControllerDecision)}
        self.assertEqual(fields, {"action", "diagnosis_label", "diagnosis_confidence", "used_probe",
                                  "probe_overrode_diagnosis", "sample_id", "needs_compensation"})
        source = (ROOT / "src" / "controller" / "controller.py").read_text(encoding="utf-8")
        for token in ("target_label", "metadata", "needs_manual_review", "gold_", "offline_evaluation"):
            self.assertNotIn(token, source)

    def test_deterministic(self):
        decisions = {decide_action(failure_diagnosis(), True, probe("correct")) for _ in range(5)}
        self.assertEqual(len(decisions), 1)


class PipelineTests(unittest.TestCase):
    def test_decide_from_probed_ignores_typed_probe_label(self):
        item = nq_input()
        diagnosis = failure_diagnosis(p=0.60, label=DiagnosisLabel.EVIDENCE_ACCESSIBILITY)
        output = probe_json(answer_supported="unsupported", evidence_sufficiency="insufficient",
                            evidence_accessibility="uncertain", conflicting_ranks=[])
        probed = apply_probe(item, diagnosis, EvidenceProbe(RecordingGenerator(output)), V2_EXPANDED_TRIGGER)
        self.assertIs(probed.final_label, DiagnosisLabel.EVIDENCE_INSUFFICIENCY)
        decision = decide_from_probed(probed)
        self.assertIs(decision.diagnosis_label, DiagnosisLabel.EVIDENCE_ACCESSIBILITY)
        self.assertIs(decision.action, ControllerAction.EVIDENCE_REFINEMENT)

    def test_run_controller_end_to_end(self):
        class StubClassifier:
            def __init__(self, diagnosis):
                self.diagnosis = diagnosis

            def diagnose(self, item):
                return self.diagnosis

        item = nq_input()
        generator = RecordingGenerator(probe_json(answer_supported="supported", conflicting_ranks=[],
                                                  supporting_ranks=[1]))
        overridden = run_controller(item, StubClassifier(failure_diagnosis(p=0.60)), EvidenceProbe(generator))
        self.assertIs(overridden.action, ControllerAction.NO_ACTION)
        self.assertTrue(overridden.probe_overrode_diagnosis)
        self.assertEqual(overridden.sample_id, item.sample_id)
        self.assertEqual(len(generator.prompts), 1)

        silent = RecordingGenerator(probe_json(answer_supported="supported"))
        untouched = run_controller(item, StubClassifier(correct_diagnosis(0.30)), EvidenceProbe(silent))
        self.assertIs(untouched.action, ControllerAction.NO_ACTION)
        self.assertFalse(untouched.used_probe)
        self.assertEqual(silent.prompts, [])

        confident = RecordingGenerator(probe_json(answer_supported="supported"))
        routed = run_controller(item, StubClassifier(failure_diagnosis(p=0.90, stage2=HIGH_STAGE2,
                                                                       label=DiagnosisLabel.EVIDENCE_INSUFFICIENCY)),
                                EvidenceProbe(confident))
        self.assertIs(routed.action, ControllerAction.RETRIEVAL_ENHANCEMENT)
        self.assertFalse(routed.used_probe)
        self.assertEqual(confident.prompts, [])

    def test_controller_import_loads_no_models(self):
        code = (
            "import sys; import src.controller, src.controller.controller, src.controller.schema; "
            "loaded = [m for m in ('torch', 'transformers', 'src.generation.qwen', 'src.retrieval.e5_faiss') "
            "if m in sys.modules]; sys.exit(1 if loaded else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
