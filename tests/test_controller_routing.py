"""P62 Controller routing contract tests. No Enhancement modules, no Test data, no gold."""

from __future__ import annotations

import inspect
import unittest

from src.controller.controller import (
    ACTION_BY_LABEL,
    decide,
    decide_from_raw_selection,
    decision_from_selection,
)
from src.controller.normalization import FAILURE_TYPE_ALIASES, normalize_failure_type
from src.controller.schema import (
    ACTION_CONCEPTUAL_ALIASES,
    ControllerAction,
    ControllerDecision,
    SelectionDecision,
    SelectionLane,
)
from src.diagnosis.labels import DiagnosisLabel
from tests.test_probe_mapping import nq_input


def sel(
    *,
    needs: bool,
    label: DiagnosisLabel | str,
    lane: SelectionLane,
    score: float = 0.7,
    confidence: float = 0.55,
    p_fail: float = 0.8,
    p_insuff: float = 0.6,
    sample_id: str | None = "s1",
) -> SelectionDecision:
    if not needs:
        lane = SelectionLane.NO_ACTION
    return SelectionDecision(
        needs_enhancement=needs,
        failure_type=label,
        selection_score=score,
        selection_lane=lane,
        confidence=confidence,
        p_fail=p_fail,
        p_insuff=p_insuff,
        used_probe=False,
        sample_id=sample_id,
    )


class NormalizationTests(unittest.TestCase):
    def test_canonical_and_aliases(self):
        self.assertIs(normalize_failure_type("evidence_insufficiency"), DiagnosisLabel.EVIDENCE_INSUFFICIENCY)
        self.assertIs(normalize_failure_type("insuff"), DiagnosisLabel.EVIDENCE_INSUFFICIENCY)
        self.assertIs(normalize_failure_type("access"), DiagnosisLabel.EVIDENCE_ACCESSIBILITY)
        self.assertIs(normalize_failure_type("util"), DiagnosisLabel.EVIDENCE_UTILISATION)
        self.assertIs(normalize_failure_type("utilization"), DiagnosisLabel.EVIDENCE_UTILISATION)
        self.assertIs(normalize_failure_type(DiagnosisLabel.CORRECT), DiagnosisLabel.CORRECT)

    def test_unknown_returns_none(self):
        self.assertIsNone(normalize_failure_type("bogus_subtype"))
        self.assertIsNone(normalize_failure_type(""))
        self.assertIsNone(normalize_failure_type(None))
        self.assertIsNone(normalize_failure_type(123))

    def test_aliases_are_centralized(self):
        self.assertIn("insuff", FAILURE_TYPE_ALIASES)
        self.assertIn("access", FAILURE_TYPE_ALIASES)
        self.assertIn("util", FAILURE_TYPE_ALIASES)


class RoutingContractTests(unittest.TestCase):
    def test_no_action_when_not_selected(self):
        # Even if failure_type looks like util, not selected => no_action
        for label in (
            DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
            DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
            DiagnosisLabel.EVIDENCE_UTILISATION,
            DiagnosisLabel.CORRECT,
        ):
            d = decide(sel(needs=False, label=label, lane=SelectionLane.NO_ACTION, score=0.1))
            self.assertIs(d.action, ControllerAction.NO_ACTION)
            self.assertFalse(d.needs_enhancement)
            self.assertTrue(d.preserves_initial_answer)
            self.assertFalse(d.used_probe)
            self.assertEqual(d.reason, "not_selected")
            self.assertIs(d.selection_lane, SelectionLane.NO_ACTION)

    def test_selected_insuff_to_retrieval(self):
        d = decide(
            sel(
                needs=True,
                label=DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
                lane=SelectionLane.INSUFF_PRESERVE,
                p_insuff=0.7,
            )
        )
        self.assertIs(d.action, ControllerAction.RETRIEVAL_ENHANCEMENT)
        self.assertTrue(d.needs_enhancement)
        self.assertFalse(d.preserves_initial_answer)
        self.assertIs(d.failure_type, DiagnosisLabel.EVIDENCE_INSUFFICIENCY)

    def test_selected_access_to_evidence_refinement(self):
        d = decide(
            sel(
                needs=True,
                label=DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
                lane=SelectionLane.RESIDUAL_COMBO,
            )
        )
        self.assertIs(d.action, ControllerAction.EVIDENCE_REFINEMENT)
        self.assertEqual(ACTION_CONCEPTUAL_ALIASES["accessibility_enhancement"], d.action.value)

    def test_selected_util_to_generation_repair(self):
        d = decide(
            sel(
                needs=True,
                label=DiagnosisLabel.EVIDENCE_UTILISATION,
                lane=SelectionLane.RESIDUAL_COMBO,
            )
        )
        self.assertIs(d.action, ControllerAction.GENERATION_REPAIR)
        self.assertEqual(ACTION_CONCEPTUAL_ALIASES["utilisation_enhancement"], d.action.value)

    def test_selection_lane_does_not_override_failure_type(self):
        # Lane1 insuff_preserve but Diagnosis subtype is utilisation -> util route
        d = decide(
            sel(
                needs=True,
                label=DiagnosisLabel.EVIDENCE_UTILISATION,
                lane=SelectionLane.INSUFF_PRESERVE,
                p_insuff=0.9,
            )
        )
        self.assertIs(d.action, ControllerAction.GENERATION_REPAIR)
        self.assertIs(d.selection_lane, SelectionLane.INSUFF_PRESERVE)
        self.assertIs(d.failure_type, DiagnosisLabel.EVIDENCE_UTILISATION)

    def test_alias_normalization_on_selection_decision(self):
        d = decide(sel(needs=True, label="access", lane=SelectionLane.RESIDUAL_COMBO))
        self.assertIs(d.action, ControllerAction.EVIDENCE_REFINEMENT)
        self.assertIs(d.failure_type, DiagnosisLabel.EVIDENCE_ACCESSIBILITY)

    def test_unknown_subtype_safe_no_action(self):
        d = decide_from_raw_selection(
            needs_enhancement=True,
            failure_type="not_a_real_label",
            selection_score=0.9,
            selection_lane=SelectionLane.RESIDUAL_COMBO,
            confidence=0.5,
        )
        self.assertIs(d.action, ControllerAction.NO_ACTION)
        self.assertTrue(d.preserves_initial_answer)
        self.assertIn("unknown_failure_type", d.reason or "")
        self.assertFalse(d.used_probe)

    def test_probe_off_fields(self):
        d = decide(
            sel(needs=True, label=DiagnosisLabel.EVIDENCE_INSUFFICIENCY, lane=SelectionLane.INSUFF_PRESERVE)
        )
        self.assertFalse(d.used_probe)
        self.assertFalse(d.probe_overrode_diagnosis)

    def test_no_action_keeps_initial_answer_reference(self):
        item = nq_input()
        d = decide(sel(needs=False, label=DiagnosisLabel.EVIDENCE_ACCESSIBILITY, lane=SelectionLane.NO_ACTION))
        self.assertTrue(d.preserves_initial_answer)
        # Host contract: when preserves_initial_answer, keep item.initial_answer unchanged.
        final_answer = item.initial_answer if d.preserves_initial_answer else "<enhanced>"
        self.assertEqual(final_answer, item.initial_answer)

    def test_controller_does_not_import_enhancement_modules(self):
        import src.controller.controller as mod

        source = inspect.getsource(mod)
        self.assertNotIn("src.compensation", source)
        self.assertNotIn("from src.compensation", source)
        self.assertNotIn("import src.compensation", source)
        # Controller must not name teammate / concrete enhancement classes.
        for token in ("P320Retrieval", "EvidenceRefiner", "generation_repair(", "answer_repair"):
            self.assertNotIn(token, source)

    def test_action_by_label_complete(self):
        self.assertEqual(set(ACTION_BY_LABEL), set(DiagnosisLabel))
        self.assertIs(ACTION_BY_LABEL[DiagnosisLabel.EVIDENCE_INSUFFICIENCY], ControllerAction.RETRIEVAL_ENHANCEMENT)
        self.assertIs(ACTION_BY_LABEL[DiagnosisLabel.EVIDENCE_ACCESSIBILITY], ControllerAction.EVIDENCE_REFINEMENT)
        self.assertIs(ACTION_BY_LABEL[DiagnosisLabel.EVIDENCE_UTILISATION], ControllerAction.GENERATION_REPAIR)

    def test_decision_from_selection_alias(self):
        s = sel(needs=True, label="insuff", lane=SelectionLane.INSUFF_PRESERVE)
        self.assertEqual(decide(s), decision_from_selection(s))


class IntegrationFixtureTests(unittest.TestCase):
    """Minimal non-Test fixture: four canonical SelectionDecision -> ControllerDecision cases."""

    CASES = (
        ("A_not_selected", False, DiagnosisLabel.EVIDENCE_UTILISATION, SelectionLane.NO_ACTION, ControllerAction.NO_ACTION),
        ("B_selected_insuff", True, DiagnosisLabel.EVIDENCE_INSUFFICIENCY, SelectionLane.INSUFF_PRESERVE, ControllerAction.RETRIEVAL_ENHANCEMENT),
        ("C_selected_access", True, DiagnosisLabel.EVIDENCE_ACCESSIBILITY, SelectionLane.RESIDUAL_COMBO, ControllerAction.EVIDENCE_REFINEMENT),
        ("D_selected_util", True, DiagnosisLabel.EVIDENCE_UTILISATION, SelectionLane.RESIDUAL_COMBO, ControllerAction.GENERATION_REPAIR),
    )

    def test_four_way_fixture(self):
        results = []
        for name, needs, label, lane, expected_action in self.CASES:
            selection = sel(needs=needs, label=label, lane=lane, sample_id=name)
            decision = decide(selection)
            results.append(
                {
                    "case": name,
                    "selection": selection.to_dict(),
                    "controller": decision.to_dict(),
                }
            )
            self.assertIs(decision.action, expected_action, name)
            self.assertEqual(decision.needs_enhancement, needs and expected_action is not ControllerAction.NO_ACTION)
            self.assertFalse(decision.used_probe)
            if needs:
                self.assertIs(decision.failure_type, label)
            else:
                self.assertIs(decision.action, ControllerAction.NO_ACTION)
                self.assertTrue(decision.preserves_initial_answer)
        self.assertEqual(len(results), 4)


if __name__ == "__main__":
    unittest.main()
