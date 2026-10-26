"""FR6 — multiple failure representation (additive; single primary route unchanged)."""

from __future__ import annotations

import unittest

from src.controller.controller import decide, decision_from_selection
from src.controller.failure_representation import (
    ACTION_PRIORITY,
    STAGE2_UNIFORM_PRIOR,
    primary_among_active,
    representation_from_diagnosis,
)
from src.controller.schema import ControllerAction, SelectionDecision, SelectionLane
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
from src.diagnosis.labels import DiagnosisLabel


def _diag(
    *,
    p_fail: float,
    stage2: dict[DiagnosisLabel, float],
    final: DiagnosisLabel,
    stage1: str = "failure",
) -> HierarchicalDiagnosis:
    joint = {DiagnosisLabel.CORRECT: 1.0 - p_fail}
    joint.update({k: p_fail * v for k, v in stage2.items()})
    return HierarchicalDiagnosis(
        stage1_failure_probability=p_fail,
        stage1_label=stage1,
        stage2_class_scores=stage2 if stage1 == "failure" else None,
        final_label=final,
        confidence=joint.get(final, 1.0 - p_fail),
        class_scores=joint if stage1 == "failure" else None,
    )


class FR6RepresentationTests(unittest.TestCase):
    def test_no_failure_empty_active(self):
        d = _diag(
            p_fail=0.2,
            stage2={
                DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.5,
                DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.3,
                DiagnosisLabel.EVIDENCE_UTILISATION: 0.2,
            },
            final=DiagnosisLabel.CORRECT,
            stage1="correct",
        )
        # correct path clears stage2 in real classifier; mimic that
        d = HierarchicalDiagnosis(0.2, "correct", None, DiagnosisLabel.CORRECT, 0.8, None)
        fr = representation_from_diagnosis(d)
        self.assertEqual(fr.active_failures, ())
        self.assertIs(fr.primary_failure_type, DiagnosisLabel.CORRECT)
        self.assertIs(fr.failure_type, DiagnosisLabel.CORRECT)

    def test_insuff_only(self):
        fr = representation_from_diagnosis(
            _diag(
                p_fail=0.8,
                stage2={
                    DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.70,
                    DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.20,
                    DiagnosisLabel.EVIDENCE_UTILISATION: 0.10,
                },
                final=DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
            )
        )
        self.assertEqual(fr.active_failures, (DiagnosisLabel.EVIDENCE_INSUFFICIENCY,))
        self.assertIs(fr.primary_failure_type, DiagnosisLabel.EVIDENCE_INSUFFICIENCY)

    def test_access_only(self):
        fr = representation_from_diagnosis(
            _diag(
                p_fail=0.8,
                stage2={
                    DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.20,
                    DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.70,
                    DiagnosisLabel.EVIDENCE_UTILISATION: 0.10,
                },
                final=DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
            )
        )
        self.assertEqual(fr.active_failures, (DiagnosisLabel.EVIDENCE_ACCESSIBILITY,))

    def test_util_only(self):
        fr = representation_from_diagnosis(
            _diag(
                p_fail=0.8,
                stage2={
                    DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.15,
                    DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.15,
                    DiagnosisLabel.EVIDENCE_UTILISATION: 0.70,
                },
                final=DiagnosisLabel.EVIDENCE_UTILISATION,
            )
        )
        self.assertEqual(fr.active_failures, (DiagnosisLabel.EVIDENCE_UTILISATION,))

    def test_insuff_plus_util(self):
        fr = representation_from_diagnosis(
            _diag(
                p_fail=0.9,
                stage2={
                    DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.40,
                    DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.20,
                    DiagnosisLabel.EVIDENCE_UTILISATION: 0.40,
                },
                final=DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
            )
        )
        self.assertEqual(
            fr.active_failures,
            (
                DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
                DiagnosisLabel.EVIDENCE_UTILISATION,
            ),
        )
        self.assertGreaterEqual(
            fr.failure_scores[DiagnosisLabel.EVIDENCE_INSUFFICIENCY.value], STAGE2_UNIFORM_PRIOR * 0.9 - 1e-9
        )

    def test_access_plus_util(self):
        fr = representation_from_diagnosis(
            _diag(
                p_fail=0.85,
                stage2={
                    DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.20,
                    DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.40,
                    DiagnosisLabel.EVIDENCE_UTILISATION: 0.40,
                },
                final=DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
            )
        )
        self.assertEqual(
            fr.active_failures,
            (
                DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
                DiagnosisLabel.EVIDENCE_UTILISATION,
            ),
        )

    def test_insuff_plus_access(self):
        fr = representation_from_diagnosis(
            _diag(
                p_fail=0.85,
                stage2={
                    DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.40,
                    DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.40,
                    DiagnosisLabel.EVIDENCE_UTILISATION: 0.20,
                },
                final=DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
            )
        )
        self.assertEqual(
            fr.active_failures,
            (
                DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
                DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
            ),
        )

    def test_three_active(self):
        # Equal conditionals all == 1/3 → all three active
        fr = representation_from_diagnosis(
            _diag(
                p_fail=0.9,
                stage2={
                    DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 1.0 / 3.0,
                    DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 1.0 / 3.0,
                    DiagnosisLabel.EVIDENCE_UTILISATION: 1.0 / 3.0,
                },
                final=DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
            )
        )
        self.assertEqual(fr.active_failures, ACTION_PRIORITY)

    def test_primary_route_unique_and_priority(self):
        active = (
            DiagnosisLabel.EVIDENCE_UTILISATION,
            DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
            DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
        )
        self.assertIs(
            primary_among_active(active),
            DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
        )
        fr = representation_from_diagnosis(
            _diag(
                p_fail=0.9,
                stage2={
                    DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.40,
                    DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.35,
                    DiagnosisLabel.EVIDENCE_UTILISATION: 0.25,
                },
                final=DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
            )
        )
        # Existing Diagnosis argmax remains primary; action priority does not rewrite it.
        self.assertIs(fr.primary_failure_type, DiagnosisLabel.EVIDENCE_INSUFFICIENCY)
        d = decide(
            SelectionDecision(
                needs_enhancement=True,
                failure_type=fr.primary_failure_type,
                selection_score=0.7,
                selection_lane=SelectionLane.INSUFF_PRESERVE,
                confidence=0.55,
                p_fail=0.9,
                p_insuff=0.36,
                primary_failure_type=fr.primary_failure_type,
                active_failures=fr.active_failures,
                failure_scores=fr.failure_scores,
            )
        )
        self.assertIs(d.action, ControllerAction.RETRIEVAL_ENHANCEMENT)
        self.assertEqual(d.active_failures, fr.active_failures)
        self.assertIs(d.primary_failure_type, DiagnosisLabel.EVIDENCE_INSUFFICIENCY)
        self.assertIs(d.failure_type, d.primary_failure_type)

    def test_no_action_unchanged(self):
        fr = representation_from_diagnosis(
            _diag(
                p_fail=0.8,
                stage2={
                    DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.40,
                    DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.35,
                    DiagnosisLabel.EVIDENCE_UTILISATION: 0.25,
                },
                final=DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
            )
        )
        sel = SelectionDecision(
            needs_enhancement=False,
            failure_type=fr.primary_failure_type,
            selection_score=0.1,
            selection_lane=SelectionLane.NO_ACTION,
            confidence=0.4,
            p_fail=0.8,
            p_insuff=0.32,
            primary_failure_type=fr.primary_failure_type,
            active_failures=fr.active_failures,
            failure_scores=fr.failure_scores,
        )
        d = decision_from_selection(sel)
        self.assertIs(d.action, ControllerAction.NO_ACTION)
        self.assertTrue(d.preserves_initial_answer)
        self.assertEqual(d.reason, "not_selected")
        # Representation retained when Diagnosis was failure but not selected.
        self.assertEqual(d.active_failures, fr.active_failures)


if __name__ == "__main__":
    unittest.main()
