"""P64 end-to-end Adaptive RAG host pipeline tests (virtual Enhancement slots)."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from src.compensation.executor import EnhancementExecutor
from src.compensation.mocks import (
    MockEvidenceRefinementEnhancement,
    MockGenerationRepairEnhancement,
    MockRetrievalEnhancement,
    RaisingEnhancement,
)
from src.controller.schema import SelectionDecision, SelectionLane
from src.controller.controller import decide_from_raw_selection
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import DatasetName, DiagnosisInput, EvidenceItem
from src.pipeline import (
    AdaptiveRAGPipeline,
    PipelineStatus,
    build_adaptive_pipeline,
    run_adaptive_rag,
    run_adaptive_rag_from_item,
)


def _evidence(n: int = 2) -> list[EvidenceItem]:
    return [EvidenceItem(i + 1, f"p{i}", f"text-{i}", 1.0 - 0.1 * i) for i in range(n)]


def make_item(
    *,
    sample_id: str = "syn-1",
    query: str = "synthetic question",
    answer: str = "initial answer",
    evidence: list[EvidenceItem] | None = None,
) -> DiagnosisInput:
    return DiagnosisInput(
        sample_id=sample_id,
        dataset=DatasetName.NQ,
        query=query,
        evidence=_evidence() if evidence is None else evidence,
        initial_answer=answer,
    )


class StubReferenceRAG:
    def __init__(self, item: DiagnosisInput):
        self.item = item
        self.calls = 0

    def run(self, query: str, *, sample_id: str | None = None, dataset: str = "nq") -> DiagnosisInput:
        self.calls += 1
        return make_item(
            sample_id=sample_id or self.item.sample_id,
            query=query,
            answer=self.item.initial_answer,
            evidence=list(self.item.evidence),
        )


class StubClassifier:
    def __init__(self, diagnosis: HierarchicalDiagnosis):
        self.diagnosis = diagnosis
        self.calls = 0

    def diagnose(self, item: DiagnosisInput) -> HierarchicalDiagnosis:
        self.calls += 1
        return self.diagnosis


class StubFeatures:
    def __init__(self, n: int = 8):
        self.n = n
        self.calls = 0

    def features_for(self, item, diagnosis):
        self.calls += 1
        return np.zeros(self.n)


class StubSelection:
    """Forces a SelectionDecision without touching frozen thresholds."""

    def __init__(self, selection: SelectionDecision):
        self.selection = selection
        self.calls = 0

    def select(self, diagnosis, adapter_features, *, sample_id=None):
        self.calls += 1
        return replace(self.selection, sample_id=sample_id or self.selection.sample_id)


def failure_diag(label: DiagnosisLabel) -> HierarchicalDiagnosis:
    p_fail = 0.8
    joint = {
        DiagnosisLabel.CORRECT: 0.2,
        DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.5 if label is DiagnosisLabel.EVIDENCE_INSUFFICIENCY else 0.1,
        DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.5 if label is DiagnosisLabel.EVIDENCE_ACCESSIBILITY else 0.1,
        DiagnosisLabel.EVIDENCE_UTILISATION: 0.5 if label is DiagnosisLabel.EVIDENCE_UTILISATION else 0.1,
    }
    # renormalize failure mass
    mass = sum(v for k, v in joint.items() if k is not DiagnosisLabel.CORRECT)
    scale = p_fail / mass
    for k in list(joint):
        if k is not DiagnosisLabel.CORRECT:
            joint[k] *= scale
    stage2 = {k: joint[k] / p_fail for k in joint if k is not DiagnosisLabel.CORRECT}
    return HierarchicalDiagnosis(
        p_fail,
        "failure",
        stage2,
        label,
        joint[label],
        joint,
    )


def correct_diag() -> HierarchicalDiagnosis:
    return HierarchicalDiagnosis(0.2, "correct", None, DiagnosisLabel.CORRECT, 0.8, None)


def selection_forced(
    *,
    needs: bool,
    label: DiagnosisLabel,
    lane: SelectionLane,
) -> SelectionDecision:
    if not needs:
        lane = SelectionLane.NO_ACTION
    return SelectionDecision(
        needs_enhancement=needs,
        failure_type=label,
        selection_score=0.9 if needs else 0.1,
        selection_lane=lane,
        confidence=0.55,
        p_fail=0.8 if needs else 0.2,
        p_insuff=0.7 if label is DiagnosisLabel.EVIDENCE_INSUFFICIENCY else 0.1,
        used_probe=False,
        sample_id="forced",
    )


def build_pipeline(
    *,
    diagnosis: HierarchicalDiagnosis,
    selection: SelectionDecision,
    executor: EnhancementExecutor | None = None,
    item: DiagnosisInput | None = None,
) -> AdaptiveRAGPipeline:
    item = item or make_item()
    return build_adaptive_pipeline(
        reference_rag=StubReferenceRAG(item),
        classifier=StubClassifier(diagnosis),
        selection_adapter=StubSelection(selection),
        feature_provider=StubFeatures(),
        executor=executor if executor is not None else EnhancementExecutor(),
    )


class EndToEndFixtureTests(unittest.TestCase):
    def test_A_no_action(self):
        pipe = build_pipeline(
            diagnosis=correct_diag(),
            selection=selection_forced(
                needs=False, label=DiagnosisLabel.CORRECT, lane=SelectionLane.NO_ACTION
            ),
        )
        # Register mocks — must not be called for no_action
        pipe.executor.register(
            "retrieval_enhancement", MockRetrievalEnhancement()
        )
        result = run_adaptive_rag("q-no-action", pipeline=pipe, sample_id="A")
        self.assertIs(result.status, PipelineStatus.NO_ACTION)
        self.assertEqual(result.final_answer, result.initial_answer)
        self.assertFalse(result.changed_answer)
        self.assertEqual(result.final_evidence, result.initial_evidence)
        self.assertFalse(result.controller_decision.used_probe)
        self.assertEqual(result.enhancement_result.status.value, "no_action")
        self.assertEqual(result.metadata.get("probe"), "OFF")

    def test_B_insuff_retrieval_mock(self):
        executor = EnhancementExecutor()
        executor.register("retrieval_enhancement", MockRetrievalEnhancement())
        executor.register("evidence_refinement", MockEvidenceRefinementEnhancement())
        executor.register("generation_repair", MockGenerationRepairEnhancement())
        pipe = build_pipeline(
            diagnosis=failure_diag(DiagnosisLabel.EVIDENCE_INSUFFICIENCY),
            selection=selection_forced(
                needs=True,
                label=DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
                lane=SelectionLane.INSUFF_PRESERVE,
            ),
            executor=executor,
        )
        result = run_adaptive_rag("q-insuff", pipeline=pipe)
        self.assertEqual(result.controller_decision.action.value, "retrieval_enhancement")
        self.assertEqual(result.enhancement_result.metadata.get("mock"), "retrieval_enhancement")
        self.assertIs(result.status, PipelineStatus.SUCCESS)
        self.assertFalse(result.controller_decision.used_probe)

    def test_C_access_evidence_mock(self):
        executor = EnhancementExecutor()
        executor.register("retrieval_enhancement", MockRetrievalEnhancement())
        executor.register("evidence_refinement", MockEvidenceRefinementEnhancement())
        executor.register("generation_repair", MockGenerationRepairEnhancement())
        pipe = build_pipeline(
            diagnosis=failure_diag(DiagnosisLabel.EVIDENCE_ACCESSIBILITY),
            selection=selection_forced(
                needs=True,
                label=DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
                lane=SelectionLane.RESIDUAL_COMBO,
            ),
            executor=executor,
        )
        result = run_adaptive_rag_from_item(make_item(sample_id="C"), pipeline=pipe)
        self.assertEqual(result.controller_decision.action.value, "evidence_refinement")
        self.assertEqual(result.enhancement_result.metadata.get("mock"), "evidence_refinement")
        self.assertIs(result.status, PipelineStatus.SUCCESS)

    def test_D_util_generation_mock(self):
        executor = EnhancementExecutor()
        executor.register("retrieval_enhancement", MockRetrievalEnhancement())
        executor.register("evidence_refinement", MockEvidenceRefinementEnhancement())
        executor.register("generation_repair", MockGenerationRepairEnhancement())
        pipe = build_pipeline(
            diagnosis=failure_diag(DiagnosisLabel.EVIDENCE_UTILISATION),
            selection=selection_forced(
                needs=True,
                label=DiagnosisLabel.EVIDENCE_UTILISATION,
                lane=SelectionLane.RESIDUAL_COMBO,
            ),
            executor=executor,
        )
        result = run_adaptive_rag("q-util", pipeline=pipe)
        self.assertEqual(result.controller_decision.action.value, "generation_repair")
        self.assertEqual(result.enhancement_result.metadata.get("mock"), "generation_repair")
        self.assertIs(result.status, PipelineStatus.SUCCESS)

    def test_host_does_not_hardcode_mock_classes(self):
        from pathlib import Path

        text = Path("src/pipeline/adaptive_rag.py").read_text(encoding="utf-8")
        self.assertNotIn("MockRetrieval", text)
        self.assertNotIn("MockEvidence", text)
        self.assertNotIn("MockGeneration", text)
        self.assertNotIn("src.compensation.mocks", text)


class FailurePathTests(unittest.TestCase):
    def test_missing_implementation_unavailable(self):
        pipe = build_pipeline(
            diagnosis=failure_diag(DiagnosisLabel.EVIDENCE_INSUFFICIENCY),
            selection=selection_forced(
                needs=True,
                label=DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
                lane=SelectionLane.INSUFF_PRESERVE,
            ),
            executor=EnhancementExecutor(),  # empty
        )
        result = run_adaptive_rag("q-missing", pipeline=pipe)
        self.assertIs(result.status, PipelineStatus.ENHANCEMENT_UNAVAILABLE)
        self.assertEqual(result.final_answer, result.initial_answer)
        self.assertFalse(result.changed_answer)
        self.assertTrue(result.enhancement_result.metadata.get("enhancement_not_registered"))

    def test_enhancement_exception_fallback(self):
        executor = EnhancementExecutor()
        executor.register("retrieval_enhancement", RaisingEnhancement())
        pipe = build_pipeline(
            diagnosis=failure_diag(DiagnosisLabel.EVIDENCE_INSUFFICIENCY),
            selection=selection_forced(
                needs=True,
                label=DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
                lane=SelectionLane.INSUFF_PRESERVE,
            ),
            executor=executor,
        )
        result = run_adaptive_rag("q-raise", pipeline=pipe)
        self.assertIs(result.status, PipelineStatus.ENHANCEMENT_ERROR)
        self.assertEqual(result.final_answer, "initial answer")
        self.assertFalse(result.changed_answer)

    def test_unknown_failure_subtype_safe(self):
        # Controller raw path: selected + unknown -> no_action; pipeline still runs.
        decision = decide_from_raw_selection(
            needs_enhancement=True,
            failure_type="totally_unknown",
            selection_score=0.9,
            selection_lane=SelectionLane.RESIDUAL_COMBO,
            confidence=0.5,
        )
        self.assertEqual(decision.action.value, "no_action")
        # Wire via forced SelectionDecision that is not_selected equivalent for safety:
        pipe = build_pipeline(
            diagnosis=failure_diag(DiagnosisLabel.EVIDENCE_ACCESSIBILITY),
            selection=selection_forced(
                needs=False,
                label=DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
                lane=SelectionLane.NO_ACTION,
            ),
        )
        result = run_adaptive_rag("q-unknown", pipeline=pipe)
        self.assertIs(result.status, PipelineStatus.NO_ACTION)

    def test_empty_evidence_allowed_if_schema_ok(self):
        item = make_item(evidence=[])
        pipe = build_pipeline(
            diagnosis=correct_diag(),
            selection=selection_forced(
                needs=False, label=DiagnosisLabel.CORRECT, lane=SelectionLane.NO_ACTION
            ),
            item=item,
        )
        result = run_adaptive_rag_from_item(item, pipeline=pipe)
        self.assertEqual(result.initial_evidence, ())
        self.assertIs(result.status, PipelineStatus.NO_ACTION)

    def test_result_hides_feature_vector(self):
        pipe = build_pipeline(
            diagnosis=correct_diag(),
            selection=selection_forced(
                needs=False, label=DiagnosisLabel.CORRECT, lane=SelectionLane.NO_ACTION
            ),
        )
        payload = run_adaptive_rag("q", pipeline=pipe).to_dict()
        self.assertNotIn("adapter_features", payload)
        self.assertNotIn("feature_vector", payload)
        blob = str(payload)
        self.assertNotIn("gold_answer", blob)


class LifecycleTests(unittest.TestCase):
    def test_resources_reused_across_queries(self):
        item = make_item()
        classifier = StubClassifier(correct_diag())
        features = StubFeatures()
        selection = StubSelection(
            selection_forced(needs=False, label=DiagnosisLabel.CORRECT, lane=SelectionLane.NO_ACTION)
        )
        ref = StubReferenceRAG(item)
        pipe = AdaptiveRAGPipeline(
            reference_rag=ref,
            classifier=classifier,
            selection_adapter=selection,
            feature_provider=features,
            executor=EnhancementExecutor(),
        )
        run_adaptive_rag("q1", pipeline=pipe)
        run_adaptive_rag("q2", pipeline=pipe)
        self.assertEqual(ref.calls, 2)
        self.assertEqual(classifier.calls, 2)
        self.assertEqual(features.calls, 2)
        self.assertIs(pipe.executor, pipe.executor)


if __name__ == "__main__":
    unittest.main()
