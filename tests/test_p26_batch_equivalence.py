"""P26/P60 batch path must match legacy single-sample formulas within float tolerance."""
from __future__ import annotations

import unittest

import numpy as np

from src.controller.p26_semantic import P26SemanticRuntime, assemble_p26_result
from src.controller.p60_online_features import P60OnlineFeatureProvider
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import DiagnosisInput, EvidenceItem


def _item(query: str, answer: str, passages: list[str], sid: str = "1") -> DiagnosisInput:
    evidence = tuple(
        EvidenceItem(rank=i + 1, passage_id=f"p{i}", text=passages[i], score=1.0 - 0.01 * i)
        for i in range(5)
    )
    return DiagnosisInput(
        sample_id=sid,
        dataset="nq",
        query=query,
        evidence=evidence,
        initial_answer=answer,
    )


def _diag(label=DiagnosisLabel.CORRECT, p_fail=0.2):
    return HierarchicalDiagnosis(
        stage1_failure_probability=p_fail,
        stage1_label="correct" if label is DiagnosisLabel.CORRECT else "failure",
        stage2_class_scores=None,
        final_label=label,
        confidence=1.0 - p_fail,
        class_scores={
            DiagnosisLabel.CORRECT: 1.0 - p_fail,
            DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.0,
            DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.0,
            DiagnosisLabel.EVIDENCE_UTILISATION: 0.0,
        },
    )


class AssembleUnitTests(unittest.TestCase):
    def test_assemble_shapes(self):
        rng = np.random.default_rng(0)
        qa = rng.normal(size=8).astype(np.float32)
        qa /= np.linalg.norm(qa) + 1e-8
        ans = rng.normal(size=8).astype(np.float32)
        ans /= np.linalg.norm(ans) + 1e-8
        pas = rng.normal(size=(5, 8)).astype(np.float32)
        pas /= np.linalg.norm(pas, axis=1, keepdims=True) + 1e-8
        passages = [f"passage number {i} about Paris France" for i in range(5)]
        result = assemble_p26_result(
            query="Where is Paris?",
            initial_answer="France",
            passages=passages,
            qa_emb=qa,
            ans_emb=ans,
            pass_emb=pas,
            s5_best=0.1,
            s5_top1=0.2,
        )
        self.assertEqual(result.p60_vector().shape, (35,))


@unittest.skipUnless(
    __import__("importlib").util.find_spec("sentence_transformers") is not None,
    "sentence_transformers required",
)
class P26BatchGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = P26SemanticRuntime(device=None)
        cls.passages = [
            "Paris is the capital of France and a major European city.",
            "Lyon is another large city in France.",
            "The Seine river flows through Paris.",
            "France is a country in Western Europe.",
            "Berlin is the capital of Germany.",
        ]
        cls.samples = [
            ("Where is Paris?", "France", cls.passages),
            ("Who wrote Hamlet?", "William Shakespeare", cls.passages),
            ("When was the UN founded?", "1945", cls.passages),
            ("How many states are in the USA?", "50", cls.passages),
        ]

    def test_batch_matches_legacy_within_tol(self):
        legacy = [
            self.runtime.compute_legacy_separate_calls(q, a, p) for q, a, p in self.samples
        ]
        batched = self.runtime.compute_batch(self.samples)
        self.assertEqual(len(legacy), len(batched))
        max_abs = 0.0
        for old, new in zip(legacy, batched):
            diff = np.max(np.abs(old.p60_vector() - new.p60_vector()))
            max_abs = max(max_abs, float(diff))
            np.testing.assert_allclose(old.p60_vector(), new.p60_vector(), atol=2e-3, rtol=0)
        self.assertLessEqual(max_abs, 2e-3)

    def test_compute_equals_batch_of_one(self):
        q, a, p = self.samples[0]
        single = self.runtime.compute(q, a, p).p60_vector()
        batched = self.runtime.compute_batch([(q, a, p)])[0].p60_vector()
        np.testing.assert_array_equal(single, batched)

    def test_p60_provider_batch_matches_serial(self):
        provider = P60OnlineFeatureProvider(semantic_runtime=self.runtime)
        items = [
            _item(q, a, list(p), sid=str(i)) for i, (q, a, p) in enumerate(self.samples)
        ]
        diags = [_diag() for _ in items]
        serial = [provider.features_for(item, diag) for item, diag in zip(items, diags)]
        # Fresh provider path via batch (same runtime models)
        batched = provider.features_for_batch(items, diags)
        max_abs = 0.0
        for s, b in zip(serial, batched):
            diff = float(np.max(np.abs(s - b)))
            max_abs = max(max_abs, diff)
            np.testing.assert_allclose(s, b, atol=2e-3, rtol=0)
            self.assertEqual(s.shape, (78,))
        self.assertLessEqual(max_abs, 2e-3)

    def test_shared_bge_matches_dedicated(self):
        """Shared RAG BGE instance with max_seq_length restore must match dedicated P26 BGE."""
        from sentence_transformers import SentenceTransformer

        from src.controller.p26_semantic import BGE_MODEL, BGE_MAX_SEQ_LENGTH

        dedicated = self.runtime.compute_batch(self.samples)
        shared_model = SentenceTransformer(BGE_MODEL, device=self.runtime.device)
        # Mimic Reference RAG default seq length
        shared_model.max_seq_length = 512
        shared_rt = type(self.runtime)(device=self.runtime.device)
        shared_rt.attach_shared_bge(shared_model)
        # Force CE load; reuse dedicated CE weights path via ensure
        shared = shared_rt.compute_batch(self.samples)
        max_abs = 0.0
        for a, b in zip(dedicated, shared):
            diff = float(np.max(np.abs(a.p60_vector() - b.p60_vector())))
            max_abs = max(max_abs, diff)
            np.testing.assert_allclose(a.p60_vector(), b.p60_vector(), atol=2e-3, rtol=0)
        self.assertLessEqual(max_abs, 2e-3)
        # RAG-facing max_seq_length must be restored after P26 encode
        self.assertEqual(int(shared_model.max_seq_length), 512)
        self.assertNotEqual(int(shared_model.max_seq_length), BGE_MAX_SEQ_LENGTH)


if __name__ == "__main__":
    unittest.main()
