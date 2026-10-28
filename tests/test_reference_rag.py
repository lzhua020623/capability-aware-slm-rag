"""Formal matched Reference RAG adapter + Host wiring (mocked inference)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.compensation.executor import EnhancementExecutor
from src.controller.schema import SelectionDecision, SelectionLane
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import DiagnosisInput, EvidenceItem
from src.generation.prompts import (
    CONTEXT_FALLBACK_INSTRUCTION,
    build_baseline_rag_prompt,
    build_context_fallback_rag_prompt,
)
from src.pipeline.adaptive_rag import AdaptiveRAGPipeline, build_adaptive_pipeline
from src.pipeline.reference_rag import (
    MatchedReferenceRAG,
    evidence_from_hits,
    load_reference_rag_config,
    validate_reference_rag_config,
)
from src.retrieval.matched_nq_train import assert_not_dev_index


ROOT = Path(__file__).resolve().parents[1]


def _hits(n: int = 5) -> list[dict]:
    return [
        {
            "passage_id": f"pass-{i}",
            "text": f"Passage text number {i} about the topic.",
            "score": 1.0 - 0.05 * i,
            "rank": i,
        }
        for i in range(1, n + 1)
    ]


class FakeLookup:
    def __init__(self, rows: dict[int, dict]):
        self.rows = rows

    def get(self, faiss_id: int) -> dict:
        return self.rows[faiss_id]

    def close(self) -> None:
        return None


class FakeRetriever:
    def __init__(self, scores, indices):
        self._scores = np.asarray(scores, dtype=np.float32)
        self._indices = np.asarray(indices, dtype=np.int64)
        self.encode_calls = 0

    def encode_queries(self, questions, batch_size=16):
        self.encode_calls += 1
        return np.zeros((len(questions), 8), dtype=np.float32)

    def search(self, query_embeddings, top_k: int):
        return self._scores[:, :top_k], self._indices[:, :top_k]

    def unload(self) -> None:
        return None


def _adapter(*, answer: str = "canberra", sample_hits: list[dict] | None = None) -> MatchedReferenceRAG:
    hits = sample_hits if sample_hits is not None else _hits()
    scores = [[h["score"] for h in hits]]
    indices = [[i for i in range(len(hits))]]
    rows = {
        i: {"passage_id": h["passage_id"], "text": h["text"]}
        for i, h in enumerate(hits)
    }
    passage_ids = [h["passage_id"] for h in hits]

    def generate(_tok, _model, prompt: str, max_new_tokens: int):
        assert max_new_tokens == 64
        assert CONTEXT_FALLBACK_INSTRUCTION in prompt
        return answer, 0.01

    return MatchedReferenceRAG(
        retriever=FakeRetriever(scores, indices),
        lookup=FakeLookup(rows),
        passage_ids=passage_ids,
        tokenizer=object(),
        model=object(),
        generate_fn=generate,
    )


class StubClassifier:
    def diagnose(self, item: DiagnosisInput) -> HierarchicalDiagnosis:
        return HierarchicalDiagnosis(
            0.1,
            "correct",
            {
                DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.3,
                DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.3,
                DiagnosisLabel.EVIDENCE_UTILISATION: 0.4,
            },
            DiagnosisLabel.CORRECT,
            0.9,
            {
                DiagnosisLabel.CORRECT: 0.9,
                DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.03,
                DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.03,
                DiagnosisLabel.EVIDENCE_UTILISATION: 0.04,
            },
        )


class StubFeatures:
    def features_for(self, item, diagnosis):
        return np.zeros(8)


class StubSelection:
    def select(self, diagnosis, adapter_features, *, sample_id=None):
        return SelectionDecision(
            needs_enhancement=False,
            failure_type=DiagnosisLabel.CORRECT,
            selection_score=0.1,
            selection_lane=SelectionLane.NO_ACTION,
            confidence=0.9,
            p_fail=0.1,
            p_insuff=0.1,
            used_probe=False,
            sample_id=sample_id or "x",
        )


class TestReferenceRAGConfig(unittest.TestCase):
    def test_repo_config_matches_matched_nq_train_contract(self):
        cfg = load_reference_rag_config(root=ROOT)
        ret = cfg["retrieval"]
        self.assertEqual(ret["embedding_model"], "BAAI/bge-base-en-v1.5")
        self.assertEqual(ret["top_k"], 5)
        self.assertEqual(ret["backend"], "corpus_offsets")
        self.assertIn("nq_train_bge_base.index", ret["index_path"].replace("\\", "/"))
        self.assertNotIn("nq_bge_base.index", Path(ret["index_path"]).name)
        self.assertEqual(cfg["rag_prompt"]["default_style"], "context_fallback")
        self.assertEqual(cfg["generator"]["model_name"], "Qwen/Qwen2.5-7B-Instruct")
        for key in ("index_path", "ids_path", "offsets_path", "corpus_path"):
            self.assertTrue((ROOT / ret[key]).is_file(), msg=ret[key])

    def test_legacy_dev_index_forbidden(self):
        with self.assertRaises(ValueError):
            assert_not_dev_index(ROOT / "data/processed/indexes/nq_bge_base.index")
        cfg = load_reference_rag_config(root=ROOT)
        bad = json.loads(json.dumps(cfg))
        bad["retrieval"]["index_path"] = "data/processed/indexes/nq_bge_base.index"
        with self.assertRaises(ValueError):
            validate_reference_rag_config(bad, root=ROOT)

    def test_context_fallback_prompt_contains_policy(self):
        prompt = build_context_fallback_rag_prompt("Q?", [{"text": "A passage."}])
        self.assertIn(CONTEXT_FALLBACK_INSTRUCTION, prompt)
        self.assertIn("[1] A passage.", prompt)
        styled = build_baseline_rag_prompt("Q?", [{"text": "A passage."}], prompt_style="context_fallback")
        self.assertEqual(prompt, styled)


class TestReferenceRAGAdapter(unittest.TestCase):
    def test_schema_top5_and_diagnosis_input(self):
        rag = _adapter(answer="Canberra")
        item = rag.run("What is the capital of Australia?", sample_id="train-smoke-1")
        self.assertIsInstance(item, DiagnosisInput)
        self.assertEqual(item.sample_id, "train-smoke-1")
        self.assertEqual(item.dataset.value, "nq")
        self.assertEqual(item.query, "What is the capital of Australia?")
        self.assertEqual(item.initial_answer, "Canberra")
        self.assertEqual(len(item.evidence), 5)
        self.assertEqual([e.rank for e in item.evidence], [1, 2, 3, 4, 5])
        self.assertTrue(all(e.text.strip() for e in item.evidence))
        self.assertTrue(all(isinstance(e.score, float) and e.score == e.score for e in item.evidence))

    def test_runtime_sample_id_when_omitted(self):
        item = _adapter().run("Who invented the telephone?")
        self.assertTrue(item.sample_id.startswith("runtime-"))

    def test_evidence_from_hits_rejects_wrong_count(self):
        with self.assertRaises(ValueError):
            evidence_from_hits(_hits(4))

    def test_empty_answer_rejected(self):
        rag = _adapter(answer="   ")
        with self.assertRaises(RuntimeError):
            rag.run("empty?")


class TestAdaptivePipelineWiring(unittest.TestCase):
    def test_run_query_builds_diagnosis_input_then_diagnosis(self):
        classifier = StubClassifier()
        seen = {}

        class TrackingClassifier(StubClassifier):
            def diagnose(self, item: DiagnosisInput):
                seen["item"] = item
                return super().diagnose(item)

        host = build_adaptive_pipeline(
            classifier=TrackingClassifier(),
            selection_adapter=StubSelection(),
            feature_provider=StubFeatures(),
            executor=EnhancementExecutor(),
            reference_rag=_adapter(answer="short answer"),
        )
        result = host.run("synthetic query for wiring", sample_id="wire-1")
        self.assertEqual(seen["item"].sample_id, "wire-1")
        self.assertEqual(len(seen["item"].evidence), 5)
        self.assertEqual(seen["item"].initial_answer, "short answer")
        self.assertEqual(result.initial_answer, "short answer")
        self.assertEqual(result.diagnosis.final_label, DiagnosisLabel.CORRECT)

    def test_run_from_item_unaffected_without_reference_rag(self):
        item = DiagnosisInput(
            sample_id="legacy-1",
            dataset="nq",
            query="already retrieved",
            evidence=[EvidenceItem(i, f"p{i}", f"t{i}", 0.9 - 0.1 * i) for i in range(1, 6)],
            initial_answer="cached",
        )
        host = build_adaptive_pipeline(
            classifier=StubClassifier(),
            selection_adapter=StubSelection(),
            feature_provider=StubFeatures(),
            executor=EnhancementExecutor(),
            reference_rag=None,
        )
        with self.assertRaises(ValueError):
            host.run("should fail")
        result = host.run_from_item(item)
        self.assertEqual(result.initial_answer, "cached")
        self.assertEqual(result.sample_id, "legacy-1")


class TestCurrentPipelineBoundaryFlag(unittest.TestCase):
    def test_run_from_item_accepts_enforce_boundary_kwarg(self):
        import inspect

        from src.pipeline.current import CurrentPipeline

        params = inspect.signature(CurrentPipeline.run_from_item).parameters
        self.assertIn("enforce_boundary", params)
        self.assertTrue(params["enforce_boundary"].default is True)


if __name__ == "__main__":
    unittest.main()
