"""Frozen P3-20 wiring/selection/answer guard; no model libraries or weights."""

import ast
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from src.compensation.retrieval_enhancement import (
    NQFaissBackend, P320RetrievalEnhancer, build_c1_prompt, select_p3_20,
)
from src.compensation.retrieval_enhancement.p3 import (
    RERANKER_MODEL, RERANKER_REVISION, answer_rejection,
)
from tests.test_retrieval_enhancement_adapters import FakeLoadedRetriever


def candidates():
    return [{"passage_id": f"p{i}", "text": f"Passage {i}: Ada lived in London and New York.",
             "score": 1.0 - i / 100, "rank": i} for i in range(1, 21)]


def item():
    return {"sample_id": "9007199254740993123", "dataset": "nq",
            "query": "Where was Ada born?", "evidence": candidates()[:5],
            "initial_answer": "original prediction"}


class Backend:
    def __init__(self):
        self.calls = []
        self.hits = candidates()

    def retrieve(self, query, *, dataset, top_k):
        self.calls.append((query, dataset, top_k))
        if isinstance(self.hits, Exception):
            raise self.hits
        return self.hits


class Scorer:
    model_name = RERANKER_MODEL
    revision = RERANKER_REVISION

    def __init__(self):
        self.calls = []
        self.scores = list(range(20))

    def __call__(self, query, texts):
        self.calls.append((query, list(texts)))
        if isinstance(self.scores, Exception):
            raise self.scores
        return self.scores


class SelectionTests(unittest.TestCase):
    def test_exact_rrf_ties_use_bge_order(self):
        r = select_p3_20(candidates(), list(range(20)))
        self.assertEqual([x["passage_id"] for x in r["evidence"]], ["p1", "p20", "p2", "p19", "p3"])
        self.assertEqual(r["scores"][0]["exact_fraction"], r["scores"][19]["exact_fraction"])

    def test_ce_ties_use_bge_order(self):
        r = select_p3_20(candidates(), [1.0] * 20)
        self.assertEqual([x["ce_rank"] for x in r["scores"]], list(range(1, 21)))
        self.assertEqual([x["passage_id"] for x in r["evidence"]], [f"p{i}" for i in range(1, 6)])

    def test_raw_retrieval_scores_and_text_are_preserved(self):
        pool = candidates()
        before = deepcopy(pool)
        r = select_p3_20(pool, list(range(20)))
        lookup = {x["passage_id"]: x for x in pool}
        self.assertEqual([x["rank"] for x in r["evidence"]], [1, 2, 3, 4, 5])
        for row in r["evidence"]:
            self.assertEqual(row["score"], lookup[row["passage_id"]]["score"])
            self.assertEqual(row["text"], lookup[row["passage_id"]]["text"])
        self.assertEqual(pool, before)

    def test_identical_text_with_different_ids_is_not_deduplicated(self):
        pool = candidates()
        for row in pool:
            row["text"] = "same literal text"
        r = select_p3_20(pool, [0] * 20)
        self.assertEqual(len(r["candidates"]), 20)
        self.assertEqual(len(r["evidence"]), 5)

    def test_invalid_counts_scores_ids_and_order_raise(self):
        for count in [0, 5, 19, 21]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                select_p3_20((candidates() + candidates())[:count], [0] * count)
        for bad in [True, "score", float("nan"), float("inf"), -float("inf")]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                select_p3_20(candidates(), [bad] + [0] * 19)
        for scores in [[0] * 19, "invalid"]:
            with self.assertRaises(ValueError):
                select_p3_20(candidates(), scores)
        pool = candidates()
        pool[1]["passage_id"] = pool[0]["passage_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_p3_20(pool, [0] * 20)
        with self.assertRaisesRegex(ValueError, "descending"):
            select_p3_20(candidates()[::-1], [0] * 20)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.scorer = Scorer()
        self.generated = []
        self.answer = "London"

        def generate(prompt):
            self.generated.append(prompt)
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer

        self.pipeline = P320RetrievalEnhancer(self.backend, self.scorer, generate)
        self.decision = {"action": "retrieval_enhancement"}

    def run_pipeline(self, value=None):
        return self.pipeline.run(item() if value is None else value, self.decision)

    def test_not_routed_calls_no_providers(self):
        for action in ["no_action", "generation_repair", "evidence_refinement"]:
            r = self.pipeline.run(item(), {"action": action})
            self.assertEqual(r["status"], "not_routed")
            self.assertEqual(r["final_answer"], item()["initial_answer"])
            self.assertEqual(r["evidence"], r["original_evidence"])
        self.assertFalse(self.backend.calls or self.scorer.calls or self.generated)

    def test_no_change_keeps_prediction_without_generation(self):
        self.scorer.scores = list(range(20, 0, -1))
        r = self.run_pipeline()
        self.assertEqual(r["status"], "no_change")
        self.assertEqual(r["final_answer"], item()["initial_answer"])
        self.assertEqual((r["retrieval_calls"], r["scoring_calls"], r["generation_calls"]), (1, 1, 0))
        self.assertEqual(r["scored_pairs"], 20)
        self.assertIsNone(r["prompt"])
        self.assertFalse(self.generated)

    def test_changed_order_with_same_members_requires_generation(self):
        self.scorer.scores = [19, 20] + list(range(18, 0, -1))
        r = self.run_pipeline()
        self.assertEqual([x["passage_id"] for x in r["evidence"]], ["p1", "p2", "p3", "p4", "p5"])
        # Symmetric first two RRF sums tie back to BGE order.
        self.assertEqual(r["status"], "no_change")
        self.scorer.scores = [1, 50, 49, 48, 47] + [0] * 15
        r = self.run_pipeline()
        self.assertEqual({x["passage_id"] for x in r["evidence"]}, {f"p{i}" for i in range(1, 6)})
        self.assertEqual(r["status"], "answer_replaced")
        self.assertEqual(r["generation_calls"], 1)

    def test_one_generation_and_full_trace(self):
        with patch('src.compensation.retrieval_enhancement.rewriting.RuleQueryRewriter.rewrite',
                   side_effect=AssertionError("rewriting must stay off")):
            r = self.run_pipeline()
        self.assertEqual(r["status"], "answer_replaced")
        self.assertEqual(r["final_answer"], "London")
        self.assertEqual(self.backend.calls, [(item()["query"], "nq", 20)])
        self.assertEqual(len(self.scorer.calls), 1)
        self.assertEqual(len(self.scorer.calls[0][1]), 20)
        self.assertEqual(len(self.generated), 1)
        self.assertEqual(len(r["scores"]), 20)
        self.assertEqual((r["generation_calls"], r["llm_calls"]), (1, 1))
        self.assertFalse(r["rewriting_enabled"] or r["rewritten_queries"])
        self.assertTrue(all(v >= 0 for v in r["latency"].values()))
        self.assertGreaterEqual(r["latency"]["total"], sum(v for k, v in r["latency"].items() if k != "total"))
        json.dumps(r, allow_nan=False)

    def test_two_argument_callable_interface(self):
        retrieval_enhancement = self.pipeline.retrieval_enhancement
        self.assertEqual(retrieval_enhancement(item(), self.decision), "London")
        self.assertEqual(self.pipeline(item(), self.decision), "London")

    def test_empty_refusal_and_unsupported_answers_keep_original(self):
        for answer in ["", "  ", "UNKNOWN", "I don't know", "I cannot answer this question", "Paris"]:
            with self.subTest(answer=answer):
                self.answer = answer
                r = self.run_pipeline()
                self.assertEqual(r["status"], "kept_original")
                self.assertEqual(r["final_answer"], item()["initial_answer"])
                self.assertEqual(r["generation_calls"], 1)
                self.assertIsNotNone(r["reject_reason"])

    def test_grounding_uses_contiguous_tokens_within_one_passage(self):
        evidence = [{"text": "Yorkshire. New and York."}, {"text": "city"}]
        for answer in ["New York", "York city"]:
            self.assertIsNotNone(answer_rejection(answer, evidence))
        self.assertIsNotNone(answer_rejection("York", [{"text": "Yorkshire"}]))
        self.assertIsNone(answer_rejection("THE New York!", [{"text": "Lives in New York today."}]))
        self.assertIsNotNone(answer_rejection("unknown", [{"text": "unknown"}]))
        self.assertEqual(answer_rejection("I cannot determine from context.",
                                         [{"text": "I cannot determine from context."}]), "refusal_or_empty")

    def test_provider_failures_keep_answer_and_count_attempt(self):
        self.backend.hits = RuntimeError("backend unavailable")
        r = self.run_pipeline()
        self.assertEqual(r["status"], "retrieval_failed")
        self.assertEqual((r["retrieval_calls"], r["scoring_calls"]), (1, 0))
        self.backend.hits = candidates()
        for scores in [RuntimeError("scorer failed"), [0] * 19, [float("nan")] * 20]:
            self.scorer.scores = scores
            r = self.run_pipeline()
            self.assertEqual(r["status"], "scoring_failed")
            self.assertEqual(r["generation_calls"], 0)
        self.scorer.scores = list(range(20))
        for answer in [RuntimeError("generation failed"), None, ("London", 0.1)]:
            self.answer = answer
            r = self.run_pipeline()
            self.assertEqual(r["status"], "generation_failed")
            self.assertEqual(r["final_answer"], item()["initial_answer"])
            self.assertEqual(r["generation_calls"], 1)

    def test_conflicting_text_including_unicode_escape_fails_before_scoring(self):
        for text in ["different body", "literal \\u0041da"]:
            self.backend.hits[0]["text"] = text
            r = self.run_pipeline()
            self.assertEqual(r["status"], "retrieval_failed")
            self.assertIn("conflicting text", r["error"]["message"])
            self.assertEqual(r["candidates"][0]["text"], text)
            self.assertEqual(r["scoring_calls"], 0)
            self.assertEqual(r["final_answer"], item()["initial_answer"])

    def test_input_not_modified_or_given_to_providers(self):
        x = item(); before = deepcopy(x); hits_before = deepcopy(self.backend.hits)
        r = self.run_pipeline(x)
        r["evidence"][0]["text"] = "change output"
        self.assertEqual(x, before)
        self.assertEqual(self.backend.hits, hits_before)
        self.assertTrue(all(isinstance(t, str) for t in self.scorer.calls[0][1]))
        self.assertNotIn(x["initial_answer"], self.generated[0])

    def test_whitelist_guards_all_runtime_fields_on_mappings_and_objects(self):
        class Guard(dict):
            def __getitem__(self, key):
                if key not in self.allowed:
                    raise AssertionError(f"forbidden read {key}")
                return super().__getitem__(key)
            def __iter__(self):
                raise AssertionError("must not enumerate runtime records")
            def keys(self):
                raise AssertionError("must not enumerate runtime records")
        def guarded(data, allowed):
            r = Guard(data); r.allowed = allowed
            return r
        x = item()
        x["evidence"] = [guarded(r, {"passage_id", "text", "score"}) for r in x["evidence"]]
        x.update(gold_answer="FORBIDDEN", short_answer="FORBIDDEN", target_label="FORBIDDEN",
                 gold_passage_id="FORBIDDEN", offline_evaluation="FORBIDDEN")
        x = guarded(x, {"sample_id", "dataset", "query", "evidence", "initial_answer"})
        decision = guarded({"action": "retrieval_enhancement", "diagnosis_label": "FORBIDDEN"}, {"action"})
        r = self.pipeline.run(x, decision)
        self.assertEqual(r["status"], "answer_replaced")
        self.assertNotIn("FORBIDDEN", json.dumps(r))

        class RuntimeObject:
            def __init__(self, values):
                object.__setattr__(self, "values", values)
            def __getattribute__(self, name):
                values = object.__getattribute__(self, "values")
                if name not in values and name != "__class__":
                    raise AssertionError(f"forbidden attribute {name}")
                return type(self) if name == "__class__" else values[name]
        plain = item()
        plain["evidence"] = [RuntimeObject({k: v for k, v in row.items() if k != "rank"}) for row in plain["evidence"]]
        r = self.pipeline.run(RuntimeObject(plain), RuntimeObject({"action": "retrieval_enhancement"}))
        self.assertEqual(r["status"], "answer_replaced")

    def test_frozen_dataclass_and_str_enum_shapes(self):
        class Action(str, Enum):
            RETRIEVAL_ENHANCEMENT = "retrieval_enhancement"
        @dataclass
        class Decision:
            action: Action
        @dataclass
        class EvidenceItem:
            rank: int
            passage_id: str
            text: str
            score: float
        @dataclass
        class DiagnosisInput:
            sample_id: str
            dataset: str
            query: str
            evidence: list
            initial_answer: str
        x = item()
        x["evidence"] = [EvidenceItem(**r) for r in x["evidence"]]
        self.assertEqual(self.pipeline(DiagnosisInput(**x), Decision(Action.RETRIEVAL_ENHANCEMENT)), "London")

    def test_wrong_revision_and_model_are_rejected(self):
        for attr, value in [("revision", "main"), ("revision", RERANKER_REVISION[:8]), ("model_name", "other")]:
            scorer = Scorer(); setattr(scorer, attr, value)
            with self.assertRaises(ValueError):
                P320RetrievalEnhancer(self.backend, scorer, lambda _: "London")

    def test_invalid_runtime_shape_and_unsupported_dataset(self):
        for field, value in [("sample_id", 123), ("query", ""), ("evidence", []), ("initial_answer", None)]:
            x = item(); x[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_pipeline(x)
        x = item(); x["dataset"] = "fever"
        self.assertEqual(self.run_pipeline(x)["status"], "unsupported_dataset")
        self.assertFalse(self.backend.calls or self.scorer.calls or self.generated)


class CompatibilityTests(unittest.TestCase):
    def test_prompt_byte_matches_actual_frozen_c1_without_importing_models(self):
        path = Path(__file__).resolve().parents[1] / "src/generation/prompts.py"
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if (
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PROMPT_INSTRUCTION" for t in n.targets)
        ) or isinstance(n, ast.FunctionDef) and n.name == "build_rag_prompt"]
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        for question in ["Where was Ada born?", "Unicode café?\nSecond line"]:
            self.assertEqual(build_c1_prompt(question, candidates()[:5]), namespace["build_rag_prompt"](question, candidates()[:5]))

    def test_backend_top20_is_explicit_default_still_top5(self):
        r = FakeLoadedRetriever()
        r.metadata = candidates(); r.passage_ids = [e["passage_id"] for e in r.metadata]
        r.index.ntotal = 20
        r.index.response = ([[e["score"] for e in r.metadata]], [list(range(20))])
        hits = NQFaissBackend(r, max_top_k=20).retrieve("query", dataset="nq", top_k=20)
        self.assertEqual(len(hits), 20)
        self.assertEqual(r.index.calls[-1][1], 20)
        with self.assertRaises(ValueError):
            NQFaissBackend(r).retrieve("query", dataset="nq", top_k=20)
        for limit in [True, 0, 21, 1.0]:
            with self.assertRaises(ValueError):
                NQFaissBackend(r, max_top_k=limit)

    def test_import_and_end_to_end_need_no_model_or_team_modules(self):
        code = '''
import sys
for name in ['torch', 'transformers', 'sentence_transformers', 'faiss', 'numpy',
             'src.retrieval', 'src.generation', 'src.diagnosis', 'src.controller', 'src.probe']:
    sys.modules[name] = None
from tests.test_retrieval_enhancement_p3 import Backend, Scorer, item
from src.compensation.retrieval_enhancement import P320RetrievalEnhancer
p = P320RetrievalEnhancer(Backend(), Scorer(), lambda prompt: 'London')
assert p(item(), {'action': 'retrieval_enhancement'}) == 'London'
'''
        r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
