"""End-to-end tests using only injected Python fakes and synthetic evidence."""

import ast
import inspect
import math
import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path

from src.compensation.retrieval_enhancement import EnhancementBudget, RetrievalEnhancer


def passage(passage_id, text=None):
    return {"passage_id": passage_id, "text": text or f"Evidence {passage_id}", "score": 0.8}


class FakeRetriever:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def retrieve(self, query, *, dataset, top_k):
        self.calls.append({"query": query, "dataset": dataset, "top_k": top_k})
        response = self.responses[query]
        if isinstance(response, Exception):
            raise response
        return response


class ScriptedRewriter:
    llm_calls_per_rewrite = 0

    def __init__(self, output):
        self.output = output
        self.calls = []

    def rewrite(self, query, dataset):
        self.calls.append((query, dataset))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class RetrievalEnhancerTests(unittest.TestCase):
    def setUp(self):
        self.query = "Where was Ada Lovelace born?"
        self.rewrite = "Ada Lovelace birthplace"
        self.base = [passage(name) for name in "ABCDE"]

    def execute(self, output=None, responses=None, budget=None):
        rewriter = ScriptedRewriter([self.rewrite] if output is None else output)
        retriever = FakeRetriever({self.rewrite: [passage("F"), passage("A")]}
                                  if responses is None else responses)
        result = RetrievalEnhancer(retriever, rewriter).enhance(self.query, "nq", self.base, budget)
        return result, retriever, rewriter

    def test_rule_rewriting_retrieval_fusion_and_complete_trace(self):
        retriever = FakeRetriever({self.rewrite: [passage("F"), passage("A")]})
        result = RetrievalEnhancer(retriever).enhance(self.query, "nq", self.base)
        trace = result["trace"]
        self.assertEqual(trace["original_query"], self.query)
        self.assertEqual(trace["rewritten_queries"], [self.rewrite])
        self.assertEqual(trace["original_evidence_ids"], list("ABCDE"))
        self.assertEqual(trace["candidate_ids"], list("AFBCDE"))
        self.assertEqual(trace["new_candidate_ids"], ["F"])
        self.assertEqual(trace["selected_ids"], list("AFBCD"))
        self.assertEqual(trace["retrieval_queries"], [self.rewrite])
        self.assertEqual(trace["retrieval_calls"], 1)
        self.assertEqual(trace["llm_calls"], 0)
        self.assertEqual(trace["status"], "completed")
        self.assertEqual(trace["errors"], [])
        self.assertTrue(math.isfinite(trace["latency"]) and trace["latency"] >= 0)
        self.assertEqual(retriever.calls, [{"query": self.rewrite, "dataset": "nq", "top_k": 5}])

    def test_empty_rewrite_returns_exact_base_without_retrieval(self):
        result, retriever, _ = self.execute(output=[])
        self.assertEqual(result["evidence"], self.base)
        self.assertEqual(result["trace"]["status"], "no_rewrite")
        self.assertEqual(retriever.calls, [])

    def test_duplicate_original_and_blank_queries_are_not_retrieved(self):
        result, retriever, _ = self.execute(output=[self.query, "", self.rewrite,
                                                     self.rewrite.upper(), "  " + self.rewrite])
        self.assertEqual(result["trace"]["rewritten_queries"], [self.rewrite])
        self.assertEqual(len(retriever.calls), 1)

    def test_candidate_recovery_does_not_imply_top5_recovery(self):
        result, _, _ = self.execute(responses={self.rewrite: [passage(name) for name in "ABCDF"]})
        self.assertEqual(result["trace"]["new_candidate_ids"], ["F"])
        self.assertNotIn("F", result["trace"]["selected_ids"])
        self.assertEqual(result["trace"]["status"], "completed")

    def test_multiple_rewrites_are_fused_in_one_round(self):
        second = "Ada Lovelace birth location"
        result, retriever, rewriter = self.execute(output=[self.rewrite, second], responses={
            self.rewrite: [passage("F"), passage("A")], second: [passage("F"), passage("B")]})
        self.assertEqual(result["trace"]["retrieval_calls"], 2)
        self.assertEqual(len(rewriter.calls), 1)
        self.assertEqual(result["trace"]["selected_ids"][0], "F")
        self.assertEqual(len(retriever.calls), 2)
        f = next(item for item in result["evidence"] if item["passage_id"] == "F")
        self.assertAlmostEqual(f["fusion_score"], 2 / 61)

    def test_zero_budget_prevents_all_provider_calls(self):
        result, retriever, rewriter = self.execute(budget=EnhancementBudget(max_retrieval_calls=0))
        self.assertEqual(result["evidence"], self.base)
        self.assertEqual(result["trace"]["status"], "budget_exhausted")
        self.assertEqual(retriever.calls, [])
        self.assertEqual(rewriter.calls, [])

    def test_one_call_budget_blocks_second_query(self):
        second = "Ada Lovelace birth location"
        result, retriever, _ = self.execute(output=[self.rewrite, second],
                                          budget=EnhancementBudget(max_retrieval_calls=1))
        self.assertEqual(result["trace"]["rewritten_queries"], [self.rewrite, second])
        self.assertEqual(result["trace"]["retrieval_queries"], [self.rewrite])
        self.assertEqual(len(retriever.calls), 1)

    def test_rewriter_cannot_force_more_than_two_calls(self):
        second = "Ada Lovelace birth location"
        result, retriever, _ = self.execute(output=[self.rewrite, second, "Ada birth place"],
                                          responses={self.rewrite: [], second: []})
        self.assertEqual(result["trace"]["rewritten_queries"], [self.rewrite, second])
        self.assertEqual(len(retriever.calls), 2)

    def test_retriever_failure_returns_exact_base_and_records_error(self):
        result, retriever, _ = self.execute(responses={self.rewrite: RuntimeError("backend unavailable")})
        self.assertEqual(result["evidence"], self.base)
        self.assertEqual(result["trace"]["status"], "retrieval_failed")
        self.assertEqual(result["trace"]["retrieval_calls"], 1)
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(result["trace"]["errors"][0]["message"], "backend unavailable")

    def test_failed_calls_still_spend_budget_without_retry(self):
        result, retriever, _ = self.execute(output=[self.rewrite, "second query"],
            responses={self.rewrite: RuntimeError("failed")}, budget=EnhancementBudget(max_retrieval_calls=1))
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(result["evidence"], self.base)

    def test_partial_failure_preserves_success_and_provenance_indices(self):
        second = "Ada Lovelace birth location"
        for failed_first in [True, False]:
            with self.subTest(failed_first=failed_first):
                responses = {self.rewrite: RuntimeError("failed") if failed_first else [passage("F")],
                             second: [passage("F")] if failed_first else RuntimeError("failed")}
                result, _, _ = self.execute(output=[self.rewrite, second], responses=responses)
                self.assertEqual(result["trace"]["status"], "partial")
                self.assertEqual(result["trace"]["new_candidate_ids"], ["F"])
                self.assertIn("F", result["trace"]["selected_ids"])
                f = next(item for item in result["evidence"] if item["passage_id"] == "F")
                self.assertEqual(f["provenance"][0]["source"], "additional:1" if failed_first else "additional:0")

    def test_rewriter_failure_or_invalid_output_safely_falls_back(self):
        for output in [RuntimeError("rewrite failed"), "not a list", [None]]:
            with self.subTest(output=output):
                result, retriever, _ = self.execute(output=output)
                self.assertEqual(result["evidence"], self.base)
                self.assertEqual(result["trace"]["status"], "rewrite_failed")
                self.assertEqual(retriever.calls, [])

    def test_empty_retrieval_and_duplicate_only_results_return_exact_base(self):
        for hits in [[], [passage("E"), passage("D")], [passage("alias", "Evidence E")]]:
            with self.subTest(hits=hits):
                result, _, _ = self.execute(responses={self.rewrite: hits})
                self.assertEqual(result["evidence"], self.base)
                self.assertEqual(result["trace"]["status"], "no_new_evidence")
                self.assertEqual(result["trace"]["new_candidate_ids"], [])
                if hits and hits[0]["passage_id"] == "alias":
                    e = next(item for item in result["candidates"] if item["passage_id"] == "E")
                    self.assertIn("alias", e["alias_passage_ids"])

    def test_malformed_or_conflicting_retrieval_safely_falls_back(self):
        for hits in [None, {}, [{"passage_id": "F"}], [passage("A", "Conflict")],
                     [passage("F") for _ in range(6)]]:
            with self.subTest(hits=hits):
                result, _, _ = self.execute(responses={self.rewrite: hits})
                self.assertEqual(result["evidence"], self.base)
                self.assertEqual(result["trace"]["status"], "retrieval_failed")
                self.assertTrue(result["trace"]["errors"])

    def test_second_call_conflict_does_not_discard_first_valid_evidence(self):
        second = "Ada Lovelace birth location"
        result, _, _ = self.execute(output=[self.rewrite, second], responses={
            self.rewrite: [passage("F")], second: [passage("F", "Different fact")]})
        self.assertEqual(result["trace"]["status"], "partial")
        f = next(item for item in result["candidates"] if item["passage_id"] == "F")
        self.assertEqual(f["text"], "Evidence F")

    def test_query_top_k_and_empty_base(self):
        retriever = FakeRetriever({self.rewrite: [passage("F")]})
        result = RetrievalEnhancer(retriever).enhance(self.query, "nq", [],
                                                     EnhancementBudget(top_k_per_query=1))
        self.assertEqual(retriever.calls[0]["top_k"], 1)
        self.assertEqual(result["trace"]["selected_ids"], ["F"])

    def test_fever_rule_rewriter_does_not_retrieve_or_change_claim(self):
        retriever = FakeRetriever({})
        result = RetrievalEnhancer(retriever).enhance("Ada was not born in 1900.", "fever", self.base)
        self.assertEqual(result["trace"]["status"], "no_rewrite")
        self.assertEqual(result["evidence"], self.base)
        self.assertEqual(retriever.calls, [])

    def test_inputs_and_nested_metadata_are_not_modified_or_shared(self):
        self.base[0]["provenance"] = [{"metadata": {"tokens": ["original"]}}]
        hits = [passage("F"), passage("A")]
        before = deepcopy((self.base, hits))
        result, _, _ = self.execute(responses={self.rewrite: hits})
        self.assertEqual((self.base, hits), before)
        a = next(item for item in result["evidence"] if item["passage_id"] == "A")
        a["provenance"][0]["metadata"]["tokens"].append("mutated")
        self.assertEqual((self.base, hits), before)
        candidate_a = next(item for item in result["candidates"] if item["passage_id"] == "A")
        self.assertEqual(candidate_a["provenance"][0]["metadata"]["tokens"], ["original"])

    def test_fallback_is_an_independent_copy_of_base(self):
        result, _, _ = self.execute(output=[])
        result["evidence"][0]["text"] = "changed"
        self.assertEqual(self.base[0]["text"], "Evidence A")

    def test_invalid_caller_input_errors_before_provider_calls(self):
        retriever = FakeRetriever({})
        rewriter = ScriptedRewriter([])
        enhancer = RetrievalEnhancer(retriever, rewriter)
        for query, dataset, base, budget in [("", "nq", self.base, None),
                (self.query, "", self.base, None), (self.query, "nq", None, None),
                (self.query, "nq", self.base * 2, None), (self.query, "nq", [{}], None),
                (self.query, "nq", self.base, {"max_retrieval_calls": 1})]:
            with self.subTest(query=query, dataset=dataset, base=base), self.assertRaises((TypeError, ValueError)):
                enhancer.enhance(query, dataset, base, budget)
        self.assertEqual(retriever.calls, [])
        self.assertEqual(rewriter.calls, [])

    def test_invalid_budgets_and_llm_provider_are_rejected(self):
        for kwargs in [{"max_retrieval_calls": -1}, {"max_retrieval_calls": 3},
                       {"max_retrieval_calls": True}, {"top_k_per_query": 0}, {"top_k_per_query": 6}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EnhancementBudget(**kwargs)
        rewriter = ScriptedRewriter([])
        rewriter.llm_calls_per_rewrite = 1
        with self.assertRaisesRegex(ValueError, "llm_calls_per_rewrite=0"):
            RetrievalEnhancer(FakeRetriever({}), rewriter)

    def test_runtime_interface_has_no_gold_diagnosis_or_controller_input(self):
        self.assertEqual(list(inspect.signature(RetrievalEnhancer.enhance).parameters),
                         ["self", "query", "dataset", "base_evidence", "budget"])
        result, retriever, rewriter = self.execute()
        self.assertEqual(rewriter.calls, [(self.query, "nq")])
        self.assertEqual(set(retriever.calls[0]), {"query", "dataset", "top_k"})
        self.assertEqual(result["trace"]["llm_calls"], 0)

    def test_package_and_execution_work_with_model_and_team_modules_blocked(self):
        code = '''
import sys
for name in ["torch", "transformers", "faiss", "sentence_transformers",
             "src.diagnosis", "src.controller", "src.probe", "src.retrieval", "src.generation"]:
    sys.modules[name] = None
from src.compensation.retrieval_enhancement import RetrievalEnhancer
class Fake:
    def retrieve(self, query, *, dataset, top_k):
        return [{"passage_id": "F", "text": "Synthetic evidence", "score": 0.8}]
r = RetrievalEnhancer(Fake()).enhance("Who wrote Hamlet?", "nq", [])
assert r["trace"]["status"] == "completed"
assert r["trace"]["llm_calls"] == 0
'''
        completed = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                                   cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_output_works_with_actual_qwen_prompt_builder_bodies(self):
        path = Path(__file__).resolve().parents[1] / "src/generation/prompts.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {"PROMPT_INSTRUCTION", "FEVER_PROMPT_INSTRUCTION", "build_rag_prompt", "build_fever_rag_prompt"}
        nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
                 or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        result, _, _ = self.execute()
        for name in ("build_rag_prompt", "build_fever_rag_prompt"):
            prompt = namespace[name](self.query, result["evidence"])
            self.assertIn("[2] Evidence F", prompt)
            self.assertIn(self.query, prompt)
            self.assertNotIn("fusion_score", prompt)


if __name__ == "__main__":
    unittest.main()
