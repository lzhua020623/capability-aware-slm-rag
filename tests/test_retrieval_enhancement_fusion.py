"""Synthetic CPU-only evidence tests; targets exist only in assertions."""

import ast
import math
import unittest
from copy import deepcopy
from pathlib import Path

from src.compensation.retrieval_enhancement import deduplicate_evidence, rrf_fuse


def passage(passage_id, text=None, score=0.8, **metadata):
    return {"passage_id": passage_id, "text": text or f"Evidence {passage_id}",
            "score": score, **metadata}


def ids(evidence):
    return [item["passage_id"] for item in evidence]


class RetrievalEnhancementFusionTests(unittest.TestCase):
    def setUp(self):
        self.base = [passage(name, score=0.9 - i * 0.1) for i, name in enumerate("ABCDE")]

    def test_missing_target_is_selected_after_additional_retrieval(self):
        result = rrf_fuse(self.base, [[passage("F"), passage("A")]])
        self.assertNotIn("F", ids(self.base))
        self.assertEqual(ids(result["evidence"]), ["A", "F", "B", "C", "D"])
        self.assertEqual(result["new_evidence_count"], 1)

    def test_new_target_in_candidates_can_be_excluded_from_top5(self):
        extra = [passage(name) for name in "ABCDF"]
        result = rrf_fuse(self.base, [extra])
        self.assertIn("F", ids(result["candidates"]))
        self.assertNotIn("F", ids(result["evidence"]))
        self.assertEqual(ids(result["evidence"]), list("ABCDE"))
        self.assertEqual(len(result["candidates"]), 6)

    def test_duplicate_id_votes_once_per_list(self):
        result = rrf_fuse([passage("A"), passage("A", score=0.99)],
                          [[passage("A"), passage("F")]])
        a = next(item for item in result["candidates"] if item["passage_id"] == "A")
        self.assertEqual(ids(result["evidence"]).count("A"), 1)
        self.assertAlmostEqual(a["fusion_score"], 2 / 61)
        self.assertEqual(len(a["provenance"]), 3)

    def test_same_text_different_ids_preserves_aliases_and_provenance(self):
        result = deduplicate_evidence([passage("A", "Same fact."),
                                       passage("alias", "Same fact.", score=0.95)], dataset="nq")
        self.assertEqual(ids(result), ["A"])
        self.assertEqual(result[0]["alias_passage_ids"], ["alias"])
        self.assertEqual([p["passage_id"] for p in result[0]["provenance"]], ["A", "alias"])
        self.assertEqual(result[0]["score"], 0.8)
        self.assertEqual(result[0]["dataset"], "nq")

    def test_unicode_and_whitespace_dedup_retains_original_text(self):
        original = "Cafe\u0301\n is\topen."
        result = deduplicate_evidence([passage("A", original),
                                       passage("B", "Caf\u00e9\u00a0is open.")])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], original)
        self.assertEqual(result[0]["alias_passage_ids"], ["B"])

    def test_different_facts_are_not_semantically_merged(self):
        texts = ["X is open.", "X is not open.", "X is open!", "X is Open.",
                 "X opened in 2020.", "X opened in 2021.", "X has 2 items.", "X has \u00b2 items."]
        result = deduplicate_evidence([passage(str(i), text) for i, text in enumerate(texts)])
        self.assertEqual(len(result), len(texts))

    def test_same_id_conflicting_text_reports_error(self):
        with self.assertRaisesRegex(ValueError, "Conflicting text.*'A'.*'nq'"):
            rrf_fuse([passage("A", "X is open.")],
                     [[passage("A", "X is not open.")]], dataset="nq")

    def test_alias_id_cannot_hide_conflicting_text(self):
        with self.assertRaisesRegex(ValueError, "Conflicting text"):
            deduplicate_evidence([passage("A", "First", alias_passage_ids=["alias"]),
                                  passage("alias", "Second")])

    def test_datasets_have_separate_id_and_text_namespaces(self):
        result = deduplicate_evidence([passage("A", "Same", dataset="nq"),
                                       passage("A", "Same", dataset="fever"),
                                       passage("B", "Different", dataset="other")])
        self.assertEqual(len(result), 3)
        self.assertEqual([item["dataset"] for item in result], ["nq", "fever", "other"])

    def test_conflicting_dataset_argument_reports_error(self):
        with self.assertRaisesRegex(ValueError, "dataset conflicts"):
            rrf_fuse([passage("A", dataset="fever")], dataset="nq")

    def test_raw_scores_and_occurrence_scores_are_preserved(self):
        result = rrf_fuse([passage("A", score=-0.2)],
                          [[passage("A", score=0.95), passage("F", score=0.99)]])
        a = result["evidence"][0]
        self.assertEqual(a["score"], -0.2)
        self.assertEqual([p["score"] for p in a["provenance"]], [-0.2, 0.95])
        self.assertEqual([p["source"] for p in a["provenance"]], ["base", "additional:0"])
        self.assertAlmostEqual(a["fusion_score"], 2 / 61)

    def test_multiple_lists_use_standard_rrf_and_stable_ties(self):
        base = [passage("A", rank=99), passage("B", rank=1)]
        extra = [[passage("B"), passage("A")], [passage("F")]]
        first = rrf_fuse(base, extra)
        self.assertEqual(ids(first["evidence"]), ["A", "B", "F"])
        expected = math.fsum([1 / 61, 1 / 62])
        self.assertEqual(first["evidence"][0]["fusion_score"], expected)
        self.assertEqual(first["evidence"][1]["fusion_score"], expected)
        for _ in range(5):
            self.assertEqual(rrf_fuse(base, extra), first)

    def test_base_precedes_new_candidate_at_equal_rrf_score(self):
        result = rrf_fuse([passage("A")], [[passage("F")]])
        self.assertEqual(ids(result["evidence"]), ["A", "F"])

    def test_empty_additional_lists_preserve_base_order(self):
        for additional in ((), [], [[]], [[], []]):
            with self.subTest(additional=additional):
                result = rrf_fuse(self.base, additional)
                self.assertEqual(ids(result["evidence"]), list("ABCDE"))
                self.assertEqual(result["new_evidence_count"], 0)

    def test_duplicate_only_additional_results_cannot_reorder_base(self):
        result = rrf_fuse(self.base, [[passage("E"), passage("D")]])
        self.assertEqual(ids(result["evidence"]), list("ABCDE"))
        self.assertGreater(result["evidence"][4]["fusion_score"], result["evidence"][0]["fusion_score"])
        self.assertEqual(result["new_evidence_count"], 0)

    def test_text_alias_is_not_counted_as_new_evidence(self):
        result = rrf_fuse(self.base, [[passage("alias-E", "Evidence E")]])
        self.assertEqual(result["new_evidence_count"], 0)
        self.assertEqual(ids(result["evidence"]), list("ABCDE"))
        self.assertEqual(result["evidence"][4]["alias_passage_ids"], ["alias-E"])

    def test_input_and_nested_metadata_are_never_modified(self):
        self.base[0]["provenance"] = [{"query": {"tokens": ["original"]}}]
        self.base[0]["alias_passage_ids"] = ["old-A"]
        extra = [[passage("F")]]
        before = deepcopy((self.base, extra))
        result = rrf_fuse(self.base, extra)
        dedup = deduplicate_evidence(self.base)
        self.assertEqual((self.base, extra), before)
        selected_a = next(item for item in result["evidence"] if item["passage_id"] == "A")
        selected_a["provenance"][0]["query"]["tokens"].append("changed")
        result["candidates"][0]["text"] = "changed candidate"
        dedup[0]["alias_passage_ids"].append("changed alias")
        self.assertEqual((self.base, extra), before)
        self.assertEqual(result["evidence"][0]["text"], "Evidence A")

    def test_empty_base_and_output_cap(self):
        self.assertEqual(rrf_fuse([], []), {"evidence": [], "candidates": [], "new_evidence_count": 0})
        result = rrf_fuse([], [[passage(str(i)) for i in range(8)]])
        self.assertEqual(len(result["evidence"]), 5)
        self.assertEqual(len(result["candidates"]), 8)
        self.assertEqual(len(rrf_fuse([], [self.base], top_k=2)["evidence"]), 2)

    def test_zero_k_and_original_positions(self):
        result = rrf_fuse([passage("A"), passage("A"), passage("B")], [[passage("F")]], k=0)
        b = next(item for item in result["candidates"] if item["passage_id"] == "B")
        self.assertAlmostEqual(b["fusion_score"], 1 / 3)
        self.assertEqual(b["provenance"][0]["rank"], 3)

    def test_invalid_required_fields_and_scores_report_errors(self):
        invalid = [None, "text", {}, {"text": "x", "score": 0.1},
                   {"passage_id": "A", "score": 0.1}, {"passage_id": "A", "text": "x"},
                   passage(" "), {"passage_id": 3, "text": "x", "score": 0.1},
                   {"passage_id": "A", "text": "\t", "score": 0.1},
                   {"passage_id": "A", "text": 3, "score": 0.1}]
        invalid += [passage("A", score=score) for score in (float("nan"), float("inf"),
                                                            -float("inf"), True, "0.8", None)]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises((TypeError, ValueError)):
                rrf_fuse([item])

    def test_invalid_metadata_and_containers_report_errors(self):
        metadata = [{"dataset": " "}, {"dataset": None}, {"dataset": 1},
                    {"rank": 0}, {"rank": True}, {"fusion_score": float("nan")},
                    {"provenance": {}}, {"provenance": ["source"]},
                    {"alias_passage_ids": "A"}, {"alias_passage_ids": [""]}]
        for extra in metadata:
            with self.subTest(extra=extra), self.assertRaises((TypeError, ValueError)):
                rrf_fuse([passage("A", **extra)])
        for base, extra in [(None, []), ("ABCDE", []), ([], None), ([], "extra"),
                            ([], [None]), ([], [passage("A")])]:
            with self.subTest(base=base, extra=extra), self.assertRaises(TypeError):
                rrf_fuse(base, extra)

    def test_invalid_rrf_parameters_report_errors(self):
        for kwargs in ({"k": -1}, {"k": 1.5}, {"k": True}, {"top_k": 0},
                       {"top_k": 6}, {"top_k": True}, {"top_k": 2.5}, {"dataset": " "}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                rrf_fuse(self.base, **kwargs)

    def test_evidence_is_consumed_by_existing_qwen_prompt_builder_bodies(self):
        # Execute the actual pure Prompt builders without importing GPU libraries.
        path = Path(__file__).resolve().parents[1] / "src/generation/qwen.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {"PROMPT_INSTRUCTION", "FEVER_PROMPT_INSTRUCTION",
                 "build_rag_prompt", "build_fever_rag_prompt"}
        nodes = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names
                 or isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        evidence = rrf_fuse(self.base, [[passage("F"), passage("A")]])["evidence"]
        for name in ("build_rag_prompt", "build_fever_rag_prompt"):
            prompt = namespace[name]("Synthetic query", evidence)
            self.assertIn("Synthetic query", prompt)
            self.assertIn("[2] Evidence F", prompt)
            self.assertNotIn("fusion_score", prompt)


if __name__ == "__main__":
    unittest.main()
