"""Pure-Python encoder/index fakes: no model loading or actual Index required."""

import ast
import inspect
import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path

from src.compensation.retrieval_enhancement import NQFaissBackend, RetrievalEnhancer
from src.compensation.retrieval_enhancement.adapters import BGE_MODEL_NAME, BGE_QUERY_INSTRUCTION


class ArrayLike:
    """Exercise ndarray-style iteration without importing numpy."""

    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


class FakeEncoder:
    def __init__(self):
        self.calls = []
        self.embeddings = ArrayLike([[1.0, 0.0]])

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        if isinstance(self.embeddings, Exception):
            raise self.embeddings
        return self.embeddings


class FakeIndex:
    d = 2
    metric_type = 0
    is_trained = True

    def __init__(self, ntotal=6, response=None):
        self.ntotal = ntotal
        self.calls = []
        self.response = response if response is not None else (
            ArrayLike([[0.95, 0.85, 0.75, 0.65, 0.55]]), ArrayLike([[4, 1, 5, 0, 2]]))

    def search(self, embeddings, top_k):
        self.calls.append((embeddings, top_k))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeLoadedRetriever:
    """Same public attributes and methods as the repository's loaded Retriever."""

    model_name = BGE_MODEL_NAME
    query_prefix = BGE_QUERY_INSTRUCTION
    passage_prefix = ""

    def __init__(self):
        self.model = FakeEncoder()
        self.index = FakeIndex()
        self.metadata = [{"passage_id": pid, "text": f"Evidence {pid}",
                          "document_title": f"Document {pid}"} for pid in "ABCDEF"]
        self.passage_ids = [record["passage_id"] for record in self.metadata]
        self.query_calls = []

    def encode_queries(self, questions, batch_size):
        self.query_calls.append((list(questions), batch_size))
        return self.model.encode([self.query_prefix + q for q in questions],
                                 batch_size=batch_size, normalize_embeddings=True)

    def search(self, embeddings, top_k):
        return self.index.search(embeddings, top_k)

    def load(self, *args):
        raise AssertionError("Adapter must not load resources")

    def unload(self):
        raise AssertionError("Adapter must not unload resources")

    def build_index(self, *args):
        raise AssertionError("Adapter must not build an Index")


class NQFaissBackendTests(unittest.TestCase):
    def setUp(self):
        self.retriever = FakeLoadedRetriever()
        self.backend = NQFaissBackend(self.retriever)

    def retrieve(self, top_k=5):
        return self.backend.retrieve("Who wrote Hamlet?", dataset="natural_questions", top_k=top_k)

    def test_positions_map_to_passage_ids_text_and_scores_in_faiss_order(self):
        hits = self.retrieve()
        self.assertEqual([hit["passage_id"] for hit in hits], list("EBFAC"))
        self.assertEqual([hit["text"] for hit in hits], [f"Evidence {pid}" for pid in "EBFAC"])
        self.assertEqual([hit["score"] for hit in hits], [0.95, 0.85, 0.75, 0.65, 0.55])
        self.assertTrue(all(set(hit) == {"passage_id", "text", "score"} for hit in hits))

    def test_raw_query_and_bge_instruction_are_encoded_exactly_once(self):
        self.retrieve()
        self.assertEqual(self.retriever.query_calls, [(["Who wrote Hamlet?"], 16)])
        self.assertEqual(self.retriever.model.calls,
                         [([BGE_QUERY_INSTRUCTION + "Who wrote Hamlet?"],
                           {"batch_size": 16, "normalize_embeddings": True})])

    def test_requested_top_k_is_forwarded(self):
        self.retriever.index.response = ([[0.9, 0.8]], [[5, 0]])
        self.assertEqual(len(self.retrieve(top_k=2)), 2)
        self.assertEqual(self.retriever.index.calls[0][1], 2)

    def test_too_many_backend_hits_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceed top_k"):
            self.retrieve(top_k=2)

    def test_invalid_query_dataset_and_top_k_are_rejected_before_encoding(self):
        for query, dataset, top_k in [("", "nq", 5), ({"question": "x"}, "nq", 5),
                ("x", "fever", 5), ("x", "nq", 0), ("x", "nq", 6),
                ("x", "nq", True), ("x", "nq", 1.5)]:
            with self.subTest(query=query, dataset=dataset, top_k=top_k), self.assertRaises(ValueError):
                self.backend.retrieve(query, dataset=dataset, top_k=top_k)
        self.assertEqual(self.retriever.query_calls, [])

    def test_nq_dataset_alias_is_supported(self):
        self.assertEqual(self.backend.retrieve("query", dataset="nq", top_k=5)[0]["passage_id"], "E")

    def test_wrong_bge_model_or_prefix_configuration_is_rejected(self):
        for name, value in [("model_name", "intfloat/e5-base-v2"),
                            ("query_prefix", "query:"), ("passage_prefix", "passage:")]:
            with self.subTest(name=name):
                retriever = FakeLoadedRetriever()
                setattr(retriever, name, value)
                with self.assertRaisesRegex(ValueError, name):
                    NQFaissBackend(retriever)

    def test_missing_untrained_or_incompatible_index_is_rejected(self):
        for name, value in [("ntotal", -1), ("d", 0), ("d", True),
                            ("metric_type", 1), ("metric_type", False), ("is_trained", False)]:
            with self.subTest(name=name):
                retriever = FakeLoadedRetriever()
                setattr(retriever.index, name, value)
                with self.assertRaises(ValueError):
                    NQFaissBackend(retriever)
        self.retriever.index = None
        with self.assertRaisesRegex(RuntimeError, "external Runner"):
            self.retrieve()

    def test_external_unload_is_detected_without_reloading(self):
        encoder = self.retriever.model
        self.retriever.model = None
        with self.assertRaisesRegex(RuntimeError, "unloaded"):
            self.retrieve()
        self.assertEqual(encoder.calls, [])

    def test_metadata_and_passage_id_counts_match_index(self):
        for name in ("metadata", "passage_ids"):
            with self.subTest(name=name):
                retriever = FakeLoadedRetriever()
                setattr(retriever, name, getattr(retriever, name)[:-1])
                with self.assertRaisesRegex(ValueError, "count"):
                    NQFaissBackend(retriever)

    def test_metadata_records_ids_text_and_mapping_are_validated(self):
        invalid = [None, {}, {"passage_id": "A", "text": " "},
                   {"passage_id": " ", "text": "text"}, {"passage_id": 0, "text": "text"},
                   {"passage_id": "wrong-ID", "text": "text"}]
        for record in invalid:
            with self.subTest(record=record):
                retriever = FakeLoadedRetriever()
                retriever.metadata[0] = record
                with self.assertRaises(ValueError):
                    NQFaissBackend(retriever)
        self.retriever.metadata[1] = dict(self.retriever.metadata[0])
        self.retriever.passage_ids[1] = "A"
        with self.assertRaisesRegex(ValueError, "Duplicate passage_id"):
            NQFaissBackend(self.retriever)

    def test_selected_metadata_in_place_change_is_detected(self):
        self.retriever.metadata[4]["passage_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "disagree at position 4"):
            self.retrieve()

    def test_replaced_metadata_is_fully_revalidated(self):
        self.retriever.metadata = deepcopy(self.retriever.metadata)
        self.retriever.metadata[3]["text"] = ""
        with self.assertRaisesRegex(ValueError, "invalid text"):
            self.retrieve()

    def test_empty_index_returns_no_hits_without_encoding_or_search(self):
        self.retriever.index = FakeIndex(ntotal=0)
        self.retriever.metadata = []
        self.retriever.passage_ids = []
        self.assertEqual(self.retrieve(), [])
        self.assertEqual(self.retriever.query_calls, [])
        self.assertEqual(self.retriever.index.calls, [])

    def test_empty_results_and_minus_one_padding(self):
        for response in [([], []), ([[]], [[]]), ([[-float("inf")]], [[-1]])]:
            with self.subTest(response=response):
                self.retriever.index.response = response
                self.assertEqual(self.retrieve(), [])
        self.retriever.index.response = ([[0.9, -float("inf"), 0.7]], [[5, -1, 0]])
        self.assertEqual([hit["passage_id"] for hit in self.retrieve()], ["F", "A"])

    def test_invalid_positions_and_valid_hit_scores_raise_clear_errors(self):
        for position in [-2, 6, 1.5, True, "0"]:
            with self.subTest(position=position), self.assertRaises(ValueError):
                self.retriever.index.response = ([[0.8]], [[position]])
                self.retrieve()
        for score in [float("nan"), float("inf"), True, "0.8"]:
            with self.subTest(score=score), self.assertRaises(ValueError):
                self.retriever.index.response = ([[score]], [[0]])
                self.retrieve()

    def test_malformed_faiss_shapes_are_not_silently_zipped(self):
        for response in [([[0.9, 0.8]], [[0]]), ([], [[0]]), ([[0.9]], []),
                         ([[0.9], [0.8]], [[0], [1]]), ([0.9], [0]), (None, None)]:
            with self.subTest(response=response), self.assertRaises(ValueError):
                self.retriever.index.response = response
                self.retrieve()

    def test_embedding_dimension_normalization_and_shape_are_checked(self):
        for embeddings in [[[1.0]], [[2.0, 0.0]], [[0.0, 0.0]],
                           [[float("nan"), 0.0]], [], [[1.0, 0.0], [1.0, 0.0]]]:
            with self.subTest(embeddings=embeddings), self.assertRaises(ValueError):
                self.retriever.model.embeddings = embeddings
                self.retrieve()
        self.assertEqual(self.retriever.index.calls, [])

    def test_backend_reuses_resources_and_does_not_modify_metadata(self):
        before = deepcopy(self.retriever.metadata)
        model, index = self.retriever.model, self.retriever.index
        first = self.retrieve()
        second = self.retrieve()
        self.assertEqual(first, second)
        first[0]["text"] = "changed output"
        self.assertEqual(self.retriever.metadata, before)
        self.assertIs(self.retriever.model, model)
        self.assertIs(self.retriever.index, index)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len(index.calls), 2)

    def test_encoder_and_search_errors_propagate_without_retry(self):
        self.retriever.model.embeddings = RuntimeError("encoder unavailable")
        with self.assertRaisesRegex(RuntimeError, "encoder unavailable"):
            self.retrieve()
        self.assertEqual(len(self.retriever.model.calls), 1)
        self.retriever.model.embeddings = [[1.0, 0.0]]
        self.retriever.index.response = RuntimeError("search unavailable")
        with self.assertRaisesRegex(RuntimeError, "search unavailable"):
            self.retrieve()
        self.assertEqual(len(self.retriever.index.calls), 1)

    def test_backend_is_compatible_with_enhancer_and_safe_failure_fallback(self):
        base = [{"passage_id": pid, "text": f"Evidence {pid}", "score": 0.8} for pid in "ABCDE"]
        self.retriever.index.response = ([[0.9, 0.8]], [[5, 0]])
        enhancer = RetrievalEnhancer(self.backend)
        result = enhancer.enhance("Where was Ada Lovelace born?", "nq", base)
        self.assertEqual(result["trace"]["selected_ids"], list("AFBCD"))
        self.assertEqual(result["trace"]["new_candidate_ids"], ["F"])
        self.assertEqual(result["trace"]["retrieval_calls"], 1)
        self.assertEqual(result["trace"]["llm_calls"], 0)
        self.retriever.index.response = RuntimeError("failed search")
        fallback = enhancer.enhance("Where was Ada Lovelace born?", "nq", base)
        self.assertEqual(fallback["evidence"], base)
        self.assertEqual(fallback["trace"]["status"], "retrieval_failed")

    def test_gold_fields_are_neither_read_nor_copied_into_evidence(self):
        class WhitelistRecord(dict):
            def get(self, key, default=None):
                if key not in {"passage_id", "text"}:
                    raise AssertionError(f"Non-whitelisted metadata access: {key}")
                return super().get(key, default)
        self.retriever.metadata = [WhitelistRecord(record, gold_evidence="DO NOT READ",
                                                   reference_answer="DO NOT READ")
                                   for record in self.retriever.metadata]
        hits = self.retrieve()
        self.assertTrue(all(set(hit) == {"passage_id", "text", "score"} for hit in hits))
        self.assertEqual(list(inspect.signature(NQFaissBackend.retrieve).parameters),
                         ["self", "query", "dataset", "top_k"])

    def test_matches_actual_base_rag_retrieval_wrapper(self):
        path = Path(__file__).resolve().parents[1] / "scripts/run_nq_base_rag.py"
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if
                 isinstance(node, ast.ImportFrom) and node.module == "__future__"
                 or isinstance(node, ast.FunctionDef) and node.name == "retrieve_top_k"]
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        base_hits = namespace["retrieve_top_k"](self.retriever, ["Who wrote Hamlet?"], 5)[0]
        self.assertEqual(self.retrieve(), base_hits)

    def test_constants_match_repository_bge_configuration(self):
        root = Path(__file__).resolve().parents[1]
        for name in ["scripts/run_nq_base_rag.py", "scripts/run_nq_bge_retriever.py"]:
            tree = ast.parse((root / name).read_text())
            constants = {target.id: ast.literal_eval(node.value) for node in tree.body
                         if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                         for target in node.targets if isinstance(target, ast.Name)}
            self.assertEqual(constants["BGE_QUERY_INSTRUCTION"], BGE_QUERY_INSTRUCTION)
            model_key = "RETRIEVER_MODEL" if "RETRIEVER_MODEL" in constants else "MODEL_NAME"
            self.assertEqual(constants[model_key], BGE_MODEL_NAME)
        for name in ["configs/base.yaml", "configs/preliminary.yaml"]:
            config = (root / name).read_text()
            self.assertIn(f"  embedding_model: {BGE_MODEL_NAME}\n", config)
            self.assertIn("  normalize_embeddings: true\n", config)
            self.assertIn("  top_k: 5\n", config)
            self.assertIn("index_type: faiss_flat_ip\n", config)

    def test_execution_requires_no_faiss_numpy_models_or_team_modules(self):
        code = '''
import sys
for name in ["torch", "transformers", "faiss", "numpy", "sentence_transformers",
             "src.retrieval", "src.generation", "src.diagnosis", "src.controller", "src.probe"]:
    sys.modules[name] = None
from tests.test_retrieval_enhancement_adapters import FakeLoadedRetriever
from src.compensation.retrieval_enhancement import NQFaissBackend
r = NQFaissBackend(FakeLoadedRetriever()).retrieve("Who wrote Hamlet?", dataset="nq", top_k=5)
assert [hit["passage_id"] for hit in r] == list("EBFAC")
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
