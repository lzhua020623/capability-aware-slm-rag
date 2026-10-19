"""Real-output regressions and generation-time termination; no CUDA required."""
import copy
import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.compensation.answer_repair import Generation, repair_answer
from src.compensation.conservative_repair import parse_candidate, parse_check, repair_from_v8_trace
from src.compensation.line_contract import (
    CHECK_LABELS, LEGACY_CHECK_LABELS, CANDIDATE_LABELS, complete_labeled_output, parse_evidence_rank,
)
from src.compensation.qwen_repair import QwenRepairGenerator
from src.diagnosis import DiagnosisInput, EvidenceItem
from tests.test_answer_repair import request
from tests.test_answer_repair_v8 import check, WRONG, CANDIDATE
from scripts.check_answer_repair_v8_smoke import evaluate_smoke


def fixtures():
    return json.loads((Path(__file__).parent / "fixtures/answer_repair_v8r1_contract_cases.json").read_text())["cases"]


def real_request(case):
    row = case["runtime"]
    return DiagnosisInput(row["sample_id"], row["dataset"], row["query"],
                          [EvidenceItem(**e) for e in row["evidence"]], row["initial_answer"])


class ContractTests(unittest.TestCase):
    def test_cannot_stop_in_middle_of_reason_or_answer(self):
        for text, labels in ((check(), CHECK_LABELS), (CANDIDATE, CANDIDATE_LABELS)):
            self.assertFalse(complete_labeled_output(text, labels))
            self.assertTrue(complete_labeled_output(text + "\n", labels))
            for prefix in (text[:i] for i in range(text.rindex(":") + 1, len(text))):
                self.assertFalse(complete_labeled_output(prefix, labels))
        self.assertTrue(complete_labeled_output("INSUFFICIENT", CANDIDATE_LABELS, insufficient=True))
        self.assertFalse(complete_labeled_output("INSUFFICIENT", CHECK_LABELS))
        self.assertFalse(complete_labeled_output("INSUFFICIENT because", CANDIDATE_LABELS, insufficient=True))

    def test_real_repetition_is_rejected_but_online_boundary_is_detected(self):
        case = fixtures()[0]
        r = real_request(case)
        with self.assertRaises(ValueError):
            parse_check(case["text"], r)
        self.assertFalse(complete_labeled_output(case["text"], LEGACY_CHECK_LABELS))
        boundary = next(i for i in range(1, len(case["text"]) + 1)
                        if complete_labeled_output(case["text"][:i], LEGACY_CHECK_LABELS))
        # This simulates token-by-token stopping; it is NOT a new model result.
        prefix = case["text"][:boundary]
        self.assertEqual(len(prefix.strip().splitlines()), 9)
        # A synthetic FACT prefix adapts this parser-only regression to r4.
        # The stored r1 output is unchanged and is not an r4 model result.
        current = "FACT: Synthetic contract adapter for a legacy source fixture.\n" + prefix
        self.assertEqual(parse_check(current, r)["status"], "supported")
        with self.assertRaises(ValueError):
            parse_check(prefix + prefix.replace("ENTITY: yes", "ENTITY: no"), r)
        self.assertFalse(complete_labeled_output(prefix + "ENTITY", CHECK_LABELS))

    def test_real_quote_and_rank_display_variants_keep_source_slices(self):
        quoted, bracketed = fixtures()[1:]
        for case, parser in ((quoted, parse_check), (bracketed, parse_candidate)):
            text = ("FACT: Synthetic contract adapter for a legacy source fixture.\n" + case["text"]
                    if parser is parse_check else case["text"])
            parsed = parser(text, real_request(case))
            citation = parsed.get("citation") or parsed["citations"][0]
            evidence = next(e for e in case["runtime"]["evidence"] if e["rank"] == citation["rank"])
            start, end = parsed["source_offsets"]
            self.assertEqual(evidence["text"][start:end], citation["quote"])

    def test_rank_is_unambiguous_and_zero_only_valid_for_uncertainty(self):
        for value in ("2", "[2]", "[ 2 ]"):
            self.assertEqual(parse_evidence_rank(value), 2)
        for value in ("1 or 2", "[1,2]", "[[2]]", "block 2", "-1", "01", "[2", "2.0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_evidence_rank(value)
        with self.assertRaises(ValueError):
            parse_check(check().replace("RANK: 1", "RANK: [0]"), request())
        uncertain = check(SUPPORT="uncertain").replace("RANK: 1", "RANK: [0]").replace(
            "QUOTE: The capital of France is Paris.", "QUOTE: NONE")
        self.assertEqual(parse_check(uncertain, request())["status"], "uncertain")

    def test_quote_wrapping_does_not_relax_entities_negation_or_token_boundaries(self):
        for bad in ('"The capital of France is London."', '"Paris is the capital of France."',
                    '"The capital of France is Paris." extra', '"The capital of France is Paris.'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_check(check().replace("QUOTE: The capital of France is Paris.", "QUOTE: " + bad), request())
        with self.assertRaises(ValueError):
            parse_candidate('RANK: [1]\nQUOTE: "York"\nANSWER: York', request(text="Yorkshire is a county."))
        with self.assertRaises(ValueError):
            parse_candidate('RANK: 1\nQUOTE: "Paris is the capital."\nANSWER: Paris',
                            request(text="Paris is not the capital."))
        parsed = parse_candidate('RANK: [1]\nQUOTE: “The capital of France is Paris.”\nANSWER: "Paris"', request())
        self.assertEqual(parsed["answer"], "Paris")
        literal = parse_candidate('RANK: 1\nQUOTE: The title is "Paris".\nANSWER: "Paris"',
                                  request(text='The title is "Paris".'))
        self.assertEqual(literal["answer"], '"Paris"')

    def test_three_stage_dispatch_and_replay_preserve_contract_observations(self):
        class Backend:
            def __init__(self):
                self.outputs = iter([WRONG, CANDIDATE, check()])
                self.contracts = []

            def generate(self, *args):
                raise AssertionError("v8 must request labeled stopping from this backend")

            def generate_labeled(self, prompt, max_new_tokens, *, labels, insufficient):
                self.contracts.append((labels, insufficient))
                text = next(self.outputs) + "\n"
                return Generation(text, .1, 10, 20, raw_text=text, contract_stopped=True)

        backend = Backend()
        result = repair_answer(request(), backend, strategy="conservative_repair")
        self.assertEqual(result.final_answer, "Paris")
        self.assertEqual(backend.contracts, [(CHECK_LABELS, False), (CANDIDATE_LABELS, True), (CHECK_LABELS, False)])
        self.assertEqual(asdict(repair_from_v8_trace(request(), result.repair_trace)), asdict(result))
        for stage in result.repair_trace["stages"]:
            self.assertTrue(stage["generation"]["contract_stopped"])
        tampered = copy.deepcopy(result.repair_trace)
        tampered["stages"][0]["generation_contract"]["labels"] = ["ANSWER"]
        with self.assertRaises(ValueError):
            repair_from_v8_trace(request(), tampered)

    def test_qwen_stop_is_installed_only_for_explicit_labeled_generation(self):
        stream = check() + "\n" + check(ENTITY="no") + "\n"

        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(map(ord, text))

            def decode(self, ids, **kwargs):
                return "".join(map(chr, ids))

        class Batch:
            def __init__(self, text):
                self.text = text

            def __getitem__(self, key):
                return list(map(ord, self.text))

        def generated(tokenizer, model, prompt, max_new_tokens, stopping_criteria=None):
            for i in range(1, len(stream) + 1):
                if stopping_criteria and stopping_criteria[0](Batch(stream[:i]), None):
                    return stream[:i], .1
            return stream, .1

        backend = QwenRepairGenerator.__new__(QwenRepairGenerator)
        backend.json_output = False
        backend.tokenizer = Tokenizer()
        backend.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=10000))
        modules = {"transformers": SimpleNamespace(StoppingCriteria=object, StoppingCriteriaList=list),
                   "src.generation.qwen": SimpleNamespace(apply_qwen_chat_template=lambda t, p: p,
                                                           generate_greedy=generated)}
        with patch.dict(sys.modules, modules):
            result = backend.generate_labeled("audit", 384, labels=CHECK_LABELS)
            self.assertEqual(result.text, check() + "\n")
            self.assertEqual(result.raw_text, result.text)
            self.assertTrue(result.contract_stopped)
            v7 = backend.generate("answer", 384)
            self.assertEqual(v7.text, stream)
            self.assertIsNone(v7.contract_stopped)
        backend.json_output = True
        with self.assertRaises(ValueError):
            backend.generate_labeled("audit", 384, labels=CHECK_LABELS)


class SmokeGateTests(unittest.TestCase):
    def test_generated_notebook_compiles_before_archive_is_saved(self):
        from scripts.prepare_answer_repair_v8_r2 import notebook
        loader = (Path(__file__).parents[1] / "scripts/load_answer_repair_bundle.py").read_bytes()
        raw = notebook(Path("answer_repair_v8r2_smoke8_gpu.tar.gz"),
                       {"scripts/load_answer_repair_bundle.py": loader, "bundle_checksums.json": b"{}"},
                       archive_raw=b"test-only-not-an-archive")
        cells = json.loads(raw)["cells"]
        self.assertEqual(len(cells), 6)
        for c in cells:
            if c["cell_type"] == "code":
                compile("".join(c["source"]), "<smoke-notebook>", "exec")

    def inputs(self):
        protocol = json.loads((Path(__file__).parents[1] / "configs/answer_repair_v8r2_smoke8_v1.json").read_text())
        protocol["sample_ids"] = ["test-only"]
        stages = [
            {"stage": name, "validation_error": None, "parsed": {"status": "supported"},
             "generation": {"text": "fixture", "raw_text": "fixture", "contract_stopped": True}}
            for name in ("original_check", "minimal_repair", "candidate_check")]
        rows = {}
        for arm, selected in zip(protocol["protocol"], ([], stages, stages[1:], stages[:2])):
            rows[arm["strategy"]] = [{"sample_id": "test-only", "status": "fixture",
                "extra_calls": len(selected) or 1, "repair_trace": {"stages": copy.deepcopy(selected)}}]
        return protocol, rows

    def test_operational_gate_does_not_require_metric_improvement(self):
        manifest, rows = self.inputs()
        # No EM or human correctness is supplied to the operational gate.
        report = evaluate_smoke(manifest, rows)
        self.assertTrue(report["engineering_gate_passed"])
        self.assertIsNone(report["human_final_gain"])
        self.assertFalse(report["semantic_correctness_verified"])

    def test_missing_paths_invalid_stages_and_cropping_block_full_run(self):
        for issue in ("invalid", "crop", "backend", "path", "incomplete"):
            with self.subTest(issue=issue):
                manifest, rows = self.inputs()
                full = rows["conservative_repair"][0]
                stage = full["repair_trace"]["stages"][0]
                if issue == "invalid":
                    stage["validation_error"] = "bad quote"
                elif issue == "crop":
                    stage["generation"]["raw_text"] += "\nENTITY: no"
                elif issue == "backend":
                    stage["generation"]["contract_stopped"] = None
                elif issue == "path":
                    full["repair_trace"]["stages"].pop()
                else:
                    rows["short_answer_quote"] = []
                self.assertFalse(evaluate_smoke(manifest, rows)["ready_for_full_nq75_run"])


if __name__ == "__main__":
    unittest.main()
