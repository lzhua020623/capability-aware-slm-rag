"""v8 gate contracts, unsafe candidate rejection, replay/resume and offline isolation."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_answer_repair import main, runtime_input
from scripts.summarize_answer_repair_v8 import summarize
from src.compensation.answer_repair import repair_answer
from src.compensation.conservative_repair import (
    CHECK_KEYS, V8_STRATEGIES, build_check_prompt, build_minimal_repair_prompt,
    parse_check, repair_from_v8_trace,
)
from src.compensation.controller_adapter import generation_repair
from src.diagnosis import DatasetName
from tests.test_answer_repair import c1_row, request
from tests.test_answer_repair_v5 import SequenceGenerator


def check(**changes):
    values = {**dict.fromkeys(CHECK_KEYS, "yes"), **changes}
    return "\n".join(f"{key}: {values[key]}" for key in CHECK_KEYS) + (
        "\nRANK: 1\nQUOTE: The capital of France is Paris.\nREASON: The evidence establishes the requested capital.")


CANDIDATE = "RANK: 1\nQUOTE: The capital of France is Paris.\nANSWER: Paris"
WRONG = check(RELATION="no", SUPPORT="no")


class ConservativeTests(unittest.TestCase):
    def test_supported_alias_is_preserved_with_one_call(self):
        r = request(answer="the city of Paris")
        gen = SequenceGenerator([check()])
        result = repair_answer(r, gen, strategy="conservative_repair")
        self.assertEqual((result.final_answer, result.extra_calls, result.status),
                         (r.initial_answer, 1, "original_supported"))
        self.assertFalse(result.changed)

    def test_full_repair_costs_and_cpu_replay(self):
        gen = SequenceGenerator([WRONG, CANDIDATE, check()])
        result = repair_answer(request(), gen, strategy="conservative_repair")
        self.assertEqual((result.final_answer, result.status, result.extra_calls), ("Paris", "repaired", 3))
        self.assertEqual((result.input_tokens, result.output_tokens_estimate, result.latency), (300, 60, .75))
        self.assertEqual([c[1] for c in gen.calls], [384, 384, 384])
        self.assertEqual(result.independent_support_status, "not_checked")
        self.assertEqual(result.verifier_calls, 0)
        trace = json.loads(json.dumps(result.repair_trace))
        self.assertEqual(asdict(repair_from_v8_trace(request(), trace)), asdict(result))

    def test_uncertainty_or_any_invalid_stage_keeps_original(self):
        cases = [
            ([check(SUPPORT="uncertain")], "original_uncertain", 1),
            (["{}"], "original_check_invalid", 1),
            ([WRONG, "INSUFFICIENT"], "candidate_insufficient", 2),
            ([WRONG, CANDIDATE.replace("RANK: 1", "RANK: 9")], "candidate_invalid", 2),
            ([WRONG, CANDIDATE.replace("Paris.", "Paris is not the capital.")], "candidate_invalid", 2),
            ([WRONG, CANDIDATE, "{}"], "candidate_check_invalid", 3),
        ]
        for outputs, status, calls in cases:
            with self.subTest(status=status):
                result = repair_answer(request(), SequenceGenerator(outputs), strategy="conservative_repair")
                self.assertEqual((result.final_answer, result.status, result.extra_calls), ("London", status, calls))
                self.assertEqual(result.citations, [])

    def test_verbatim_candidate_is_not_accepted_without_semantic_checks(self):
        for key in CHECK_KEYS:
            for value in ("no", "uncertain"):
                with self.subTest(key=key, value=value):
                    result = repair_answer(request(), SequenceGenerator([WRONG, CANDIDATE, check(**{key: value})]),
                                           strategy="conservative_repair")
                    self.assertEqual((result.final_answer, result.status, result.candidate_answer),
                                     ("London", "candidate_rejected", "Paris"))

    def test_no_normalization_only_rewrite(self):
        result = repair_answer(request(answer="PARIS"), SequenceGenerator([WRONG, CANDIDATE]),
                               strategy="conservative_repair")
        self.assertEqual((result.final_answer, result.extra_calls, result.status), ("PARIS", 2, "candidate_unchanged"))

    def test_ablations_remove_only_the_named_gate(self):
        for strategy, outputs, expected_stages in (
            ("conservative_no_original_gate", [CANDIDATE, check()], ["minimal_repair", "candidate_check"]),
            ("conservative_no_candidate_gate", [WRONG, CANDIDATE], ["original_check", "minimal_repair"]),
        ):
            gen = SequenceGenerator(outputs)
            result = repair_answer(request(), gen, strategy=strategy)
            self.assertEqual((result.final_answer, result.extra_calls), ("Paris", 2))
            self.assertEqual([s["stage"] for s in result.repair_trace["stages"]], expected_stages)
            repair_prompt = next(prompt for (prompt, _), stage in zip(gen.calls, expected_stages) if stage == "minimal_repair")
            self.assertEqual(repair_prompt, build_minimal_repair_prompt(request()))

    def test_extractively_complete_explanation_can_exceed_v7_twenty_words(self):
        text = "The name came from an old roadside sign whose first letter had fallen off, leaving the words Doors Down, with three added for the original members."
        # Direct parser test: explanation spans are not mechanically truncated to a name.
        from src.compensation.conservative_repair import parse_candidate
        parsed = parse_candidate("RANK: 1\nQUOTE: " + text + "\nANSWER: " + text, request(text=text))
        self.assertEqual(parsed["answer"], text)

    def test_grounding_contract_cannot_splice_or_remove_negation(self):
        for bad in (check().replace("Paris.", "London."), check().replace("ENTITY: yes", "ENTITY: true"),
                    check() + "\nExtra: yes", check().replace("RANK: 1", "RANK: 0").replace("QUOTE: The capital of France is Paris.", "QUOTE: NONE")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_check(bad, request())
        uncertain = check(SUPPORT="uncertain").replace("RANK: 1", "RANK: 0").replace("QUOTE: The capital of France is Paris.", "QUOTE: NONE")
        self.assertEqual(parse_check(uncertain, request())["status"], "uncertain")

    def test_runtime_failures_propagate_at_each_stage(self):
        for outputs in ([RuntimeError("GPU failed")], [WRONG, RuntimeError("GPU failed")],
                        [WRONG, CANDIDATE, RuntimeError("GPU failed")]):
            with self.assertRaisesRegex(RuntimeError, "GPU failed"):
                repair_answer(request(), SequenceGenerator(outputs), strategy="conservative_repair")

    def test_trace_cannot_change_stages_decisions_or_prompts(self):
        result = repair_answer(request(), SequenceGenerator([WRONG, CANDIDATE, check()]), strategy="conservative_repair")
        for mutate in (
            lambda t: t["stages"].pop(),
            lambda t: t["stages"][0].update(stage="candidate_check"),
            lambda t: t["stages"][0].update(prompt_sha256="bad"),
            lambda t: t["stages"][0]["parsed"].update(status="supported"),
        ):
            trace = copy.deepcopy(result.repair_trace)
            mutate(trace)
            with self.assertRaises(ValueError):
                repair_from_v8_trace(request(), trace)

    def test_offline_labels_cannot_enter_prompts_and_non_e_route_costs_zero(self):
        row = c1_row("1")
        row.update(gold_answer_or_label="SECRET_GOLD", human_correct=False, diagnosis_label="SECRET_LABEL")
        r = runtime_input(row, DatasetName.NQ)
        for prompt in (build_check_prompt(r), build_check_prompt(r, "London"), build_minimal_repair_prompt(r)):
            self.assertNotIn("SECRET_", prompt)
        decision = SimpleNamespace(action="no_action", diagnosis_label="correct", diagnosis_confidence=.8,
                                   used_probe=False, probe_overrode_diagnosis=False, needs_compensation=False, sample_id="1")
        result = generation_repair(r, decision, SequenceGenerator([]), strategy="conservative_repair")
        self.assertEqual((result.extra_calls, result.status), (0, "skipped_controller_action"))

    def test_nq_only(self):
        for strategy in V8_STRATEGIES:
            with self.assertRaisesRegex(ValueError, "NQ only"):
                repair_answer(request(dataset=DatasetName.FEVER), SequenceGenerator([]), strategy=strategy)


class RunnerTests(unittest.TestCase):
    def test_line_backend_resume_and_saved_cost_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder) / "c1.jsonl", Path(folder) / "out.jsonl"
            source.write_text(json.dumps(c1_row("1")) + "\n")
            args = ["--dataset", "nq", "--c1", str(source), "--all", "--strategy", "conservative_repair", "--out", str(out)]
            gen = SequenceGenerator([WRONG, CANDIDATE, check()])
            with contextlib.redirect_stdout(io.StringIO()), patch("src.compensation.qwen_repair.QwenRepairGenerator", return_value=gen) as factory:
                main(args)
                self.assertEqual(factory.call_args.kwargs, {"json_output": False})
                main(args + ["--check-only"])
                self.assertEqual(len(gen.calls), 3)
            row = json.loads(out.read_text())
            self.assertEqual(row["settings"]["max_calls_per_sample"], 3)
            row["latency"] = 0
            out.write_text(json.dumps(row) + "\n")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "v8 decision/trace/cost mismatch"):
                main(args + ["--check-only"])

    def test_offline_comparison_preserves_pending_semantics_and_rejects_fingerprint_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle, outputs = Path(folder) / "bundle", Path(folder) / "results"
            (bundle / "inputs").mkdir(parents=True)
            (bundle / "offline").mkdir()
            source = bundle / "inputs/nq75_c1.jsonl"
            source.write_text(json.dumps(c1_row("1")) + "\n")
            (bundle / "manifest.json").write_text(json.dumps({"version": "test-only", "sample_ids": ["1"],
                "protocol": [{"strategy": "conservative_repair", "max_new_tokens": 384}]}))
            (bundle / "offline/initial_reviews.jsonl").write_text(json.dumps({"sample_id": "1", "review_number": 6,
                "group": "include_e_core", "initial_correct": False}) + "\n")
            out = outputs / "conservative_repair.jsonl"
            args = ["--dataset", "nq", "--c1", str(source), "--all", "--strategy", "conservative_repair", "--out", str(out)]
            with contextlib.redirect_stdout(io.StringIO()), patch("src.compensation.qwen_repair.QwenRepairGenerator",
                    return_value=SequenceGenerator([WRONG, CANDIDATE, check()])):
                main(args)
            report = summarize(bundle, outputs)
            self.assertTrue(report["all_arms_complete"])
            self.assertEqual(report["pending_changed_outputs"], 1)
            self.assertEqual(report["arms"]["conservative_repair"]["fixed"], 1)
            self.assertIsNone(report["human_final_gain"])
            queue = json.loads((outputs / "output_review_queue.jsonl").read_text())
            self.assertEqual(queue["human_output_review"]["status"], "pending")
            self.assertIsNone(queue["human_output_review"]["final_correct"])
            row = json.loads(out.read_text())
            row["run_fingerprint"] = "tampered"
            out.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                summarize(bundle, outputs)


if __name__ == "__main__":
    unittest.main()
