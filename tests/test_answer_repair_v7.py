"""v7: lexical anchors, anchored NLI decisions, short-answer-in-quote parsing, runner wiring."""
import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_answer_repair import main
from src.compensation.anchors import (
    anchor_coverage, claim_anchors, fever_window_anchored, nq_passage_anchored, question_anchors, subject_chunk,
)
from src.compensation.answer_repair import repair_answer
from src.compensation.controller_adapter import generation_repair
from src.compensation.nli_repair import (
    NLI_MODEL, NLI_REVISION, NLI_THRESHOLD, repair_from_nli_trace, repair_from_nli_trace_anchored,
)
from src.compensation.short_answer_repair import (
    INSUFFICIENT, MAX_ANSWER_WORDS, MAX_QUOTE_WORDS, build_short_answer_prompt, locate_excerpt, parse_short_answer,
)
from src.diagnosis import DatasetName, DiagnosisInput, EvidenceItem
from tests.test_answer_repair import c1_row, request
from tests.test_answer_repair_v5 import SequenceGenerator


def lines(rank, quote, answer):
    return SequenceGenerator([f"RANK: {rank}\nQUOTE: {quote}\nANSWER: {answer}"])


def window(item, probabilities, start=0, end=None):
    end = len(item.text) if end is None else end
    return {"rank": item.rank, "passage_id": item.passage_id, "quote": item.text[start:end],
            "start": start, "end": end, "input_tokens": 30, "probabilities": probabilities}


def trace(windows):
    return {"model_name": NLI_MODEL, "model_revision": NLI_REVISION, "threshold": NLI_THRESHOLD,
            "calls": (len(windows) + 3) // 4, "latency": 0.5,
            "input_tokens": sum(w["input_tokens"] for w in windows), "windows": windows}


ENTAIL = {"ENTAILMENT": .96, "CONTRADICTION": .02, "NEUTRAL": .02}
CONTRA = {"ENTAILMENT": .01, "CONTRADICTION": .97, "NEUTRAL": .02}
NEUTRAL = {"ENTAILMENT": .05, "CONTRADICTION": .05, "NEUTRAL": .90}


class AnchorTests(unittest.TestCase):
    def test_claim_anchors_keep_names_and_numbers_and_drop_stopwords(self):
        self.assertEqual(claim_anchors("Henry Cavill starred in two Superman films."), ["henry", "cavill", "superman"])
        self.assertEqual(claim_anchors("The 1974 crime film The Sugarland Express was co-written by Steven Spielberg."),
                         ["1974", "sugarland", "express", "steven", "spielberg"])
        self.assertEqual(claim_anchors("Daggering is not a type of dance."), ["daggering"])
        self.assertEqual(claim_anchors("it is cold."), [])

    def test_subject_chunk_is_the_leading_capitalised_run(self):
        self.assertEqual(subject_chunk("Henry Cavill starred in two Superman films."), ["henry", "cavill"])
        self.assertEqual(subject_chunk("The 1974 crime film The Sugarland Express was co-written by Steven Spielberg."),
                         ["sugarland", "express"])
        self.assertEqual(subject_chunk("In 2003, Eva Green debuted in film."), ["eva", "green"])
        self.assertEqual(subject_chunk("Men in Black II is a film from America."), ["men"])
        self.assertEqual(subject_chunk("Part of Dilwale Dulhania Le Jayenge was shot in August."),
                         ["dilwale", "dulhania", "le", "jayenge"])
        self.assertEqual(subject_chunk("Winter's Tale was published in 1970."), ["winter", "tale"])
        self.assertEqual(subject_chunk("The New Jersey Turnpike has shoulders."), ["new", "jersey", "turnpike"])
        self.assertEqual(subject_chunk("Richard Fortus is a Guns N' Roses member."), ["richard", "fortus"])
        self.assertEqual(subject_chunk("it is cold."), [])

    def test_subject_rule_blocks_namesakes_but_keeps_refuting_pages(self):
        # The disputed attribute (Australian) is absent from the refuting page and must not be required.
        self.assertTrue(fever_window_anchored("Brubaker is a solely Australian film.",
                                              "Brubaker is a 1980 American prison drama film .")["anchored"])
        self.assertFalse(fever_window_anchored("Bea Arthur was born under a different name.",
                                               "Bea Wain -LRB- born Beatrice Weinsier -RRB- is a singer .")["anchored"])
        self.assertFalse(fever_window_anchored("Henry Cavill starred in two Superman films.",
                                               "Henry 's Crime is a 2010 American romantic comedy film .")["anchored"])
        self.assertFalse(fever_window_anchored("Men in Black II is a film from America.",
                                               "Black is a 2015 Belgian crime film .")["anchored"])
        report = fever_window_anchored("In 1974 Alan Shepard retired from the US Navy.",
                                       "Alan Shepard retired from the United States Navy and NASA on August 1 , 1974 .")
        self.assertTrue(report["anchored"])
        self.assertEqual((report["subject"], report["subject_missing"]), (["alan", "shepard"], []))
        self.assertFalse(fever_window_anchored("it is cold.", "It is cold.")["anchored"])

    def test_plural_variants_and_question_anchors(self):
        self.assertEqual(anchor_coverage(["touchscreens"], "A touchscreen is an input device .")["coverage"], 1.0)
        self.assertEqual(question_anchors("who sings good night on the beatles white album"),
                         ["sings", "good", "night", "beatles", "white", "album"])
        self.assertTrue(nq_passage_anchored("when did end of the road come out",
                                            "End of the Road was released on June 30 , 1992 .")["anchored"])
        self.assertFalse(nq_passage_anchored("who wrote the song", "Paris is the capital of France .")["anchored"])


def fever_request(**kwargs):
    return request(dataset=DatasetName.FEVER, answer="SUPPORTS", **kwargs)


class AnchoredNLITests(unittest.TestCase):
    def setUp(self):
        self.r = DiagnosisInput("1", DatasetName.FEVER, "Henry Cavill starred in two Superman films.", [
            EvidenceItem(1, "Man_of_Steel", "Man of Steel is a 2013 superhero film . It stars Henry Cavill as Superman .", .9),
            EvidenceItem(2, "Henrys_Crime", "Henry 's Crime is a 2010 American romantic comedy film .", .8)], "SUPPORTS")

    def test_v6_rule_is_unchanged_and_v7_ignores_unanchored_contradiction(self):
        t = trace([window(self.r.evidence[0], NEUTRAL), window(self.r.evidence[1], CONTRA)])
        self.assertEqual((repair_from_nli_trace(self.r, t).final_answer, repair_from_nli_trace(self.r, t).anchor_filter),
                         ("REFUTES", None))
        v7 = repair_from_nli_trace_anchored(self.r, t)
        self.assertEqual((v7.final_answer, v7.status, v7.citations), ("SUPPORTS", "nli_uncertain", []))
        self.assertEqual(v7.anchor_filter["excluded_decisive_windows"], [1])
        self.assertEqual([w["anchored"] for w in v7.anchor_filter["windows"]], [True, False])

    def test_anchored_window_resolves_a_v6_conflict(self):
        t = trace([window(self.r.evidence[0], ENTAIL), window(self.r.evidence[1], CONTRA)])
        self.assertEqual(repair_from_nli_trace(self.r, t).status, "nli_conflict")
        v7 = repair_from_nli_trace_anchored(self.r, t)
        self.assertEqual((v7.status, v7.verdict, v7.citations[0].passage_id), ("kept", "keep", "Man_of_Steel"))
        self.assertEqual((v7.verifier_calls, v7.extra_calls, v7.independent_support_status), (1, 0, "nli_entailment"))

    def test_anchored_repair_and_trace_tampering(self):
        r = DiagnosisInput("2", DatasetName.FEVER, "Daggering is not a type of dance.",
                           [EvidenceItem(1, "Daggering", "Daggering is a form of dance originating from Jamaica .", .9)], "SUPPORTS")
        t = trace([window(r.evidence[0], CONTRA)])
        v7 = repair_from_nli_trace_anchored(r, t)
        self.assertEqual((v7.final_answer, v7.status), ("REFUTES", "repaired"))
        t["windows"][0]["quote"] = "made up"
        with self.assertRaises(ValueError):
            repair_from_nli_trace_anchored(r, t)

    def test_strategy_dispatch_requires_verifier_and_fever(self):
        verifier = SimpleNamespace(score=lambda data: trace([window(data.evidence[0], ENTAIL)]))
        r = fever_request(text="Mira is a botanist.")
        result = repair_answer(r, strategy="fever_nli_anchored", verifier=verifier)
        self.assertEqual((result.final_answer, result.extra_calls, result.verifier_calls), ("SUPPORTS", 0, 1))
        with self.assertRaises(ValueError):
            repair_answer(r, strategy="fever_nli_anchored")
        with self.assertRaises(ValueError):
            repair_answer(request(), strategy="fever_nli_anchored", verifier=verifier)


class ExcerptTests(unittest.TestCase):
    text = "`` Good Night '' is a song by the Beatles , composed by Lennon - McCartney . Ca n't Help Falling in Love ."

    def test_spacing_and_quote_marks_are_normalised_but_slices_are_source_text(self):
        for quote in ('"Good Night" is a song', "``Good Night'' is a song", "“Good Night” is a song"):
            self.assertEqual(locate_excerpt(quote, self.text)[0], "`` Good Night '' is a song")
        self.assertEqual(locate_excerpt("Lennon-McCartney", self.text)[0], "Lennon - McCartney")
        self.assertEqual(locate_excerpt("Can't Help Falling in Love.", self.text)[0], "Ca n't Help Falling in Love .")

    def test_no_subwords_deletions_or_splices(self):
        for quote in ("Beat", "Night is a song", "song by Beatles", "Good Night is a song by the Beatles composed", ""):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                locate_excerpt(quote, self.text)
        with self.assertRaises(ValueError):
            locate_excerpt("York", "Yorkshire is not in Paris.")


class ShortAnswerTests(unittest.TestCase):
    def setUp(self):
        self.r = DiagnosisInput("1", DatasetName.NQ, "who sings good night on the beatles white album", [
            EvidenceItem(1, "p1", "`` Good Night '' is a song by the Beatles . It is sung by Ringo Starr , "
                                  "the only Beatle to appear on the track .", .9),
            EvidenceItem(2, "p2", "Paris is the capital of France .", .5)], "John Lennon")

    def test_prompt_contains_only_runtime_fields(self):
        prompt = build_short_answer_prompt(self.r)
        self.assertIn("[2] Paris", prompt)
        self.assertNotIn("John Lennon", prompt)
        self.assertIn(INSUFFICIENT, prompt)

    def test_short_answer_is_a_source_slice_inside_the_quote(self):
        result = repair_answer(self.r, lines(1, "It is sung by Ringo Starr, the only Beatle to appear on the track.",
                                              "Ringo Starr"), strategy="short_answer_quote")
        self.assertEqual((result.final_answer, result.status, result.verdict, result.extra_calls),
                         ("Ringo Starr", "repaired", "repair", 1))
        self.assertEqual(result.citations[0].quote, "It is sung by Ringo Starr , the only Beatle to appear on the track .")
        self.assertTrue(result.anchor_filter["anchored"])
        self.assertEqual(result.independent_support_status, "not_checked")

    def test_normalised_equality_is_keep(self):
        r = DiagnosisInput("1", DatasetName.NQ, self.r.query, self.r.evidence, "ringo starr")
        result = repair_answer(r, lines(1, "It is sung by Ringo Starr", "Ringo Starr"), strategy="short_answer_quote")
        self.assertEqual((result.final_answer, result.status, result.changed), ("ringo starr", "kept", False))

    def test_insufficient_and_invalid_outputs_preserve_initial(self):
        result = repair_answer(self.r, SequenceGenerator([INSUFFICIENT]), strategy="short_answer_quote")
        self.assertEqual((result.status, result.final_answer), ("insufficient_evidence", "John Lennon"))
        bad = [
            "RANK: 1\nQUOTE: It is sung by Ringo Starr\nANSWER: Ringo Starr\nNote: done",   # trailing text
            "QUOTE: It is sung by Ringo Starr\nRANK: 1\nANSWER: Ringo Starr",              # order
            "RANK: 7\nQUOTE: It is sung by Ringo Starr\nANSWER: Ringo Starr",              # rank
            "RANK: 1\nQUOTE: It is sung by Ringo Starr\nANSWER: Ringo",                    # subword? no: whole token ok -> use Beatle
            "RANK: 1\nQUOTE: It is sung by Ringo Starr\nANSWER: Paul McCartney",           # answer not in quote
            "RANK: 1\nQUOTE: It is sung by Paul\nANSWER: Paul",                             # quote not verbatim
            "RANK: 1\nQUOTE: " + " ".join(["word"] * (MAX_QUOTE_WORDS + 1)) + "\nANSWER: word",
        ]
        for text in bad[:3] + bad[4:]:
            with self.subTest(text=text[:40]):
                result = repair_answer(self.r, SequenceGenerator([text]), strategy="short_answer_quote")
                self.assertEqual((result.status, result.final_answer), ("invalid_output", "John Lennon"))
                self.assertTrue(result.validation_error)
        long_answer = "RANK: 2\nQUOTE: Paris is the capital of France.\nANSWER: " + " ".join(["Paris"] * (MAX_ANSWER_WORDS + 1))
        self.assertEqual(repair_answer(self.r, SequenceGenerator([long_answer]), strategy="short_answer_quote").status,
                         "invalid_output")
        self.assertIsNone(parse_short_answer(" INSUFFICIENT \n"))

    def test_unanchored_passage_keeps_initial_answer(self):
        result = repair_answer(self.r, lines(2, "Paris is the capital of France.", "Paris"), strategy="short_answer_quote")
        self.assertEqual((result.status, result.final_answer, result.citations), ("unanchored_candidate", "John Lennon", []))
        self.assertFalse(result.anchor_filter["anchored"])
        with self.assertRaises(ValueError):
            repair_answer(fever_request(), lines(1, "x", "x"), strategy="short_answer_quote")

    def test_adapter_runs_line_strategy_without_json_stop(self):
        decision = SimpleNamespace(action="generation_repair", diagnosis_label="evidence_utilisation",
                                   diagnosis_confidence=.4, used_probe=False, probe_overrode_diagnosis=False,
                                   needs_compensation=True, sample_id="1")
        gen = lines(1, "It is sung by Ringo Starr", "Ringo Starr")
        result = generation_repair(self.r, decision, gen, strategy="short_answer_quote")
        self.assertEqual((result.final_answer, len(gen.calls)), ("Ringo Starr", 1))
        decision = SimpleNamespace(**{**vars(decision), "action": "no_action", "diagnosis_label": "correct",
                                      "needs_compensation": False})
        self.assertEqual(generation_repair(self.r, decision, gen, strategy="short_answer_quote").status,
                         "skipped_controller_action")


class RunnerTests(unittest.TestCase):
    def test_short_answer_run_resume_and_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, out = root / "c1.jsonl", root / "out.jsonl"
            rows = [c1_row("1", "London"), c1_row("2", "Paris")]
            source.write_text("".join(json.dumps(r) + "\n" for r in rows))
            args = ["--dataset", "nq", "--c1", str(source), "--all", "--strategy", "short_answer_quote", "--out", str(out)]
            outputs = ["RANK: 1\nQUOTE: The capital of France is Paris.\nANSWER: Paris", RuntimeError("interrupted")]
            gen = SequenceGenerator(outputs)
            with contextlib.redirect_stdout(io.StringIO()), patch("src.compensation.qwen_repair.QwenRepairGenerator", return_value=gen) as factory:
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    main(args)
                self.assertEqual(factory.call_args.kwargs, {"json_output": False})
            gen = SequenceGenerator(["RANK: 1\nQUOTE: The capital of France is Paris.\nANSWER: Paris"])
            with contextlib.redirect_stdout(io.StringIO()), patch("src.compensation.qwen_repair.QwenRepairGenerator", return_value=gen):
                main(args)
            saved = [json.loads(l) for l in out.read_text().splitlines()]
            self.assertEqual([r["status"] for r in saved], ["repaired", "kept"])
            self.assertEqual(saved[0]["settings"]["prompt_version"], "v7_short_answer_in_quote_lines")
            self.assertIsNone(saved[0]["settings"]["json_stop"])
            self.assertEqual(saved[0]["settings"]["anchor_policy"]["version"], "v7_lexical_subject_anchor")
            self.assertEqual(saved[0]["final_metrics"]["correct"], True)
            with contextlib.redirect_stdout(io.StringIO()):
                main(args + ["--check-only"])

    def test_anchored_nli_resume_validates_v7_decision(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, out = root / "c1.jsonl", root / "out.jsonl"
            row = c1_row("1")
            row.update(dataset="fever", question_or_claim="Mira is a botanist.", prediction="REFUTES",
                       gold_answer_or_label="SUPPORTS", correct=False)
            row["evidence"] = [{"passage_id": "Mira", "text": "Mira is a botanist .", "score": .9},
                               {"passage_id": "Other", "text": "Leon is a baker .", "score": .5}]
            source.write_text(json.dumps(row) + "\n")
            args = ["--dataset", "fever", "--c1", str(source), "--all", "--strategy", "fever_nli_anchored", "--out", str(out)]

            def score(data):
                return trace([window(data.evidence[0], ENTAIL), window(data.evidence[1], CONTRA)])
            env = {"id2label": {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}}
            verifier = SimpleNamespace(score=score, runtime_environment=env)
            with contextlib.redirect_stdout(io.StringIO()), patch("src.compensation.nli_repair.MNLIVerifier", return_value=verifier):
                main(args)
            saved = [json.loads(l) for l in out.read_text().splitlines()]
            self.assertEqual((saved[0]["final_answer"], saved[0]["status"]), ("SUPPORTS", "repaired"))
            self.assertEqual(saved[0]["anchor_filter"]["excluded_decisive_windows"], [1])
            self.assertEqual(saved[0]["settings"]["prompt_version"], "v7_mnli_windows_threshold_090_subject_anchor")
            with contextlib.redirect_stdout(io.StringIO()):
                main(args + ["--check-only"])
            saved[0]["final_answer"] = "REFUTES"
            saved[0]["final_metrics"] = {"correct": False, "parsed_prediction": "REFUTES"}
            saved[0]["changed"] = False
            out.write_text(json.dumps(saved[0]) + "\n")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "trace/cost mismatch"):
                main(args + ["--check-only"])


if __name__ == "__main__":
    unittest.main()
