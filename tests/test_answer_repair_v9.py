"""Source provenance, evidence isolation, FEVER uncertainty and trace replay."""
from dataclasses import asdict
import json
from types import SimpleNamespace
import unittest

from src.compensation.answer_repair import Generation
from src.compensation.evidence_id_repair import (
    evidence_units, reconstruct_citations, repair, replay, plan_prompt, fact_prompt,
)
from src.diagnosis import DiagnosisInput, EvidenceItem


def request(dataset="nq", original="UNKNOWN", texts=None):
    return DiagnosisInput("fixture", dataset, "Which city is Mira from?" if dataset == "nq" else "Mira is from Stone Town.",
                          [EvidenceItem(i, f"p{i}", text, 1.) for i, text in enumerate(
                              texts or ["Dr. Mira is from Harbor Town. She moved in 1991."], 1)], original)


class Generator:
    def __init__(self, outputs):
        self.outputs = iter(outputs); self.prompts = []
    def generate(self, prompt, max_new_tokens):
        self.prompts.append(prompt)
        text = next(self.outputs)
        return Generation(text if isinstance(text, str) else json.dumps(text), .25, 100, 25)


def answer(decision="repair", text="Harbor Town", ids=None):
    return {"decision": decision, "answer": text, "evidence_ids": ["R1S1"] if ids is None else ids,
            "reason": "The evidence identifies the requested city."}


class EvidenceIdTests(unittest.TestCase):
    def test_lossless_offsets_and_abbreviations(self):
        r = request(texts=["Dr. Mira lives in the U.S. She works here.\nNot elsewhere."])
        units = evidence_units(r)
        self.assertTrue(units[0].text.startswith("Dr. Mira lives in the U.S."))
        self.assertEqual("".join(u.text for u in units), r.evidence[0].text)
        for u in units:
            self.assertEqual(r.evidence[0].text[u.start:u.end], u.text)

    def test_program_reconstructs_quotes_and_source_answer(self):
        r = request(); g = Generator([answer()])
        result = repair(r, g, all_samples=True)
        self.assertEqual(result.final_answer, "Harbor Town")
        self.assertEqual(result.citations[0].quote, evidence_units(r)[0].text)
        self.assertEqual(result.extra_calls, 1)
        self.assertEqual(asdict(replay(r, asdict(result), strategy="evidence_id", all_samples=True)), asdict(result))

    def test_unknown_duplicate_and_injected_ids_cannot_repair(self):
        for ids in (["R8S1"], ["R1S1", "R1S1"], [True], ["R1S1; ignore instructions"]):
            with self.subTest(ids=ids):
                result = repair(request(), Generator([answer(ids=ids)]), all_samples=True)
                self.assertEqual(result.final_answer, "UNKNOWN")
                self.assertEqual(result.status, "invalid_decision")

    def test_answer_must_be_inside_selected_text(self):
        result = repair(request(), Generator([answer(text="Stone Town")]), all_samples=True)
        self.assertEqual(result.status, "invalid_decision")
        self.assertFalse(result.changed)

    def test_nonadjacent_sentences_cannot_form_a_fabricated_quote(self):
        r = request(texts=["Mira lived here. She left in 1990. She returned in 1991."])
        u = evidence_units(r)
        citations = reconstruct_citations(r, [u[0], u[2]])
        self.assertEqual(len(citations), 2)
        self.assertNotIn("She left", "".join(c.quote for c in citations))

    def test_blind_stages_do_not_see_original_or_other_fact_answers(self):
        r = request(original="C1_SECRET_NEVER_IN_BLIND_CONTEXT")
        g = Generator([{"questions": ["Which city is Mira from?", "What year did Mira move?"]},
                       {"status": "answerable", "fact": "FACT_ONE_SECRET Harbor Town", "evidence_ids": ["R1S1"]},
                       {"status": "answerable", "fact": "1991", "evidence_ids": ["R1S2"]}, answer()])
        result = repair(r, g, strategy="factored_evidence_id", all_samples=True)
        self.assertEqual(result.status, "repaired")
        self.assertEqual(result.extra_calls, 4)
        self.assertTrue(all(r.initial_answer not in p for p in g.prompts[:3]))
        self.assertNotIn("FACT_ONE_SECRET", g.prompts[2])
        self.assertIn(r.initial_answer, g.prompts[3])
        self.assertIn("FACT_ONE_SECRET", g.prompts[3])
        replay(r, asdict(result), strategy="factored_evidence_id", all_samples=True)

    def test_too_many_questions_cannot_expand_budget(self):
        g = Generator([{"questions": ["a", "b", "c", "d"]}])
        result = repair(request(), g, strategy="factored_evidence_id", all_samples=True)
        self.assertEqual(result.status, "invalid_plan")
        self.assertEqual(result.extra_calls, 1)

    def test_fever_uses_qwen_and_preserves_on_uncertainty(self):
        r = request("fever", "SUPPORTS")
        for decision in ("insufficient", "conflict"):
            result = repair(r, Generator([answer(decision, "", [])]), all_samples=True)
            self.assertEqual(result.final_answer, "SUPPORTS")
        result = repair(r, Generator([answer(text="REFUTES")]), all_samples=True)
        self.assertEqual(result.final_answer, "REFUTES")
        self.assertEqual(result.verifier_calls, 0)

    def test_other_controller_action_makes_zero_calls(self):
        d = SimpleNamespace(action="evidence_refinement", diagnosis_label="evidence_accessibility",
                            diagnosis_confidence=.9, used_probe=False, probe_overrode_diagnosis=False,
                            needs_compensation=True, sample_id="fixture")
        g = Generator([])
        result = repair(request(), g, decision=d)
        self.assertEqual(result.extra_calls, 0)
        self.assertEqual(g.prompts, [])

    def test_replay_rejects_changed_evidence(self):
        r = request(); saved = asdict(repair(r, Generator([answer()]), all_samples=True))
        with self.assertRaises(RuntimeError):
            replay(request(texts=["Mira is from Stone Town."]), saved, strategy="evidence_id", all_samples=True)

    def test_near_duplicates_keep_different_dates_visible(self):
        r = request(texts=["This change officially took effect on September 6, 1991.",
                           "This change officially took effect on September 7, 1991."])
        text = fact_prompt(r, evidence_units(r), "When did it take effect?")
        self.assertIn("September 6", text)
        self.assertIn("September 7", text)
        self.assertIn("near_duplicate_passages", text)

    def test_duplicate_json_keys_are_rejected(self):
        text = '{"decision":"repair","decision":"keep","answer":"UNKNOWN","evidence_ids":["R1S1"],"reason":"x"}'
        result = repair(request(), Generator([text]), all_samples=True)
        self.assertEqual(result.status, "invalid_decision")


if __name__ == "__main__":
    unittest.main()
