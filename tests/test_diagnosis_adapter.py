"""Tests for the C1 record -> DiagnosisInput adapter. No models are loaded."""

from __future__ import annotations

import copy
import dataclasses
import unittest

from src.diagnosis.adapter import c1_record_to_input
from src.diagnosis.schema import FORBIDDEN_RUNTIME_FIELDS, DatasetName, DiagnosisInput

GOLD_MARKER = "GOLD-ONLY-VALUE"


def nq_record() -> dict:
    return {
        "sample_id": "-7299076255108845857",
        "dataset": "natural_questions",
        "condition": "C1",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "cpu_offload": True,
        "cpu_offload_modules": ["lm_head"],
        "question_or_claim": "what is the sentencing reform act of 1984",
        "gold_answer_or_label": GOLD_MARKER,
        "evidence": [
            {"passage_id": "nq_p_083952", "text": "The Sentencing Reform Act ...", "score": 0.7904664874076843},
            {"passage_id": "nq_p_021156", "text": "In 1984 , the Bilingual ...", "score": 0.6171275973320007},
        ],
        "prediction": "U.S. Sentencing Commission",
        "correct": False,
        "latency": 1.34,
        "em": 0,
        "f1": 0.2667,
        "retrieved_top5_ids": ["nq_p_083952", "nq_p_021156"],
        "gold_evidence_in_top5": True,
    }


def fever_record() -> dict:
    return {
        "sample_id": 106023,
        "dataset": "fever",
        "condition": "C1",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "experiment_fingerprint": "ad716bd0",
        "cpu_offload": False,
        "question_or_claim": "Boeing 787 Dreamliner is a machine.",
        "gold_answer_or_label": GOLD_MARKER,
        "evidence": [
            {"passage_id": "Boeing_787_Dreamliner", "text": "The Boeing 787 Dreamliner is ...", "score": 0.5653},
            {"passage_id": "Boeing_707", "text": "The Boeing 707 is ...", "score": 0.4211},
        ],
        "retrieved_top5_ids": ["Boeing_787_Dreamliner", "Boeing_707"],
        "gold_evidence_in_top5": True,
        "raw_output": "SUPPORTS\n\nThe evidence says ...",
        "prediction": "SUPPORTS",
        "correct": True,
        "latency": 2.67,
    }


def input_values(item: DiagnosisInput) -> list:
    values = [item.sample_id, item.dataset, item.query, item.initial_answer]
    for evidence in item.evidence:
        values.extend(dataclasses.astuple(evidence))
    return values


class AdapterConversionTests(unittest.TestCase):
    def test_nq_record(self):
        item = c1_record_to_input(nq_record())
        self.assertEqual(item.sample_id, "-7299076255108845857")
        self.assertIs(item.dataset, DatasetName.NQ)
        self.assertEqual(item.query, "what is the sentencing reform act of 1984")
        self.assertEqual(item.initial_answer, "U.S. Sentencing Commission")
        self.assertEqual(
            [(e.rank, e.passage_id, e.score) for e in item.evidence],
            [(1, "nq_p_083952", 0.7904664874076843), (2, "nq_p_021156", 0.6171275973320007)],
        )
        self.assertEqual(item.evidence[0].text, "The Sentencing Reform Act ...")

    def test_fever_record(self):
        item = c1_record_to_input(fever_record())
        self.assertIs(item.dataset, DatasetName.FEVER)
        self.assertEqual(item.query, "Boeing 787 Dreamliner is a machine.")
        self.assertEqual(item.initial_answer, "SUPPORTS")
        self.assertEqual([e.passage_id for e in item.evidence], ["Boeing_787_Dreamliner", "Boeing_707"])

    def test_fever_int_sample_id_becomes_string(self):
        item = c1_record_to_input(fever_record())
        self.assertEqual(item.sample_id, "106023")
        self.assertIsInstance(item.sample_id, str)

    def test_natural_questions_maps_to_nq(self):
        self.assertIs(c1_record_to_input(nq_record()).dataset, DatasetName.NQ)
        record = nq_record()
        record["dataset"] = "nq"
        self.assertIs(c1_record_to_input(record).dataset, DatasetName.NQ)

    def test_unknown_dataset_rejected(self):
        record = nq_record()
        record["dataset"] = "hotpotqa"
        with self.assertRaises(ValueError):
            c1_record_to_input(record)

    def test_fever_raw_output_ignored(self):
        item = c1_record_to_input(fever_record())
        self.assertNotIn("The evidence says", item.initial_answer)


class AdapterLeakageTests(unittest.TestCase):
    def test_gold_and_evaluation_fields_not_copied(self):
        for record in (nq_record(), fever_record()):
            item = c1_record_to_input(record)
            field_names = {f.name for f in dataclasses.fields(item)}
            self.assertFalse(field_names & FORBIDDEN_RUNTIME_FIELDS)
            self.assertNotIn(GOLD_MARKER, [str(value) for value in input_values(item)])

    def test_output_does_not_depend_on_gold_fields(self):
        for record in (nq_record(), fever_record()):
            changed = copy.deepcopy(record)
            changed["gold_answer_or_label"] = "something else"
            changed["correct"] = not record["correct"]
            changed["gold_evidence_in_top5"] = False
            changed.pop("latency")
            self.assertEqual(c1_record_to_input(record), c1_record_to_input(changed))


class AdapterValidationTests(unittest.TestCase):
    def test_wrong_condition_rejected(self):
        for condition in ("C0", "C3", None):
            record = nq_record()
            record["condition"] = condition
            with self.assertRaises(ValueError):
                c1_record_to_input(record)

    def test_wrong_model_rejected(self):
        for model in ("Qwen/Qwen2.5-32B-Instruct", "meta-llama/llama-3.3-70b-instruct", "Qwen/Qwen2.5-3B-Instruct"):
            record = fever_record()
            record["model_name"] = model
            with self.assertRaises(ValueError):
                c1_record_to_input(record)

    def test_retrieved_ids_mismatch_rejected(self):
        record = nq_record()
        record["retrieved_top5_ids"] = list(reversed(record["retrieved_top5_ids"]))
        with self.assertRaises(ValueError):
            c1_record_to_input(record)
        record = nq_record()
        record["retrieved_top5_ids"] = record["retrieved_top5_ids"][:1]
        with self.assertRaises(ValueError):
            c1_record_to_input(record)

    def test_missing_required_field_rejected(self):
        for key in ("sample_id", "question_or_claim", "evidence", "prediction", "retrieved_top5_ids"):
            record = nq_record()
            record.pop(key)
            with self.assertRaises(ValueError):
                c1_record_to_input(record)


if __name__ == "__main__":
    unittest.main()
