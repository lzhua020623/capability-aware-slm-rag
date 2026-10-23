"""CPU-only validation tests for the frozen full-index FEVER C1 runner."""

from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import run_fever_c1_preliminary as runner


class FeverC1PreliminaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = copy.deepcopy(runner.load_preliminary_config())
        manifest_source = runner.ROOT / self.config["samples"]["fever"]["manifest"]
        manifest_target = self.root / self.config["samples"]["fever"]["manifest"]
        manifest_target.parent.mkdir(parents=True)
        manifest_target.write_text(manifest_source.read_text(), encoding="utf-8")
        self.manifest = json.loads(manifest_target.read_text())
        source = self.root / self.config["samples"]["fever"]["source"]
        source.parent.mkdir(parents=True, exist_ok=True)
        self.source_records = [
            {
                "id": item["id"],
                "claim": f"Synthetic claim {item['id']}",
                "label": item["label"],
                "gold_page_ids": [f"Page_{item['id']}"],
                "gold_evidence_text": [f"Gold evidence {item['id']}"],
            }
            for item in self.manifest["samples"]
        ]
        source.write_text(
            "".join(json.dumps(record) + "\n" for record in self.source_records),
            encoding="utf-8",
        )
        self.addCleanup(patch.stopall)
        patch.object(runner, "ROOT", self.root).start()
        self.samples = runner.load_frozen_samples(self.config)

    def invoke(self, *args):
        with (
            patch.object(sys, "argv", ["runner", *args]),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            return runner.main()

    def test_frozen_manifest_order_and_balance(self):
        self.assertEqual(len(self.samples), 500)
        self.assertEqual(
            [str(sample["id"]) for sample in self.samples],
            [str(item["id"]) for item in self.manifest["samples"]],
        )
        self.assertEqual(sum(s["label"] == "SUPPORTS" for s in self.samples), 250)
        self.assertEqual(sum(s["label"] == "REFUTES" for s in self.samples), 250)

    def test_conditions_c1_is_explicitly_required(self):
        with self.assertRaises(SystemExit) as raised:
            self.invoke("--check-only")
        self.assertEqual(raised.exception.code, 2)

    def test_resolve_index_positions_and_full_corpus_text(self):
        ids = self.root / "ids.jsonl"
        corpus = self.root / "corpus.jsonl"
        ids.write_text(
            "".join(json.dumps({"passage_id": f"p{i}"}) + "\n" for i in range(6))
        )
        corpus.write_text(
            "".join(
                json.dumps({"passage_id": f"p{i}", "text": f"Full page {i}"}) + "\n"
                for i in range(6)
            )
        )
        passage_ids = runner.resolve_passage_ids(ids, [[4, 1, 5, 0, 2]], 6)
        self.assertEqual(passage_ids, [["p4", "p1", "p5", "p0", "p2"]])
        texts = runner.resolve_passage_texts(corpus, passage_ids, 6)
        self.assertEqual(texts["p4"], "Full page 4")

    def test_checkpoint_rejects_duplicates_wrong_fingerprint_and_non_top5(self):
        path = self.root / "checkpoint.jsonl"
        records = []
        for sample in self.samples:
            records.append(
                {
                    "sample_id": sample["id"],
                    "retrieval_fingerprint": "fp",
                    "retrieved_top5": [
                        {"passage_id": f"p{i}", "text": f"text {i}", "score": 1.0 - i / 10}
                        for i in range(5)
                    ],
                }
            )
        runner.atomic_write_jsonl(path, records)
        ordered = runner.validate_checkpoint(path, self.samples, "fp")
        self.assertEqual(len(ordered), 500)
        for change in ("duplicate", "fingerprint", "top5"):
            changed = copy.deepcopy(records)
            if change == "duplicate":
                changed[1]["sample_id"] = changed[0]["sample_id"]
            elif change == "fingerprint":
                changed[0]["retrieval_fingerprint"] = "wrong"
            else:
                changed[0]["retrieved_top5"].pop()
            runner.atomic_write_jsonl(path, changed)
            with self.assertRaises(RuntimeError):
                runner.validate_checkpoint(path, self.samples, "fp")

    def test_result_resume_rejects_duplicates_and_metric_changes(self):
        retrieval = []
        for sample in self.samples:
            retrieval.append(
                {
                    "sample_id": sample["id"],
                    "retrieved_top5": [
                        {"passage_id": f"p{i}", "text": f"text {i}", "score": 1.0 - i / 10}
                        for i in range(5)
                    ],
                }
            )
        sample = self.samples[0]
        hits = retrieval[0]["retrieved_top5"]
        result = {
            "sample_id": sample["id"],
            "dataset": "fever",
            "condition": "C1",
            "model_name": self.config["primary_model"]["model_name"],
            "experiment_fingerprint": "result-fp",
            "question_or_claim": sample["claim"],
            "gold_answer_or_label": sample["label"],
            "evidence": hits,
            "retrieved_top5_ids": [hit["passage_id"] for hit in hits],
            "gold_evidence_in_top5": False,
            "raw_output": sample["label"],
            "prediction": sample["label"],
            "correct": True,
            "latency": 0.1,
        }
        path = self.root / "result.jsonl"
        runner.atomic_write_jsonl(path, [result])
        self.assertEqual(
            runner.completed_ids(path, self.samples, retrieval, self.config, "result-fp"),
            {str(sample["id"])},
        )
        runner.atomic_write_jsonl(path, [result, result])
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            runner.completed_ids(path, self.samples, retrieval, self.config, "result-fp")
        changed = dict(result, correct=False)
        runner.atomic_write_jsonl(path, [changed])
        with self.assertRaisesRegex(RuntimeError, "invalid prediction"):
            runner.completed_ids(path, self.samples, retrieval, self.config, "result-fp")


if __name__ == "__main__":
    unittest.main()
