"""Small checksum journal fixtures for the model-free consistency CLI."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.check_nq_p320_saved_scores import check_saved_scores
from src.compensation.retrieval_enhancement.p3 import RERANKER_REVISION
from tests.test_retrieval_enhancement_p3 import candidates


class SavedScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scores = self.root / "scores.jsonl"
        self.reference = self.root / "reference.jsonl"
        self.out = self.root / "new_output"
        self.row = {"sample_id": "9007199254740993123", "dataset": "nq",
                    "reranker_revision": RERANKER_REVISION,
                    "candidates": [{"rank": i, "passage_id": c["passage_id"], "text": c["text"],
                                    "retrieval_score": c["score"], "cross_encoder_score": i}
                                   for i, c in enumerate(candidates(), 1)]}
        self.ref = {"sample_id": self.row["sample_id"], "policy": "P3", "depth": 20,
                    "selected_ids": ["p1", "p20", "p2", "p19", "p3"]}
        self.write()

    def write(self):
        payload = json.dumps(self.row, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()
        self.scores.write_text(json.dumps({**self.row, "_record_sha256": hashlib.sha256(payload).hexdigest()}) + "\n")
        self.reference.write_text(json.dumps(self.ref) + "\n")

    def check(self):
        return check_saved_scores(self.scores, self.reference, self.out, expected_count=1)

    def test_matching_saved_predictions_and_no_overwrite(self):
        report = self.check()
        self.assertTrue(report["passed"])
        self.assertEqual(report["model_calls"], 0)
        self.assertEqual(report["mismatch_count"], 0)
        with self.assertRaises(FileExistsError):
            self.check()

    def test_selection_mismatch_is_failure_with_audit(self):
        self.ref["selected_ids"] = ["wrong"]
        self.write()
        with self.assertRaisesRegex(ValueError, "consistency failed"):
            self.check()
        report = json.loads((self.out / "report.json").read_text())
        self.assertEqual(report["mismatch_count"], 1)
        self.assertFalse(report["passed"])

    def test_bad_checksum_is_rejected(self):
        self.scores.write_text(self.scores.read_text().replace('"p1"', '"corrupted"'))
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.check()

    def test_missing_or_duplicate_samples_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "consistency failed"):
            check_saved_scores(self.scores, self.reference, self.out, expected_count=2)
        self.out = self.root / "second_output"
        self.scores.write_text(self.scores.read_text() * 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check()

    def test_wrong_model_revision_is_rejected(self):
        self.row["reranker_revision"] = "main"
        self.write()
        with self.assertRaisesRegex(ValueError, "revision"):
            self.check()


if __name__ == "__main__":
    unittest.main()
