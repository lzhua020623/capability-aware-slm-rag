"""Review presentation must not turn assistance into human labels or invent evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_answer_repair_historical_review import load, render


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class ReviewIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        result = {"citations": [{"rank": 1, "passage_id": "p", "quote": "A named fact."}]}
        digest = sha(json.dumps(result, ensure_ascii=False, sort_keys=True).encode())
        self.row = {"review_number": 1, "sample_id": "s", "dataset": "nq",
                    "review_set_id": "test", "strategy": "conservative_repair", "cohort": "nq75",
                    "query": "Which fact?", "initial_answer": "right", "final_answer": "wrong",
                    "result_sha256": digest, "model_result": result,
                    "evidence": [{"passage_id": "p", "text": "A named fact."}],
                    "human_verified": False, "frozen_reference": ["right"],
                    "initial_human_review": {"initial_correct": True},
                    "pair_review": {"status": "pending", "initial_correct": None, "final_correct": None,
                                    "initial_supported": None, "final_supported": None}}
        raw = (json.dumps(self.row) + "\n").encode()
        (self.directory / "pending_pairs.jsonl").write_bytes(raw)
        (self.directory / "manifest.json").write_text(json.dumps({"queue_sha256": sha(raw), "selected_pairs": 1}))
        self.note = {"review_number": 1, "sample_id": "s", "dataset": "nq", "review_set_id": "test",
                     "source_queue_sha256": sha(raw), "source_result_sha256": digest, "human_verified": False,
                     "agent_review": {"status": "assisted_pending_human", "initial_correct": True,
                                      "final_correct": False, "initial_supported": True, "final_supported": False,
                                      "outcome": "suggested_regression", "reference_visible": True,
                                      "correctness_basis": "frozen_reference_and_evidence", "reason_zh": "Explicit opinion",
                                      "evidence_anchors": [{"rank": 1, "passage_id": "p", "quote": "named fact",
                                                            "start": 2, "end": 12,
                                                            "evidence_sha256": sha(b"A named fact.")}]}}

    def save(self, note):
        (self.directory / "assisted_partial.jsonl").write_text(json.dumps(note) + "\n")

    def test_valid_assistance_leaves_human_fields_pending(self):
        self.save(self.note)
        rows, _ = load(self.directory)
        self.assertEqual(rows[0]["pair_review"], self.row["pair_review"])
        self.assertFalse(rows[0]["human_verified"])

    def test_invented_evidence_anchor_rejected(self):
        note = copy.deepcopy(self.note)
        note["agent_review"]["evidence_anchors"][0]["quote"] = "imagined fact"
        self.save(note)
        with self.assertRaisesRegex(ValueError, "Evidence anchor mismatch"):
            load(self.directory)

    def test_agent_cannot_mark_human_confirmation(self):
        note = copy.deepcopy(self.note); note["human_verified"] = True
        self.save(note)
        with self.assertRaisesRegex(ValueError, "human status mismatch"):
            load(self.directory)

    def test_confirmed_initial_correctness_cannot_be_overridden(self):
        note = copy.deepcopy(self.note); note["agent_review"]["initial_correct"] = False
        self.save(note)
        with self.assertRaisesRegex(ValueError, "cannot override confirmed"):
            load(self.directory)

    def test_untrusted_text_is_escaped_and_final_arm_identified(self):
        self.save(self.note); rows, _ = load(self.directory)
        rows[0]["query"] = "</h2><script>unexpected()</script>"
        page = render(rows, {}, prefix="F", title="Final review", scope="Final scope")
        self.assertIn('id="F1"', page)
        self.assertIn("最终 r3 改答", page)
        self.assertIn("人工确认 0/1", page)
        self.assertNotIn("<script>unexpected()</script>", page)
        self.assertIn("&lt;script&gt;unexpected()&lt;/script&gt;", page)


if __name__ == "__main__":
    unittest.main()
