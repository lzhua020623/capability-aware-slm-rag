"""Review UI must preserve source text and exclude exactly the confirmed scope."""
from copy import deepcopy
import json
from pathlib import Path
import re
import unittest

from scripts.build_nq_train_pending_review import build_packet, render

ROOT = Path(__file__).resolve().parents[1]


class PendingReviewPageTests(unittest.TestCase):
    def setUp(self):
        paths = ["artifacts/answer_repair/nq_train_dev_v1/review/review_queue.jsonl",
                 "data/annotations/assisted/answer_repair_nq_train_dev_v1_20261017.jsonl",
                 "data/annotations/human/answer_repair_nq_train_dev_v1_batch01_20261017.jsonl",
                 "data/annotations/human/answer_repair_nq_train_dev_v1_batch01_20261017.summary.json"]
        self.raw = [(ROOT / path).read_bytes() for path in paths]

    def test_pending_scope_and_all_evidence_preserved_without_human_prefill(self):
        packet = build_packet(*self.raw)
        original = {r["review_number"]: r for r in map(json.loads, self.raw[0].splitlines())}
        self.assertEqual(packet["counts"], {"pending": 92, "disputed": 23, "suggested_correct": 69})
        confirmed = {6, 29, 31, 48, 49, 74, 81, 96}
        self.assertEqual({r["review_number"] for r in packet["records"]}, set(range(1, 101)) - confirmed)
        self.assertTrue(all(r["agent_assessment"]["suggested_diagnosis_label"] is None for r in packet["records"][:23]))
        for record in packet["records"]:
            source = original[record["review_number"]]
            self.assertEqual(record["evidence"], source["evidence"])
            self.assertEqual(record["offline_reference"], source["annotation_reference"])
            self.assertNotIn("human_review", record)
        self.assertEqual(sum(len(r["evidence"]) for r in packet["records"]), 460)

    def test_modified_source_or_confirmation_cannot_silently_change_pending_scope(self):
        changed = deepcopy(self.raw)
        queue = [json.loads(line) for line in changed[0].splitlines()]
        queue[0]["query"] = "changed question"
        changed[0] = b"\n".join(json.dumps(r).encode() for r in queue)
        with self.assertRaises(ValueError):
            build_packet(*changed)
        changed = deepcopy(self.raw)
        summary = json.loads(changed[3]); summary["pending_review_numbers"].remove(3)
        changed[3] = json.dumps(summary).encode()
        with self.assertRaises(ValueError):
            build_packet(*changed)

    def test_html_data_cannot_break_script_or_substitute_template_placeholders(self):
        packet = build_packet(*self.raw)
        hostile = '</script><script>alert("data")</script> __REVIEW_SCRIPT__'
        packet["records"][0]["query"] = hostile
        html = render(packet)
        self.assertNotIn(hostile, html)
        embedded = re.search(r'<script id="packet" type="application/json">(.*?)</script>', html, re.S)
        self.assertIsNotNone(embedded)
        self.assertEqual(json.loads(embedded[1])["records"][0]["query"], hostile)
        self.assertEqual(html.count('<script>'), 1)


if __name__ == '__main__':
    unittest.main()
