"""Protect held-out IDs and frozen artifacts in portable E preparation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_answer_repair_experiment import select_c1, select_development_ids, write_frozen
from tests.test_answer_repair import c1_row


def split(train, validation=(), test=()):
    return {"splits": {"nq": {"train": list(train), "validation": list(validation), "test": list(test)}}}


class ExperimentPreparationTests(unittest.TestCase):
    def test_reserves_ids_moved_to_new_validation_or_test(self):
        v1 = split(["a", "b", "c", "d"], ["v"], ["t"])
        v2 = split(["a", "d", "new"], ["b", "v"], ["c", "t"])
        self.assertEqual(set(select_development_ids([v1, v2], "nq", 2)), {"a", "d"})
        with self.assertRaisesRegex(ValueError, "Only 2"):
            select_development_ids([v1, v2], "nq", 3)

    def test_order_is_stable_and_smoke_is_a_prefix(self):
        ids = [str(i) for i in range(30)]
        full = select_development_ids([split(ids)], "nq", 20)
        reversed_input = select_development_ids([split(list(reversed(ids)))], "nq", 20)
        self.assertEqual(full, reversed_input)
        self.assertEqual(full[:10], select_development_ids([split(ids)], "nq", 10))

    def test_overlapping_and_duplicate_split_ids_are_rejected(self):
        for document in (split(["a", "a"]), split(["a"], ["a"]), split(["a"], ["b"], ["b"])):
            with self.subTest(document=document), self.assertRaises(ValueError):
                select_development_ids([document], "nq", 1)

    def test_selection_does_not_read_correctness_or_annotations(self):
        document = split([str(i) for i in range(20)])
        before = select_development_ids([document], "nq", 10)
        document["correctness"] = {str(i): i % 2 == 0 for i in range(20)}
        document["diagnosis_labels"] = {str(i): "evidence_utilisation" for i in range(20)}
        self.assertEqual(before, select_development_ids([document], "nq", 10))

    def test_c1_order_and_evidence_preserved_missing_ids_rejected(self):
        rows = [c1_row("1", "Paris"), c1_row("2", "London")]
        raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
        selected, stats = select_c1(raw, "nq", ["2", "1"])
        self.assertEqual([json.loads(line) for line in selected.splitlines()], list(reversed(rows)))
        self.assertEqual(stats, {"samples": 2, "initial_correct": 1, "initial_wrong": 1})
        with self.assertRaisesRegex(ValueError, "missing"):
            select_c1(raw, "nq", ["not-present"])

    def test_existing_manifest_cannot_be_silently_changed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            write_frozen(path, b"original")
            write_frozen(path, b"original")
            with self.assertRaisesRegex(ValueError, "Frozen artifact differs"):
                write_frozen(path, b"changed")
            self.assertEqual(path.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
