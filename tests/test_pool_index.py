"""Tests for the stand-in candidate pool builder. No encoder is loaded."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_nq15_pool_index import collect_passages, write_jsonl


def record(sample_id: str, passages: list[tuple[str, str]]) -> dict:
    return {
        "sample_id": sample_id,
        "query": "q",
        "evidence": [
            {"rank": rank, "passage_id": passage_id, "text": text, "score": 1.0}
            for rank, (passage_id, text) in enumerate(passages, start=1)
        ],
    }


class CollectPassagesTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "records.jsonl"

    def write(self, records: list[dict]) -> Path:
        self.path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        return self.path

    def test_it_collects_every_passage(self):
        path = self.write([record("s1", [("p1", "one"), ("p2", "two")])])
        self.assertEqual(collect_passages(path), {"p1": "one", "p2": "two"})

    def test_passages_shared_between_samples_appear_once(self):
        path = self.write([record("s1", [("p1", "one")]), record("s2", [("p1", "one"), ("p3", "three")])])
        self.assertEqual(collect_passages(path), {"p1": "one", "p3": "three"})

    def test_the_first_text_wins_on_a_repeated_id(self):
        path = self.write([record("s1", [("p1", "first")]), record("s2", [("p1", "second")])])
        self.assertEqual(collect_passages(path)["p1"], "first")

    def test_ids_are_strings(self):
        path = self.write([record("s1", [(7, "seven")])])
        self.assertEqual(collect_passages(path), {"7": "seven"})

    def test_limit_stops_early(self):
        path = self.write([record("s1", [("p1", "one")]), record("s2", [("p2", "two")])])
        self.assertEqual(collect_passages(path, limit=1), {"p1": "one"})

    def test_blank_lines_are_skipped(self):
        self.path.write_text(json.dumps(record("s1", [("p1", "one")])) + "\n\n", encoding="utf-8")
        self.assertEqual(collect_passages(self.path), {"p1": "one"})

    def test_an_empty_file_gives_nothing(self):
        self.path.write_text("", encoding="utf-8")
        self.assertEqual(collect_passages(self.path), {})


class WriteJsonlTest(unittest.TestCase):
    def test_it_writes_one_object_per_line_and_creates_the_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "out.jsonl"
            write_jsonl(path, ({"passage_id": key} for key in ("p1", "p2")))
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line)["passage_id"] for line in lines], ["p1", "p2"])

    def test_row_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.jsonl"
            write_jsonl(path, ({"passage_id": key} for key in ("p9", "p1", "p5")))
            ids = [json.loads(line)["passage_id"] for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(ids, ["p9", "p1", "p5"])


if __name__ == "__main__":
    unittest.main()
