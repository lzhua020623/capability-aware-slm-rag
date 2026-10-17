"""Protect runtime supervision boundaries and reproducible NQ Train selection."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_nq_train_answer_repair import jsonl, partition, prepare, sha
from scripts.run_answer_repair import prepare_records
from src.compensation.answer_repair import MODEL_NAME, build_repair_prompt
from src.compensation.nq_train_adapter import c1_record, runtime_record, runtime_request
from src.compensation.short_answer_repair import build_short_answer_prompt
from src.diagnosis import DatasetName


def delivery_row(number=0):
    evidence = [{"rank": i, "passage_id": f"p{number}-{i}",
                 "text": f"Retrieved passage {i} for question {number}.", "score": 1 / i}
                for i in range(1, 6)]
    return {"sample_id": str(number), "dataset": "nq", "query": f"Question number {number}?",
            "initial_answer": "London" if number % 2 == 0 else "Paris", "evidence": evidence,
            "annotation_reference": {"gold_answer": "Paris", "gold_passage_id": evidence[0]["passage_id"],
                "gold_evidence_text": evidence[0]["text"], "gold_in_top1": True,
                "gold_in_top3": True, "gold_in_top5": True}, "label": None}


def source_stats(count=80):
    return {"generation_model": MODEL_NAME, "quantization": "nf4",
            "decoding": "greedy do_sample=False", "max_new_tokens": 64,
            "annotation_candidate_count": count}


def adapt(row, stats=None):
    return c1_record(row, stats or source_stats(), source_shard="fixture.jsonl",
                     source_sha256="fixture-hash", data_role="development")


def fixture(folder, rows=None):
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    source = root / "raw"
    source.mkdir()
    rows = rows if rows is not None else [delivery_row(n) for n in range(80)]
    raw = jsonl(rows)
    (source / "fixture.jsonl").write_bytes(raw)
    manifest = {"total_samples": len(rows), "shard_count": 1, "shards": [{
        "file": "fixture.jsonl", "n_samples": len(rows), "first_sample_id": rows[0]["sample_id"],
        "last_sample_id": rows[-1]["sample_id"], "sha256": sha(raw)}]}
    manifest_raw = json.dumps(manifest).encode()
    (root / "delivery.json").write_bytes(manifest_raw)
    (root / "stats.json").write_text(json.dumps(source_stats(len(rows))))
    audit = {"repository_manifest_sha256": sha(manifest_raw), "total_samples": len(rows),
             "all_shard_hashes_match": True, "schema_issue_counts": {},
             "shards": [{"file": "fixture.jsonl", "sha256": sha(raw)}],
             "overlap_checks": {"old": {"id_overlap_new_ids": ["0"], "query_overlap_new_ids": []}},
             "semantically_previewed_sample_ids": ["1"]}
    (root / "audit.json").write_text(json.dumps(audit))
    return {"source_dir": source, "delivery_manifest": root / "delivery.json",
            "stats_path": root / "stats.json", "audit_path": root / "audit.json",
            "manifest_out": root / "config.json", "out_dir": root / "packet",
            "full_out": root / "local/c1.jsonl", "screen_count": 4, "control_count": 1}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class TrainAdapterTests(unittest.TestCase):
    def test_preserves_evidence_and_is_accepted_by_existing_runner(self):
        row = delivery_row()
        adapted = adapt(row)
        request = prepare_records([adapted], DatasetName.NQ)[0][1]
        self.assertEqual(request, runtime_request(row))
        self.assertEqual(request.initial_answer, row["initial_answer"])
        self.assertEqual([e.text for e in request.evidence], [e["text"] for e in row["evidence"]])
        self.assertFalse(adapted["correct"])
        self.assertEqual(adapted["provenance"]["c1_model_revision"], None)

    def test_supervision_and_injected_extra_fields_never_enter_runtime_or_prompts(self):
        row = delivery_row()
        row["annotation_reference"]["gold_answer"] = "OFFLINE_SENTINEL"
        row.update(label="OFFLINE_SENTINEL", gold_answer="OFFLINE_SENTINEL", em="OFFLINE_SENTINEL")
        row["evidence"][0]["gold_label"] = "OFFLINE_SENTINEL"
        adapted = adapt(row)
        request = prepare_records([adapted], DatasetName.NQ)[0][1]
        self.assertEqual(set(runtime_record(row)),
                         {"sample_id", "dataset", "query", "evidence", "initial_answer"})
        for content in (json.dumps(runtime_record(row)), build_repair_prompt(request),
                        build_short_answer_prompt(request)):
            self.assertNotIn("OFFLINE_SENTINEL", content)
        row["annotation_reference"]["gold_answer"] = row["initial_answer"]
        self.assertTrue(adapt(row)["correct"])
        self.assertEqual(runtime_request(row), request)

    def test_rejects_incompatible_generation_and_corrupted_evidence_references(self):
        for key, value in (("generation_model", "other-model"), ("quantization", "fp16"),
                           ("decoding", "sampling"), ("max_new_tokens", 128)):
            stats = source_stats(); stats[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                adapt(delivery_row(), stats)
        bad = []
        row = delivery_row(); row["evidence"].reverse(); bad.append(row)
        row = delivery_row(); row["evidence"][1]["passage_id"] = row["evidence"][0]["passage_id"]; bad.append(row)
        row = delivery_row(); row["evidence"][0]["score"] = float("inf"); bad.append(row)
        row = delivery_row(); row["annotation_reference"]["gold_in_top5"] = False; bad.append(row)
        row = delivery_row(); row["annotation_reference"]["gold_evidence_text"] = "wrong passage"; bad.append(row)
        for row in bad:
            with self.subTest(row=row), self.assertRaises(ValueError):
                adapt(row)

    def test_partition_depends_on_question_only_and_respects_reservations(self):
        row = delivery_row()
        original = partition(row, 42, set(), set())
        changed = deepcopy(row)
        changed.update(sample_id="different-id", query="  QUESTION   number 0? ",
                       initial_answer="changed", annotation_reference={}, label="changed")
        self.assertEqual(partition(changed, 42, set(), set()), original)
        self.assertEqual(partition(row, 42, {"question number 0?"}, set()), "development")
        self.assertEqual(partition(row, 42, {"question number 0?"}, {"0"}), "excluded_by_team")


class TrainPreparationTests(unittest.TestCase):
    def test_full_pool_preserved_review_is_development_only_pending_and_repeatable(self):
        with tempfile.TemporaryDirectory() as folder:
            args = fixture(folder)
            reserved = Path(folder) / "reserved.json"
            reserved.write_text('["2"]')
            args["reserved_ids"] = reserved
            first = prepare(**args)
            frozen = {p: p.read_bytes() for p in Path(folder).rglob("*") if p.is_file()}
            self.assertEqual(first, prepare(**args))
            self.assertTrue(all(p.read_bytes() == raw for p, raw in frozen.items()))
            full = read_rows(args["full_out"])
            review = read_rows(args["out_dir"] / "review/review_queue.jsonl")
            pure = read_rows(args["out_dir"] / "inputs/nq_train_review_runtime.jsonl")
            manifest = json.loads(args["manifest_out"].read_bytes())
            self.assertEqual(len(full), 80)
            self.assertEqual(sum(first["partition_counts"].values()), 80)
            self.assertEqual(first["partition_counts"]["excluded_by_team"], 2)
            self.assertGreater(first["partition_counts"]["reserved_evaluation"], 0)
            self.assertEqual(len(review), 5)
            self.assertEqual([r["screening_group"] for r in review], ["screen_candidate"] * 4 + ["em_match_control"])
            for row in review:
                self.assertEqual(row["source"]["data_role"], "development")
                self.assertNotIn(row["sample_id"], {"0", "2"})
                self.assertIs(row["human_verified"], False)
                self.assertEqual(row["human_review"]["status"], "pending")
                self.assertIsNone(row["human_review"]["suggested_diagnosis_label"])
            self.assertTrue(all("annotation_reference" not in row for row in pure))
            self.assertFalse(manifest["human_labels_assigned"])
            self.assertEqual(next(r for r in full if r["sample_id"] == "1")["provenance"]["data_role"], "development")

    def test_review_selection_does_not_depend_on_source_row_order(self):
        with tempfile.TemporaryDirectory() as folder:
            a = fixture(Path(folder) / "a")
            b = fixture(Path(folder) / "b", list(reversed([delivery_row(n) for n in range(80)])))
            prepare(**a); prepare(**b)
            manifests = [json.loads(args["manifest_out"].read_bytes()) for args in (a, b)]
            self.assertEqual(manifests[0]["sample_ids"], manifests[1]["sample_ids"])

    def test_rejects_changed_shard_without_publishing(self):
        with tempfile.TemporaryDirectory() as folder:
            args = fixture(folder)
            with (args["source_dir"] / "fixture.jsonl").open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                prepare(**args)
            self.assertFalse(args["full_out"].exists())
            self.assertFalse(args["manifest_out"].exists())

    def test_duplicate_ids_and_insufficient_candidates_fail_without_partial_output(self):
        for duplicate in (True, False):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as folder:
                rows = [delivery_row(n) for n in range(80)]
                if duplicate:
                    rows[-1]["sample_id"] = rows[0]["sample_id"]
                args = fixture(folder, rows)
                if not duplicate:
                    args["screen_count"] = 81
                with self.assertRaises(ValueError):
                    prepare(**args)
                self.assertFalse(args["full_out"].exists())
                self.assertFalse(args["manifest_out"].exists())
                self.assertEqual(list(args["full_out"].parent.iterdir()), [])

    def test_changed_frozen_output_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            args = fixture(folder)
            prepare(**args)
            original_full = args["full_out"].read_bytes()
            path = args["out_dir"] / "review/review_queue.jsonl"
            path.write_bytes(b"manual edits must be saved separately\n")
            with self.assertRaisesRegex(ValueError, "Frozen artifact differs"):
                prepare(**args)
            self.assertEqual(path.read_bytes(), b"manual edits must be saved separately\n")
            self.assertEqual(args["full_out"].read_bytes(), original_full)

    def test_output_aliases_cannot_overwrite_sources(self):
        for name in ("source", "metadata", "alias"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                args = fixture(folder)
                if name == "source":
                    args["full_out"] = args["source_dir"] / "new.jsonl"
                elif name == "metadata":
                    args["full_out"] = args["audit_path"]
                else:
                    args["manifest_out"] = args["full_out"]
                with self.assertRaisesRegex(ValueError, "Output paths"):
                    prepare(**args)
                self.assertFalse(args["out_dir"].exists())


if __name__ == "__main__":
    unittest.main()
