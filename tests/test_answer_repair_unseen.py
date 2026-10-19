"""Protect blind deterministic selection, exposure exclusions and final-source binding."""
from __future__ import annotations

from collections import Counter
import gzip
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.prepare_answer_repair_experiment import encode_json
from scripts.prepare_answer_repair_final_dev import SELECTED_SHA
from scripts.prepare_answer_repair_unseen import COUNTS, frozen_selection, prepare
from scripts.prepare_nq_train_answer_repair import jsonl, partition, sha
from tests.test_nq_train_answer_repair import delivery_row, source_stats


def fixture(folder):
    root = Path(folder); source = root / "raw"; source.mkdir()
    rows = [delivery_row(n) for n in range(3200)]
    # Distinct IDs of the same question must also be excluded when one was exposed.
    calibration = [r for r in rows if partition(r, 42, set(), set()) == "calibration"]
    calibration[1]["query"] = calibration[0]["query"]
    previewed = {"question number 1?"}
    roles = Counter(partition(r, 42, previewed, {"0"}) for r in rows)
    for k in ("development", "calibration", "reserved_evaluation", "excluded_by_team"):
        roles.setdefault(k, 0)
    development = [r["sample_id"] for r in rows if partition(r, 42, previewed, {"0"}) == "development"]
    raw = jsonl(rows); (source / "fixture.jsonl").write_bytes(raw)
    stats = encode_json(source_stats(len(rows))); (root / "stats.json").write_bytes(stats)
    split = {"version": "answer_repair_nq_train_dev_v1", "seed": 42, "source_total": len(rows),
             "sources_sha256": {"source_stats": sha(stats)},
             "source_shards": [{"file": "fixture.jsonl", "sha256": sha(raw)}],
             "partition_counts": dict(roles), "sample_ids": development[:100],
             "partition_policy": {"previously_previewed_ids_forced_to_development": ["1"],
                                  "excluded_old_overlap_and_team_ids": ["0"]}}
    (root / "split.json").write_bytes(encode_json(split))
    files = {"src/fixture.py": b"# Synthetic provenance fixture; no inference\n"}
    manifest = {"version": "answer_repair_v8r3_nq75_final_v1", "selected_final_revision": "r3",
                "samples": 75, "sample_ids": development[:75],
                "final_inference_bundle_sha256": SELECTED_SHA["r3"], "frozen_final": True,
                "tuning_allowed_after_run": False,
                "runtime_source_sha256": {p: sha(raw) for p, raw in files.items()}}
    files["manifest.json"] = encode_json(manifest)
    files["bundle_checksums.json"] = encode_json({p: sha(raw) for p, raw in files.items()})
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for name, raw in files.items():
                info = tarfile.TarInfo("synthetic/" + name); info.size = len(raw)
                tar.addfile(info, io.BytesIO(raw))
    archive = root / "synthetic.tar.gz"; archive.write_bytes(buf.getvalue())
    receipt = {"version": manifest["version"], "selected_final_revision": "r3",
               "frozen_inference_source_unchanged": True, "predictions_expected": 300,
               "bundle_sha256": sha(buf.getvalue())}
    (root / "receipt.json").write_bytes(encode_json(receipt))
    return {"source_dir": source, "split_path": root / "split.json", "stats_path": root / "stats.json",
            "final_receipt": root / "receipt.json", "final_archive": archive, "out_dir": root / "challenge"}, rows


class UnseenPreparationTests(unittest.TestCase):
    def test_reproducible_125_calibration_only_and_runtime_allowlist(self):
        with tempfile.TemporaryDirectory() as folder:
            args, rows = fixture(folder)
            receipt = prepare(**args)
            snapshot = {p: p.read_bytes() for p in args["out_dir"].rglob("*") if p.is_file()}
            self.assertEqual(receipt, prepare(**args))
            self.assertTrue(all(p.read_bytes() == raw for p, raw in snapshot.items()))
            manifest = json.loads((args["out_dir"] / "manifest.json").read_bytes())
            strata = [json.loads(line) for line in (args["out_dir"] / "offline/strata.jsonl").read_bytes().splitlines()]
            runtime = [json.loads(line) for line in (args["out_dir"] / "inputs/nq125_runtime.jsonl").read_bytes().splitlines()]
            self.assertEqual(Counter(r["screening_group"] for r in strata), COUNTS)
            by_id = {r["sample_id"]: r for r in rows}
            for r in runtime:
                self.assertEqual(set(r), {"sample_id", "dataset", "query", "evidence", "initial_answer"})
                self.assertEqual(partition(by_id[r["sample_id"]], 42, {"question number 1?"}, {"0"}), "calibration")
            self.assertTrue(all(r["diagnosis_label"] is None and not r["human_verified"] for r in strata))
            self.assertFalse(manifest["post_result_tuning_allowed"])
            self.assertEqual(manifest["team_registration"].split(';')[0], "pending")
            self.assertNotIn("sample_ids", receipt)

    def test_same_group_hash_order_independent_of_source_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / "a").mkdir(); (root / "b").mkdir()
            a, _ = fixture(root / "a"); b, rows = fixture(root / "b")
            raw = jsonl(list(reversed(rows)))
            (b["source_dir"] / "fixture.jsonl").write_bytes(raw)
            split = json.loads(b["split_path"].read_bytes()); split["source_shards"][0]["sha256"] = sha(raw)
            b["split_path"].write_bytes(encode_json(split))
            prepare(**a); prepare(**b)
            ids = [json.loads((x["out_dir"] / "manifest.json").read_bytes())["sample_ids"] for x in (a, b)]
            self.assertEqual(*ids)

    def test_exposed_id_excludes_its_entire_normalized_query_group(self):
        with tempfile.TemporaryDirectory() as folder:
            args, rows = fixture(folder)
            cal = [r for r in rows if partition(r, 42, set(), set()) == "calibration"]
            exposed = Path(folder) / "exposed.json"; exposed.write_text(json.dumps([cal[0]["sample_id"]]))
            prepare(**args, exposed_ids_path=exposed)
            manifest = json.loads((args["out_dir"] / "manifest.json").read_bytes())
            self.assertEqual(manifest["excluded_exposed_calibration_rows"], 2)
            ids = set(manifest["sample_ids"])
            self.assertTrue(ids.isdisjoint({cal[0]["sample_id"], cal[1]["sample_id"]}))

    def test_source_tamper_and_missing_final_selection_publish_nothing(self):
        for reason in ("source", "selection"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as folder:
                args, _ = fixture(folder)
                if reason == "source":
                    with (args["source_dir"] / "fixture.jsonl").open("ab") as f:
                        f.write(b"\n")
                else:
                    args["final_receipt"].write_text("{}")
                with self.assertRaises(ValueError):
                    prepare(**args)
                self.assertFalse(args["out_dir"].exists())

    def test_selected_source_archive_hash_and_no_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _ = fixture(folder)
            selected = frozen_selection(args["final_receipt"], args["final_archive"])
            self.assertEqual(selected["final_inference_bundle_sha256"], SELECTED_SHA["r3"])
            prepare(**args)
            path = args["out_dir"] / "inputs/nq125_runtime.jsonl"; path.write_text("preserve this\n")
            with self.assertRaisesRegex(ValueError, "do not resample"):
                prepare(**args)
            self.assertEqual(path.read_text(), "preserve this\n")
            args["final_archive"].write_bytes(b"changed archive")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                frozen_selection(args["final_receipt"], args["final_archive"])


if __name__ == "__main__":
    unittest.main()
