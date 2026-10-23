"""Exercise recovery against real small FAISS indexes and corrupted inputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from scripts.recover_nq_train_index_sidecars import NAMES, encode_bge_in_child, recover, sample_positions


@unittest.skipIf(faiss is None, "FAISS CPU required for binary-index recovery checks")
class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus = self.root / "语料.jsonl"
        self.index_path = self.root / "source.index"
        self.output = self.root / "sidecars"
        self.records = [
            {"passage_id": "p_三", "text": "Café and 汉字"},
            {"passage_id": "p_one", "text": "first passage"},
            {"passage_id": "p_emoji", "text": "🌏 evidence"},
            {"passage_id": "p_zero", "text": "last passage"},
        ]
        self.vectors = np.eye(4, dtype=np.float32)
        self.write_corpus(self.records)
        self.write_index(self.vectors)

    def write_corpus(self, records):
        # Mixed CRLF/LF and whitespace lines test real UTF-8 byte positions.
        raw = b"\r\n" + b"\n \r\n".join(
            json.dumps(r, ensure_ascii=False).encode() for r in records) + b"\r\n"
        self.corpus.write_bytes(raw)

    def write_index(self, vectors, *, metric="ip"):
        index = faiss.IndexFlatIP(vectors.shape[1]) if metric == "ip" else faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(self.index_path))

    def run_recovery(self, **kwargs):
        options = {"index_read_mode": "heap", **kwargs}
        return recover(self.corpus, self.index_path, self.output,
                       expected_count=4, dimension=4, sample_count=4, **options)

    def encoder(self, texts):
        mapping = {r["text"]: v for r, v in zip(self.records, self.vectors)}
        return np.stack([mapping[t] for t in texts])

    def assert_not_published(self):
        self.assertTrue(all(not (self.output / n).exists() for n in NAMES))
        if self.output.exists():
            self.assertEqual(list(self.output.iterdir()), [])

    def test_real_index_recovery_preserves_order_unicode_and_text_mode_seeking(self):
        source_bytes = self.corpus.read_bytes()
        index_hash = hashlib.sha256(self.index_path.read_bytes()).hexdigest()
        report = self.run_recovery(encoder=self.encoder, encoder_provenance={"revision": "test-only"})
        ids = [json.loads(line)["passage_id"] for line in (self.output / NAMES[0]).read_text().splitlines()]
        self.assertEqual(ids, [r["passage_id"] for r in self.records])
        offsets = np.load(self.output / NAMES[1], allow_pickle=False)
        self.assertEqual(offsets.dtype, np.dtype("int64"))
        with self.corpus.open("r", encoding="utf-8") as handle:
            recovered = []
            for offset in offsets:
                handle.seek(int(offset))
                recovered.append(json.loads(handle.readline()))
        self.assertEqual(recovered, self.records)
        self.assertEqual(self.corpus.read_bytes(), source_bytes)
        self.assertEqual(hashlib.sha256(self.index_path.read_bytes()).hexdigest(), index_hash)
        self.assertTrue(report["embedding_alignment"]["passed"])
        self.assertFalse(report["original_build_metadata_recovered"])
        self.assertIsNone(report["original_model_revision"])
        self.assertFalse(report["full_T2_ready"])
        self.assertEqual(report["index_fingerprint"]["sha256"], index_hash)

    def test_same_count_permuted_index_is_rejected_by_independent_embedding_check(self):
        self.write_index(self.vectors[[1, 0, 2, 3]])
        with self.assertRaisesRegex(ValueError, "disagree with index row order"):
            self.run_recovery(encoder=self.encoder)
        self.assert_not_published()

    def test_structural_only_recovery_does_not_claim_embedding_alignment(self):
        report = self.run_recovery()
        self.assertFalse(report["embedding_alignment"]["performed"])
        self.assertNotIn("passed", report["embedding_alignment"])

    @unittest.skipUnless(faiss is not None and hasattr(faiss, "IO_FLAG_MMAP_IFC"), "Flat mmap not supported")
    def test_read_only_mmap_recovers_without_changing_original_index(self):
        before = self.index_path.read_bytes()
        report = self.run_recovery(index_read_mode="mmap", encoder=self.encoder)
        self.assertEqual(report["index_access"]["mode"], "mmap")
        self.assertTrue(report["embedding_alignment"]["passed"])
        self.assertEqual(self.index_path.read_bytes(), before)

    def test_existing_sidecar_is_preserved(self):
        self.output.mkdir()
        existing = self.output / NAMES[1]
        existing.write_bytes(b"user-owned")
        with self.assertRaises(FileExistsError):
            self.run_recovery()
        self.assertEqual(existing.read_bytes(), b"user-owned")
        self.assertFalse((self.output / NAMES[0]).exists())

    def test_duplicate_ids_do_not_publish_partial_outputs(self):
        records = [dict(r) for r in self.records]
        records[2]["passage_id"] = records[0]["passage_id"]
        self.write_corpus(records)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_recovery()
        self.assert_not_published()

    def test_missing_text_invalid_json_and_nonobject_are_rejected(self):
        for invalid in [b'{"passage_id":"p"}\n', b'{broken\n', b'[]\n']:
            with self.subTest(invalid=invalid):
                self.corpus.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    self.run_recovery()
                self.assert_not_published()

    def test_too_few_or_too_many_corpus_rows_are_rejected(self):
        for records in [self.records[:-1], self.records + [{"passage_id": "extra", "text": "extra"}]]:
            with self.subTest(count=len(records)):
                self.write_corpus(records)
                with self.assertRaisesRegex(ValueError, "Corpus contains"):
                    self.run_recovery()
                self.assert_not_published()

    def test_wrong_index_dimension_count_or_metric_is_rejected(self):
        for vectors, metric in [(np.eye(3, dtype=np.float32), "ip"),
                                (self.vectors[:3], "ip"), (self.vectors, "l2")]:
            with self.subTest(shape=vectors.shape, metric=metric):
                self.write_index(vectors, metric=metric)
                with self.assertRaisesRegex(ValueError, "Index type"):
                    self.run_recovery()
                self.assert_not_published()

    def test_nonfinite_and_unnormalized_index_vectors_are_rejected(self):
        for bad_value in [float("nan"), 2.0]:
            with self.subTest(value=bad_value):
                vectors = self.vectors.copy()
                vectors[0, 0] = bad_value
                self.write_index(vectors)
                with self.assertRaises(ValueError):
                    self.run_recovery()
                self.assert_not_published()

    def test_invalid_encoder_output_is_rejected(self):
        for vectors in [np.ones((3, 4)), np.full((4, 4), float("nan")), np.ones((4, 4))]:
            with self.subTest(shape=vectors.shape):
                with self.assertRaises(ValueError):
                    self.run_recovery(encoder=lambda texts: vectors)
                self.assert_not_published()

    def test_input_mutation_during_verification_is_rejected(self):
        def mutating_encoder(texts):
            with self.corpus.open("ab") as handle:
                handle.write(b"\n")
            return self.encoder(texts)
        with self.assertRaisesRegex(ValueError, "Input changed"):
            self.run_recovery(encoder=mutating_encoder)
        self.assert_not_published()

    def test_failed_encoding_child_does_not_publish_structural_only_outputs(self):
        with mock.patch("scripts.recover_nq_train_index_sidecars.subprocess.run",
                        side_effect=subprocess.CalledProcessError(139, ["verification-child"])):
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_recovery(encoder=lambda texts: encode_bge_in_child(texts, {}))
        self.assert_not_published()

    def test_publication_race_rolls_back_our_file_and_preserves_other_writer(self):
        real_link = os.link
        def competing_link(source, destination):
            if Path(destination).name == NAMES[1]:
                Path(destination).write_bytes(b"other-writer")
                raise FileExistsError("created concurrently")
            return real_link(source, destination)
        with mock.patch("scripts.recover_nq_train_index_sidecars.os.link", side_effect=competing_link):
            with self.assertRaises(FileExistsError):
                self.run_recovery()
        self.assertFalse((self.output / NAMES[0]).exists())
        self.assertEqual((self.output / NAMES[1]).read_bytes(), b"other-writer")
        self.assertFalse((self.output / NAMES[2]).exists())

    def test_missing_index_and_corrupt_index_do_not_publish_sidecars(self):
        self.index_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_recovery()
        self.assert_not_published()
        self.index_path.write_bytes(b"not a FAISS file")
        with self.assertRaises(RuntimeError):
            self.run_recovery()
        self.assert_not_published()


class SamplingTests(unittest.TestCase):
    def test_deterministic_bounded_sampling_covers_boundaries(self):
        sample = sample_positions(1_224_119, 64)
        self.assertEqual(sample, sample_positions(1_224_119, 64))
        self.assertEqual(len(sample), 64)
        self.assertEqual((sample[0], sample[-1]), (0, 1_224_118))
        self.assertEqual(sample_positions(2, 64), [0, 1])
        with self.assertRaises(ValueError):
            sample_positions(4, 0)


if __name__ == "__main__":
    unittest.main()
