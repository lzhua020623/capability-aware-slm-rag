"""Catch mapping corruption outside the earlier vector sample and lazy read leaks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.recover_nq_train_index_sidecars import sample_positions
from src.retrieval.offset_corpus import OffsetCorpus, file_sha, validate_complete_mapping


class OffsetCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus, self.ids, self.offsets, self.meta, self.index = [
            self.root / n for n in ('corpus.jsonl', 'ids.jsonl', 'offsets.npy', 'meta.json', 'source.index')]
        self.records = [{'passage_id': f'p{i}', 'text': f'Café 汉字 🌏 passage {i}',
                         'gold_answer': 'must never leave lookup', 'label': 'hidden'} for i in range(130)]
        lines = [json.dumps(r, ensure_ascii=False).encode() for r in self.records]
        raw = b'\r\n' + b'\r\n \n'.join(lines) + b'\n'
        self.corpus.write_bytes(raw)
        self.positions = np.asarray([raw.index(line) for line in lines], dtype=np.int64)
        np.save(self.offsets, self.positions)
        self.ids.write_text(''.join(json.dumps({'passage_id': r['passage_id']})+'\n' for r in self.records))
        self.index.write_bytes(b'known-index-for-fingerprint-binding')
        self.info = {'schema_version': 'nq_train_index_recovery_v1', 'corpus_size': len(lines),
                     'embedding_alignment': {'performed': True, 'sample_count': 64, 'passed': True}}
        self.refresh_fingerprints()

    def refresh_fingerprints(self):
        for name, path in [('corpus', self.corpus), ('ids', self.ids), ('offsets', self.offsets), ('index', self.index)]:
            self.info[f'{name}_fingerprint'] = {'bytes': path.stat().st_size, 'sha256': file_sha(path)}
        self.meta.write_text(json.dumps(self.info))

    def validate(self):
        return validate_complete_mapping(self.corpus, self.ids, self.offsets, self.meta, index_path=self.index)

    def test_every_row_is_actually_read_back_including_unicode_crlf_and_blank_lines(self):
        before = self.corpus.read_bytes()
        report = self.validate()
        self.assertTrue(report['passed'])
        self.assertEqual(report['records_checked'], 130)
        self.assertEqual(report['actual_seek_reads'], 130)
        self.assertTrue(report['index_sha256_verified'])
        self.assertFalse(report['embedding_reencoding_performed'])
        self.assertEqual(before, self.corpus.read_bytes())

    def test_corruption_outside_the_64_position_sample_cannot_pass(self):
        position = next(p for p in range(130) if p not in sample_positions(130, 64))
        bad = self.positions.copy()
        bad[position] += 1
        np.save(self.offsets, bad)
        self.refresh_fingerprints()  # Even a rehashed sidecar must obey true line boundaries.
        with self.assertRaisesRegex(ValueError, f'boundary at row {position}'):
            self.validate()

    def test_same_count_permuted_id_rows_are_rejected(self):
        rows = self.ids.read_text().splitlines()
        rows[85], rows[86] = rows[86], rows[85]
        self.ids.write_text('\n'.join(rows)+'\n')
        self.refresh_fingerprints()
        with self.assertRaisesRegex(ValueError, 'ID mapping disagrees'):
            self.validate()

    def test_resource_hash_drift_is_rejected(self):
        for path in [self.corpus, self.ids, self.offsets, self.index]:
            with self.subTest(path=path.name):
                before = path.read_bytes()
                path.write_bytes(before+b' ')
                with self.assertRaises((ValueError, OSError)):
                    self.validate()
                path.write_bytes(before)

    def test_extra_or_missing_id_rows_fail_even_with_updated_fingerprint(self):
        original = self.ids.read_text().splitlines()
        for rows in [original[:-1], original + [json.dumps({'passage_id': 'extra'})]]:
            with self.subTest(count=len(rows)):
                self.ids.write_text('\n'.join(rows)+'\n')
                self.refresh_fingerprints()
                with self.assertRaisesRegex(ValueError, 'ID mapping|extra rows'):
                    self.validate()

    def test_wrong_offset_dtype_dimension_negative_or_order_is_rejected(self):
        for bad in [self.positions.astype(np.int32), self.positions[None, :],
                    np.full(130, -1, dtype=np.int64), self.positions[::-1]]:
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                np.save(self.offsets, bad)
                self.refresh_fingerprints()
                with self.assertRaises(ValueError):
                    self.validate()

    def test_lazy_sequence_has_bounded_cache_and_only_runtime_fields(self):
        with OffsetCorpus(self.corpus, self.ids, self.offsets, cache_size=2) as store:
            self.assertEqual(len(store), 130)
            for i in range(5):
                self.assertEqual(store[i], {'passage_id': f'p{i}', 'text': self.records[i]['text']})
            self.assertEqual(len(store.cache), 2)
            self.assertEqual(store[-1]['passage_id'], 'p129')
            row = store[129]
            row['text'] = 'mutated caller copy'
            self.assertEqual(store[129]['text'], self.records[129]['text'])
            with self.assertRaises(TypeError):
                store[:]
            with self.assertRaises(IndexError):
                store[130]
        with self.assertRaises(RuntimeError):
            store[0]

    def test_cached_lookup_does_not_hide_mutated_corpus_or_sidecars(self):
        for path in [self.corpus, self.ids, self.offsets]:
            with self.subTest(path=path.name):
                before = path.read_bytes()
                with OffsetCorpus(self.corpus, self.ids, self.offsets) as store:
                    store[0]
                    path.write_bytes(before+b' ')
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        store[0]
                path.write_bytes(before)

    def test_lazy_lookup_rejects_id_mismatch_and_out_of_bounds_offsets(self):
        rows = self.ids.read_text().splitlines()
        rows[0] = json.dumps({'passage_id': 'different'})
        self.ids.write_text('\n'.join(rows)+'\n')
        with OffsetCorpus(self.corpus, self.ids, self.offsets) as store:
            with self.assertRaisesRegex(ValueError, 'disagree'):
                store[0]
        bad = self.positions.copy(); bad[-1] = self.corpus.stat().st_size
        np.save(self.offsets, bad)
        with self.assertRaisesRegex(ValueError, 'bounds'):
            OffsetCorpus(self.corpus, self.ids, self.offsets)


if __name__ == '__main__':
    unittest.main()
