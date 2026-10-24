import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from src.retrieval.resource_import import build_ordered_offsets, extract_member
from src.retrieval.offset_corpus import OffsetCorpus


class ResourceImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_selected_member_cannot_choose_output_path_or_overwrite(self):
        archive, output = self.root / 'test.zip', self.root / 'chosen.txt'
        with zipfile.ZipFile(archive, 'w') as source:
            source.writestr('../escape', b'payload')
            source.writestr('unused', b'other')
        self.assertEqual(extract_member(archive, '../escape', output)['bytes'], 7)
        self.assertEqual(output.read_bytes(), b'payload')
        self.assertFalse((self.root.parent / 'escape').exists())
        with self.assertRaises(FileExistsError):
            extract_member(archive, '../escape', output)
        with self.assertRaises(ValueError):
            extract_member(archive, 'missing', self.root / 'missing')

    def test_symbolic_link_is_rejected(self):
        archive = self.root / 'link.zip'
        link = zipfile.ZipInfo('link'); link.external_attr = 0o120777 << 16
        with zipfile.ZipFile(archive, 'w') as source:
            source.writestr(link, '../target')
        with self.assertRaisesRegex(ValueError, 'symbolic'):
            extract_member(archive, 'link', self.root / 'output')

    def make_mapping(self):
        body, ids, offsets = [self.root / name for name in ('body', 'ids', 'offsets.npy')]
        body.write_bytes(b'\n' + ''.join(json.dumps({'passage_id': f'p{i}', 'text': f'中文 {i}'},
                          ensure_ascii=False)+'\r\n' for i in range(3)).encode())
        ids.write_text(''.join(json.dumps({'passage_id': f'p{i}'})+'\n' for i in range(3)))
        return body, ids, offsets

    def test_full_mapping_and_lazy_readback(self):
        body, ids, offsets = self.make_mapping()
        report = build_ordered_offsets(body, ids, offsets, expected_count=3)
        self.assertTrue(report['full_order_verified'])
        self.assertEqual(np.load(offsets).dtype, np.int64)
        with OffsetCorpus(body, ids, offsets) as store:
            self.assertEqual(store[2], {'passage_id': 'p2', 'text': '中文 2'})
        with self.assertRaises(FileExistsError):
            build_ordered_offsets(body, ids, offsets, expected_count=3)

    def test_order_count_duplicate_and_extra_ids_fail_without_publishing(self):
        body, ids, offsets = self.make_mapping()
        original = ids.read_text()
        for content, count in [(original.replace('p1', 'different'), 3), (original, 4),
                               (original + '{"passage_id":"extra"}\n', 3),
                               (original.replace('p1', 'p0'), 3)]:
            ids.write_text(content)
            with self.assertRaises(ValueError):
                build_ordered_offsets(body, ids, offsets, expected_count=count)
            self.assertFalse(offsets.exists())


if __name__ == '__main__':
    unittest.main()
