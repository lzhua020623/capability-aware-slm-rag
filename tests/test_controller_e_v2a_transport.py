import base64
import hashlib
import json
import unittest

from scripts.prepare_controller_e_v2a_transport import RELEASE, dataset_variant


class TransportTests(unittest.TestCase):
    def test_only_the_loader_changes_and_the_exact_archive_is_preserved(self):
        old = json.loads((RELEASE / 'controller_e_v2a_development_v1.ipynb').read_text())
        new = dataset_variant(old)
        before = ''.join(old['cells'][1]['source']); after = ''.join(new['cells'][1]['source'])
        payload = next(line for line in before.splitlines() if line.startswith('raw=base64.b64decode('))
        embedded = payload.split("'", 2)[1]
        archive = RELEASE / 'controller_e_v2a_development_v1.tar.gz'
        self.assertEqual(base64.b64decode(embedded, validate=True), archive.read_bytes())
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.assertIn(digest, after)
        self.assertIn('controller_e_v2a_development_v1.tar.gz.bin', after)
        self.assertEqual(before[before.index('assert hashlib.sha256'):], after[after.index('assert hashlib.sha256'):])
        self.assertEqual(old['metadata'], new['metadata'])
        self.assertEqual(old['cells'][0], new['cells'][0])
        self.assertLess(len(json.dumps(new).encode()), 1_000_000)

    def test_missing_or_duplicate_loader_is_rejected(self):
        old = json.loads((RELEASE / 'controller_e_v2a_development_v1.ipynb').read_text())
        for code in ('print("wrong notebook")', 'raw=base64.b64decode("x")\nraw=base64.b64decode("y")'):
            old['cells'][1]['source'] = [code]
            with self.assertRaises(ValueError): dataset_variant(old)


if __name__ == '__main__': unittest.main()
