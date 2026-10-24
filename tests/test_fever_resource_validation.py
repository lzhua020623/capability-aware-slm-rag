from types import SimpleNamespace
import unittest

from scripts.check_fever_controller_resources import validate_index


class FeverResourceValidationTests(unittest.TestCase):
    def make_index(self):
        cls = type('IndexIVFPQ', (), {})
        index = cls()
        for key, value in dict(ntotal=5396106, d=768, nlist=4096, nprobe=16,
                               metric_type=0, is_trained=True, by_residual=False).items():
            setattr(index, key, value)
        index.pq = SimpleNamespace(M=64, nbits=8)
        return index

    def test_team_structure_is_accepted(self):
        report = validate_index(self.make_index(), {'corpus_size': 5396106})
        self.assertEqual(report['nprobe'], 16)
        self.assertFalse(report['by_residual'])

    def test_wrong_metric_dimension_count_probe_or_pq_are_rejected(self):
        for field, value in [('d', 1024), ('ntotal', 100000), ('metric_type', 1),
                             ('nprobe', 8), ('is_trained', False), ('by_residual', True)]:
            with self.subTest(field=field):
                index = self.make_index(); setattr(index, field, value)
                with self.assertRaises(ValueError):
                    validate_index(index, {'corpus_size': 5396106})
        index = self.make_index(); index.pq.M = 32
        with self.assertRaisesRegex(ValueError, 'PQ'):
            validate_index(index, {'corpus_size': 5396106})
        with self.assertRaisesRegex(ValueError, 'IndexIVFPQ'):
            validate_index(SimpleNamespace(), {'corpus_size': 5396106})


if __name__ == '__main__':
    unittest.main()
