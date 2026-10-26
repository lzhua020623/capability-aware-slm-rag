"""Selection export provenance, paired inputs and rejection must survive replay."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts import audit_controller_t5_selection as auditor

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'artifacts/controller/t5_selection_v1/kaggle_v1'


class SelectionOutputAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.review, cls.replay = auditor.audit(RESULT / 'outputs.zip')
        members = auditor.verified_members(RESULT / 'outputs.zip')
        cls.rows = [json.loads(line) for line in members['predictions.jsonl'].splitlines()]
        cls.frozen = json.loads((ROOT / auditor.RELEASE).read_text())

    def test_real_export_replays_saved_report_and_frozen_scope(self):
        self.assertEqual(self.report, json.loads((RESULT / 'audit.json').read_text()))
        self.assertEqual(self.report['archive_sha256'], '2139d926fb32c6071b95ceae3035ac95c06e7da10118b9c8976e302bcb62a7bd')
        self.assertEqual(self.report['archive_checked_files'], 17)
        self.assertEqual(self.report['acceptance']['n_records'], 2000)
        self.assertEqual(self.report['acceptance']['execution_errors'], 0)
        self.assertEqual(self.report['adaptive_fixed_cache_checks'], 400)
        self.assertEqual(len(self.replay), 400)

    def test_scores_and_rejected_sole_candidate_are_not_a_new_search(self):
        self.assertEqual([self.report['groups']['nq'][m]['correct'] for m in self.frozen['modes']],
                         [89, 89, 92, 89, 93])
        self.assertEqual([self.report['groups']['fever'][m]['correct'] for m in self.frozen['modes']],
                         [163, 167, 164, 163, 164])
        nq = self.report['candidate_comparison']['nq']
        self.assertEqual((nq['candidate_correct'], nq['overridden_routes'], nq['changed_answer_pairs']),
                         (90, 103, 15))
        self.assertEqual((nq['vs_original_adaptive']['fixed'], nq['vs_original_adaptive']['regressed']), (3, 6))
        self.assertEqual(self.report['decision']['status'], 'rejected')
        self.assertFalse(self.report['decision']['candidate_promoted'])
        self.assertEqual(self.report['decision']['alternate_thresholds_evaluated'], 0)
        self.assertEqual(self.report['new_GPU_calls'], 0)
        self.assertEqual(self.report['candidate_comparison']['fever']['changed_answer_pairs'], 0)

    def test_EM_does_not_fill_human_semantics_or_promote_positive_candidate(self):
        self.assertEqual(len(self.review), 15)
        self.assertTrue(all(r['semantic_before'] is None and r['semantic_after'] is None
                            and r['human_review'] == 'pending' for r in self.review))
        for net in (-3, 0):
            self.assertEqual(auditor.selection_decision(net)['status'], 'rejected')
        positive = auditor.selection_decision(3)
        self.assertEqual(positive['status'], 'pending_semantic_review')
        self.assertFalse(positive['candidate_promoted'])

    def test_missing_real_prediction_cannot_be_scored_as_complete(self):
        registration = {'datasets': self.frozen['datasets'], 'required_modes': self.frozen['modes']}
        with self.assertRaisesRegex(ValueError, 'Incomplete frozen comparison'):
            auditor.validate_predictions(self.rows[:-1], registration)

    def test_changed_adaptive_answer_cannot_be_replayed_from_another_state(self):
        rows = list(self.rows)
        index = next(i for i, r in enumerate(rows) if r['mode'] == 'adaptive')
        rows[index] = deepcopy(rows[index])
        rows[index]['final_answer'] = 'forged cached answer'
        with self.assertRaisesRegex(ValueError, 'states differ'):
            auditor.verify_cache(rows)

    def test_unavailable_E_must_not_call_a_model(self):
        row = deepcopy(next(r for r in self.rows if r['mode'] == 'always_repair'))
        row['calls'] = [{'kind': 'llm', 'attempted': True}]
        with self.assertRaisesRegex(ValueError, 'Unavailable E'):
            auditor.verify_cache([row])

    def test_checksum_corruption_and_omitted_members_are_rejected(self):
        for payload, checks in [(b'changed', {'prediction': hashlib.sha256(b'original').hexdigest()}),
                                (b'original', {})]:
            with tempfile.TemporaryDirectory() as directory:
                archive = Path(directory) / 'bad.zip'
                with zipfile.ZipFile(archive, 'w') as z:
                    z.writestr('prediction', payload)
                    z.writestr('output_checksums.json', json.dumps(checks))
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    auditor.verified_members(archive)

    def test_changed_selection_gold_is_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rel in (auditor.RELEASE, auditor.POLICY, auditor.CANDIDATE):
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / rel).read_bytes())
            # Release dataset order is FEVER then NQ. A different reference
            # must fail its registered hash, even if IDs would still match.
            target = root / 'artifacts/controller/data_v1/offline/fever/selection/gold.jsonl'
            target.parent.mkdir(parents=True)
            target.write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'reference hash changed'):
                auditor.audit(RESULT / 'outputs.zip', root=root)


if __name__ == '__main__':
    unittest.main()
