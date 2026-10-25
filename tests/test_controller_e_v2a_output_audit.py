from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.audit_controller_e_v2a_outputs import audit

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/controller/e_v2a_interface_r1'
ARCHIVE = FOLDER / 'kaggle_v1/outputs.zip'
RELEASE = FOLDER / 'gpu_release_v1'


class OutputAuditTests(unittest.TestCase):
    def setUp(self):
        with zipfile.ZipFile(ARCHIVE) as z:
            self.files = {n: z.read(n) for n in z.namelist()}

    def check_modified(self, edit, message, *, rehash=True):
        files = deepcopy(self.files)
        edit(files)
        if rehash:
            files['output_checksums.json'] = json.dumps({n: hashlib.sha256(raw).hexdigest()
                for n, raw in files.items() if n != 'output_checksums.json'}).encode()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'modified.zip'
            with zipfile.ZipFile(p, 'w') as z:
                for n, raw in files.items():
                    z.writestr(n, raw)
            with self.assertRaisesRegex(ValueError, message):
                audit(p, RELEASE)

    @staticmethod
    def update_json(files, name, **values):
        content = json.loads(files[name])
        content.update(values)
        files[name] = json.dumps(content).encode()

    def test_both_actual_failed_exports_are_intact_but_not_accepted(self):
        for folder, passed, attempted in ((ROOT / 'artifacts/controller/e_v2a_v1', 1, 2),
                                          (FOLDER, 2, 3)):
            result = audit(folder / 'kaggle_v1/outputs.zip', folder / 'gpu_release_v1')
            self.assertTrue(result['audit_passed'])
            self.assertFalse(result['engineering_passed'])
            self.assertEqual(result['gate_checks_passed'], passed)
            self.assertEqual(result['short_checks_completed'], attempted)
            self.assertEqual(result['n_development_predictions'], 0)
            self.assertEqual(result['long_checks_completed'], 0)

    def test_corrected_failure_is_verdict_taxonomy_with_safe_keep(self):
        result = audit(ARCHIVE, RELEASE)
        failed, = [c for c in result['gate_checks'] if not c['passed']]
        self.assertTrue(failed['contract_passed'])
        self.assertEqual(failed['expected']['original_verification'], 'insufficient')
        self.assertEqual(failed['observed']['original_verification'], 'unsupported')
        self.assertEqual(failed['observed']['final_action'], 'keep')
        self.assertTrue(failed['observed']['unresolved'])
        self.assertEqual(failed['observed_final_answer'], failed['expected']['final_answer'])

    def test_corruption_and_uncovered_outputs_are_rejected(self):
        self.check_modified(lambda f: f.update({'run.log': b'changed'}), 'checksum mismatch', rehash=False)
        self.check_modified(lambda f: f.update({'unregistered.txt': b'x'}), 'coverage mismatch', rehash=False)

    def test_forged_behavior_pass_is_recomputed_not_trusted(self):
        name = 'engineering/v2a-r1-fiction-nq-insufficient_unresolved.json'
        self.check_modified(lambda f: self.update_json(f, name, passed=True), 'disagrees with observed')

    def test_foreign_inputs_and_incomplete_completed_state_are_rejected(self):
        other = ROOT / 'artifacts/controller/e_v2a_v1/gpu_release_v1/execution_manifest.json'
        self.check_modified(lambda f: f.update({'input_execution_manifest.json': other.read_bytes()}),
                            'different sealed inputs')
        self.check_modified(lambda f: self.update_json(f, 'manifest.json', state='completed'),
                            'Incomplete run')
        self.check_modified(lambda f: self.update_json(f, 'manifest.json', n_predictions=1),
                            'count disagrees')

    def test_missing_six_and_long_checks_cannot_be_promoted(self):
        def edit(files):
            self.update_json(files, 'manifest.json', engineering_passed=True)
            files['engineering_receipt.json'] = b'{"passed":true}'
        self.check_modified(edit, 'all eight passing checks')

    def test_failed_gate_deployment_is_unavailable_without_fabricated_dev_rates(self):
        decision = json.loads((FOLDER / 'deployment_decision.json').read_text())
        self.assertEqual(decision['registered_policy'], 'engineering_failed')
        self.assertEqual(set(decision['modes'].values()), {'unavailable'})
        self.assertEqual(decision['real_development_contract_rates'], {'nq': None, 'fever': None})
        self.assertFalse(decision['proposal_E_completion_accepted'])
        self.assertFalse(decision['T5_candidate_frozen'])


if __name__ == '__main__':
    unittest.main()
