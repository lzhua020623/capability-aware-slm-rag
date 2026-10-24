import json
from pathlib import Path
import tempfile
import unittest
from scripts.audit_controller_fever_c1_outputs import audit


class RecordedOutputTests(unittest.TestCase):
    def test_failed_attempt_retains_645_exact_frozen_inputs_without_claiming_completion(self):
        folder=Path(__file__).resolve().parents[1]/'artifacts/controller/fever_c1_v1'
        report=audit(folder/'transport_r1/collection',folder/'attempt2_oom/raw_outputs')
        self.assertEqual(report['counts'],{'development':200,'selection':200,'final_candidate':245})
        self.assertEqual(report['n_completed'],645)
        self.assertFalse(report['complete'] or report['quality_scored'] or report['answers_printed'])
        self.assertTrue(report['runtime_gold_isolated'])

    def test_false_completion_and_wrong_manifest_binding_are_rejected(self):
        folder=Path(__file__).resolve().parents[1]/'artifacts/controller/fever_c1_v1'
        source=folder/'attempt2_oom/raw_outputs'
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            # Hard links avoid copying sealed predictions or exposing their content.
            for path in source.rglob('*'):
                target=output/path.relative_to(source)
                if path.is_dir():target.mkdir(exist_ok=True)
                else:target.hardlink_to(path)
            path=output/'manifest.json';state=json.loads(path.read_text());path.unlink()
            state['state']='completed';path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'Incomplete output'):audit(folder/'transport_r1/collection',output)
            state['state']='failed';state['collection_manifest_sha256']='0'*64;path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'provenance'):audit(folder/'transport_r1/collection',output)


if __name__=='__main__':unittest.main()
