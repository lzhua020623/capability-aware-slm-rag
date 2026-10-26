import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class SelectionLaunchTests(unittest.TestCase):
    def test_first_private_launch_binds_frozen_package_without_claiming_engineering_or_quality(self):
        folder=ROOT/'artifacts/controller/t5_selection_v1'
        launch=json.loads((folder/'launch_v1.json').read_text())
        current=json.loads((folder/'current_release.json').read_text())
        for key in ('archive_sha256','notebook_sha256','source_commit'):
            self.assertEqual(launch[key],current[key])
        nb=folder/current['release_dir']/'controller_t5_selection_v1_dataset.ipynb'
        self.assertEqual(hashlib.sha256(nb.read_bytes()).hexdigest(),launch['notebook_sha256'])
        self.assertEqual(launch['script_version_id'],362316572)
        self.assertEqual((launch['version'],launch['attempt']),(1,1))
        self.assertEqual(launch['notebook_privacy'],'Private')
        self.assertEqual(launch['input_code_dataset_privacy_at_creation'],'Private')
        self.assertEqual(launch['dataset_counts'],{'nq':200,'fever':200})
        self.assertEqual(launch['E_mode'],{'nq':'unavailable','fever':'unavailable'})
        self.assertEqual((launch['expected_records'],launch['candidate_threshold']),(2000,.2))
        self.assertIsNone(launch['engineering_outcome'])
        self.assertIsNone(launch['selection_predictions_confirmed'])
        self.assertFalse(launch['selection_started_confirmed'])
        self.assertFalse(launch['selection_quality_scored'])
        self.assertFalse(launch['candidate_promoted'])


if __name__=='__main__':unittest.main()
