from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_controller_t5_selection import prepare,NAME,notebook
from scripts.run_controller_t5_selection import (POLICY,verify,run_batches,active_engineering,main)
from scripts.run_controller_integration import smoke_inputs
from src.compensation.answer_repair import Generation
from src.controller.execution import ExecutionConfig
from src.controller.engineering import fictional_item
from tests.test_controller_integration import pipeline

ROOT=Path(__file__).resolve().parents[1]


class T5SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.folder=Path(cls.temp.name)
        # Package tests use the live working source, explicitly not a release.
        with patch('scripts.prepare_controller_t5_selection.read_source_commit',return_value='unit_test_uncommitted'):
            cls.receipt=prepare(cls.folder/'release')
        with tarfile.open(cls.folder/'release'/(NAME+'.tar.gz')) as t:t.extractall(cls.folder/'extract',filter='data')
        cls.root=cls.folder/'extract'/NAME
        cls.manifest=json.loads((cls.root/'execution_manifest.json').read_text())

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_selection_package_has_only_frozen_role_inputs_and_gold_hashes(self):
        inputs,policy=verify(self.root,self.manifest)
        self.assertEqual({d:len(v) for d,v in inputs.items()},{'nq':200,'fever':200})
        self.assertEqual(self.manifest['expected_predictions'],2000)
        self.assertFalse(self.manifest['contains_gold'])
        self.assertFalse(self.manifest['E_engineering_accepted'])
        self.assertFalse(any(p.name=='gold.jsonl' for p in self.root.rglob('*')))
        for ds in ('nq','fever'):
            self.assertEqual(policy['runtime_config'][ds+'_v2a_mode'],'unavailable')
            rows=[json.loads(l) for l in (self.root/'inputs'/(ds+'_selection_runtime.jsonl')).read_text().splitlines()]
            self.assertTrue(all(set(r)=={'sample_id','dataset','query','evidence','initial_answer'} for r in rows))
        self.assertEqual(json.loads((self.root/'inputs/candidate_policy.json').read_text())['threshold'],.2)

    def test_preflight_never_loads_model_or_creates_predictions(self):
        with patch('scripts.run_controller_t5_selection.ROOT',self.root),redirect_stdout(StringIO()):
            self.assertEqual(main(['--preflight','--output-dir',str(self.folder/'unused')]),0)
        self.assertFalse((self.folder/'unused').exists())

    def test_source_and_candidate_tampering_rejected(self):
        p=self.root/'inputs/candidate_policy.json';original=p.read_bytes()
        try:
            p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Candidate changed'):verify(self.root,self.manifest)
        finally:p.write_bytes(original)
        p=self.root/POLICY;original=p.read_bytes()
        try:
            p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'source changed'):verify(self.root,self.manifest)
        finally:p.write_bytes(original)

    def test_package_refuses_uncommitted_source_and_existing_output(self):
        with patch('scripts.prepare_controller_t5_selection.read_source_commit',side_effect=ValueError('dirty')):
            with self.assertRaisesRegex(ValueError,'dirty'):prepare(self.folder/'dirty')
        with self.assertRaises(FileExistsError):prepare(self.folder/'release')

    def test_package_rejects_changed_core_before_reading_selection(self):
        real_sha=__import__('scripts.prepare_controller_t5_selection',fromlist=['file_sha']).file_sha
        def tampered(path):
            return '0'*64 if str(path).endswith('src/controller/execution.py') else real_sha(path)
        with patch('scripts.prepare_controller_t5_selection.read_source_commit',return_value='unit'),patch(
                'scripts.prepare_controller_t5_selection.file_sha',side_effect=tampered):
            with self.assertRaisesRegex(ValueError,'Frozen core source changed'):
                prepare(self.folder/'bad_core')

    def test_notebook_checks_resource_hashes_and_always_exports_without_retry_or_scoring(self):
        nb=json.loads((self.folder/'release'/(NAME+'_dataset.ipynb')).read_text())
        code=''.join(nb['cells'][1]['source']);compile(code,'<test>','exec')
        self.assertIn('finally:',code)
        self.assertIn("registry['contains_gold'] is False",code)
        self.assertIn("sha(src)==info['sha256']",code)
        self.assertNotIn('evaluate_controller',code)
        self.assertNotIn('while True',code)

    def test_four_live_arms_plus_cached_C1_preserve_disabled_E(self):
        config=ExecutionConfig(nq_repair_strategy='evidence_id_v2a',fever_repair_strategy='evidence_id_v2a',
                               nq_v2a_mode='unavailable',fever_v2a_mode='unavailable')
        p=pipeline(config=config)
        state={'n_predictions':0};items=smoke_inputs()
        with tempfile.TemporaryDirectory() as td,redirect_stdout(StringIO()):
            lean=run_batches(p,{'nq':[items[0]],'fever':[items[1]]},td,state)
            rows=[json.loads(l) for l in (Path(td)/'predictions.jsonl').read_text().splitlines()]
        self.assertEqual((state['n_predictions'],len(lean)),(10,10))
        for row in rows:
            if row['mode'] in ('always_repair','c1'):
                self.assertEqual(row['calls'],[])
                self.assertEqual(row['final_answer'],row['initial_answer'])

    def test_active_gate_covers_six_long_paths_but_never_calls_E_or_accepts_E_semantics(self):
        config=ExecutionConfig(nq_repair_strategy='evidence_id_v2a',fever_repair_strategy='evidence_id_v2a',
                               nq_v2a_mode='unavailable',fever_v2a_mode='unavailable')
        p=pipeline(config=config)
        p.raw_generator.tokenizer=object()
        p.raw_generator.generate=lambda prompt,budget:Generation('SUPPORTS',.01,16001,1)
        with tempfile.TemporaryDirectory() as td,redirect_stdout(StringIO()),patch(
                'src.controller.engineering.long_fixture',side_effect=lambda tok,ds,n:(fictional_item(ds),[16001]*3)):
            receipt=active_engineering(p,Path(td)/'checks')
        self.assertEqual(len(receipt['active_long_checks']),6)
        self.assertEqual(len(receipt['inactive_E_checks']),2)
        self.assertTrue(receipt['passed'])
        self.assertFalse(receipt['E_engineering_accepted'])
        self.assertTrue(all(not r['calls'] for r in receipt['inactive_E_checks']))


if __name__=='__main__':unittest.main()
