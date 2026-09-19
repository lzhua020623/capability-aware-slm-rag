"""CPU-only protocol and interrupted-run regression tests (synthetic evidence)."""
from __future__ import annotations

import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from scripts import run_fever_preliminary as runner


class FeverPreliminaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = runner.load_preliminary_config()
        for name in (
            self.config['samples']['fever']['manifest'],
            'src/generation/qwen.py', 'src/generation/prompts.py',
            'src/evaluation/fever.py', 'scripts/run_fever_preliminary.py',
        ):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(runner.ROOT / name, target)
        self.manifest_path = self.root / self.config['samples']['fever']['manifest']
        self.manifest = json.loads(self.manifest_path.read_text())
        self.source = self.root / self.config['samples']['fever']['source']
        self.source.parent.mkdir(parents=True)
        self.records = [
            dict(item, claim=f"Synthetic claim {item['id']}", gold_page_ids=['Test_page'],
                 gold_evidence_text=[f"Synthetic evidence {item['id']}", 'Second sentence.'])
            for item in self.manifest['samples']
        ]
        self.write_source(self.records)
        self.addCleanup(patch.stopall)
        patch.object(runner, 'ROOT', self.root).start()
        self.samples = runner.load_frozen_samples(self.config)
        self.fingerprint = runner.experiment_fingerprint(self.config, self.samples)
        self.output = self.root / self.config['result_paths']['fever_c0_7b']

    def write_source(self, records):
        self.source.write_text(''.join(json.dumps(r) + '\n' for r in records))

    def result(self, condition='C0'):
        sample = self.samples[0]
        return dict(
            sample_id=sample['id'], dataset='fever', condition=condition,
            model_name=self.config['primary_model']['model_name'],
            experiment_fingerprint=self.fingerprint, question_or_claim=sample['claim'],
            gold_answer_or_label=sample['label'], evidence=runner.evidence_for(condition, sample),
            gold_page_ids=sample['gold_page_ids'], raw_output='SUPPORTS',
            prediction='SUPPORTS', correct=True, latency=0.1,
        )

    def completed(self):
        return runner.completed_ids(self.output, 'C0', self.samples, self.config, self.fingerprint)

    def invoke(self, *args):
        with patch.object(sys, 'argv', ['runner', *args]), redirect_stdout(io.StringIO()):
            return runner.main()

    def fake_model(self):
        module = ModuleType('src.generation.qwen')
        module.load_qwen_nf4 = Mock(return_value=(object(), SimpleNamespace(hf_device_map={'': 0})))
        module.generate_greedy = Mock(return_value=('SUPPORTS', 0.1))
        transformers = ModuleType('transformers')
        transformers.set_seed = Mock()
        patch.dict(sys.modules, {'src.generation.qwen': module, 'transformers': transformers}).start()
        return module

    def test_frozen_500_and_manifest_order(self):
        runner.validate_frozen_settings(self.config)
        self.assertEqual(len(self.samples), 500)
        self.assertEqual(sum(s['label'] == 'SUPPORTS' for s in self.samples), 250)
        self.assertEqual([s['id'] for s in self.samples], [s['id'] for s in self.manifest['samples']])

    def test_missing_source_actionable_without_torch(self):
        self.source.unlink()
        with self.assertRaisesRegex(RuntimeError, 'Missing FEVER source file'):
            self.invoke('--check-only')

    def test_check_only_does_not_import_model(self):
        with patch.dict(sys.modules, {'src.generation.qwen': None, 'transformers': None, 'torch': None}):
            self.assertEqual(self.invoke('--check-only'), 0)
        self.assertFalse(self.output.exists())

    def test_config_mismatch_rejected(self):
        self.config['primary_model']['max_new_tokens'] = 128
        with self.assertRaisesRegex(RuntimeError, 'differs'):
            runner.validate_frozen_settings(self.config)

    def test_manifest_duplicates_and_metadata_rejected(self):
        for change in ('duplicate', 'seed', 'label'):
            with self.subTest(change=change):
                manifest = copy.deepcopy(self.manifest)
                if change == 'duplicate':
                    manifest['samples'][1] = manifest['samples'][0]
                elif change == 'seed':
                    manifest['seed'] = 0
                else:
                    manifest['samples'][0]['label'] = 'REFUTES'
                self.manifest_path.write_text(json.dumps(manifest))
                with self.assertRaises(RuntimeError):
                    runner.load_frozen_samples(self.config)

    def test_missing_duplicate_and_invalid_source_rejected(self):
        variants = [self.records[1:], self.records + [self.records[0]]]
        for key, value in (
            ('label', 'REFUTES'), ('claim', ''), ('gold_evidence_text', [None]),
            ('gold_evidence_text', 'not a list'), ('gold_page_ids', ['']),
        ):
            records = copy.deepcopy(self.records)
            records[0][key] = value
            variants.append(records)
        for records in variants:
            with self.subTest(first=records[0]):
                self.write_source(records)
                with self.assertRaises(RuntimeError):
                    runner.load_frozen_samples(self.config)

    def test_c0_no_evidence_and_c3_all_evidence(self):
        sample = self.samples[0]
        self.assertEqual(runner.evidence_for('C0', sample), [])
        self.assertEqual(runner.build_prompt(self.config, 'C0', sample),
                         self.config['prompts']['fever']['C0'] + '\n\nClaim: ' + sample['claim'])
        c3 = runner.build_prompt(self.config, 'C3', sample)
        self.assertIn('[1] ' + sample['gold_evidence_text'][0], c3)
        self.assertIn('[2] Second sentence.', c3)
        self.assertTrue(c3.endswith('Claim: ' + sample['claim']))

    def test_resume_rejects_incompatible_or_corrupt_records(self):
        self.output.parent.mkdir(parents=True)
        for key, value in (
            ('model_name', 'Qwen/Qwen2.5-3B-Instruct'), ('experiment_fingerprint', None),
            ('question_or_claim', 'changed'), ('evidence', [{'text': 'leak'}]),
            ('correct', False), ('prediction', 'REFUTES'), ('latency', float('nan')),
            ('sample_id', 'not-frozen'), ('condition', 'C3'),
        ):
            with self.subTest(key=key):
                record = self.result()
                record[key] = value
                self.output.write_text(json.dumps(record) + '\n')
                with self.assertRaises(RuntimeError):
                    self.completed()
        self.output.write_text((json.dumps(self.result()) + '\n') * 2)
        with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
            self.completed()

    def test_preflight_checks_existing_results(self):
        runner.write_result(self.output, dict(self.result(), model_name='wrong'))
        with self.assertRaisesRegex(RuntimeError, 'model_name'):
            self.invoke('--check-only')

    def test_newline_less_record_resumes_without_corruption(self):
        self.output.parent.mkdir(parents=True)
        self.output.write_text(json.dumps(self.result()))
        self.assertEqual(self.completed(), {str(self.samples[0]['id'])})
        runner.write_result(self.output, {'extra': 'record'})
        self.assertEqual(len(runner.load_jsonl(self.output)), 2)

    def test_partial_json_fails_without_overwriting(self):
        self.output.parent.mkdir(parents=True)
        damaged = json.dumps(self.result()) + '\n{"sample_id":'
        self.output.write_text(damaged)
        with self.assertRaisesRegex(RuntimeError, 'Back up the file'):
            self.completed()
        self.assertEqual(self.output.read_text(), damaged)

    def test_changed_evidence_or_prompt_changes_fingerprint(self):
        samples = copy.deepcopy(self.samples)
        samples[0]['gold_evidence_text'][0] = 'Changed evidence'
        self.assertNotEqual(self.fingerprint, runner.experiment_fingerprint(self.config, samples))
        self.config['prompts']['fever']['C0'] += ' Changed'
        self.assertNotEqual(self.fingerprint, runner.experiment_fingerprint(self.config, self.samples))

    def test_smoke_does_not_write_and_unknown_fails(self):
        model = self.fake_model()
        self.assertEqual(self.invoke('--smoke-test'), 0)
        self.assertEqual(model.generate_greedy.call_count, 2)
        self.assertFalse(self.output.parent.exists())
        model.generate_greedy.return_value = ('I do not know.', 0.1)
        with self.assertRaisesRegex(RuntimeError, 'no parseable'):
            self.invoke('--smoke-test')

    def test_interrupted_run_resumes_500_per_condition_without_duplicates(self):
        model = self.fake_model()
        model.generate_greedy.side_effect = [('SUPPORTS', 0.1), RuntimeError('interrupted')]
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.invoke()
        self.assertEqual(len(runner.load_jsonl(self.output)), 1)
        model.generate_greedy.reset_mock(side_effect=True)
        self.assertEqual(self.invoke('--conditions', 'C0', 'C3', 'C0'), 0)
        self.assertEqual(model.generate_greedy.call_count, 999)
        for condition in ('C0', 'C3'):
            path = self.root / self.config['result_paths'][f'fever_{condition.lower()}_7b']
            records = runner.load_jsonl(path)
            self.assertEqual(len(records), 500)
            self.assertEqual(len({r['sample_id'] for r in records}), 500)
            self.assertEqual(sum(r['correct'] for r in records), 250)
            self.assertTrue(all(bool(r['evidence']) == (condition == 'C3') for r in records))
        model.load_qwen_nf4.reset_mock()
        self.assertEqual(self.invoke(), 0)
        model.load_qwen_nf4.assert_not_called()


if __name__ == '__main__':
    unittest.main()
