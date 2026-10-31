"""Default entry points select the complete product without changing frozen runs."""
import ast
from contextlib import redirect_stdout
from dataclasses import replace
import inspect
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import check_controller_current as check
from scripts import run_quality_eval as runner
from scripts.run_controller_integration import smoke_inputs, smoke_pipeline
from src.controller.execution import ExecutionConfig
from src.diagnosis.labels import DiagnosisLabel
from src.pipeline import current
from src.pipeline.registrations import (
    B_REGISTRATION, BM25_REGISTRATION, DEFAULT_REGISTRATION, LEGACY_REGISTRATION,
)
from tests.test_adaptive_rag_pipeline import failure_diag
from tests.test_controller_current import FictionalFeatures, ForcedSelection

ROOT = Path(__file__).resolve().parents[1]


class ProductionDefaultTests(unittest.TestCase):
    def test_fresh_linux_style_checkout_preserves_runtime_hashes(self):
        # Copy the registration dependency closure, then exercise Git's actual
        # clean/checkout filters with autocrlf disabled (as on a Linux host).
        queue = [DEFAULT_REGISTRATION, 'configs/splits/team_nq_data_v1.json']
        files = set()

        def references(value):
            if isinstance(value, str):
                if (value.split('/', 1)[0] in {'configs', 'artifacts', 'baselines', 'data', 'src', 'scripts', 'models', 'results'}
                        and (ROOT / value).is_file()):
                    queue.append(value)
            elif isinstance(value, list):
                for child in value:
                    references(child)
            elif isinstance(value, dict):
                for key, child in value.items():
                    references(key); references(child)

        while queue:
            name = queue.pop()
            if name in files:
                continue
            files.add(name)
            if name.endswith('.json'):
                references(json.loads((ROOT / name).read_text()))
        with tempfile.TemporaryDirectory() as tmp:
            repo, checkout = Path(tmp) / 'repo', Path(tmp) / 'checkout'
            repo.mkdir(); checkout.mkdir()
            for name in files | {'.gitattributes'}:
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, path)
            git = ['git', '-c', 'core.autocrlf=false', '-C', str(repo)]
            subprocess.run([*git, 'init', '-q'], check=True, capture_output=True)
            subprocess.run([*git, 'add', '--all'], check=True, capture_output=True)
            subprocess.run([*git, 'checkout-index', '--all', '--prefix=' + str(checkout) + '/'],
                           check=True, capture_output=True)
            for name in files:
                self.assertEqual((checkout / name).read_bytes(), (ROOT / name).read_bytes(), name)
            receipt = current.readiness(checkout)
            self.assertEqual(receipt['missing'], [])
            self.assertTrue(receipt['ready_for_gpu'])
            self.assertEqual(receipt['registration'], DEFAULT_REGISTRATION)

    def test_python_and_cli_default_to_the_same_complete_product(self):
        for function in (current.readiness, current.wire_latest_pipeline, current.load_current_pipeline):
            self.assertEqual(inspect.signature(function).parameters['registration'].default, DEFAULT_REGISTRATION)
        args = runner.parse(['--role', 'confirm', '--queries', 'q.jsonl', '--output-dir', 'out'])
        self.assertEqual(args.registration, DEFAULT_REGISTRATION)
        config = json.loads((ROOT / args.registration).read_text())
        self.assertTrue(config['p3_bm25'] and config['answer_selection'])
        self.assertEqual(set(config['failure_type_routes'].values()), {'evidence_insufficiency'})
        self.assertEqual(config['c1_prefill'], 'full')
        self.assertEqual(config['answer_selection']['chosen'], {'theta': 0.0, 'tau': 0.8, 'decision': 'selector'})

    def test_readiness_cli_defaults_to_latest_but_accepts_explicit_history(self):
        for args, expected in (([], DEFAULT_REGISTRATION), (['--registration', LEGACY_REGISTRATION], LEGACY_REGISTRATION)):
            with patch.object(check, 'readiness', return_value={'ready_for_gpu': True}) as readiness, redirect_stdout(io.StringIO()):
                self.assertEqual(check.main(args), 0)
            readiness.assert_called_once_with(check.ROOT, expected)

    def test_default_factory_loads_bm25_and_attaches_answer_selection(self):
        config = json.loads((ROOT / DEFAULT_REGISTRATION).read_text())
        backend, selection, result = Mock(), Mock(), object()
        provider = SimpleNamespace(runtime_receipt=lambda: {})
        with patch.object(current, 'readiness', return_value={'ready_for_gpu': True}), \
             patch('src.controller.providers.load_pipeline', return_value=(backend, {})) as load, \
             patch('src.controller.p60_online_features.build_p60_online_feature_provider', return_value=provider), \
             patch.object(current, 'load_answer_selection', return_value=selection) as load_selection, \
             patch.object(current, 'TeamNQBoundary'), \
             patch.object(current, 'wire_latest_pipeline', return_value=result) as wire:
            pipeline, environment = current.load_current_pipeline(attach_reference_rag=False)
        self.assertIs(pipeline, result)
        self.assertEqual(load.call_args.kwargs['p3_bm25'], config['p3_bm25'])
        load_selection.assert_called_once_with(config['answer_selection'], backend, ROOT)
        self.assertEqual(wire.call_args.kwargs['registration'], DEFAULT_REGISTRATION)
        self.assertIs(wire.call_args.kwargs['answer_selection'], selection)
        self.assertEqual(environment['answer_selection'], config['answer_selection'])

    def test_default_wiring_cannot_silently_omit_answer_selection(self):
        config = json.loads((ROOT / DEFAULT_REGISTRATION).read_text())
        backend = smoke_pipeline(ExecutionConfig(**config['execution_config']))
        with self.assertRaisesRegex(ValueError, 'requires answer selection'):
            current.wire_latest_pipeline(backend, FictionalFeatures())

    def test_default_B_route_and_post_controller_selection_are_both_executed(self):
        config = json.loads((ROOT / DEFAULT_REGISTRATION).read_text())
        label = DiagnosisLabel.EVIDENCE_UTILISATION
        backend = smoke_pipeline(ExecutionConfig(**config['execution_config']))
        backend.classifier.diagnose = lambda _: failure_diag(label)
        stage = Mock(side_effect=lambda r: replace(r, final_answer='Selected answer'))
        pipeline = current.wire_latest_pipeline(backend, FictionalFeatures(),
            selection_adapter=ForcedSelection(label), answer_selection=stage)
        result = pipeline.run_from_item(smoke_inputs()[0])
        self.assertEqual(result.controller_decision.action.value, 'retrieval_enhancement')
        self.assertEqual(result.final_answer, 'Selected answer')
        self.assertEqual(result.metadata['baseline'], 'quality_adaptive_v1')
        stage.assert_called_once()

    def test_historical_runners_explicitly_pin_their_registered_backend(self):
        for name in ('precompute_quality_sweep.py', 'precompute_p3_coverage.py', 'l2_generate.py', 'smoke_production_e2e.py'):
            tree = ast.parse((ROOT / 'scripts' / name).read_text())
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                     and n.func.id == 'load_current_pipeline']
            self.assertEqual(len(calls), 1, name)
            choice = next(k.value for k in calls[0].keywords if k.arg == 'registration')
            self.assertEqual(ast.literal_eval(choice) if isinstance(choice, ast.Constant) else choice.id,
                             'LEGACY_REGISTRATION', name)
        tree = ast.parse((ROOT / 'baselines/run_product_evaluation.py').read_text())
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == 'load_current_pipeline')
        choice = next(k.value for k in call.keywords if k.arg == 'registration')
        self.assertEqual(ast.unparse(choice), "cfg['production']['baseline_registration']")


class ResumeConfigurationTests(unittest.TestCase):
    def binding(self, registration):
        return {'registration': registration, 'registration_sha256': 'registered-bytes', 'query_sha256': 'query-bytes',
                'role': 'confirm', 'data_role': 'validation', 'batch_size': 8, 'shard': 0, 'shards': 1}

    def test_historical_outputs_require_the_original_explicit_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for original in (B_REGISTRATION, BM25_REGISTRATION):
                binding = self.binding(original)
                (root / 'manifest.json').write_text(json.dumps(binding))
                (root / 'results.jsonl').write_bytes(b'{"sample_id":"one"}\n{"partial":')
                before = {p.name: p.read_bytes() for p in root.iterdir()}
                with self.assertRaisesRegex(ValueError, 'Resume configuration mismatch: registration'):
                    runner.validate_resume(root, self.binding(DEFAULT_REGISTRATION))
                self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
                runner.validate_resume(root, binding)

    def test_same_path_with_changed_bytes_or_changed_queries_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding = self.binding(DEFAULT_REGISTRATION)
            (root / 'manifest.json').write_text(json.dumps(binding))
            for field in ('registration_sha256', 'query_sha256', 'batch_size', 'role', 'shards'):
                with self.assertRaisesRegex(ValueError, field):
                    runner.validate_resume(root, {**binding, field: 'changed'})

    def test_results_without_a_manifest_are_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner.validate_resume(root, self.binding(DEFAULT_REGISTRATION))
            path = root / 'results.jsonl'
            path.write_bytes(b'{"sample_id":"one"}\n')
            with self.assertRaisesRegex(ValueError, 'no manifest'):
                runner.validate_resume(root, self.binding(DEFAULT_REGISTRATION))
            self.assertEqual(path.read_bytes(), b'{"sample_id":"one"}\n')

    def test_cli_rejects_mixed_resume_before_model_load_or_output_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries = root / 'queries.jsonl'
            queries.write_text('{"sample_id":"fiction","query":"Where?"}\n')
            output = root / 'run'; output.mkdir()
            (output / 'manifest.json').write_text(json.dumps({'registration': B_REGISTRATION}))
            (output / 'results.jsonl').write_bytes(b'{"unfinished":')
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with patch.object(runner, 'expected_ids', return_value=['fiction']), \
                 patch.object(current, 'load_current_pipeline') as load:
                with self.assertRaisesRegex(ValueError, 'Resume configuration mismatch'):
                    runner.main(['--role', 'confirm', '--queries', str(queries), '--output-dir', str(output)])
                load.assert_not_called()
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})


if __name__ == '__main__':
    unittest.main()
