"""Latest Diagnosis must route through P60/P64 and execute real module code."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io

import numpy as np

from scripts.run_controller_integration import smoke_inputs, smoke_pipeline
from src.controller.execution import ExecutionConfig
from src.controller.schema import SelectionDecision, SelectionLane
from src.controller.selection import FrozenOnlineSelectionAdapter
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
from src.diagnosis.labels import DiagnosisLabel as Label
from src.pipeline.current import (REGISTRATION, CheckedFeatures, MeteredEnhancementAdapter,
                                  load_current_pipeline, provenance_issues, readiness, wire_latest_pipeline)
from src.controller.execution import fingerprint
from tests.test_adaptive_rag_pipeline import failure_diag

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / REGISTRATION).read_text())
NAMES = json.loads((ROOT / CONFIG['selector_config']).read_text())['feature_names']


class FictionalFeatures:
    """Only for controlled tests; never registered as the production provider."""
    feature_names = tuple(NAMES)

    def __init__(self, values=None):
        self.values = np.arange(78) / 100 if values is None else values

    def features_for(self, item, diagnosis):
        return self.values


class ForcedSelection:
    def __init__(self, label, selected=True):
        self.label, self.selected = label, selected
        self.calls = 0

    def select(self, diagnosis, features, *, sample_id=None):
        self.calls += 1
        self.last_features = features.copy()
        return SelectionDecision(needs_enhancement=self.selected, failure_type=self.label,
            selection_score=.8, selection_lane=SelectionLane.INSUFF_PRESERVE if self.selected else SelectionLane.NO_ACTION,
            confidence=.8, p_fail=.8, p_insuff=.6, sample_id=sample_id)


class CurrentIntegrationTests(unittest.TestCase):
    def backend(self, label=Label.EVIDENCE_INSUFFICIENCY):
        backend = smoke_pipeline(ExecutionConfig(**CONFIG['execution_config']))
        backend.classifier.diagnose = lambda item: failure_diag(label)
        backend.run = lambda *args, **kwargs: self.fail('Old Diagnosis/Probe/routing loop was called')
        return backend

    def run_route(self, label):
        backend = self.backend(label)
        selector = ForcedSelection(label)
        pipeline = wire_latest_pipeline(backend, FictionalFeatures(), selection_adapter=selector)
        result = pipeline.run_from_item(smoke_inputs()[0])
        self.assertFalse(result.controller_decision.used_probe)
        self.assertEqual(result.metadata['baseline'], CONFIG['baseline'])
        self.assertTrue(all(e['stage'] != 'probe' for e in result.enhancement_result.metadata.get('calls', [])))
        return result

    def test_registry_connects_real_retrieval_module_and_keeps_metering(self):
        result = self.run_route(Label.EVIDENCE_INSUFFICIENCY)
        native = result.enhancement_result.metadata
        self.assertEqual(result.controller_decision.action.value, 'retrieval_enhancement')
        self.assertEqual(native['adapter'], 'metered_real_modules_1024_v1')
        self.assertEqual(native['enhancement_cost']['retrieval_calls'], 1)
        self.assertEqual(native['enhancement_cost']['scoring_calls'], 1)

    def test_registry_connects_real_reranker_and_regeneration(self):
        result = self.run_route(Label.EVIDENCE_ACCESSIBILITY)
        cost = result.enhancement_result.metadata['enhancement_cost']
        self.assertEqual(result.controller_decision.action.value, 'evidence_refinement')
        self.assertEqual((cost['scoring_calls'], cost['retrieval_calls'], cost['llm_calls']), (1, 0, 1))
        self.assertEqual(result.final_answer, 'Amber')

    def test_E_stays_unavailable_without_model_calls_or_fabricated_verification(self):
        result = self.run_route(Label.EVIDENCE_UTILISATION)
        self.assertEqual(result.status.value, 'enhancement_unavailable')
        self.assertEqual(result.final_answer, result.initial_answer)
        self.assertEqual(result.metadata['enhancement_cost']['llm_calls'], 0)
        self.assertEqual(result.enhancement_result.metadata['native']['module_trace']['support_check']['original_verification'],
                         'incomplete')

    def test_no_action_never_executes_a_module_or_legacy_probe(self):
        backend = self.backend()
        selector = ForcedSelection(Label.EVIDENCE_INSUFFICIENCY, selected=False)
        pipeline = wire_latest_pipeline(backend, FictionalFeatures(), selection_adapter=selector)
        with patch.object(backend, 'execute', side_effect=AssertionError('must not execute')):
            result = pipeline.run_from_item(smoke_inputs()[0])
        self.assertEqual(result.status.value, 'no_action')
        self.assertEqual(result.metadata['enhancement_cost']['llm_calls'], 0)
        self.assertEqual(result.metadata['feature_provider']['cost_state'], 'not_accounted')

    def test_real_frozen_P60_assets_are_the_default_selector(self):
        pipeline = wire_latest_pipeline(self.backend(), FictionalFeatures())
        self.assertIsInstance(pipeline.host.selection_adapter, FrozenOnlineSelectionAdapter)
        self.assertEqual(pipeline.host.selection_adapter.version, 'p60_online_adapter_v1')
        self.assertFalse(pipeline.host.selection_adapter.probe_enabled)
        self.assertEqual(pipeline.host.selection_adapter.n_features, 78)
        self.assertEqual(len(pipeline.host.executor.registered_actions()), 3)

    def test_all_78_feature_positions_are_forwarded_without_transform(self):
        selector = ForcedSelection(Label.EVIDENCE_INSUFFICIENCY, selected=False)
        provider = FictionalFeatures()
        pipeline = wire_latest_pipeline(self.backend(), provider, selection_adapter=selector)
        pipeline.run_from_item(smoke_inputs()[0])
        np.testing.assert_array_equal(selector.last_features, provider.values)
        self.assertEqual((len(NAMES), len(set(NAMES))), (78, 75))

    def test_missing_or_deduplicated_feature_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'required'):
            CheckedFeatures(None, NAMES)
        provider = FictionalFeatures()
        provider.feature_names = tuple(dict.fromkeys(NAMES))
        with self.assertRaisesRegex(ValueError, 'all 78'):
            CheckedFeatures(provider, NAMES)

    def test_bad_feature_values_fail_before_selection(self):
        for values in (np.zeros(77), np.zeros(79), np.full(78, np.nan)):
            selector = ForcedSelection(Label.EVIDENCE_INSUFFICIENCY)
            pipeline = wire_latest_pipeline(self.backend(), FictionalFeatures(values), selection_adapter=selector)
            with self.assertRaisesRegex(ValueError, '78 finite'):
                pipeline.run_from_item(smoke_inputs()[0])
            self.assertEqual(selector.calls, 0)

    def test_latest_factory_does_not_silently_reenable_E_defaults(self):
        backend = self.backend()
        backend.config = ExecutionConfig()
        with self.assertRaisesRegex(ValueError, 'E stays unavailable'):
            wire_latest_pipeline(backend, FictionalFeatures())

    def test_operational_failure_retains_answer_and_reports_actual_budget_block(self):
        backend = self.backend(Label.EVIDENCE_ACCESSIBILITY)
        backend.config = replace(backend.config, max_llm_calls=0)
        from src.controller.controller import decide
        selection = ForcedSelection(Label.EVIDENCE_ACCESSIBILITY).select(None, np.arange(78), sample_id='fiction')
        from src.compensation.schema import EnhancementRequest
        item = replace(smoke_inputs()[0], sample_id='fiction')
        request = EnhancementRequest.from_runtime(query=item.query, initial_answer=item.initial_answer,
            evidence=item.evidence, controller_decision=decide(selection), sample_id=item.sample_id, dataset='nq')
        result = MeteredEnhancementAdapter(backend).enhance(request)
        self.assertEqual(result.status.value, 'error')
        self.assertFalse(result.changed_answer)
        self.assertEqual(result.metadata['enhancement_cost']['budget_blocks'], 1)

    def test_production_stops_before_GPU_load_when_original_deliverables_missing(self):
        report = readiness()
        self.assertEqual(report['upstream_commit'], '138d52aa22e0eee78637e01f981a030d7ad388df')
        self.assertFalse(report['ready_for_gpu'])
        self.assertFalse(report['legacy_fallback'])
        self.assertIn('runtime_78_feature_provider_not_delivered', report['missing'])
        self.assertIn('P60_training_tuning_calibration_ID_provenance_not_delivered', report['missing'])
        with patch('src.controller.providers.load_pipeline', side_effect=AssertionError('GPU must not load')):
            with self.assertRaisesRegex(RuntimeError, 'not ready'):
                load_current_pipeline()

    def test_branch_audit_binds_exact_upstream_models_instead_of_substituting_weights(self):
        import hashlib
        report = json.loads((ROOT/'artifacts/controller/diagnosis_1024_v1/upstream_material_audit.json').read_text())
        self.assertEqual(report['commit'], CONFIG['upstream_commit'])
        self.assertEqual(report['tracked_files'], 142)
        self.assertFalse(report['tree_truncated'])
        self.assertEqual(report['feature_provider_definition_files'],
                         ['src/pipeline/adaptive_rag.py', 'tests/test_adaptive_rag_pipeline.py'])
        for row in report['file_tree']:
            if row['path'].startswith('models/controller/p60_online_adapter/'):
                raw = (ROOT/row['path']).read_bytes()
                self.assertEqual(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(), row['sha'])

    def test_asset_hash_mismatch_blocks_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {**CONFIG, 'frozen_assets': {'missing_model.joblib': 'wrong'}}
            target = root / REGISTRATION
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(config))
            self.assertIn('asset_missing_or_changed:missing_model.joblib', readiness(root)['missing'])

    def test_current_CLI_saves_missing_deliverables_without_starting_legacy_run(self):
        from scripts.check_controller_current import main
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / 'readiness.json'
            self.assertEqual(main(['--output', str(output)]), 2)
            report = json.loads(output.read_text())
            self.assertFalse(report['legacy_fallback'])
            self.assertEqual(report['baseline'], CONFIG['baseline'])
            with self.assertRaises(FileExistsError):
                main(['--output', str(output)])

    def test_production_requires_current_input_bound_feature_costs(self):
        selector = ForcedSelection(Label.EVIDENCE_INSUFFICIENCY)
        pipeline = wire_latest_pipeline(self.backend(), FictionalFeatures(), selection_adapter=selector,
                                       require_feature_costs=True)
        with self.assertRaisesRegex(ValueError, 'input-bound'):
            pipeline.run_from_item(smoke_inputs()[0])
        self.assertEqual(selector.calls, 0)

    def test_explicit_zero_feature_calls_are_recorded_without_inventing_costs(self):
        provider = FictionalFeatures()
        item = smoke_inputs()[0]
        provider.runtime_receipt = lambda: {'input_sha256': fingerprint(asdict(item)), 'calls': []}
        selector = ForcedSelection(Label.EVIDENCE_INSUFFICIENCY, selected=False)
        pipeline = wire_latest_pipeline(self.backend(), provider, selection_adapter=selector,
                                       require_feature_costs=True)
        result = pipeline.run_from_item(item)
        self.assertEqual(result.metadata['feature_provider']['cost_state'], 'reported_by_provider')
        self.assertEqual(result.metadata['feature_provider']['provider_cost_receipt']['calls'], [])

    def test_provenance_checks_raw_per_model_IDs_against_entire_reserve_and_selection(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'artifacts/controller/data_v1'
            data.mkdir(parents=True)
            (data/'evaluation_reserve_ids.json').write_text(json.dumps({'nq':['protected'], 'fever':[]}))
            (data/'selection_ids.json').write_text(json.dumps({'nq':{'selection':['selected']}, 'fever':{'selection':[]}}))
            models = {p:h for p,h in CONFIG['frozen_assets'].items() if p.startswith('models/controller/p60_online_adapter/')}
            rows = [{'model':p, 'dataset':'nq', 'sample_id':'training-only', 'role':'training'} for p in models]
            payload = {'complete':True, 'model_sha256':models, 'records':rows}
            path = root / 'provenance.json'
            def check():
                path.write_text(json.dumps(payload))
                spec = {'path':'provenance.json','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                return provenance_issues(root, spec, CONFIG)
            self.assertEqual(check(), [])
            for forbidden in ('protected', 'selected'):
                payload['records'][0]['sample_id'] = forbidden
                self.assertIn('P60_training_or_tuning_overlaps_protected_or_selection_IDs', check())
            payload['records'][0]['sample_id'] = 'training-only'
            payload['records'] = payload['records'][:1]
            self.assertIn('P60_provenance_does_not_cover_all_three_models', check())


if __name__ == '__main__':
    unittest.main()
