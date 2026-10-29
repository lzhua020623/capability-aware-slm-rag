"""Quality round production wiring: label routes, the separate registration, finalize and local scoring."""
import json
from pathlib import Path
import tempfile
import unittest

from src.compensation.executor import EnhancementExecutor
from src.compensation.mocks import (
    MockEvidenceRefinementEnhancement, MockGenerationRepairEnhancement, MockRetrievalEnhancement,
)
from src.controller.schema import ControllerAction, SelectionDecision, SelectionLane
from src.diagnosis.labels import DiagnosisLabel
from src.pipeline import build_adaptive_pipeline
from src.pipeline.current import REGISTRATION, readiness
from scripts import finalize_quality_config as finalize
from scripts import run_quality_eval as runner
from scripts import score_quality_eval as scorer
from tests.test_adaptive_rag_pipeline import StubClassifier, StubFeatures, StubSelection, failure_diag, make_item

ROOT = Path(__file__).resolve().parents[1]
UTIL = DiagnosisLabel.EVIDENCE_UTILISATION


def pipeline(routes, needs=True, gate=None, diagnosis=None, label=UTIL):
    executor = EnhancementExecutor()
    executor.register(ControllerAction.RETRIEVAL_ENHANCEMENT, MockRetrievalEnhancement())
    executor.register(ControllerAction.EVIDENCE_REFINEMENT, MockEvidenceRefinementEnhancement())
    executor.register(ControllerAction.GENERATION_REPAIR, MockGenerationRepairEnhancement())
    selection = SelectionDecision(needs_enhancement=needs, failure_type=label, selection_score=1.0,
                                  selection_lane=SelectionLane.RESIDUAL_COMBO if needs else SelectionLane.NO_ACTION,
                                  confidence=0.6, p_fail=0.8, p_insuff=0.1, sample_id='syn-1')
    return build_adaptive_pipeline(classifier=StubClassifier(diagnosis or failure_diag(UTIL)),
                                   selection_adapter=StubSelection(selection), feature_provider=StubFeatures(),
                                   executor=executor, failure_type_routes=routes, correct_gate=gate)


def correct_diag(p_fail):
    from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
    return HierarchicalDiagnosis(p_fail, 'correct', None, DiagnosisLabel.CORRECT, 1 - p_fail, None)


GATE = {'stage1_p_fail_at_least': 0.39, 'route': 'evidence_insufficiency'}


class RouteTests(unittest.TestCase):
    def test_routes_change_the_action_of_selected_items_only(self):
        item = make_item()
        self.assertEqual(pipeline(None).run_from_item(item).controller_decision.action, ControllerAction.GENERATION_REPAIR)
        routed = pipeline({'evidence_utilisation': 'evidence_insufficiency'}).run_from_item(item)
        self.assertEqual(routed.controller_decision.action, ControllerAction.RETRIEVAL_ENHANCEMENT)
        self.assertEqual(routed.diagnosis.final_label, UTIL)
        self.assertEqual(routed.metadata['failure_type_route'], 'evidence_insufficiency')
        off = pipeline({'evidence_utilisation': 'no_action'}).run_from_item(item)
        self.assertEqual((off.controller_decision.action, off.final_answer), (ControllerAction.NO_ACTION, item.initial_answer))
        self.assertTrue(off.selection_decision.needs_enhancement)
        unselected = pipeline({'evidence_utilisation': 'evidence_insufficiency'}, needs=False).run_from_item(item)
        self.assertEqual((unselected.controller_decision.action, unselected.metadata['failure_type_route']),
                         (ControllerAction.NO_ACTION, None))
        for bad in ({'correct': 'evidence_insufficiency'}, {'evidence_utilisation': 'generation_repair'}):
            with self.assertRaises(ValueError):
                pipeline(bad)

    def test_correct_gate_sends_high_p_fail_correct_items_to_p3_only(self):
        item = make_item()
        opened = pipeline(None, needs=False, gate=GATE, diagnosis=correct_diag(0.45), label=DiagnosisLabel.CORRECT).run_from_item(item)
        self.assertEqual(opened.controller_decision.action, ControllerAction.RETRIEVAL_ENHANCEMENT)
        self.assertTrue(opened.metadata['correct_gate_opened'])
        self.assertFalse(opened.selection_decision.needs_enhancement)
        for p_fail, gate in ((0.2, GATE), (0.45, None)):
            closed = pipeline(None, needs=False, gate=gate, diagnosis=correct_diag(p_fail),
                              label=DiagnosisLabel.CORRECT).run_from_item(item)
            self.assertEqual((closed.controller_decision.action, closed.metadata['correct_gate_opened']),
                             (ControllerAction.NO_ACTION, False))
        unselected_failure = pipeline(None, needs=False, gate=GATE).run_from_item(item)
        self.assertEqual(unselected_failure.controller_decision.action, ControllerAction.NO_ACTION)
        for bad in ({'stage1_p_fail_at_least': 1.5, 'route': 'evidence_insufficiency'},
                    {'stage1_p_fail_at_least': 0.4, 'route': 'no_action'}, {'route': 'evidence_insufficiency'}):
            with self.assertRaises(ValueError):
                pipeline(None, gate=bad)

    def test_a_separate_registration_is_checked_like_the_original(self):
        base = json.loads((ROOT / REGISTRATION).read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'reg.json'
            path.write_text(json.dumps({**base, 'baseline': 'x', 'c1_prefill': 'full'}))
            self.assertEqual(readiness(ROOT, path)['missing'], readiness(ROOT)['missing'])
            broken = {**base, 'frozen_assets': {**base['frozen_assets'], 'configs/controller/p60_quality_v1.json': '0' * 64}}
            path.write_text(json.dumps(broken))
            self.assertIn('asset_missing_or_changed:configs/controller/p60_quality_v1.json', readiness(ROOT, path)['missing'])


class FinalizeTests(unittest.TestCase):
    def test_committed_quality_registration_is_ready_and_matches_the_report(self):
        reg = json.loads((ROOT / finalize.REGISTRATION_OUT).read_text(encoding='utf-8'))
        report = json.loads((ROOT / 'artifacts/controller/quality_sweep_v1/offline_report.json').read_text())
        self.assertTrue(readiness(ROOT, finalize.REGISTRATION_OUT)['ready_for_gpu'])
        self.assertEqual(reg['quality_sweep']['chosen'], report['chosen'])
        self.assertEqual(reg['quality_sweep']['offline_report_sha256'],
                         finalize.sha(ROOT / 'artifacts/controller/quality_sweep_v1/offline_report.json'))
        self.assertEqual((report['partial'], report['metric'], report['records']), (False, 'net', 6000))
        self.assertEqual(set(reg['failure_type_routes'].values()), {'evidence_insufficiency'})
        self.assertEqual(reg['correct_gate'], GATE)
        p60 = json.loads((ROOT / finalize.P60_OUT).read_text(encoding='utf-8'))
        self.assertEqual((p60['coverage_target'], p60['calibration']['failure_pool_n']), (1.0, 2078))
        self.assertEqual(reg['data_boundary'], json.loads((ROOT / REGISTRATION).read_text())['data_boundary'])

    def test_action_maps_become_routes(self):
        sweep = json.loads((ROOT / finalize.SWEEP).read_text(encoding='utf-8'))
        maps = sweep['grid']['action_maps']
        self.assertEqual(finalize.routes_for(maps['frozen']), {'evidence_insufficiency': 'evidence_insufficiency',
            'evidence_accessibility': 'evidence_accessibility', 'evidence_utilisation': 'evidence_utilisation'})
        self.assertEqual(set(finalize.routes_for(maps['all_p3']).values()), {'evidence_insufficiency'})
        self.assertEqual(finalize.routes_for(maps['e_off'])['evidence_utilisation'], 'no_action')

    def test_new_config_keeps_models_and_features_and_registration_keeps_the_original(self):
        base = json.loads((ROOT / 'configs/controller/p60_online_adapter.json').read_text())
        chosen = {'coverage': 0.75, 'insuff_share': 0.25, 'action_map': 'all_p3', 'correct_gate': 'closed'}
        cal = {'t_insuff': 0.3, 't_combo': -0.2, 'z_stats': {'p_fail': {'mean': 0.7, 'std': 0.1},
               'rec': {'mean': 0.1, 'std': 0.1}, 'p_corr': {'mean': 0.4, 'std': 0.3}}}
        new = finalize.p60_config(base, chosen, cal, 2000)
        for key in ('models', 'feature_names', 'd_combo', 'policy', 'p_corr_class_index'):
            self.assertEqual(new[key], base[key])
        self.assertEqual((new['thresholds'], new['coverage_target']), ({'t_insuff': 0.3, 't_combo': -0.2}, 0.75))
        reg = finalize.registration_for(json.loads((ROOT / REGISTRATION).read_text()), chosen, 'abc')
        self.assertEqual((reg['selector_config'], reg['c1_prefill'], reg['correct_gate']), (finalize.P60_OUT, 'full', None))
        opened = finalize.registration_for(json.loads((ROOT / REGISTRATION).read_text()),
                                           {**chosen, 'correct_gate': 'p_fail_at_least_0.39_to_p3'}, 'abc')
        self.assertEqual(opened['correct_gate'], GATE)
        self.assertEqual(reg['data_boundary'], json.loads((ROOT / REGISTRATION).read_text())['data_boundary'])
        self.assertIn(finalize.P60_OUT, reg['frozen_assets'])


def result(sid, initial, final, action='retrieval_enhancement'):
    evidence = [{'rank': 1, 'passage_id': 'p1', 'text': 'Paris is the capital', 'score': 0.9}]
    return {'sample_id': sid, 'query': 'q', 'initial_answer': initial, 'final_answer': final,
            'initial_evidence': evidence, 'final_evidence': evidence,
            'diagnosis': {'final_label': 'evidence_utilisation'}, 'selection_decision': {'needs_enhancement': True},
            'controller_decision': {'action': action}, 'enhancement_result': {'status': 'success'},
            'metadata': {'failure_type_route': 'evidence_insufficiency'}}


class ScoringTests(unittest.TestCase):
    def test_prediction_rows_match_the_team_fields(self):
        row = scorer.prediction_row(result('a', 'Lyon', 'Paris'), ['Paris'])
        self.assertEqual((row['prediction'], row['em'], row['normalized_match'], row['answer_in_context']), ('Paris', 1, 1, 1))
        self.assertEqual((row['controller_action'], row['selected'], row['mode']), ('retrieval_enhancement', True, 'adaptive_product'))

    def test_judge_inputs_skip_prior_strings_and_score_counts_net(self):
        rows = [scorer.prediction_row(result('a', 'Lyon', 'Paris'), ['Paris']),
                scorer.prediction_row(result('b', 'X', 'X', 'no_action'), ['Y'])]
        prior = {('b', 'X'): False}
        todo = scorer.judge_rows(rows, prior)
        self.assertEqual([(r['source_sample_id'], r['prediction']) for r in todo], [('a', 'Paris'), ('a', 'Lyon')])
        labels = {**prior, ('a', 'Paris'): True, ('a', 'Lyon'): False}
        report = scorer.score(rows, labels)
        self.assertEqual((report['judge_assisted_correct'], report['fixes'], report['harms'], report['net_vs_c1']), (1, 1, 0, 1))
        with self.assertRaises(ValueError):
            scorer.score(rows, prior)

    def test_prior_labels_read_hashed_and_plain_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'j.jsonl'
            rows = [{'sample_id': 'a::0123456789', 'prediction': 'Paris', 'judge_status': 'llm_ok', 'auto_correct': False,
                     'judge_label': 'CORRECT'},
                    {'sample_id': 'b', 'prediction': 'X', 'judge_status': 'auto_correct', 'auto_correct': True, 'judge_label': None},
                    {'sample_id': 'c', 'prediction': 'Z', 'judge_status': 'llm_failed', 'auto_correct': False, 'judge_label': None}]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            self.assertEqual(scorer.prior_labels([path]), {('a', 'Paris'): True, ('b', 'X'): True})


class RunnerTests(unittest.TestCase):
    def test_confirm_ids_are_the_registered_list_and_results_resume_as_a_prefix(self):
        ids = runner.expected_ids('confirm')
        self.assertEqual(len(ids), 2440)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'results.jsonl'
            path.write_bytes(b'{"sample_id": "a"}\n{"sample_id": "b"')
            self.assertEqual(runner.done_ids(path), ['a'])
            q = Path(tmp) / 'q.jsonl'
            q.write_text('{"sample_id": "a", "query": "x", "gold": "y"}\n')
            with self.assertRaises(ValueError):
                runner.read_queries(q)


if __name__ == '__main__':
    unittest.main()
