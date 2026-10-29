"""Quality round: registered Validation split, full-prefill C1 and the sweep pass's candidate actions."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from scripts import build_quality_sweep_split as split_builder
from scripts import precompute_quality_sweep as sweep
from src.controller.schema import SelectionDecision, SelectionLane
from src.data.team_nq import TeamNQBoundary
from src.diagnosis.labels import DiagnosisLabel
import src.pipeline.reference_rag as reference_rag

ROOT = Path(__file__).resolve().parents[1]


def selection(label, needs=False):
    return SelectionDecision(needs_enhancement=needs, failure_type=label, selection_score=0.0,
                             selection_lane=SelectionLane.NO_ACTION, confidence=0.6, p_fail=0.7,
                             p_insuff=0.1, sample_id='s1')


def diagnosis(label, p_fail=0.7, stage1='failure'):
    return {'final_label': label, 'stage1_label': stage1, 'stage1_failure_probability': p_fail}


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registration = json.loads((ROOT / sweep.REGISTRATION).read_text(encoding='utf-8'))
        cls.boundary = TeamNQBoundary(ROOT)
        cls.roles = split_builder.split(cls.boundary)

    def test_registered_lists_are_the_continued_ordering(self):
        for role in ('sweep', 'confirm'):
            spec = self.registration['split'][role]
            data = (ROOT / spec['path']).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), spec['sha256'])
            self.assertEqual(data, split_builder.ids_bytes(self.roles[role]))
            self.assertEqual(json.loads(data), self.roles[role])
            self.assertEqual(len(self.roles[role]), spec['n'])

    def test_roles_are_disjoint_validation_and_skip_development_and_selection(self):
        sweep_ids, confirm = set(self.roles['sweep']), set(self.roles['confirm'])
        used = set(self.roles['development_selection'])
        self.assertEqual(len(used), 1000)
        self.assertFalse(sweep_ids & confirm or sweep_ids & used or confirm & used)
        self.assertEqual(sweep_ids | confirm | used, set(self.boundary.allowed_ids('validation')))
        self.boundary.require(sorted(sweep_ids | confirm), 'validation')
        self.assertFalse(any(self.boundary.membership[sid] == 'test' for sid in sweep_ids | confirm))

    def test_loader_checks_the_registered_hash(self):
        registration, ids = sweep.load_registration()
        self.assertEqual(ids, self.roles['sweep'])
        self.assertEqual(registration['precompute']['correct_p_fail_cut'], 0.39)


class QueryTests(unittest.TestCase):
    def test_query_file_must_be_query_only_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'q.jsonl'
            path.write_text('{"sample_id": "a", "query": "who"}\n', encoding='utf-8')
            self.assertEqual(sweep.read_queries(path, ['a']), [('a', 'who')])
            with self.assertRaises(ValueError):
                sweep.read_queries(path, ['a', 'b'])
            path.write_text('{"sample_id": "a", "query": "who", "gold": ["x"]}\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                sweep.read_queries(path, ['a'])

    def test_interrupted_last_line_is_dropped_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'records.jsonl'
            path.write_bytes(b'{"sample_id": "a"}\n{"sample_id": "b", "act')
            self.assertEqual(sweep.done_ids(path), {'a'})
            self.assertEqual(path.read_bytes(), b'{"sample_id": "a"}\n')


class ActionTests(unittest.TestCase):
    def test_candidates_per_label(self):
        self.assertEqual(sweep.actions_for(diagnosis('evidence_insufficiency'), 0.39), ['retrieval_enhancement'])
        self.assertEqual(sweep.actions_for(diagnosis('evidence_accessibility'), 0.39),
                         ['retrieval_enhancement', 'evidence_refinement'])
        self.assertEqual(sweep.actions_for(diagnosis('evidence_utilisation'), 0.39),
                         ['retrieval_enhancement', 'generation_repair'])
        self.assertEqual(sweep.actions_for(diagnosis('correct', 0.40, 'correct'), 0.39), ['retrieval_enhancement'])
        self.assertEqual(sweep.actions_for(diagnosis('correct', 0.20, 'correct'), 0.39), [])
        self.assertEqual(sweep.actions_for(diagnosis('evidence_utilisation', 0.45, 'correct'), 0.39),
                         ['retrieval_enhancement'])

    def test_forced_decision_routes_any_selection_to_the_requested_action(self):
        for label in DiagnosisLabel:
            for action in sweep.LABEL_FOR_ACTION:
                decision = sweep.forced_decision(selection(label), action)
                self.assertEqual(decision.action.value, action)
                self.assertEqual(decision.diagnosis_label.value, sweep.LABEL_FOR_ACTION[action])
                self.assertEqual(decision.sample_id, 's1')

    def test_compare_reports_drift(self):
        base = {'initial_answer': 'x', 'diagnosis': {'final_label': 'correct', 'stage1_failure_probability': 0.2},
                'frozen_selection': {'needs_enhancement': False}, 'features': [0.0, 1.0]}
        other = {**base, 'initial_answer': 'y', 'features': [0.0, 1.5]}
        gap = sweep.compare(base, other)
        self.assertFalse(gap['c1_same'])
        self.assertTrue(gap['label_same'] and gap['selected_same'])
        self.assertEqual(gap['max_feature_gap'], 0.5)


class FullPrefillTests(unittest.TestCase):
    def test_chunking_is_cleared_for_c1_and_restored(self):
        config = SimpleNamespace(prefill_chunk_size=256)
        model = SimpleNamespace(generation_config=config)
        seen = []
        original = reference_rag._default_generate
        try:
            reference_rag._default_generate = lambda t, m, p, n: (seen.append(m.generation_config.prefill_chunk_size), ('ans', 0.1))[1]
            self.assertEqual(reference_rag.full_prefill_generate(None, model, 'p', 64), ('ans', 0.1))
            self.assertEqual((seen, config.prefill_chunk_size), ([None], 256))

            def boom(*_):
                raise RuntimeError('oom')
            reference_rag._default_generate = boom
            with self.assertRaises(RuntimeError):
                reference_rag.full_prefill_generate(None, model, 'p', 64)
            self.assertEqual(config.prefill_chunk_size, 256)
        finally:
            reference_rag._default_generate = original


if __name__ == '__main__':
    unittest.main()
