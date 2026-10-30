"""P3-BM25 in production: the rule on a P3-20 record, its agreement with the offline rule, controller wiring and budgets, and the registration."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts import bm25_gate, l2_offline, run_quality_eval
from src.compensation.retrieval_enhancement import p3_bm25
from src.compensation.retrieval_enhancement.p3 import build_c1_prompt
from src.controller.execution import CallLog, ExecutionConfig, decision_for
from src.controller.schema import ControllerAction
from src.pipeline.current import readiness
from tests.test_controller_integration import Scorer, pipeline

ROOT = Path(__file__).resolve().parents[1]
sha = lambda path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def p3_record(status='answer_replaced', n=20):
    candidates = [{'passage_id': f'b{i}', 'text': f'bge passage {i} about Amber', 'score': 1.0 - i / 100, 'rank': i + 1}
                  for i in range(n)]
    return {'dataset': 'nq', 'query': 'q', 'status': status, 'initial_answer': 'Oslo', 'final_answer': 'Rome',
            'generated_answer': 'Rome', 'reject_reason': None, 'prompt': 'p3 prompt', 'evidence': candidates[:5],
            'candidates': candidates, 'retrieval_calls': 1, 'scoring_calls': 1, 'scored_pairs': 20,
            'generation_calls': 1, 'llm_calls': 1}


class Fakes:
    def __init__(self, hits, scores, answer='Amber'):
        self.hits, self.scores, self.answer, self.prompts = hits, scores, answer, []
        self.corpus = {i: (f'x{i}', f'bm25 passage {i} about Amber') for i in range(100)}

    def rule(self):
        return p3_bm25.P3Bm25(lambda q, k: self.hits[:k], lambda pos: self.corpus[pos],
                              lambda q, texts: [self.scores.get(t, 0.0) for t in texts], self.generate)

    def generate(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class RuleTests(unittest.TestCase):
    def test_a_bm25_passage_in_the_top5_replaces_the_answer(self):
        fakes = Fakes([(3, 7.0), (4, 6.0)], {'bm25 passage 3 about Amber': 9.0}, answer=' Amber ')
        out = fakes.rule().run(p3_record())
        self.assertEqual(out['bm25']['selected'], ['x3', 'b0', 'b1', 'b2', 'b3'])
        self.assertEqual((out['bm25']['bm25_only'], out['bm25']['applied']), (['x3'], True))
        self.assertEqual((out['final_answer'], out['status'], out['policy']), ('Amber', 'answer_replaced', 'P3-BM25'))
        self.assertEqual(fakes.prompts, [build_c1_prompt('q', out['evidence'])])
        self.assertEqual([(e['passage_id'], e['rank'], e['score']) for e in out['evidence'][:2]], [('x3', 1, 7.0), ('b0', 2, 1.0)])
        self.assertEqual(out['p3_20']['final_answer'], 'Rome')
        self.assertEqual((out['retrieval_calls'], out['scoring_calls'], out['llm_calls'], out['scored_pairs']), (2, 2, 2, 42))

    def test_a_rejected_answer_keeps_c1_even_when_p3_20_replaced_it(self):
        fakes = Fakes([(3, 7.0)], {'bm25 passage 3 about Amber': 9.0}, answer='Paris')
        out = fakes.rule().run(p3_record())
        self.assertEqual((out['final_answer'], out['status'], out['reject_reason']),
                         ('Oslo', 'kept_original', 'not_grounded_in_selected_evidence'))

    def test_without_a_bm25_passage_p3_20_stands(self):
        for record in (p3_record(), p3_record('no_change'), p3_record('kept_original')):
            fakes = Fakes([(3, 7.0)], {})
            out = fakes.rule().run(record)
            self.assertEqual({k: v for k, v in out.items() if k != 'bm25'}, record)
            self.assertEqual((out['bm25']['reason'], fakes.prompts), ('no_bm25_passage', []))

    def test_failed_or_unrouted_p3_and_errors_keep_p3_20(self):
        out = Fakes([(3, 7.0)], {'bm25 passage 3 about Amber': 9.0}).rule().run(p3_record('retrieval_failed'))
        self.assertEqual((out['bm25']['reason'], out['final_answer']), ('p3_20_not_completed', 'Rome'))
        out = Fakes([(3, 7.0)], {'bm25 passage 3 about Amber': 9.0}, answer=RuntimeError('gpu')).rule().run(p3_record())
        self.assertEqual((out['bm25']['reason'], out['final_answer'], out['status']), ('generation_failed', 'Rome', 'answer_replaced'))
        self.assertEqual(out['bm25']['error']['type'], 'RuntimeError')

    def test_selection_matches_the_offline_rule(self):
        """Same scores in, same Top-5 and the same replace/keep decision as scripts/bm25_gate.py."""
        text = lambda pid: f"{'bm25' if pid[0] == 'x' else 'bge'} passage {pid[1:]} about Amber"
        for lift, hits in (('x5', [(5, 3.0), (6, 2.0)]), ('b7', [(5, 3.0)]), ('b2', [(1, 3.0), (2, 2.5)])):
            record = p3_record()
            scores = {text(f'x{i}'): -1.0 for i in range(100)} | {text(f'b{i}'): 0.0 for i in range(20)} | {text(lift): 5.0}
            fakes = Fakes(hits, scores)
            out = fakes.rule().run(record)
            row = {'sample_id': 's', 'bge': [[c['passage_id'], c['score']] for c in record['candidates']],
                   'bm25': [[f'x{p}', s] for p, s in hits],
                   'ce': {c['passage_id']: scores[c['text']] for c in record['candidates']} |
                         {f'x{p}': scores[f'bm25 passage {p} about Amber'] for p, _ in hits}}
            offline = l2_offline.select_ids(row, bm25_gate.SETTING)
            self.assertEqual(tuple(out['bm25']['selected']), offline)
            self.assertEqual(bool(out['bm25']['bm25_only']), bm25_gate.brings_bm25(row, offline))


class WiringTests(unittest.TestCase):
    """The production path: MeteredEnhancementAdapter calls execute() with the registration's budget."""

    def setUp(self):
        from scripts.run_controller_integration import smoke_inputs
        self.nq = smoke_inputs()[0]

    def execute(self, config, with_bm25=True):
        controller = pipeline(config=config)
        if with_bm25:
            controller.p3_bm25 = p3_bm25.P3Bm25Resources(
                index=SimpleNamespace(search=lambda q, k: [(0, 4.0)]),
                record=lambda p: ('bm25_0', 'The capital of Luma is Amber.'), scorer=Scorer(), receipt={})
        calls = CallLog(config)
        decision = decision_for(ControllerAction.RETRIEVAL_ENHANCEMENT, self.nq.sample_id)
        return controller.execute(self.nq, decision, calls=calls), calls.summary(), calls.events

    def test_route_1_is_unchanged_without_p3_bm25(self):
        r, cost, _ = self.execute(ExecutionConfig(), with_bm25=False)
        self.assertNotIn('bm25', r['module_trace'])
        self.assertEqual(tuple(cost[k] for k in ('retrieval_calls', 'scoring_calls', 'llm_calls')), (1, 1, 1))

    def test_p3_bm25_runs_inside_the_budget_and_reports_its_calls(self):
        r, cost, events = self.execute(ExecutionConfig(max_llm_calls=2, max_retrieval_calls=2, max_scoring_calls=2))
        self.assertEqual(r['module_trace']['policy'], 'P3-BM25')
        self.assertEqual(r['module_trace']['bm25']['bm25_only'], ['bm25_0'])
        self.assertEqual(tuple(cost[k] for k in ('retrieval_calls', 'scoring_calls', 'llm_calls')), (2, 2, 2))
        self.assertEqual([e['stage'] for e in events if e['kind'] == 'retrieval'], ['additional_retrieval', 'bm25'])
        self.assertEqual(r['final_evidence'][0]['passage_id'], 'bm25_0')
        json.dumps(r, allow_nan=False)

    def test_route_1_limits_block_the_extra_calls(self):
        r, cost, _ = self.execute(ExecutionConfig(max_llm_calls=1, max_retrieval_calls=1, max_scoring_calls=1))
        self.assertEqual(r['status'], 'budget_exceeded')
        self.assertEqual((r['module_trace']['bm25']['reason'], cost['budget_blocks']), ('bm25_failed', 1))


class RegistrationTests(unittest.TestCase):
    def test_registration_is_route_1_plus_p3_bm25(self):
        new = json.loads((ROOT / 'configs/controller/quality_bm25_v1.json').read_text(encoding='utf-8'))
        base = json.loads((ROOT / 'configs/controller/quality_baseline_v1.json').read_text(encoding='utf-8'))
        changed = {k for k in set(new) | set(base) if new.get(k) != base.get(k)}
        self.assertEqual(changed, {'baseline', 'registered_date', 'user_instruction', 'execution_config', 'frozen_assets',
                                   'evaluation_status', 'route_1_registration', 'p3_bm25'})
        self.assertEqual({k for k in new['execution_config'] if new['execution_config'][k] != base['execution_config'][k]},
                         {'max_llm_calls', 'max_retrieval_calls', 'max_scoring_calls'})
        self.assertTrue(set(base['frozen_assets'].items()) <= set(new['frozen_assets'].items()))
        for path, digest in new['frozen_assets'].items():
            self.assertEqual(sha(path), digest, path)
        self.assertEqual(new['route_1_registration']['sha256'], sha(new['route_1_registration']['path']))
        spec = new['p3_bm25']
        self.assertEqual(spec['rule_registration_sha256'], sha(spec['rule_registration']))
        p3_bm25.check_spec(spec)
        self.assertEqual(readiness(ROOT, 'configs/controller/quality_bm25_v1.json')['missing'],
                         readiness(ROOT, 'configs/controller/quality_baseline_v1.json')['missing'])
        self.assertIn('configs/controller/quality_bm25_v1.json', run_quality_eval.REGISTRATIONS)

    def test_spec_and_index_checks(self):
        spec = json.loads((ROOT / 'configs/controller/quality_bm25_v1.json').read_text(encoding='utf-8'))['p3_bm25']
        for change in ({'k_bm25': 50}, {'fusion': 'rrf'}, {'scorer': {**spec['scorer'], 'device': 'cpu'}}):
            with self.assertRaises(ValueError):
                p3_bm25.check_spec({**spec, **change})
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'index').mkdir()
            (Path(tmp) / 'index' / 'a.npy').write_bytes(b'1')
            (Path(tmp) / 'receipt.json').write_bytes(b'{}')
            first = p3_bm25.directory_sha(tmp)
            self.assertEqual(first, p3_bm25.directory_sha(tmp))
            (Path(tmp) / 'index' / 'a.npy').write_bytes(b'2')
            self.assertNotEqual(first, p3_bm25.directory_sha(tmp))
            with self.assertRaises(ValueError):
                p3_bm25.load_resources({**spec, 'index_dir': tmp}, SimpleNamespace(), '/')


if __name__ == '__main__':
    unittest.main()
