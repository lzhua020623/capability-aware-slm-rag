"""Adaptive answer selection in production: parity with the offline evaluation, the module, pipeline wiring and the registration."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts import adaptive_selection as ad
from scripts import bm25_gate, l2_offline, run_quality_eval
from src.compensation import answer_selection as sel
from src.compensation.retrieval_enhancement.p3 import answer_rejection, build_c1_prompt
from src.controller.execution import ExecutionConfig
from src.diagnosis.labels import DiagnosisLabel as Label
from src.pipeline.current import MeteredAnswerSelection, load_answer_selection, readiness, wire_latest_pipeline
from tests.test_controller_current import FictionalFeatures, ForcedSelection
from tests.test_adaptive_rag_pipeline import failure_diag

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / ad.POLICY).read_text(encoding='utf-8'))
REGISTRATION = 'configs/controller/quality_adaptive_v1.json'
CONFIG = json.loads((ROOT / REGISTRATION).read_text(encoding='utf-8'))
sha = lambda path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
WORDS = ('Paris', 'Rome', 'Lille', 'Nice')


def text(pid):
    """Passage p mentions one of WORDS, chosen by its number, so answer support varies."""
    return f'{pid} is about {WORDS[int(pid[1:]) % len(WORDS)]}'


class World:
    """One query: BGE b0..b99, BM25 x0..x(n-1) (corpus positions 100+), cross-encoder scores, and a fake 7B."""

    def __init__(self, seed, n_bm25=12):
        rng = int(hashlib.sha256(str(seed).encode()).hexdigest(), 16)
        self.bge = [f'b{i}' for i in range(100)]
        self.bm25 = [f'x{i}' for i in range(n_bm25)]
        self.ce = {p: ((rng >> (i % 200)) % 97) / 10 - 4 for i, p in enumerate(self.bge + self.bm25)}
        self.corpus = self.bge + self.bm25
        self.prompts = []

    def answer_of(self, top5):
        """The fake 7B's raw answer over one Top-5 and its answer_rejection outcome."""
        h = int(hashlib.sha256(','.join(top5).encode()).hexdigest(), 16)
        answer = ' ' + ('I do not know' if h % 7 == 0 else WORDS[h % len(WORDS)])
        return answer, answer_rejection(answer, [{'passage_id': p, 'text': text(p)} for p in top5])

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.by_prompt[prompt]

    def selection(self, policy, query='q'):
        self.by_prompt = {}
        for name in policy['order']:
            top5 = sel.select_context(self.bge, self.bm25, self.ce, name)
            self.by_prompt[build_c1_prompt(query, [{'passage_id': p, 'text': text(p)} for p in top5])] = self.answer_of(top5)[0]
        position = {p: i for i, p in enumerate(self.corpus)}
        return sel.AdaptiveSelection(
            policy, lambda q, k: [{'passage_id': p, 'text': text(p), 'score': 1.0} for p in self.bge[:k]],
            lambda q, k: [[100 + i, 3.0 - i / 10] for i in range(min(k, len(self.bm25)))],
            lambda pos: (self.corpus[pos], text(self.corpus[pos])), position.__getitem__,
            lambda q, texts: [self.ce[t.split(' ')[0]] for t in texts], self.generate)

    def candidate_row(self, sid):
        return {'sample_id': sid, 'bge': [[p, 1.0] for p in self.bge], 'bm25': [[p, 3.0 - i / 10] for i, p in enumerate(self.bm25)],
                'ce': dict(self.ce)}


def offline_and_online(policy, seed, routed, p_fail, c1='Paris', c1_top5=('b0', 'b1', 'b2', 'b3', 'b4')):
    """The offline item path (build_items + apply_stages) and AdaptiveSelection.run on the same world."""
    world, sid = World(seed), f's{seed}'
    cand = world.candidate_row(sid)
    p3_top5 = l2_offline.select_ids(cand, {'k_bge': 20, 'k_bm25': 0, 'fusion': 'rrf'})
    l2 = l2_offline.select_ids(cand, bm25_gate.SETTING)
    brings = routed and bm25_gate.brings_bm25(cand, l2)
    answered = {}
    if routed:
        answered[p3_top5] = world.answer_of(p3_top5)
        if brings:
            answered[l2] = world.answer_of(l2)
    p3 = sel.context_answer(c1, p3_top5, c1_top5, answered) if routed else c1
    base = sel.context_answer(c1, l2, c1_top5, answered) if brings else p3
    native = ({'module_trace': {'evidence': [{'passage_id': p} for p in p3_top5], 'generated_answer': answered[p3_top5][0],
                                'reject_reason': answered[p3_top5][1]}} if routed else {})
    result = {'sample_id': sid, 'query': 'q', 'initial_answer': c1, 'initial_evidence': [{'passage_id': p} for p in c1_top5],
              'controller_decision': {'action': ad.P3 if routed else 'no_action'},
              'enhancement_result': {'metadata': {'native': native} if native else {}},
              'diagnosis': {'stage1_failure_probability': p_fail}}
    stored = ad.stored_generations([result], [])
    for name in policy['order']:
        top5 = l2_offline.select_ids(cand, ad.settings({'contexts': {'order': [name]}})[0])
        if top5 != tuple(c1_top5) and (sid, top5) not in stored:
            stored[(sid, top5)] = world.answer_of(top5)
    if brings:
        stored[(sid, l2)] = answered[l2]
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / 'corpus.jsonl'
        corpus.write_text(''.join(json.dumps({'passage_id': p, 'text': text(p)}) + '\n' for p in world.corpus), encoding='utf-8')
        row = ad.build_items([result], {sid: base}, {sid: cand}, stored, corpus, policy['order'])[0]
    offline = ad.apply_stages(row, policy['chosen'], policy['models'])
    online = world.selection(policy).run({'query': 'q', 'c1': c1, 'c1_top5': list(c1_top5), 'base': base, 'routed': routed,
                                          'p_fail': p_fail, 'answered': answered})
    return offline, online, row, world


class ParityTests(unittest.TestCase):
    def test_contexts_select_the_offline_top5(self):
        for seed in range(6):
            world = World(seed)
            row = world.candidate_row('s')
            for config in ad.settings(ad.load_registration()):
                self.assertEqual(sel.select_context(world.bge, world.bm25, world.ce, l2_offline.name(config)),
                                 l2_offline.select_ids(row, config))

    def test_online_run_equals_the_offline_policy(self):
        """Same answer, stage and extra generations as the evaluated offline path, for the chosen policy and a vote variant."""
        policies = (POLICY, {**POLICY, 'chosen': {'theta': 0.0, 'tau': 0.6, 'decision': 'vote'}},
                    {**POLICY, 'chosen': {'theta': 0.5, 'tau': 0.8, 'decision': 'selector'}})
        seen = set()
        for policy in policies:
            for seed in range(24):
                offline, online, row, world = offline_and_online(policy, seed, routed=seed % 3 == 0, p_fail=(seed % 10) / 10)
                if offline[1]:
                    self.assertEqual(online['groups'] and [g['x'] for g in online['groups']],
                                     [g['x'] for g in row['stages'][str(offline[1])]['groups']])
                self.assertEqual((online['final_answer'], online['stage'], online['generations']), offline, (seed, policy['chosen']))
                self.assertEqual(online['entered'], offline[1] > 0)
                seen.add((offline[1], online['changed']))
        self.assertTrue({(0, False), (3, False), (8, False)} <= seen and any(changed for _, changed in seen), seen)

    def test_the_offline_script_uses_the_product_functions(self):
        for name in ('groups', 'decide_groups', 'probability', 'supports', 'norm'):
            self.assertIs(getattr(ad, name), getattr(sel, name))
        self.assertEqual((ad.FEATURES, ad.STAGES), (sel.FEATURES, sel.STAGES))


class ModuleTests(unittest.TestCase):
    item = {'query': 'q', 'c1': 'Paris', 'c1_top5': ['b0', 'b1', 'b2', 'b3', 'b4'], 'base': 'Paris', 'routed': False,
            'p_fail': 0.2, 'answered': {}}

    def test_items_below_the_gate_make_no_calls(self):
        world = World(1)
        selection = world.selection({**POLICY, 'chosen': {'theta': 0.5, 'tau': 0.8, 'decision': 'selector'}})
        selection.bge = lambda q, k: self.fail('retrieval below the gate')
        out = selection.run(self.item)
        self.assertEqual((out['entered'], out['final_answer'], out['reason'], world.prompts), (False, 'Paris', 'below_gate', []))

    def test_reused_top5_lists_are_not_generated_again(self):
        world = World(2)
        selection = world.selection(POLICY)
        top5 = sel.select_context(world.bge, world.bm25, world.ce, POLICY['order'][0])
        out = selection.run({**self.item, 'answered': {top5: ('Rome', None)}})
        self.assertEqual(out['contexts'][POLICY['order'][0]], {'top5': list(top5), 'answer': 'Rome', 'source': 'reused'})
        self.assertEqual(out['generations'], len(world.prompts))
        self.assertNotIn(build_c1_prompt('q', [{'passage_id': p, 'text': text(p)} for p in top5]), world.prompts)

    def test_errors_keep_the_product_answer(self):
        world = World(3)
        selection = world.selection(POLICY)
        selection.scorer = lambda q, texts: (_ for _ in ()).throw(RuntimeError('gpu'))
        out = selection.run(self.item)
        self.assertEqual((out['final_answer'], out['reason'], out['error']['type'], out['changed']),
                         ('Paris', 'scoring_failed', 'RuntimeError', False))
        selection = world.selection(POLICY)
        selection.generate_fn = lambda prompt: 7
        self.assertEqual(selection.run(self.item)['reason'], 'generation_failed')

    def test_a_switch_reports_the_evidence_of_the_chosen_context(self):
        vote = {**POLICY, 'chosen': {'theta': 0.0, 'tau': 0.5, 'decision': 'vote'}}
        for seed in range(40):
            world = World(seed)
            out = world.selection(vote).run({**self.item, 'c1': 'Oslo', 'base': 'Oslo'})
            if out['changed']:
                break
        else:
            self.fail('no switch in the fixture worlds')
        source = out['contexts'][out['evidence_from']]
        self.assertEqual(source['answer'], out['final_answer'])
        self.assertEqual([e['passage_id'] for e in out['evidence']], source['top5'])
        self.assertEqual([e['score'] for e in out['evidence']], [world.ce[p] for p in source['top5']])

    def test_policy_must_match_the_module(self):
        with self.assertRaises(ValueError):
            World(0).selection({**POLICY, 'stages': [3, 5]})


class FakeDeep:
    """Stands in for the top-100 NQ backend: passages p0..p119, every one stating the capital of Luma is Amber."""

    def __init__(self):
        self.retriever = SimpleNamespace(passage_ids=[f'p{i}' for i in range(120)])

    def retrieve(self, query, *, dataset, top_k):
        return [{'rank': i + 1, 'passage_id': f'p{i}', 'text': f'p{i}: The capital of Luma is Amber.', 'score': 1 - i / 1000}
                for i in range(top_k)]

    def _record(self, position):
        return f'p{position}', f'p{position}: The capital of Luma is Amber.'


class PipelineTests(unittest.TestCase):
    def stage(self, config=ExecutionConfig(max_llm_calls=8, max_retrieval_calls=2, max_scoring_calls=1,
                                           max_total_output_tokens=512), policy=None):
        bm25 = SimpleNamespace(index=SimpleNamespace(search=lambda q, k: [(100 + i, 2.0) for i in range(min(k, 20))]),
                               scorer=lambda q, texts: [float(len(t) % 7) for t in texts])
        backend = self.backend()
        policy = policy or {**POLICY, 'chosen': {'theta': 0.0, 'tau': 0.8, 'decision': 'vote'}}
        return MeteredAnswerSelection(policy, FakeDeep(), bm25, backend.raw_generator, config), backend

    def backend(self):
        from scripts.run_controller_integration import smoke_pipeline
        backend = smoke_pipeline(ExecutionConfig(**CONFIG['execution_config']))
        backend.classifier.diagnose = lambda item: failure_diag(Label.EVIDENCE_INSUFFICIENCY)
        return backend

    def run_item(self, stage, backend):
        from scripts.run_controller_integration import smoke_inputs
        pipeline = wire_latest_pipeline(backend, FictionalFeatures(),
                                        selection_adapter=ForcedSelection(Label.EVIDENCE_INSUFFICIENCY, selected=False),
                                        answer_selection=stage)
        return pipeline.run_from_item(smoke_inputs()[0])

    def test_the_selection_replaces_the_controller_answer_and_reports_its_calls(self):
        stage, backend = self.stage()
        result = self.run_item(stage, backend)
        trace = result.metadata['answer_selection']
        self.assertEqual((result.controller_decision.action.value, result.initial_answer), ('no_action', 'Blue'))
        self.assertEqual((trace['entered'], trace['final_answer'], result.final_answer, result.changed_answer),
                         (True, 'Amber', 'Amber', True))
        cost = trace['cost']
        self.assertEqual((cost['retrieval_calls'], cost['scoring_calls'], cost['llm_calls']), (2, 1, trace['generations']))
        self.assertEqual([e['stage'] for e in trace['calls'] if e['kind'] == 'retrieval'], ['additional_retrieval', 'bm25'])
        self.assertEqual([e.passage_id for e in result.final_evidence], trace['contexts'][trace['evidence_from']]['top5'])
        json.dumps(result.to_dict(), allow_nan=False, default=str)

    def test_a_blocked_budget_keeps_the_controller_answer(self):
        stage, backend = self.stage(ExecutionConfig(max_llm_calls=1, max_retrieval_calls=2, max_scoring_calls=1))
        result = self.run_item(stage, backend)
        trace = result.metadata['answer_selection']
        self.assertEqual((result.final_answer, trace['reason'], trace['cost']['budget_blocks']), ('Blue', 'generation_failed', 1))

    def test_route_1_and_p3_bm25_generations_are_reused(self):
        evidence = lambda ids: [{'passage_id': p} for p in ids]
        record = {'evidence': evidence('xyzuv'), 'generated_answer': 'Amber', 'reject_reason': None,
                  'p3_20': {'evidence': evidence('abcde'), 'generated_answer': ' Rome', 'reject_reason': 'not_grounded'}}
        result = SimpleNamespace(
            controller_decision=SimpleNamespace(action=sel_action()), query='q', initial_answer='Blue', final_answer='Amber',
            initial_evidence=[SimpleNamespace(passage_id=p) for p in 'fghij'],
            enhancement_result=SimpleNamespace(metadata={'native': {'module_trace': record}}),
            diagnosis=SimpleNamespace(stage1_failure_probability=0.7))
        item = MeteredAnswerSelection.item(result)
        self.assertEqual(item['answered'], {tuple('xyzuv'): ('Amber', None), tuple('abcde'): (' Rome', 'not_grounded')})
        self.assertEqual((item['routed'], item['base'], item['c1_top5'], item['p_fail']), (True, 'Amber', list('fghij'), 0.7))


def sel_action():
    from src.controller.schema import ControllerAction
    return ControllerAction.RETRIEVAL_ENHANCEMENT


class RegistrationTests(unittest.TestCase):
    def test_registration_is_b_bm25_plus_the_selection(self):
        new = json.loads((ROOT / REGISTRATION).read_text(encoding='utf-8'))
        base = json.loads((ROOT / 'configs/controller/quality_bm25_v1.json').read_text(encoding='utf-8'))
        changed = {k for k in set(new) | set(base) if new.get(k) != base.get(k)}
        self.assertEqual(changed, {'baseline', 'user_instruction', 'frozen_assets', 'evaluation_status',
                                   'product_registration', 'answer_selection'})
        self.assertEqual(set(new['frozen_assets']) - set(base['frozen_assets']),
                         {'src/compensation/answer_selection.py', ad.POLICY})
        self.assertTrue(set(base['frozen_assets'].items()) <= set(new['frozen_assets'].items()))
        for path, digest in new['frozen_assets'].items():
            self.assertEqual(sha(path), digest, path)
        self.assertEqual(new['product_registration']['sha256'], sha(new['product_registration']['path']))
        spec = new['answer_selection']
        self.assertEqual((spec['registration_sha256'], spec['policy_sha256']), (sha(ad.REGISTRATION), sha(ad.POLICY)))
        self.assertEqual((spec['chosen'], spec['order'], spec['stages']), (POLICY['chosen'], POLICY['order'], POLICY['stages']))
        self.assertEqual(spec['budget']['max_total_output_tokens'], 8 * spec['budget']['answer_max_new_tokens'])
        ExecutionConfig(**spec['budget'])
        self.assertEqual(readiness(ROOT, REGISTRATION)['missing'], readiness(ROOT, 'configs/controller/quality_bm25_v1.json')['missing'])
        self.assertIn(REGISTRATION, run_quality_eval.REGISTRATIONS)

    def test_loading_checks_the_policy_and_the_bm25_resources(self):
        spec = json.loads((ROOT / REGISTRATION).read_text(encoding='utf-8'))['answer_selection']
        with self.assertRaises(ValueError):
            load_answer_selection(spec, SimpleNamespace(p3_bm25=None), ROOT)
        with self.assertRaises(ValueError):
            load_answer_selection({**spec, 'policy_sha256': '0' * 64}, SimpleNamespace(p3_bm25=object()), ROOT)
        with self.assertRaises(ValueError):
            load_answer_selection({**spec, 'chosen': {**spec['chosen'], 'tau': 0.6}}, SimpleNamespace(p3_bm25=object()), ROOT)


if __name__ == '__main__':
    unittest.main()
