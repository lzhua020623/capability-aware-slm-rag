"""T3 state flow, counterfactual prerequisites, mechanical labels and budgets."""
from dataclasses import asdict, replace
from types import SimpleNamespace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from scripts.run_controller_integration import smoke_inputs
from src.compensation.answer_repair import Generation
from src.controller.execution import CallLog, BudgetExceeded, ExecutionConfig, fingerprint
from src.controller.providers import shared_repair_views
from src.controller.providers import load_pipeline
from src.controller.runtime import configure_generation_runtime, unresolved_assessment
from tests.test_controller_integration import pipeline, diagnosis, Scorer


class T3ExecutionTests(unittest.TestCase):
    def test_fever_label_adapter_preserves_original_and_works_for_every_arm(self):
        fever = replace(smoke_inputs()[1], initial_answer='The verdict is REFUTES. Extra explanation.')
        for mode in ('c1', 'adaptive', 'adaptive_no_probe'):
            r = pipeline().run(fever, mode=mode)
            self.assertEqual(r['initial_answer'], 'REFUTES')
            self.assertEqual(r['initial_answer_raw'], fever.initial_answer)
            self.assertEqual(r['original_input_sha256'], fingerprint(asdict(fever)))
            self.assertTrue(r['input_adapter']['changed'])

    def test_no_probe_keeps_diagnosis_without_invoking_probe(self):
        r = pipeline(diag=diagnosis('evidence_utilisation', p=.6, low=True)).run(
            smoke_inputs()[0], mode='adaptive_no_probe')
        self.assertFalse(r['routing']['decision']['used_probe'])
        self.assertEqual(r['cost']['llm_calls'], 1)
        self.assertFalse(any(c['stage'] == 'probe' for c in r['calls']))

    def test_full_passes_actual_accepted_state_and_uses_two_scoring_calls(self):
        item = smoke_inputs()[0]
        r = pipeline(config=ExecutionConfig(max_scoring_calls=2)).run(item, mode='full_enhancement')
        stages = r['module_trace']['stages']
        self.assertEqual(len(stages), 3)
        self.assertEqual(r['status'], 'sequence_completed')
        self.assertEqual(r['cost']['scoring_calls'], 2)
        self.assertEqual(stages[1]['initial_answer'], stages[0]['final_answer'])
        self.assertEqual(stages[2]['initial_answer'], stages[1]['final_answer'])
        self.assertEqual(stages[1]['input_sha256'], fingerprint(asdict(replace(
            item, initial_answer=stages[0]['final_answer'],
            evidence=[type(item.evidence[0])(**e) for e in stages[0]['final_evidence']]))))
        self.assertLessEqual(sum(c['arguments']['max_new_tokens'] for c in r['calls']
                                 if c['kind'] == 'llm' and c['attempted']), 512)
        self.assertEqual(r['final_answer'], 'Amber')

    def test_full_failure_keeps_entering_state_and_continues_without_budget_reset(self):
        class FailFirstScorer(Scorer):
            def __call__(self, query, texts):
                if not self.calls:
                    self.calls += 1
                    raise RuntimeError('first scoring fails')
                return super().__call__(query, texts)
        item = smoke_inputs()[0]
        r = pipeline(config=ExecutionConfig(max_scoring_calls=2), scorer=FailFirstScorer()).run(
            item, mode='full_enhancement')
        stages = r['module_trace']['stages']
        self.assertEqual(stages[1]['input_sha256'], fingerprint(asdict(item)))
        self.assertEqual(r['cost']['failed_calls'], 1)
        self.assertEqual(r['cost']['scoring_calls'], 2)
        self.assertEqual(r['status'], 'sequence_failed')
        self.assertEqual(r['final_answer'], 'Amber')
        self.assertTrue(r['unresolved_failure']['flagged'])

    def test_each_arm_receives_its_protocol_budget(self):
        r = pipeline(config=ExecutionConfig(max_scoring_calls=2)).run(smoke_inputs()[0], mode='always_repair')
        self.assertEqual(r['config']['max_llm_calls'], 1)
        self.assertEqual(r['config']['max_scoring_calls'], 0)
        self.assertEqual(r['config']['max_total_output_tokens'], 384)

    def test_aggregate_token_allocations_include_failed_attempts(self):
        calls = CallLog(ExecutionConfig(max_total_output_tokens=64))
        with self.assertRaises(RuntimeError):
            calls.invoke('llm', 'first', {'max_new_tokens': 64}, lambda: (_ for _ in ()).throw(RuntimeError()))
        with self.assertRaises(BudgetExceeded):
            calls.invoke('llm', 'second', {'max_new_tokens': 1}, lambda: Generation('', 0))
        self.assertEqual(calls.summary()['llm_calls'], 1)
        self.assertEqual(calls.summary()['budget_blocks'], 1)

    def test_human_type_mode_requires_explicit_label_and_never_changes_prompt_from_reason(self):
        p = pipeline()
        with self.assertRaises(ValueError):
            p.run(smoke_inputs()[0], mode='human_type_routed')
        r = p.run(smoke_inputs()[0], mode='human_type_routed', human_type='correct')
        self.assertEqual(r['routing']['source'], 'offline_human_type')
        self.assertEqual(r['calls'], [])

    def test_keep_is_unverified_not_a_semantic_success(self):
        r = pipeline().run(smoke_inputs()[0])
        self.assertFalse(r['unresolved_failure']['flagged'])
        self.assertEqual(r['unresolved_failure']['state'], 'unverified')
        self.assertEqual(r['unresolved_failure']['semantic_correctness'], 'not_checked')
        invalid = pipeline(repair='bad output').run(smoke_inputs()[0], mode='always_repair')
        self.assertTrue(invalid['unresolved_failure']['flagged'])
        self.assertFalse(invalid['answer_changed'])

    def test_all_shared_model_views_inherit_same_prefill_configuration(self):
        raw = SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(prefill_chunk_size=None)),
                              tokenizer=object(), runtime_environment={}, json_output=False)
        configure_generation_runtime(raw)
        views = shared_repair_views(raw, ExecutionConfig())
        for view in views.values():
            self.assertIs(view.model, raw.model)
            self.assertEqual(view.model.generation_config.prefill_chunk_size, 256)
        with self.assertRaises(ValueError):
            configure_generation_runtime(raw, 512)

    def test_real_loader_contract_has_shared_runtime_and_cpu_FP32_scoring(self):
        raw = SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(prefill_chunk_size=None)),
                              tokenizer=object(), runtime_environment={}, json_output=False)
        observed = {}
        class Encoder:
            def __init__(self, *args, **kwargs):
                observed['init'] = kwargs
            def predict(self, pairs, **kwargs):
                observed['predict'] = kwargs
                return [1.] * len(pairs)
        fake_torch = SimpleNamespace(nn=SimpleNamespace(Identity=lambda: 'Identity'), float32='FP32')
        with patch.dict(sys.modules, {'torch': fake_torch, 'sentence_transformers': SimpleNamespace(CrossEncoder=Encoder)}), \
             patch('src.compensation.qwen_repair.QwenRepairGenerator', return_value=raw), \
             patch('src.diagnosis.hierarchical_classifier.HierarchicalDiagnosisClassifier.load', return_value=None):
            p, env = load_pipeline(ExecutionConfig(), Path('fictional-model'))
            self.assertEqual(p.scorer('query', ['one', 'two']), [1., 1.])
        self.assertEqual(observed['init']['device'], 'cpu')
        self.assertEqual(observed['init']['model_kwargs']['torch_dtype'], 'FP32')
        self.assertEqual(observed['predict']['batch_size'], 4)
        self.assertEqual(env['controller_generation_runtime']['prefill_chunk_size'], 256)
        self.assertIs(p.repair_generators['nq'].model, p.raw_generator.model)

    def test_negative_probe_applies_only_to_the_unchanged_answer(self):
        record = {'cost': {'failed_calls': 0, 'budget_blocks': 0}, 'answer_changed': False,
                  'routing': {'probe': {'answer_supported': 'unsupported', 'evidence_sufficiency': 'insufficient'}}}
        result = unresolved_assessment(record)
        self.assertTrue(result['flagged'])
        self.assertIn('probe_rejects_unchanged_answer', result['reasons'])
        record['answer_changed'] = True
        self.assertFalse(unresolved_assessment(record)['flagged'])
        record['error'] = {'type': 'StructuralFailure'}
        self.assertTrue(unresolved_assessment(record)['flagged'])


if __name__ == '__main__':
    unittest.main()
