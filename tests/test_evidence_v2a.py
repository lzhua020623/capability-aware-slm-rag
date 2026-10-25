from dataclasses import asdict, replace
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.compensation.answer_repair import Generation
from src.compensation.evidence_v2a import repair, apply_mode, schema, prompt
from src.compensation.evidence_id_repair import evidence_units
from src.controller.execution import ExecutionConfig, fingerprint
from src.controller.providers import shared_repair_views
from scripts.run_controller_integration import smoke_inputs
from tests.test_controller_integration import pipeline


class Backend:
    def __init__(self, value, metadata=None):
        self.value, self.calls, self.metadata = value, [], metadata or {}

    def generate_schema(self, p, budget, s):
        self.calls.append((p, budget, s))
        if isinstance(self.value, Exception):
            raise self.value
        text = self.value if isinstance(self.value, str) else json.dumps(self.value)
        return Generation(text, .1, 100, 30), {'decoder': 'test_stub', **self.metadata}


def output(verdict='unsupported', answer='Amber', ids=None, original_ids=None):
    return {'original_verification': verdict, 'verdict_evidence_ids': ['R1S1'] if original_ids is None else original_ids,
            'replacement_answer': answer, 'replacement_evidence_ids': (['R1S1'] if answer else []) if ids is None else ids,
            'reason': 'The evidence establishes the requested fact.'}


class V2ATests(unittest.TestCase):
    def setUp(self):
        self.nq, self.fever = smoke_inputs()

    def test_unsupported_valid_span_repairs_with_one_call_and_original_quotes(self):
        b = Backend(output()); r = repair(self.nq, b)
        self.assertEqual(r.final_answer, 'Amber')
        self.assertEqual(r.support_check['original_verification'], 'unsupported')
        self.assertEqual(r.support_check['candidate_check'], 'passed')
        self.assertEqual(r.support_check['final_action'], 'replace')
        self.assertFalse(r.support_check['unresolved'])
        self.assertTrue(r.support_check['contract_passed'])
        self.assertEqual(len(b.calls), 1)
        self.assertEqual(b.calls[0][1], 384)
        self.assertEqual(r.citations[0].quote, self.nq.evidence[0].text)

    def test_supported_always_protects_entering_answer_including_expansion(self):
        r = repair(self.nq, Backend(output('supported', self.nq.evidence[0].text)))
        self.assertEqual(r.final_answer, self.nq.initial_answer)
        self.assertEqual(r.support_check['candidate_check'], 'protected')
        self.assertFalse(r.support_check['unresolved'])

    def test_unsupported_expansion_can_add_a_required_qualifier(self):
        r = replace(self.nq, initial_answer='Luma')
        b = Backend(output(answer='The capital of Luma is Amber'))
        result = repair(r, b)
        self.assertEqual(result.final_answer, 'The capital of Luma is Amber')
        self.assertTrue(result.changed)

    def test_missing_or_conflicting_evidence_preserves_answer_and_flags(self):
        for v, ids in [('unsupported', ['R1S1']), ('insufficient', [])]:
            r = repair(self.nq, Backend(output(v, '', original_ids=ids)))
            self.assertEqual(r.final_answer, self.nq.initial_answer)
            self.assertTrue(r.support_check['unresolved'])
            self.assertTrue(r.support_check['contract_passed'])

    def test_candidate_failure_does_not_erase_completed_original_verification(self):
        for o, expected in [(output(ids=['R99S1']), 'invalid_source'),
                            (output(ids=['R1S1', 'R1S1']), 'invalid_source'),
                            (output(answer='invented'), 'invalid_span'),
                            (output(answer='', ids=['R1S1']), 'invalid_source')]:
            r = repair(self.nq, Backend(o))
            self.assertEqual(r.support_check['original_verification'], 'unsupported')
            self.assertEqual(r.support_check['candidate_check'], expected)
            self.assertFalse(r.support_check['contract_passed'])
            self.assertEqual(r.final_answer, self.nq.initial_answer)

    def test_original_failure_is_incomplete_and_never_fabricated(self):
        for value, meta in [('not JSON', {}), (output(original_ids=[]), {}),
                            (output(), {'hit_token_budget': True}), (RuntimeError('CUDA fake failure'), {})]:
            r = repair(self.nq, Backend(value, meta))
            self.assertEqual(r.support_check['original_verification'], 'incomplete')
            self.assertTrue(r.support_check['unresolved'])
            self.assertFalse(r.support_check['contract_passed'])
            self.assertEqual(r.final_answer, self.nq.initial_answer)

    def test_duplicate_keys_and_blank_reasons_fail_closed(self):
        o = output(); o['reason'] = ''
        for value in (o, json.dumps(output())[:-1] + ',"reason":"duplicate"}'):
            self.assertEqual(repair(self.nq, Backend(value)).verdict, 'incomplete')

    def test_equivalent_answer_is_not_replaced_or_counted_fixed(self):
        r = replace(self.nq, initial_answer='the Amber')
        result = repair(r, Backend(output()))
        self.assertFalse(result.changed)
        self.assertEqual(result.support_check['candidate_check'], 'protected')

    def test_verify_only_keeps_diagnostic_candidate_without_replacement(self):
        raw = repair(self.nq, Backend(output()))
        gated = apply_mode(raw, 'verify_only')
        self.assertEqual(gated.final_answer, self.nq.initial_answer)
        self.assertEqual(gated.support_check['diagnostic_final_answer'], 'Amber')
        self.assertEqual(gated.support_check['candidate_check_before_policy'], 'passed')
        self.assertTrue(gated.support_check['unresolved'])
        self.assertEqual(raw.final_answer, 'Amber')
        self.assertEqual(apply_mode(gated, 'allow_repair').final_answer, 'Amber')
        self.assertEqual(asdict(apply_mode(gated, 'verify_only')), asdict(gated))

    def test_unavailable_does_not_call_backend(self):
        b = Backend(output()); r = repair(self.nq, b, mode='unavailable')
        self.assertFalse(b.calls)
        self.assertEqual(r.verdict, 'incomplete')
        self.assertEqual(r.final_answer, self.nq.initial_answer)

    def test_fever_verifies_label_not_absence_of_claim_support(self):
        r = repair(self.fever, Backend(output(answer='SUPPORTS')))
        self.assertEqual(r.final_answer, 'SUPPORTS')
        r = repair(self.fever, Backend(output('insufficient', '', original_ids=[])))
        self.assertEqual(r.final_answer, self.fever.initial_answer)
        self.assertIn('missing/conflicting evidence', prompt(self.fever, evidence_units(self.fever)))
        expanded = replace(self.fever, initial_answer='SUPPORTS\nThe supplied passage supports it.')
        self.assertFalse(repair(expanded, Backend(output(answer='SUPPORTS'))).changed)

    def test_schema_only_exposes_legal_source_keys_and_no_offline_labels(self):
        units = evidence_units(self.nq); s = schema(units)
        self.assertTrue(all(not b['additionalProperties'] for b in s['anyOf']))
        self.assertEqual(s['$defs']['source_id']['enum'], [u.id for u in units])
        self.assertTrue(all('gold' not in b['properties'] for b in s['anyOf']))

    def test_conditional_schema_rejects_observed_empty_ids_before_generation(self):
        from jsonschema import Draft202012Validator
        validator = Draft202012Validator(schema(evidence_units(self.nq), self.nq))
        for valid in (output(), output('supported', ''), output('unsupported', ''),
                      output('insufficient', '', original_ids=[])):
            self.assertTrue(validator.is_valid(valid), valid)
        for invalid in (output(original_ids=[]), output(ids=[]), output('supported', 'Amber'),
                        output('insufficient', 'Amber')):
            self.assertFalse(validator.is_valid(invalid), invalid)
        fever = Draft202012Validator(schema(evidence_units(self.fever), self.fever))
        self.assertFalse(fever.is_valid(output(answer='REFUTES')))

    def test_interface_r1_gate_uses_new_fictional_facts(self):
        from scripts.run_controller_e_v2a import fictional_cases
        cases = fictional_cases()
        self.assertEqual(len(cases), 6)
        text = json.dumps(cases)
        self.assertNotIn('Velora', text)
        self.assertNotIn('Nerin Sol', text)
        self.assertTrue(all(c['case_id'].startswith('v2a-r1-fiction-') for c in cases))

    def test_registered_budget_and_modes_only(self):
        for kwargs in ({'mode': 'other'}, {'max_new_tokens': 385}):
            with self.assertRaises(ValueError): repair(self.nq, Backend(output()), **kwargs)

    def test_controller_meters_schema_call_and_sets_unresolved(self):
        config = ExecutionConfig(nq_repair_strategy='evidence_id_v2a')
        p = pipeline(config=config)
        p.repair_generators['nq'] = Backend(output('unsupported', ''))
        r = p.run(self.nq, mode='always_repair')
        self.assertEqual(r['cost']['llm_calls'], 1)
        self.assertTrue(r['unresolved_failure']['flagged'])
        self.assertEqual(r['calls'][0]['arguments']['schema_sha256'], fingerprint(schema(evidence_units(self.nq))))
        self.assertEqual(r['calls'][0]['structured_decoder']['decoder'], 'test_stub')
        self.assertEqual(r['module_trace']['support_check']['original_verification'], 'unsupported')

    def test_full_repair_uses_actual_entering_state_not_original_C1(self):
        p = pipeline(config=ExecutionConfig(nq_repair_strategy='evidence_id_v2a', max_scoring_calls=2))
        b = Backend(output('supported', '')); p.repair_generators['nq'] = b
        r = p.run(self.nq, mode='full_enhancement')
        self.assertNotEqual(r['module_trace']['stages'][-1]['input_sha256'], r['input_sha256'])
        self.assertIn('Entering answer: "Amber"', b.calls[0][0])

    def test_shared_view_created_once_for_both_datasets(self):
        raw = SimpleNamespace(model=object(), tokenizer=object())
        view = object()
        cfg = ExecutionConfig(nq_repair_strategy='evidence_id_v2a', fever_repair_strategy='evidence_id_v2a')
        with patch('src.compensation.qwen_v2a.QwenV2AGenerator.from_shared', return_value=view) as make:
            views = shared_repair_views(raw, cfg)
        make.assert_called_once_with(raw)
        self.assertIs(views['nq'], views['fever'])

    def test_structured_backend_view_reuses_real_model_objects_without_loading(self):
        from src.compensation.qwen_v2a import QwenV2AGenerator
        raw = SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(prefill_chunk_size=256),
                                                    config=SimpleNamespace(vocab_size=100)),
                              tokenizer=object(), runtime_environment={'seed': 42}, json_output=False)
        xgr = SimpleNamespace(TokenizerInfo=SimpleNamespace(from_huggingface=lambda *a, **kw: 'info'),
                              GrammarCompiler=lambda *a, **kw: object())
        modules = {'xgrammar': xgr, 'xgrammar.contrib': SimpleNamespace(),
                   'xgrammar.contrib.hf': SimpleNamespace(LogitsProcessor=object)}
        with patch.dict(sys.modules, modules), patch('importlib.metadata.version', return_value='0.2.8'):
            view = QwenV2AGenerator.from_shared(raw)
        self.assertIs(view.model, raw.model)
        self.assertIs(view.tokenizer, raw.tokenizer)
        self.assertFalse(raw.json_output)
        self.assertTrue(view.json_output)
        raw.model.generation_config.prefill_chunk_size = None
        with self.assertRaises(RuntimeError): view.generate_schema('text', 384, {})


if __name__ == '__main__': unittest.main()
