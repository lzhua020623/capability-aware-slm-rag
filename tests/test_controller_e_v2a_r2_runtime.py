"""r2 gate: run every sealed check, classify failures, follow the registered branch."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.audit_controller_e_v2a_outputs import audit_r2
from scripts.run_controller_e_v2a import (FIXTURES, check_expected, classify, development_entries, execute_one,
                                          gate, long_case, r2_fixtures)
from src.compensation.evidence_v2a import MODEL_LABELS
from src.controller.execution import ExecutionConfig, fingerprint
from tests.test_controller_integration import pipeline
from tests.test_evidence_v2a import Backend, output

ROOT = Path(__file__).resolve().parents[1]
TO_MODEL = {v: k for k, v in MODEL_LABELS.items()}


def model_output(case):
    """The ideal model JSON for a sealed case, as a fake backend would return it."""
    e = case['expected']
    key = case['key_evidence_ids'] or ['R1S1']
    replace = e['final_action'] == 'replace'
    return output(TO_MODEL[e['original_verification']], e['final_answer'] if replace else '',
                  ids=key[:1] if replace else [], original_ids=[] if e['original_verification'] == 'insufficient' else key)


def record_for(case, *, contract=True, label=None, final=None, tokens=15000, decoder='xgrammar', failed=0):
    e = case['expected']
    return {'status': 'kept', 'final_answer': e['final_answer'] if final is None else final,
            'cost': {'failed_calls': failed, 'budget_blocks': 0},
            'calls': [{'kind': 'llm', 'attempted': True, 'status': 'completed', 'output': {'input_tokens': tokens},
                       'structured_decoder': {'decoder': decoder, 'prefill_chunk_size': 256, 'hit_token_budget': False}}],
            'module_trace': {'raw_output': '{}', 'support_check': {
                'contract_passed': contract, 'original_verification': label or e['original_verification'],
                'final_action': e['final_action'], 'unresolved': e['unresolved'],
                'verdict_evidence_ids': case['key_evidence_ids'], 'model_label': 'x', 'type_warnings': []}}}


class WordTokenizer:
    def encode(self, text): return text.split()
    def apply_chat_template(self, messages, **kw): return messages[0]['content']


class R2RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = r2_fixtures()
        cls.cases = cls.fixtures['cases']

    def test_ideal_outputs_pass_every_sealed_case_end_to_end(self):
        config = ExecutionConfig(nq_repair_strategy='evidence_id_v2a', fever_repair_strategy='evidence_id_v2a')
        for case in self.cases:
            p = pipeline(config=config)
            p.repair_generators[case['input']['dataset']] = Backend(model_output(case))
            record = execute_one(p, {'input': case['input'], 'arm': 'engineering_unit_test'})
            self.assertTrue(check_expected(record, case), case['case_id'])
            self.assertEqual(classify(record, case), (None, []), case['case_id'])
            self.assertEqual(record['module_trace']['support_check']['model_label'],
                             TO_MODEL[case['expected']['original_verification']])

    def test_old_r1_label_on_missing_fact_is_now_a_structure_failure_not_a_silent_pass(self):
        case = next(c for c in self.cases if c['kind'] == 'missing_insufficient' and c['input']['dataset'] == 'nq')
        p = pipeline(config=ExecutionConfig(nq_repair_strategy='evidence_id_v2a'))
        p.repair_generators['nq'] = Backend(output('unsupported', '', original_ids=['R1S1']))
        record = execute_one(p, {'input': case['input'], 'arm': 'engineering_unit_test'})
        self.assertEqual(classify(record, case)[0], 'structure')
        p.repair_generators['nq'] = Backend(output('contradicted', '', original_ids=['R1S1']))
        record = execute_one(p, {'input': case['input'], 'arm': 'engineering_unit_test'})
        self.assertEqual(classify(record, case)[0], 'semantic')

    def test_classification_order_execution_then_structure_then_semantic(self):
        case = self.cases[0]
        self.assertEqual(classify(record_for(case, failed=1, contract=False), case)[0], 'execution')
        self.assertEqual(classify(record_for(case, contract=False), case)[0], 'structure')
        self.assertEqual(classify(record_for(case, label='insufficient'), case)[0], 'semantic')
        self.assertEqual(classify(record_for(case, final='someone else'), case)[0], 'semantic')
        self.assertEqual(classify(record_for(case, tokens=900), case, 14000)[0], 'execution')
        self.assertEqual(classify(record_for(case, decoder='hf'), case, 14000)[0], 'execution')
        repair = next(c for c in self.cases if c['case_id'] == 'v2a-r2-fiction-nq-contradicted_repair-2')
        self.assertIsNone(classify(record_for(repair, final='in 1887'), repair)[0])

    def run_gate(self, broken=None):
        def fake(_, entry):
            case = next(c for c in self.cases if c['input']['sample_id'] == entry['input']['sample_id'])
            return broken(case) if broken and broken(case) else record_for(case)
        stub = type('P', (), {'raw_generator': type('R', (), {'tokenizer': WordTokenizer()})()})()
        with tempfile.TemporaryDirectory() as td, patch('scripts.run_controller_e_v2a.execute_one', side_effect=fake) as ex:
            receipt = gate(stub, self.fixtures, Path(td))
            files = sorted(p.name for p in Path(td).iterdir())
        return receipt, ex.call_count, files

    def test_gate_runs_all_twenty_checks_even_after_failures(self):
        receipt, calls, files = self.run_gate()
        self.assertEqual((receipt['outcome'], calls, len(files)), ('passed', 20, 20))
        self.assertEqual(sum(c['long'] for c in receipt['checks']), 2)
        semantic = lambda c: record_for(c, label='supported') if c['kind'] == 'missing_insufficient' else None
        receipt, calls, _ = self.run_gate(semantic)
        self.assertEqual((receipt['outcome'], calls, receipt['failure_counts']['semantic']), ('semantic_only', 20, 4))
        structure = lambda c: record_for(c, contract=False) if c['case_id'].endswith('supported_keep-1') else None
        receipt, calls, _ = self.run_gate(structure)
        self.assertEqual((receipt['outcome'], calls), ('failed', 20))
        crash = lambda c: (_ for _ in ()).throw(RuntimeError('fake CUDA')) if c['kind'] == 'conflict_insufficient' else None
        receipt, calls, _ = self.run_gate(crash)
        self.assertEqual(receipt['outcome'], 'failed')
        self.assertEqual(receipt['failure_counts']['execution'], 4)

    def test_branches_select_exact_registered_entries(self):
        from scripts.prepare_controller_e_v2a import frozen_plan
        plan, _, _ = frozen_plan()
        self.assertEqual([len(development_entries(plan, o)) for o in ('passed', 'semantic_only', 'failed')], [800, 400, 0])
        self.assertEqual({e['arm'] for e in development_entries(plan, 'semantic_only')}, {'always_repair'})

    def test_long_cases_follow_the_sealed_spec(self):
        spec = self.fixtures['long']
        by_id = {c['case_id']: c for c in self.cases}
        for base_id in spec['base_case_ids']:
            case, n = long_case(WordTokenizer(), by_id[base_id], spec['minimum_input_tokens'], spec)
            self.assertGreaterEqual(n, 14000)
            self.assertLessEqual(n, spec['maximum_input_tokens'])
            self.assertEqual(case['case_id'], base_id + '-long')
            self.assertEqual(case['input']['evidence'][:-1], by_id[base_id]['input']['evidence'][:-1])

    def test_fixture_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in (FIXTURES, 'configs/controller/e_v2a_r2_fixture_seal.json', 'configs/controller/e_v2a_engineering_r2.json'):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_bytes((ROOT / rel).read_bytes())
            r2_fixtures(root)
            changed = json.loads((root / FIXTURES).read_text()); changed['cases'][0]['expected']['final_answer'] = 'x'
            (root / FIXTURES).write_text(json.dumps(changed))
            with self.assertRaises(ValueError): r2_fixtures(root)


class R2AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures_raw = (ROOT / FIXTURES).read_bytes()
        cls.fixtures = json.loads(cls.fixtures_raw)
        item = cls.fixtures['cases'][0]['input']
        cls.plan = [{'input': item, 'input_sha256': fingerprint(item), 'arm': arm} for arm in ('always_repair', 'full_E')]
        cls.plan_raw = ''.join(json.dumps(r) + '\n' for r in cls.plan).encode()
        cls.expected = {'engineering_revision': 'evidence_id_v2a_v3_definition_r2', 'source_commit': 'c0ffee',
                        'cases_sha256': hashlib.sha256(cls.fixtures_raw).hexdigest(),
                        'plan_sha256': hashlib.sha256(cls.plan_raw).hexdigest()}
        cls.expected_raw = json.dumps(cls.expected).encode()

    def export(self, outcome='passed', edit=None, predictions=None):
        long = self.fixtures['long']
        by_id = {c['case_id']: c for c in self.fixtures['cases']}
        files = {}
        planned = [(c, c['case_id']) for c in self.fixtures['cases']] + [(by_id[i], i + '-long') for i in long['base_case_ids']]
        for case, cid in planned:
            record = record_for(case, label='supported' if outcome == 'semantic_only' and case['kind'] == 'missing_insufficient' else None)
            record['input'] = deepcopy(case['input'])
            failure = classify(record, case, 14000 if cid.endswith('-long') else None)[0]
            files['engineering/%s.json' % cid] = {'case_id': cid, 'expected': case['expected'], 'record': record,
                                                 'passed': failure is None, 'failure_class': failure}
        if predictions is None:
            predictions = development_entries(self.plan, outcome)
        rows = []
        for entry in predictions:
            initial = entry['input']['initial_answer']
            rows.append({'dataset': entry['input']['dataset'], 'sample_id': entry['input']['sample_id'], 'mode': entry['arm'],
                         'input': entry['input'], 'input_sha256': entry['input_sha256'], 'final_answer': initial,
                         'module_trace': {'support_check': {'deployment_mode': 'verify_only' if outcome == 'semantic_only' else 'allow_repair'}}})
        files['manifest.json'] = {'state': 'completed', 'source_commit': 'c0ffee', 'role': 'development', 'quality_scored': False,
                                  'engineering_outcome': outcome, 'n_predictions': len(rows)}
        files['engineering_receipt.json'] = {'outcome': outcome}
        if edit: edit(files, rows)
        raw = {k: json.dumps(v).encode() for k, v in files.items()}
        raw['predictions.jsonl'] = ''.join(json.dumps(r) + '\n' for r in rows).encode()
        raw['input_execution_manifest.json'] = self.expected_raw
        raw['output_checksums.json'] = json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}).encode()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as z:
            for k, v in raw.items(): z.writestr(k, v)
        td = tempfile.mkdtemp()
        path = Path(td) / 'out.zip'; path.write_bytes(buffer.getvalue())
        return audit_r2(path, self.expected_raw, self.expected, self.fixtures_raw, self.plan_raw)

    def test_passed_and_semantic_only_exports_audit_with_recomputed_outcomes(self):
        result = self.export('passed')
        self.assertEqual((result['engineering_outcome'], result['n_development_predictions'], result['gate_checks_passed']), ('passed', 2, 20))
        result = self.export('semantic_only')
        self.assertEqual((result['engineering_outcome'], result['n_development_predictions']), ('semantic_only', 1))
        self.assertFalse(result['engineering_passed'])

    def test_forged_outcomes_and_off_branch_outputs_are_rejected(self):
        def forge(files, rows):
            files['engineering/v2a-r2-fiction-nq-supported_keep-1.json']['failure_class'] = 'semantic'
            files['engineering/v2a-r2-fiction-nq-supported_keep-1.json']['passed'] = False
        with self.assertRaisesRegex(ValueError, 'disagrees with observed'): self.export('passed', forge)
        with self.assertRaisesRegex(ValueError, 'outside the registered branch'):
            self.export('semantic_only', predictions=self.plan)
        def replaced(files, rows): rows[0]['final_answer'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'replaced an answer'): self.export('semantic_only', replaced)
        def claim(files, rows): files['manifest.json']['engineering_outcome'] = 'passed'; files['engineering_receipt.json']['outcome'] = 'passed'
        with self.assertRaisesRegex(ValueError, 'disagrees with recomputed'): self.export('semantic_only', claim)



class R2ReleaseTests(unittest.TestCase):
    def test_sealed_r2_bundle_binds_registration_fixtures_and_transport(self):
        import tarfile
        folder = ROOT / 'artifacts/controller/e_v2a_interface_r2'
        receipt = json.loads((folder / 'gpu_release_v1/bundle_receipt.json').read_text())
        archive = folder / 'gpu_release_v1' / (receipt['bundle_name'] + '.tar.gz')
        raw = archive.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), receipt['archive_sha256'])
        self.assertEqual((folder / 'transport_v1' / (receipt['bundle_name'] + '.tar.gz.bin')).read_bytes(), raw)
        self.assertFalse(receipt['GPU_started'])
        with tarfile.open(archive) as t:
            prefix = receipt['bundle_name'] + '/'
            manifest = json.load(t.extractfile(prefix + 'execution_manifest.json'))
            fixtures = t.extractfile(prefix + FIXTURES).read()
            prompt_source = t.extractfile(prefix + 'src/compensation/evidence_v2a.py').read()
        self.assertEqual(manifest['engineering_revision'], 'evidence_id_v2a_v3_definition_r2')
        self.assertEqual(manifest['source_commit'], receipt['source_commit'])
        self.assertEqual(hashlib.sha256(fixtures).hexdigest(), json.loads(
            (ROOT / 'configs/controller/e_v2a_engineering_r2.json').read_text())['gate']['fixtures_sha256'])
        self.assertEqual(manifest['cases_sha256'], hashlib.sha256(fixtures).hexdigest())
        self.assertIn(b"'contradicted': 'unsupported'", prompt_source)
        self.assertEqual((manifest['gate_checks'], manifest['contains_real_gold']), (20, False))


if __name__ == '__main__':
    unittest.main()
