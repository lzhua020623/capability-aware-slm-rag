"""Sealed r2 fictional fixtures: written and blind-reviewed before any r2 prompt change."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

from scripts.run_controller_integration import runtime_input
from src.compensation.evidence_id_repair import evidence_units, near_duplicates
from src.compensation.short_answer_repair import locate_excerpt
from src.evaluation.qa import normalize_answer

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'configs/controller/e_v2a_r2_fixtures.json'
SEAL = ROOT / 'configs/controller/e_v2a_r2_fixture_seal.json'
REVIEW = ROOT / 'artifacts/controller/e_v2a_interface_r2/fixture_review'
KINDS = {'nq': ['supported_keep', 'contradicted_repair', 'contradicted_no_candidate',
                'missing_insufficient', 'conflict_insufficient'],
         'fever': ['supported_keep', 'contradicted_repair', 'missing_insufficient', 'conflict_insufficient']}
MODEL = {'supported': 'supported', 'contradicted': 'unsupported', 'insufficient': 'insufficient'}


class R2FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = FIXTURES.read_bytes()
        cls.fixtures = json.loads(cls.raw)
        cls.cases = cls.fixtures['cases']
        cls.seal = json.loads(SEAL.read_text())

    def test_seal_matches_bytes_and_precedes_prompt_change(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), self.seal['fixtures_sha256'])
        self.assertTrue(self.fixtures['written_before_r2_prompt_change'])
        self.assertEqual(self.seal['n_short'], 18)
        self.assertEqual(self.seal['n_long'], 2)

    def test_registered_classes_two_each(self):
        counts = Counter((c['input']['dataset'], c['kind']) for c in self.cases)
        self.assertEqual(counts, Counter({(ds, k): 2 for ds, kinds in KINDS.items() for k in kinds}))
        self.assertEqual(len({c['case_id'] for c in self.cases}), 18)
        self.assertTrue(all(c['case_id'].startswith('v2a-r2-fiction-') and c['synthetic'] for c in self.cases))

    def test_new_fiction_and_clean_runtime_inputs(self):
        text = json.dumps(self.cases)
        for name in self.fixtures['retired_names'] + ['Ostrava']:
            self.assertNotIn(name, text)
        for c in self.cases:
            item = runtime_input(c['input'])
            self.assertEqual(len(item.evidence), 5)
            self.assertFalse(near_duplicates(item), c['case_id'])
            self.assertEqual(set(c['input']), {'sample_id', 'dataset', 'query', 'evidence', 'initial_answer'})

    def test_expectations_follow_the_definitions(self):
        for c in self.cases:
            e, initial = c['expected'], c['input']['initial_answer']
            replace = c['kind'] == 'contradicted_repair'
            self.assertEqual(e['original_verification'], {'supported_keep': 'supported'}.get(
                c['kind'], 'unsupported' if c['kind'].startswith('contradicted') else 'insufficient'))
            self.assertEqual(e['final_action'], 'replace' if replace else 'keep')
            self.assertEqual(e['unresolved'], not (replace or c['kind'] == 'supported_keep'))
            self.assertEqual(e['final_answer'] == initial, not replace)
            self.assertEqual(e['acceptable_final_answers'][0], e['final_answer'])

    def test_replacements_are_legal_and_key_ids_exist(self):
        for c in self.cases:
            item = runtime_input(c['input'])
            units = {u.id: u for u in evidence_units(item)}
            self.assertTrue(set(c['key_evidence_ids']) <= set(units), c['case_id'])
            self.assertEqual(not c['key_evidence_ids'], c['kind'].startswith('missing'))
            if c['kind'] != 'contradicted_repair':
                continue
            if item.dataset.value == 'fever':
                self.assertEqual({item.initial_answer, c['expected']['final_answer']}, {'SUPPORTS', 'REFUTES'})
                continue
            for answer in c['expected']['acceptable_final_answers']:
                self.assertLessEqual(len(answer.split()), 20)
                located, _, _ = locate_excerpt(answer, units[c['key_evidence_ids'][0]].text)
                self.assertNotEqual(normalize_answer(located), normalize_answer(item.initial_answer))

    def test_long_spec_uses_registered_repair_bases(self):
        long = self.fixtures['long']
        ids = {c['case_id']: c for c in self.cases}
        self.assertEqual([ids[i]['input']['dataset'] for i in long['base_case_ids']], ['nq', 'fever'])
        self.assertTrue(all(ids[i]['kind'] == 'contradicted_repair' for i in long['base_case_ids']))
        self.assertEqual((long['minimum_input_tokens'], long['maximum_input_tokens']), (14000, 32768 - 384))

    def test_every_sealed_case_was_blind_reviewed_in_its_final_form(self):
        latest = {}
        for round_ in self.seal['blind_review']['rounds']:
            folder = REVIEW / round_['folder']
            key = json.loads((folder / 'blind_key.json').read_text())
            review = json.loads((folder / 'blind_review.json').read_text())
            packet = json.loads((folder / 'blind_packet.json').read_text())
            reviewed = {c['case_id']: c for c in json.loads((folder / 'fixtures_reviewed.json').read_text())['cases']}
            text = json.dumps(packet)
            self.assertFalse(any(k in text for k in ('expected', 'key_evidence', 'v2a-r2-fiction')))
            agree = 0
            for r in review['cases']:
                case = reviewed[key[r['blind_id']]]
                e = case['expected']
                agree += (MODEL[r['verdict']] == e['original_verification'] and r['final_action'] == e['final_action']
                          and r['unresolved'] == e['unresolved'] and normalize_answer(r['final_answer'])
                          in {normalize_answer(a) for a in e.get('acceptable_final_answers', [e['final_answer']])})
                latest[case['case_id']] = case
            self.assertEqual((agree, len(review['cases'])), (round_['agreement'], round_['n']))
            self.assertEqual(round_['major_flags'], 0)
        self.assertEqual(set(latest), {c['case_id'] for c in self.cases})
        for c in self.cases:
            self.assertEqual(latest[c['case_id']]['input'], c['input'], c['case_id'])
            self.assertEqual(latest[c['case_id']]['expected']['original_verification'], c['expected']['original_verification'])


if __name__ == '__main__':
    unittest.main()
