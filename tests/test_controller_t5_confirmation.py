from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.record_controller_t5_confirmation import (KEYS, NOTES, ROOT, SOURCE,
    apply_confirmation, prepare, render, summarize)

FOLDER = ROOT / 'artifacts/controller/t5_human_review_v1'


class T5ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.raw = (ROOT / SOURCE).read_bytes()
        self.cases = [json.loads(l) for l in self.raw.splitlines()]
        self.confirmation = json.loads((FOLDER / 'confirmation.json').read_text())

    def test_explicit_adoption_confirms_eleven_but_does_not_resolve_six_ambiguous_cases(self):
        result = apply_confirmation(self.cases, self.confirmation)
        stats = summarize(result)
        self.assertEqual((stats['cases'], stats['assistance_adopted'], stats['fully_adjudicated']), (17, 17, 11))
        self.assertEqual((stats['partially_adjudicated'], stats['unresolved_all_fields']), (2, 4))
        self.assertEqual(stats['not_fully_adjudicated_ids'], ['R03', 'R06', 'R09', 'R11', 'R16', 'R17'])
        self.assertEqual((stats['both_initial_and_retrieval_correct'], stats['initial_wrong_retrieval_correct']), (8, 3))
        self.assertFalse(stats['candidate_frozen'])
        for old, new in zip(self.cases, result):
            for key in KEYS:
                self.assertIs(new['human_review'][key], old['assisted_review'][key])
            self.assertEqual(old['human_review']['status'], 'pending')
            self.assertEqual(new['variants'], old['variants'])
            self.assertFalse(new['human_review']['other_fixed_actions_adjudicated'])
            self.assertFalse(new['human_review']['independent_blind_review'])

    def test_missing_authorization_duplicate_ids_and_scope_expansion_are_rejected(self):
        for key, value in [('explicit_user_confirmation', False), ('user_message_exact', ''),
                           ('judged_fields', [*KEYS, 'always_rerank_correct']),
                           ('independent_blind_review', True), ('review_ids', ['R01'] * 17)]:
            receipt = {**self.confirmation, key: value}
            with self.assertRaises(ValueError):
                apply_confirmation(self.cases, receipt)
        with self.assertRaises(ValueError):
            apply_confirmation([*self.cases[:-1], self.cases[0]], self.confirmation)

    def test_previous_human_decisions_and_non_boolean_suggestions_cannot_be_overwritten(self):
        cases = deepcopy(self.cases)
        cases[0]['human_review']['status'] = 'confirmed'
        with self.assertRaisesRegex(ValueError, 'must not be overwritten'):
            apply_confirmation(cases, self.confirmation)
        cases = deepcopy(self.cases)
        cases[0]['assisted_review']['initial_correct'] = 'true'
        with self.assertRaisesRegex(ValueError, 'boolean or unresolved'):
            apply_confirmation(cases, self.confirmation)

    def test_changed_assistance_after_adoption_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in (SOURCE, NOTES):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes((ROOT / name).read_bytes())
            (root / SOURCE).write_bytes(self.raw + b'\n')
            output = root / 'out'
            with self.assertRaisesRegex(ValueError, 'changed after'):
                prepare(output, FOLDER / 'confirmation.json', root)
            self.assertFalse(output.exists())

    def test_actual_confirmation_snapshot_is_reproducible_and_preserves_original_assistance(self):
        before = hashlib.sha256(self.raw).hexdigest()
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / 'review'
            summary = prepare(output, FOLDER / 'confirmation.json')
            self.assertEqual(summary, json.loads((FOLDER / 'review/summary.json').read_text()))
            for name in ('review17.jsonl', 'review17.html', 'summary.json'):
                self.assertEqual((output / name).read_bytes(), (FOLDER / 'review' / name).read_bytes())
            with self.assertRaises(FileExistsError):
                prepare(output, FOLDER / 'confirmation.json')
        self.assertEqual(hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest(), before)

    def test_confirmed_page_escapes_text_and_keeps_other_variants_unadjudicated(self):
        cases = apply_confirmation(self.cases, self.confirmation)
        cases[0]['query'] = '<script>bad()</script>'
        page = render(cases, summarize(cases))
        self.assertNotIn('<script>', page)
        self.assertIn('&lt;script&gt;', page)
        self.assertIn('完整定案 11 题', page)
        self.assertIn('未在本次确认范围内', page)
        self.assertEqual(page.count('<article id='), 17)


if __name__ == '__main__':
    unittest.main()
