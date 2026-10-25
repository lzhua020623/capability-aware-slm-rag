from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_controller_t5_assisted_review import ROOT, bind, prepare, render


class AssistedReviewTests(unittest.TestCase):
    def first_case(self):
        path = ROOT / 'artifacts/controller/e_v2a_v1/t5_development_review/routing15_review.jsonl'
        case = json.loads(path.read_text().splitlines()[0])
        all_notes = json.loads((ROOT / 'artifacts/controller/t5_assisted_review_v1/source_notes.json').read_text())
        return case, {**all_notes, 'notes': all_notes['notes'][:1]}

    def test_suggestions_cannot_fabricate_sources_ids_or_human_confirmation(self):
        case, notes = self.first_case()
        original_human = deepcopy(case['human_review'])
        self.assertEqual(bind([case], notes)[0]['human_review'], original_human)
        changed = deepcopy(notes); changed['notes'][0]['anchors'][0]['quote'] = 'not in the supplied original evidence'
        with self.assertRaises(ValueError): bind([case], changed)
        changed = deepcopy(notes); changed['notes'][0]['human_review'] = {'status': 'confirmed'}
        with self.assertRaises(ValueError): bind([case], changed)
        changed = deepcopy(notes); changed['notes'].append(changed['notes'][0])
        with self.assertRaises(ValueError): bind([case], changed)
        changed = deepcopy(notes); changed['human_confirmation'] = True
        with self.assertRaises(ValueError): bind([case], changed)

    def test_actual_queue_includes_both_candidate_harms_and_keeps_all_humans_pending(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / 'review'
            receipt = prepare(output)
            rows = [json.loads(l) for l in (output / 'review17.jsonl').read_text().splitlines()]
            self.assertEqual((receipt['cases'], receipt['human_confirmed']), (17, 0))
            self.assertEqual(len({r['sample_id'] for r in rows}), 17)
            self.assertEqual([r['sample_id'] for r in rows[-2:]], ['-5328057821735888756', '538314753001203199'])
            self.assertTrue(all(r['human_review']['status'] == 'pending' for r in rows))
            self.assertTrue(all(r['assisted_review']['source'] == 'agent_only_not_human' for r in rows))
            self.assertFalse(receipt['candidate_frozen'])
            with self.assertRaises(FileExistsError): prepare(output)

    def test_auxiliary_text_is_escaped_and_does_not_preselect_a_human_label(self):
        case, notes = self.first_case()
        notes['notes'][0]['agent_judgment']['reason'] = '<script>bad()</script>'
        text = render(bind([case], notes))
        self.assertNotIn('<script>bad()', text)
        self.assertIn('&lt;script&gt;', text)
        self.assertNotIn('<option selected', text)


if __name__ == '__main__':
    unittest.main()
