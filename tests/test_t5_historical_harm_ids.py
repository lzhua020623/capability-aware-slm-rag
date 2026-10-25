"""T5.7 ID check: candidates for the five registered historical E harms, pending the user's choice."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / 'artifacts/controller/e_v2a_v1/t5_development_review/historical_harm_id_check.json'


class HistoricalHarmIdTests(unittest.TestCase):
    def setUp(self):
        self.sheet = json.loads(SHEET.read_text())

    def test_sources_are_pinned_and_status_is_pending_user(self):
        self.assertEqual(self.sheet['status'], 'pending_user_choice')
        for source in ('source_A_user_confirmed', 'source_B_agent_read_only'):
            s = self.sheet[source]
            self.assertEqual(hashlib.sha256((ROOT / s['file']).read_bytes()).hexdigest(), s['sha256'])

    def test_user_confirmed_items_match_human_records(self):
        a = self.sheet['source_A_user_confirmed']
        human = {r['review_number']: r for r in map(json.loads, (ROOT / a['file']).read_text().splitlines())}
        first = {r['review_number']: r for r in map(json.loads, (ROOT / a['batch02_file']).read_text().splitlines())}
        for item in a['items']:
            record = human[item['review_number']]
            self.assertEqual((record['sample_id'], record['human_review']['outcome']), (item['sample_id'], 'regressed'))
            self.assertEqual(first[item['review_number']]['human_review']['outcome'] == 'regressed', item['in_batch02_five'])
        self.assertEqual(sum(i['in_batch02_five'] for i in a['items']), 5)
        self.assertEqual(len(a['items']), 6)

    def test_agent_reads_are_never_labelled_human(self):
        items = self.sheet['source_B_agent_read_only']['items']
        self.assertEqual(len(items), 5)
        self.assertTrue(all(i['confirmed_by_user'] is False for i in items))


if __name__ == '__main__':
    unittest.main()
