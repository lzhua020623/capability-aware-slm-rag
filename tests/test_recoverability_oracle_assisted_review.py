"""Provenance and scope checks for the 100-question assistant review."""
import collections
import hashlib
import json
from pathlib import Path
import tarfile
import unittest
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'artifacts/controller/recoverability_oracle_v1'
REVIEW = ROOT / 'artifacts/controller/recoverability_oracle_review_100_v1'
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def workbook(path):
    with ZipFile(path) as package:
        strings = []
        if 'xl/sharedStrings.xml' in package.namelist():
            strings = [''.join(x.itertext()) for x in ET.fromstring(package.read('xl/sharedStrings.xml'))]
        sheet = ET.fromstring(package.read('xl/worksheets/sheet1.xml'))
        values = {}
        for cell in sheet.findall('.//m:sheetData/m:row/m:c', NS):
            kind = cell.get('t')
            value = cell.find('m:v', NS)
            inline = cell.find('m:is', NS)
            if kind == 'inlineStr':
                result = ''.join(inline.itertext()) if inline is not None else ''
            elif kind == 's':
                result = strings[int(value.text)]
            else:
                result = value.text if value is not None else ''
            values[cell.get('r')] = result
        book = ET.fromstring(package.read('xl/workbook.xml'))
        return values, sheet, book


class OracleAssistedReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((REVIEW / 'manifest.json').read_text())
        cls.reviews = [json.loads(line) for line in (REVIEW / 'reviews.jsonl').read_text().splitlines()]
        cls.summary = json.loads((REVIEW / 'summary.json').read_text())
        cls.source, cls.source_xml, cls.source_book = workbook(REVIEW / 'source.xlsx')
        cls.output, cls.output_xml, cls.output_book = workbook(REVIEW / 'reviewed.xlsx')

    def test_source_and_experiment_archive_fingerprints(self):
        self.assertEqual(hashlib.sha256((REVIEW / 'source.xlsx').read_bytes()).hexdigest(), self.manifest['input_sha256'])
        self.assertEqual(hashlib.sha256((ARCHIVE / 'results.tar.gz').read_bytes()).hexdigest(), self.manifest['archive_sha256'])

    def test_one_review_per_original_id_without_float_conversion(self):
        self.assertEqual(len(self.reviews), 100)
        self.assertEqual(len({r['sample_id'] for r in self.reviews}), 100)
        self.assertEqual(collections.Counter(r['group'] for r in self.reviews), {'A':50,'B':50})
        for i, r in enumerate(self.reviews, 2):
            self.assertEqual(r['source_row'], i)
            self.assertIsInstance(r['sample_id'], str)
            self.assertEqual(r['sample_id'], self.source[f'B{i}'])
            self.assertEqual(r['question'], self.source[f'C{i}'])
            self.assertEqual(r['source_passage_sha256'], hashlib.sha256(self.source[f'N{i}'].encode()).hexdigest())

    def test_assistant_labels_are_not_human_confirmation(self):
        self.assertFalse(self.manifest['human_confirmed'])
        for r in self.reviews:
            self.assertEqual(r['reviewer'], 'assistant')
            self.assertIs(r['human_confirmed'], False)
            self.assertIn('未人工确认', r['workbook_values'][3])
            self.assertEqual(r['needs_adjudication'], '不确定' in (r['l4_correctness'], r['c1_correctness']))
            self.assertTrue(r['reason'])
            for url in r['external_sources']:
                self.assertTrue(url.startswith('https://'))

    def test_only_review_values_change_and_all_are_populated(self):
        for col in 'ABCDEFGHIJKLMNO':
            for row in range(1, 102):
                key = f'{col}{row}'
                self.assertEqual(self.source.get(key,''), self.output.get(key,''), key)
        for col in 'PQRS':
            self.assertEqual(self.source[f'{col}1'], self.output[f'{col}1'])
        for r in self.reviews:
            row = r['source_row']
            for col, value in zip('PQRS', r['workbook_values']):
                self.assertEqual(self.output[f'{col}{row}'], value)
            self.assertIn(r['l4_correctness'], {'对','错','不确定'})
            self.assertIn(r['c1_correctness'], {'对','错','不确定'})

    def test_sheet_identity_filter_and_freeze_are_preserved(self):
        def sheets(book):
            return [(s.get('name'),s.get('state','visible')) for s in book.find('m:sheets', NS)]
        self.assertEqual(sheets(self.source_book), sheets(self.output_book))
        for tag in ['autoFilter','sheetViews/sheetView/pane']:
            xpath = './'+'/'.join('m:'+part for part in tag.split('/'))
            a, b = self.source_xml.find(xpath,NS), self.output_xml.find(xpath,NS)
            self.assertEqual(a is None,b is None,tag)
            if a is not None:
                for key,value in a.attrib.items():
                    if key != 'activePane':
                        self.assertEqual(b.get(key),value,(tag,key))

    def test_review_inputs_match_frozen_generation_archive(self):
        with tarfile.open(ARCHIVE / 'results.tar.gz','r:gz') as tar:
            def records(name):
                return [json.loads(line) for line in tar.extractfile(name)]
            inputs = {r['sample_id']:r for r in records('ladder_inputs.jsonl')}
            refs = {r['sample_id']:r['gold'] for r in records('references.jsonl')}
            generations = records('generations/generations.jsonl')
            l4 = {r['sample_id']:r['generated_answer'] for r in generations if len(r['passage_ids'])==1}
        for r in self.reviews:
            sid,row = r['sample_id'],r['source_row']
            self.assertEqual(self.source[f'C{row}'],inputs[sid]['query'])
            self.assertEqual(self.source[f'E{row}'],inputs[sid]['c1'])
            self.assertEqual(self.source[f'I{row}'],l4[sid])
            self.assertIn(self.source[f'D{row}'], refs[sid])

    def test_summary_does_not_drop_uncertainty_or_change_strata(self):
        self.assertEqual(self.summary['n'],100)
        for group in ['A','B']:
            rs=[r for r in self.reviews if r['group']==group]
            for label in ['l4','c1']:
                self.assertEqual(self.summary['groups'][group][label],dict(collections.Counter(r[f'{label}_correctness'] for r in rs)))
        self.assertEqual(self.summary['adjudication_ids'],[r['review_id'] for r in self.reviews if r['needs_adjudication']])
        b_failed=[r for r in self.reviews if r['group']=='B' and self.source[f"L{r['source_row']}"]=='是']
        self.assertEqual(len(b_failed),37)
        self.assertEqual(self.summary['B_all_conditions_failed']['l4_correct_ids'],[r['review_id'] for r in b_failed if r['l4_correctness']=='对'])


if __name__ == '__main__':
    unittest.main()
