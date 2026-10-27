"""Actual role IDs with wholly fictional content; no paid API/model calls."""
import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jsonschema

from src.data.team_nq import BoundaryError, TeamNQBoundary
from src.distillation.e_prep import (PROTOCOL, ROOT, RUNTIME_FIELDS, audit_teacher_output,
    build_sample_manifest, digest, export_pilot, load_protocol, teacher_schema,
    teacher_task, to_input, validate_sample_manifest)
from tests.test_team_nq_boundary import fictional


def target(verdict='contradicted', answer='Aster', verdict_ids=None, replacement_ids=None):
    return json.dumps({'original_verification': verdict,
        'verdict_evidence_ids': ['R1S1'] if verdict_ids is None else verdict_ids,
        'replacement_answer': answer,
        'replacement_evidence_ids': (['R1S1'] if answer else []) if replacement_ids is None else replacement_ids,
        'reason': 'The supplied passage establishes the requested emblem.'})


class EDistillationPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boundary = TeamNQBoundary(ROOT)
        cls.protocol = load_protocol(ROOT)
        cls.manifest = build_sample_manifest(cls.boundary, cls.protocol)
        cls.sid = cls.manifest['sample_ids']['teacher_pilot'][0]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def runtime(self):
        return self.boundary.teacher_input(fictional(self.sid))

    def test_frozen_pilot_is_balanced_AI_metadata_not_semantic_gold(self):
        m = self.manifest
        self.assertEqual(m['pilot_AI_strata'], {label:125 for label in self.protocol['sampling']['strata']})
        self.assertEqual({r:len(ids) for r, ids in m['sample_ids'].items()},
                         {'teacher_pilot':500,'development':500,'selection':500})
        self.assertFalse(m['pilot_is_population_representative'])
        self.assertFalse(m['AI_strata_are_semantic_gold'])
        self.assertEqual(m['teacher_API_calls'], 0)

    def test_roles_are_disjoint_and_do_not_use_test_reserve_or_exposed_validation(self):
        m = self.manifest['sample_ids']
        self.assertEqual(len(set().union(*[set(v) for v in m.values()])), 1500)
        for role, ids in m.items():
            self.boundary.require(ids, 'train' if role == 'teacher_pilot' else 'validation')
            self.assertFalse(set(ids) & self.boundary.legacy)
            self.assertFalse(set(ids) & set(self.boundary.splits['test']))
            if role != 'teacher_pilot':
                self.assertFalse(set(ids) & self.boundary.exposed)

    def test_saved_manifest_is_the_reproducible_registered_manifest(self):
        saved = json.loads((ROOT/'artifacts/controller/e_distillation_v1/sample_manifest.json').read_text())
        self.assertEqual(saved,self.manifest)

    def test_sampling_reproducible_even_if_in_memory_ID_order_changes(self):
        reverse = lambda role: list(reversed(self.boundary.allowed_ids(role)))
        fake = copy.copy(self.boundary)
        fake.allowed_ids = reverse
        self.assertEqual(build_sample_manifest(fake, self.protocol), self.manifest)

    def test_changed_manifest_and_test_injection_are_rejected(self):
        for modify in [lambda m:m['sample_ids']['teacher_pilot'].__setitem__(0,self.boundary.splits['test'][0]),
                       lambda m:m['sample_ids']['selection'].reverse(),
                       lambda m:m.update(unregistered_extra='not allowed')]:
            m = copy.deepcopy(self.manifest); modify(m)
            with self.assertRaises(BoundaryError): validate_sample_manifest(self.boundary, self.protocol, m)

    def test_protocol_cannot_enable_API_expansion_or_change_runtime_budget(self):
        for key in ('teacher', 'expansion', 'student'):
            protocol = copy.deepcopy(self.protocol)
            protocol[key]['enabled' if key != 'student' else 'max_new_tokens'] = True if key != 'student' else 1024
            with patch('src.distillation.e_prep.read_json', return_value=protocol):
                with self.assertRaises(BoundaryError): load_protocol(ROOT)

    def test_modified_contract_asset_blocks_preparation(self):
        protocol = copy.deepcopy(self.protocol)
        protocol['contract_assets']['src/compensation/evidence_v2a.py'] = '0'*64
        with patch('src.distillation.e_prep.read_json', return_value=protocol):
            with self.assertRaisesRegex(BoundaryError, 'asset missing or changed'): load_protocol(ROOT)

    def test_teacher_task_strips_offline_reference_label_and_passage_extras(self):
        row = fictional(self.sid)
        row['annotation_reference']['gold_answer'] = 'OFFLINE_SENTINEL'
        row['primary_label'] = 'OFFLINE_LABEL_SENTINEL'
        row['evidence'][0]['gold_answer'] = 'OFFLINE_SENTINEL'
        task = teacher_task(self.boundary, row)
        self.assertEqual(set(task['runtime']), RUNTIME_FIELDS)
        self.assertNotIn('OFFLINE_SENTINEL', json.dumps(task))
        self.assertNotIn('OFFLINE_LABEL_SENTINEL', json.dumps(task))
        self.assertEqual(task['fingerprints']['teacher_messages'], digest(task['teacher_messages']))
        self.assertEqual(task['fingerprints']['runtime'], digest(task['runtime']))
        self.assertEqual(task['fingerprints']['student_prompt'], digest(task['student_prompt']))
        self.assertFalse(task['teacher_model_and_total_budget_registered'])

    def test_teacher_input_rejects_nontrain_and_extra_runtime_fields(self):
        for sid in (self.boundary.allowed_ids('validation')[0], self.boundary.splits['test'][0]):
            with self.assertRaises(BoundaryError): teacher_task(self.boundary, fictional(sid))
        runtime = self.runtime(); runtime['gold_answer'] = 'OFFLINE'
        with self.assertRaises(BoundaryError): to_input(runtime)

    def test_replayed_runtime_cannot_change_Top5_shape_ID_types_or_finite_scores(self):
        for mutate in [lambda r:r.update(sample_id=int(self.sid)),
                       lambda r:r['evidence'].pop(),
                       lambda r:r['evidence'][0].update(score=float('inf')),
                       lambda r:r['evidence'][1].update(passage_id=r['evidence'][0]['passage_id']),
                       lambda r:r['evidence'][0].update(gold_answer='OFFLINE')]:
            runtime=self.runtime();mutate(runtime)
            with self.assertRaises(BoundaryError): to_input(runtime)

    def test_API_schema_has_object_root_and_legal_local_source_IDs(self):
        schema = teacher_schema(to_input(self.runtime()))
        self.assertEqual(schema['type'], 'object')
        self.assertNotIn('anyOf', schema)
        self.assertFalse(schema['additionalProperties'])
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(json.loads(target()), schema)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(json.loads(target(verdict_ids=['R99S1'])), schema)

    def test_supported_keeps_and_is_never_auto_approved_as_semantic_gold(self):
        report = audit_teacher_output(self.runtime(), target('supported', ''))
        self.assertTrue(report['structural_provenance_passed'])
        self.assertEqual(report['candidate_action'], 'keep')
        self.assertEqual(report['candidate_final_answer'], 'Wrong')
        # This fictional original answer is actually wrong: legal JSON alone
        # must not turn the teacher's mistaken judgement into semantic truth.
        self.assertFalse(report['semantic_verified'])
        self.assertFalse(report['approved_for_SFT'])

    def test_valid_repair_is_traceable_but_still_needs_semantic_and_length_checks(self):
        report = audit_teacher_output(self.runtime(), target())
        self.assertEqual(report['candidate_action'], 'replace')
        self.assertEqual(report['candidate_final_answer'], 'Aster')
        self.assertIn('Aster', report['citations'][0]['quote'])
        self.assertFalse(report['approved_for_SFT'])
        self.assertFalse(report['student_target_token_length_checked'])

    def test_missing_conflicting_and_no_candidate_outcomes_keep_and_flag(self):
        for verdict, ids in [('insufficient', []), ('insufficient', ['R1S1','R2S1']),
                             ('contradicted', ['R1S1'])]:
            report = audit_teacher_output(self.runtime(), target(verdict, '', verdict_ids=ids))
            self.assertEqual(report['candidate_action'], 'keep')
            self.assertTrue(report['candidate_unresolved'])
            self.assertFalse(report['approved_for_SFT'])

    def test_structurally_legal_but_wrong_span_is_rejected_locally(self):
        for text in [target(answer='invented ungrounded value'),
                     target(verdict_ids=['R1S1','R1S1']),
                     target(replacement_ids=['R99S1']),
                     target('supported', 'Aster'),
                     target('insufficient', 'Aster'),
                     target('contradicted', '', replacement_ids=['R1S1'])]:
            with self.assertRaises(BoundaryError): audit_teacher_output(self.runtime(), text)

    def test_equivalent_repair_and_long_reason_are_not_training_targets(self):
        runtime = self.runtime(); runtime['initial_answer'] = 'the Aster'
        with self.assertRaisesRegex(BoundaryError, 'unchanged/equivalent'):
            audit_teacher_output(runtime, target())
        value = json.loads(target()); value['reason'] = 'word '*61
        with self.assertRaisesRegex(BoundaryError, '1-60 words'):
            audit_teacher_output(self.runtime(), json.dumps(value))

    def test_pilot_export_separates_gold_and_skips_test_validation_content_before_decode(self):
        source = self.folder / 'fictional.jsonl'
        with source.open('w') as handle:
            for sid in self.manifest['sample_ids']['teacher_pilot']:
                row = fictional(sid); row['annotation_reference']['gold_answer'] = 'OFFLINE_ONLY_REFERENCE'
                handle.write(json.dumps(row)+'\n')
            for sid in (self.boundary.splits['test'][0], self.boundary.allowed_ids('validation')[0]):
                handle.write('{"sample_id":"'+sid+'","gold":MUST_NOT_DECODE}\n')
        out = self.folder / 'pilot'
        report = export_pilot(self.boundary,self.protocol,self.manifest,source,out)
        self.assertEqual(report['n_tasks'],500)
        self.assertFalse(report['ready_for_API']); self.assertFalse(report['ready_for_SFT'])
        self.assertNotIn('OFFLINE_ONLY_REFERENCE',(out/'teacher_tasks.jsonl').read_text())
        self.assertIn('OFFLINE_ONLY_REFERENCE',(out/'offline_references.jsonl').read_text())
        self.assertEqual(len((out/'teacher_tasks.jsonl').read_text().splitlines()),500)
        self.assertEqual(report['source_layout'],'derived_ID_checked_file')
        with self.assertRaises(FileExistsError):
            export_pilot(self.boundary,self.protocol,self.manifest,source,out)

    def test_incomplete_source_creates_no_export_directory(self):
        source = self.folder / 'missing.jsonl'
        source.write_text(json.dumps(fictional(self.sid))+'\n')
        out = self.folder / 'rejected'
        with self.assertRaisesRegex(BoundaryError,'missing registered pilot IDs'):
            export_pilot(self.boundary,self.protocol,self.manifest,source,out)
        self.assertFalse(out.exists())

    def test_cli_validates_existing_manifest_and_does_not_silently_overwrite_it(self):
        from scripts.prepare_e_distillation import main
        manifest=self.folder/'samples.json'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--manifest',str(manifest)]),0)
            before=manifest.read_bytes()
            self.assertEqual(main(['--manifest',str(manifest)]),0)
            self.assertEqual(manifest.read_bytes(),before)
        changed=json.loads(manifest.read_text());changed['sample_ids']['teacher_pilot'].reverse()
        manifest.write_text(json.dumps(changed));tampered=manifest.read_bytes()
        with self.assertRaises(BoundaryError): main(['--manifest',str(manifest)])
        self.assertEqual(manifest.read_bytes(),tampered)

    def test_cli_rejects_unused_source_argument(self):
        from scripts.prepare_e_distillation import main
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['--manifest',str(self.folder/'samples.json'),'--source','unused.jsonl'])


if __name__ == '__main__':
    unittest.main()
