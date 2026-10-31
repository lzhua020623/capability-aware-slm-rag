"""Real ID membership, fictional content: no benchmark questions/gold used."""
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.data.team_nq import (BoundaryError, EXPECTED_COUNTS, POLICY, REGISTRATION,
    TeamNQBoundary, prepare_training_bundle, query_key, sha)

ROOT = Path(__file__).resolve().parents[1]


def fictional(sid, name='Aster', *, role='train'):
    return {'sample_id': sid, 'dataset': 'nq', 'query': f'Which emblem belongs to fictional {role} unit {name}?',
        'initial_answer': 'Wrong', 'primary_label': 'evidence_accessibility',
        'evidence': [{'rank': i, 'passage_id': f'fiction-{i}',
            'text': (f'The emblem of fictional unit {name} is {name}.' if i == 1
                     else f'Unrelated fictional distractor passage number {i}.'), 'score': 1 - i / 10}
            for i in range(1, 6)], 'annotation_reference': {'gold_answer': name}}


def dump(path, rows):
    Path(path).write_text(''.join(json.dumps(r) + '\n' for r in rows))


class TeamBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boundary = TeamNQBoundary(ROOT)
        cls.train = cls.boundary.allowed_ids('train')[:3]
        cls.validation = cls.boundary.allowed_ids('validation')[:3]
        cls.test_id = cls.boundary.splits['test'][0]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def source(self):
        p = self.folder / 'source.jsonl'
        dump(p, [fictional(sid, name, role=role) for role, ids in (
            ('train', self.train[:2]), ('validation', self.validation[:2]))
            for sid, name in zip(ids, ('Aster', 'Beryl'))])
        return p

    def mirror(self):
        root = self.folder / 'mirror'
        policy = self.boundary.policy
        exposure = json.loads((ROOT / policy['known_exposure']['path']).read_text())
        specs = [policy[k] for k in ('team_split', 'label_index', 'legacy_reserve',
                                    'legacy_selection', 'known_exposure')]
        specs += [policy['source_delivery']['manifest']] + exposure['sources']
        paths = {REGISTRATION, POLICY} | {s['path'] for s in specs}
        for rel in paths:
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, target)
        return root

    def bundle(self, *, component='answer_selector'):
        if component == 'answer_selector':
            from scripts.build_selector_training_data import build_examples
        else:
            from scripts.build_rerank_training_data import build_examples as rerank
            build_examples = lambda rows: rerank(rows, 'evidence_accessibility', 4)
        p = self.folder / component
        prepare_training_bundle(self.boundary, self.source(), p, build_examples, component=component)
        return p

    def test_team_counts_disjointness_and_tightened_eligible_counts(self):
        receipt = self.boundary.receipt()
        self.assertEqual(receipt['official_counts'], EXPECTED_COUNTS)
        self.assertEqual(receipt['eligible_counts'], {'train': 46421, 'validation': 9440})
        self.assertEqual(receipt['test_known_exposed'], 603)
        self.assertEqual(len(self.boundary.membership), 74097)
        self.assertFalse(receipt['labels_semantically_verified'])

    def test_train_and_validation_allow_only_their_own_roles(self):
        for role, own, other in [('train', self.train, self.validation),
                                 ('validation', self.validation, self.train)]:
            self.boundary.require(own, role)
            for sid in other + [self.test_id]:
                with self.assertRaises(BoundaryError):
                    self.boundary.require([sid], role)

    def test_unknown_numeric_and_non_string_IDs_are_not_coerced(self):
        for sid in ['999999999999999999999999999', int(self.train[0]), float(self.train[0]), None]:
            with self.assertRaises(BoundaryError):
                self.boundary.require([sid], 'train')

    def test_legacy_reserve_and_legacy_selection_stay_protected(self):
        for role in ('train', 'validation'):
            blocked = set(self.boundary.splits[role]) & self.boundary.legacy
            self.assertTrue(blocked)
            with self.assertRaises(BoundaryError):
                self.boundary.require([next(iter(blocked))], role)

    def test_registered_validation_exposure_is_not_reused_for_selection(self):
        sid = next(iter(set(self.boundary.splits['validation']) & self.boundary.exposed))
        with self.assertRaises(BoundaryError):
            self.boundary.require([sid], 'validation')

    def test_test_access_cannot_be_requested_as_an_ordinary_role(self):
        for role in ['test', 'final', 'calibration_typo']:
            with self.assertRaises(BoundaryError):
                self.boundary.allowed_ids(role)

    def test_changed_split_or_registry_fails_closed(self):
        root = self.mirror()
        path = root / self.boundary.policy['team_split']['path']
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(BoundaryError, 'changed'):
            TeamNQBoundary(root)
        shutil.copyfile(ROOT / self.boundary.policy['team_split']['path'], path)
        (root / POLICY).write_text('{}')
        with self.assertRaisesRegex(BoundaryError, 'changed'):
            TeamNQBoundary(root)

    def test_missing_registry_cannot_silently_select_another_split(self):
        root = self.mirror()
        (root / POLICY).unlink()
        with self.assertRaises(BoundaryError):
            TeamNQBoundary(root)

    def test_changed_exposure_source_is_detected(self):
        root = self.mirror()
        source = json.loads((root / self.boundary.policy['known_exposure']['path']).read_text())['sources'][0]
        p = root / source['path']
        p.write_bytes(p.read_bytes() + b' ')
        with self.assertRaisesRegex(BoundaryError, 'changed'):
            TeamNQBoundary(root)

    def test_rehashed_same_size_resplit_is_not_accepted_as_the_team_split(self):
        root = self.mirror()
        rel = self.boundary.policy['team_split']['path']
        data = json.loads((root / rel).read_text())
        a, b = data['splits']['nq']['train'], data['splits']['nq']['validation']
        a[0], b[0] = b[0], a[0]
        (root / rel).write_text(json.dumps(data))
        policy = json.loads((root / POLICY).read_text(encoding='utf-8'))
        policy['team_split']['sha256'] = sha(root / rel)
        (root / POLICY).write_text(json.dumps(policy))
        registration = json.loads((root / REGISTRATION).read_text(encoding='utf-8'))
        registration['data_boundary']['sha256'] = sha(root / POLICY)
        (root / REGISTRATION).write_text(json.dumps(registration))
        with self.assertRaisesRegex(BoundaryError, 'never re-split'):
            TeamNQBoundary(root)

    def test_protected_content_is_skipped_before_JSON_decoding(self):
        p = self.source()
        with p.open('ab') as handle:
            handle.write(b'{"sample_id":"' + self.test_id.encode() + b'","gold":MUST_NOT_DECODE}\n')
        rows, counts = self.boundary.read_role_records(p)
        self.assertEqual(counts['skipped_test'], 1)
        self.assertEqual({r: len(v) for r, v in rows.items()}, {'train': 2, 'validation': 2})

    def test_unrequested_validation_content_is_not_decoded(self):
        p = self.folder / 'only_train.jsonl'
        dump(p, [fictional(self.train[0])])
        with p.open('ab') as handle:
            handle.write(b'{"sample_id":"' + self.validation[0].encode() + b'","gold":INVALID}\n')
        rows, counts = self.boundary.read_role_records(p, roles=('train',))
        self.assertEqual(len(rows['train']), 1)
        self.assertEqual(counts['skipped_validation'], 1)

    def test_duplicate_and_ambiguous_ID_fields_are_rejected(self):
        p = self.folder / 'bad.jsonl'
        for raw in [json.dumps(fictional(self.train[0])) + '\n' + json.dumps(fictional(self.train[0])),
                    '{"sample_id":"' + self.train[0] + '","sample_id":"' + self.test_id + '"}']:
            p.write_text(raw)
            with self.assertRaises(BoundaryError):
                self.boundary.read_role_records(p)

    def test_original_shards_are_all_hashed_before_any_row_is_decoded(self):
        fake = copy.copy(self.boundary)
        folder = self.folder / 'shards'
        folder.mkdir()
        a, b = folder / 'a.jsonl', folder / 'b.jsonl'
        dump(a, [fictional(self.train[0])])
        dump(b, [fictional(self.validation[0])])
        fake.delivery = {'shards': [{'file': p.name, 'sha256': sha(p)} for p in (a, b)]}
        b.write_text('tampered')
        with patch('src.data.team_nq.json.loads', side_effect=AssertionError('must not decode')):
            with self.assertRaisesRegex(BoundaryError, 'shard missing or changed'):
                fake.read_role_records(folder)

    def test_parent_and_donor_IDs_cannot_cross_roles(self):
        for key, value in [('source_sample_ids', [self.test_id]),
                           ('parent_sample_ids', self.validation[:1]),
                           ('donor_sample_id', self.validation[0])]:
            row = fictional(self.train[0])
            row[key] = value
            with self.assertRaises(BoundaryError):
                self.boundary.require_record(row, 'train')

    def test_borrowed_answer_requires_donor_provenance(self):
        row = fictional(self.train[0]); row['source'] = 'borrowed_gold'
        with self.assertRaisesRegex(BoundaryError, 'donor provenance'):
            self.boundary.require_record(row, 'train')

    def test_synthetic_data_requires_parent_provenance(self):
        row = fictional(self.train[0]); row['source_type'] = 'counterfactual'
        with self.assertRaisesRegex(BoundaryError, 'parent IDs'):
            self.boundary.require_record(row, 'train')
        row['parent_sample_ids'] = [self.train[0]]
        self.boundary.require_record(row, 'train')

    def test_NQ_policy_is_not_a_FEVER_policy(self):
        row = fictional(self.train[0]); row['dataset'] = 'fever'
        with self.assertRaisesRegex(BoundaryError, 'NQ only'):
            self.boundary.require_record(row, 'train')

    def test_teacher_payload_excludes_offline_gold_labels_and_extra_evidence_fields(self):
        row = fictional(self.train[0])
        row['evidence'][0]['gold_answer'] = 'OFFLINE_ONLY'
        payload = self.boundary.teacher_input(row)
        self.assertEqual(set(payload), {'sample_id', 'dataset', 'query', 'initial_answer', 'evidence'})
        self.assertTrue(all(set(e) == {'rank', 'passage_id', 'text', 'score'} for e in payload['evidence']))
        self.assertNotIn('OFFLINE_ONLY', json.dumps(payload))

    def test_teacher_payload_rejects_validation_test_and_malformed_top5(self):
        for sid in [self.validation[0], self.test_id]:
            with self.assertRaises(BoundaryError):
                self.boundary.teacher_input(fictional(sid))
        for mutate in [lambda r: r['evidence'].pop(),
                       lambda r: r['evidence'][0].update(score=float('nan')),
                       lambda r: r['evidence'][0].update(rank=True)]:
            row = fictional(self.train[0]); mutate(row)
            with self.assertRaises(BoundaryError):
                self.boundary.teacher_input(row)

    def test_selector_partitions_before_borrowing_and_writes_no_test_file(self):
        path = self.bundle()
        parts, manifest = self.boundary.read_training_bundle(path, component='answer_selector')
        for role, rows in parts.items():
            ids = {r['sample_id'] for r in rows}
            self.assertEqual(ids, set(getattr(self, role)[:2]))
            for row in rows:
                if row['source'] == 'borrowed_gold':
                    self.assertIn(row['donor_sample_id'], ids)
        self.assertFalse((path / 'test.jsonl').exists())
        self.assertTrue(manifest['boundary']['test_sealed'])

    def test_reranker_uses_official_roles_without_a_second_random_split(self):
        with patch('scripts.build_rerank_training_data.split_examples', side_effect=AssertionError('no re-split')):
            path = self.bundle(component='reranker')
        parts, _ = self.boundary.read_training_bundle(path, component='reranker')
        self.assertEqual({r: len(v) for r, v in parts.items()}, {'train': 2, 'validation': 2})

    def test_normalized_query_overlap_stops_before_output_is_created(self):
        from scripts.build_selector_training_data import build_examples
        rows = [fictional(self.train[0]), fictional(self.validation[0])]
        rows[1]['query'] = rows[0]['query'].upper() + '  '
        p = self.folder / 'overlap.jsonl'; dump(p, rows)
        out = self.folder / 'rejected'
        with self.assertRaisesRegex(BoundaryError, 'Normalized query'):
            prepare_training_bundle(self.boundary, p, out, build_examples, component='answer_selector')
        self.assertFalse(out.exists())

    def test_bundle_cannot_overwrite_prior_data_or_accept_a_changed_file(self):
        path = self.bundle()
        from scripts.build_selector_training_data import build_examples
        with self.assertRaises(FileExistsError):
            prepare_training_bundle(self.boundary, self.source(), path, build_examples, component='answer_selector')
        with (path / 'train.jsonl').open('ab') as handle: handle.write(b' ')
        with self.assertRaisesRegex(BoundaryError, 'changed after'):
            self.boundary.read_training_bundle(path)

    def test_old_unregistered_or_wrong_component_bundles_fail(self):
        with self.assertRaisesRegex(BoundaryError, 'Unregistered'):
            self.boundary.read_training_bundle(self.folder)
        path = self.bundle()
        with self.assertRaisesRegex(BoundaryError, 'component'):
            self.boundary.read_training_bundle(path, component='reranker')
        (path / 'test.jsonl').write_text('Do not read this file')
        with self.assertRaisesRegex(BoundaryError, 'must not contain a test'):
            self.boundary.read_training_bundle(path)

    def test_training_role_preflight_runs_before_models_or_CUDA(self):
        from scripts import train_answer_selector, train_rerank_accessibility
        for module in (train_answer_selector, train_rerank_accessibility):
            with patch.object(module.AutoTokenizer, 'from_pretrained') as tokenizer, \
                 patch.object(module, 'set_seed') as seed:
                with self.assertRaises(BoundaryError):
                    module.main(['--data-dir', str(self.folder), '--out-dir', str(self.folder / 'model'),
                                 '--metrics-out', str(self.folder / 'metrics.json')])
                tokenizer.assert_not_called(); seed.assert_not_called()
        self.assertFalse((self.folder / 'model').exists())

    def test_modified_receipted_test_ID_is_still_blocked_before_model_loading(self):
        from scripts import train_answer_selector
        path = self.bundle()
        p = path / 'train.jsonl'; rows = [json.loads(line) for line in p.read_text().splitlines()]
        rows[0]['sample_id'] = self.test_id; dump(p, rows)
        manifest = json.loads((path / 'build_manifest.json').read_text())
        manifest['role_files']['train']['sha256'] = sha(p)
        (path / 'build_manifest.json').write_text(json.dumps(manifest))
        with patch.object(train_answer_selector.AutoTokenizer, 'from_pretrained') as tokenizer:
            with self.assertRaises(BoundaryError):
                train_answer_selector.main(['--data-dir', str(path), '--out-dir', str(self.folder / 'model'),
                                            '--metrics-out', str(self.folder / 'metrics.json')])
            tokenizer.assert_not_called()

    def test_live_host_blocks_test_and_FEVER_before_diagnosis_or_enhancement(self):
        from scripts.run_controller_integration import smoke_inputs, smoke_pipeline
        from src.controller.execution import ExecutionConfig
        from src.pipeline.current import wire_latest_pipeline
        from tests.test_controller_current import FictionalFeatures, ForcedSelection
        from src.diagnosis.labels import DiagnosisLabel
        from dataclasses import replace
        cfg = json.loads((ROOT / REGISTRATION).read_text(encoding='utf-8'))
        backend = smoke_pipeline(ExecutionConfig(**cfg['execution_config']))
        pipeline = wire_latest_pipeline(backend, FictionalFeatures(),
            selection_adapter=ForcedSelection(DiagnosisLabel.CORRECT, selected=False),
            data_boundary=self.boundary, data_role='train', registration='configs/controller/current_baseline.json')
        with patch.object(pipeline.host, 'run_from_item') as host:
            for item in [replace(smoke_inputs()[0], sample_id=self.test_id), smoke_inputs()[1]]:
                with self.assertRaises(BoundaryError): pipeline.run_from_item(item)
            host.assert_not_called()

    def test_live_host_records_the_enforced_role_and_split_fingerprints(self):
        from scripts.run_controller_integration import smoke_inputs, smoke_pipeline
        from src.controller.execution import ExecutionConfig
        from src.pipeline.current import wire_latest_pipeline
        from tests.test_controller_current import FictionalFeatures, ForcedSelection
        from src.diagnosis.labels import DiagnosisLabel
        from dataclasses import replace
        cfg = json.loads((ROOT / REGISTRATION).read_text(encoding='utf-8'))
        backend = smoke_pipeline(ExecutionConfig(**cfg['execution_config']))
        pipeline = wire_latest_pipeline(backend, FictionalFeatures(),
            selection_adapter=ForcedSelection(DiagnosisLabel.CORRECT, selected=False),
            data_boundary=self.boundary, data_role='train', registration='configs/controller/current_baseline.json')
        result = pipeline.run_from_item(replace(smoke_inputs()[0], sample_id=self.train[0]))
        self.assertEqual(result.metadata['data_role'], 'train')
        self.assertEqual(result.metadata['data_boundary'], {
            'enforced': True, 'policy_sha256': self.boundary.receipt()['policy_sha256'],
            'split_sha256': self.boundary.policy['team_split']['sha256']})

    def test_current_loader_cannot_request_test_execution_before_GPU_loading(self):
        from src.pipeline.current import load_current_pipeline
        with patch('src.controller.providers.load_pipeline') as model:
            with self.assertRaisesRegex(BoundaryError, 'T7'):
                load_current_pipeline(data_role='test')
            model.assert_not_called()

    def test_component_evaluator_cannot_bypass_roles_with_a_custom_subset(self):
        from scripts import evaluate_refinement_end_to_end as evaluator
        p = self.source()
        subset = self.folder / 'subset.jsonl'
        dump(subset, [{'sample_id': self.test_id}])
        with patch.object(evaluator, 'build_refiner') as model:
            with self.assertRaises(BoundaryError):
                evaluator.main(['--records', str(p), '--split', str(subset), '--no-split-filter',
                                '--out', str(self.folder / 'result.json')])
            model.assert_not_called()
        self.assertFalse((self.folder / 'result.json').exists())

    def test_historical_evaluation_flag_is_not_an_escape_for_team_test(self):
        from scripts import evaluate_refinement_end_to_end as evaluator
        p = self.folder / 'fake_historical.jsonl'
        dump(p, [fictional(self.test_id)])
        with patch.object(evaluator, 'frozen_500_records') as reader, patch.object(evaluator, 'build_refiner') as model:
            with self.assertRaisesRegex(ValueError, 'original frozen'):
                evaluator.main(['--records', str(p), '--frozen-500', '--out', str(self.folder / 'result.json')])
            reader.assert_not_called(); model.assert_not_called()

    def test_comparison_cache_cannot_decode_test_answers_before_model_loading(self):
        from scripts import evaluate_refinement_end_to_end as evaluator
        cache = self.folder / 'unregistered_cache.json'
        cache.write_text('{"rows":[{"sample_id":"' + self.test_id + '","initial_answer":MUST_NOT_DECODE}]}')
        with patch.object(evaluator, 'load_before_answers') as reader, \
             patch.object(evaluator, 'build_refiner') as model, redirect_stdout(io.StringIO()):
            with self.assertRaises(BoundaryError):
                evaluator.main(['--records', str(self.source()), '--before-from', str(cache),
                                '--out', str(self.folder / 'result.json')])
            reader.assert_not_called(); model.assert_not_called()

    def test_valid_comparison_cache_is_checked_before_model_loading(self):
        from scripts import evaluate_refinement_end_to_end as evaluator
        cache = self.folder / 'validation_cache.json'
        cache.write_text(json.dumps({'rows':[{'sample_id':sid, 'initial_answer':'Fictional cache'}
                                             for sid in self.validation[:2]]}))
        with patch.object(evaluator, 'load_before_answers', wraps=evaluator.load_before_answers) as reader, \
             patch.object(evaluator, 'build_refiner', side_effect=RuntimeError('fake-model-stop')) as model, \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'fake-model-stop'):
                evaluator.main(['--records', str(self.source()), '--before-from', str(cache),
                                '--out', str(self.folder / 'result.json')])
            reader.assert_called_once(); model.assert_called_once()

    def test_reranker_comparison_cannot_select_test(self):
        from scripts import compare_rerank_models as compare
        with patch.object(compare, 'positive_ranks') as model, redirect_stdout(io.StringIO()), \
                redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): compare.main(['--split', 'test'])
            model.assert_not_called()

    def test_build_CLIs_use_the_role_boundaries_and_never_resplit(self):
        from scripts import build_rerank_training_data as reranker, build_selector_training_data as selector
        for module in (reranker, selector):
            out = self.folder / module.__name__.split('.')[-1]
            with redirect_stdout(io.StringIO()):
                module.main(['--records', str(self.source()), '--out-dir', str(out)])
            parts, _ = self.boundary.read_training_bundle(out)
            self.assertEqual({r: len({row['sample_id'] for row in rows}) for r, rows in parts.items()},
                             {'train': 2, 'validation': 2})

    def test_ID_audit_recomputes_population_labels_and_known_exposure(self):
        from scripts.audit_team_nq_data import audit
        report = audit(ROOT, inspect_source=False)
        self.assertTrue(report['official_split_byte_identical'])
        self.assertEqual(report['local_label_ID_population'], {'n': 74097, 'missing': 0, 'extra': 0, 'duplicates': 0})
        self.assertTrue(report['train_validation_label_count_metadata_matches'])
        self.assertEqual(report['known_test_exposure'], 603)
        self.assertEqual(report['test_outside_registered_exposure'], 10511)
        self.assertFalse(report['remaining_test_unseen_proven'])
        self.assertFalse(report['test_question_evidence_gold_decoded'])


if __name__ == '__main__':
    unittest.main()
