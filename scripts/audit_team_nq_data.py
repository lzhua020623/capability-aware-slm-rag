"""ID-only team alignment plus streaming structure checks on admitted roles.

No test questions, evidence or gold are decoded. This audits and never
regenerates the team split, trains models or changes prior experiment outputs.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.team_nq import (BoundaryError, TeamNQBoundary, ids_only, numeric_id,
                             query_key, read_json, sha)


def exposure_ids(root, source):
    if source['kind'] == 'E_exposure_registry':
        return set(read_json(root / source['path'])['known_exposed_ids'])
    if source['kind'] == 'controller_development_and_selection':
        ids = read_json(root / source['path'])['nq']
        return set(ids['development']) | set(ids['selection'])
    return ids_only(root / source['path'])


def audit(root=ROOT, *, inspect_source=True):
    root = Path(root)
    boundary = TeamNQBoundary(root)
    policy = boundary.policy
    original = subprocess.check_output(['git', 'show', policy['source_commit'] + ':' +
                                       policy['team_split']['path']], cwd=root)
    if hashlib.sha256(original).hexdigest() != policy['team_split']['sha256']:
        raise BoundaryError('Copied split differs from the original team Git blob')
    rows = []
    with (root / policy['label_index']['path']).open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            numeric_id(row['sample_id'])
            if row['dataset'] != 'nq' or row['primary_label'] not in {
                    'correct', 'evidence_insufficiency', 'evidence_accessibility', 'evidence_utilisation'}:
                raise BoundaryError('Invalid local label metadata')
            rows.append(row)
    by_id = {r['sample_id']: r['primary_label'] for r in rows}
    if len(by_id) != len(rows) or set(by_id) != set(boundary.membership):
        raise BoundaryError('Local 74097 ID population differs from team split')
    split = read_json(root / policy['team_split']['path'])
    label_counts = {}
    for role in ('train', 'validation'):
        actual = dict(Counter(by_id[sid] for sid in boundary.splits[role]))
        expected = split['summary']['label_distribution'][role]['counts']
        if actual != expected:
            raise BoundaryError('Train/validation label-count metadata differs from the team split')
        label_counts[role] = actual
    known = read_json(root / policy['known_exposure']['path'])
    observed, sources = set(), []
    for source in known['sources']:
        ids = exposure_ids(root, source)
        canonical = ids & boundary.membership.keys()
        observed |= canonical
        sources.append({'path': source['path'], 'canonical_n': len(canonical),
                        'by_team_role': {r: len(canonical & set(v)) for r, v in boundary.splits.items()}})
    if observed != boundary.exposed:
        raise BoundaryError('Known exposure registry does not match its source ID fields')
    old = read_json(root / policy['legacy_selection']['path'])['nq']
    report = {'schema_version': 'team_nq_alignment_audit_v1',
        'boundary': boundary.receipt(), 'official_split_byte_identical': True,
        'local_label_ID_population': {'n': len(by_id), 'missing': 0, 'extra': 0, 'duplicates': 0},
        'train_validation_label_count_metadata_matches': True, 'label_counts': label_counts,
        'eligible_AI_label_counts': {r: dict(Counter(by_id[sid] for sid in boundary.allowed_ids(r)))
                                    for r in ('train', 'validation')},
        'legacy_controller_by_team_role': {old_role: {role: len(set(ids) & set(team_ids))
            for role, team_ids in boundary.splits.items()} for old_role, ids in old.items()},
        'exposure_sources': sources, 'known_exposure_unique': len(observed),
        'known_test_exposure': len(observed & set(boundary.splits['test'])),
        'test_outside_registered_exposure': len(set(boundary.splits['test']) - observed),
        'remaining_test_unseen_proven': False, 'FEVER_team_alignment': 'not_registered',
        'test_question_evidence_gold_decoded': False,
        'model_training_or_API_inference': False}
    source_dir = root / policy['source_delivery']['default_local_directory']
    if inspect_source:
        paths = boundary.source_paths(source_dir)
        role_counts, structure = Counter(), {r: Counter() for r in ('train', 'validation')}
        query_roles, bad = {}, {r: [] for r in structure}
        for role, row in boundary.iter_role_records(source_dir, counts=role_counts):
            problems = []
            try:
                q = query_key(row.get('query'))
                digest = hashlib.sha256(q.encode()).hexdigest()
                query_roles.setdefault(digest, set()).add(role)
            except BoundaryError:
                problems.append('invalid_query')
            if not isinstance(row.get('initial_answer'), str) or not row['initial_answer'].strip():
                problems.append('empty_initial_answer')
            evidence = row.get('evidence')
            if not isinstance(evidence, list) or len(evidence) != 5:
                problems.append('invalid_top5_count')
            else:
                evidence_ids = set()
                for rank, e in enumerate(evidence, 1):
                    if (type(e.get('rank')) is not int or e['rank'] != rank
                            or not isinstance(e.get('passage_id'), str) or not e['passage_id']
                            or e['passage_id'] in evidence_ids
                            or not isinstance(e.get('text'), str) or not e['text'].strip()
                            or type(e.get('score')) not in (int, float) or not math.isfinite(e['score'])):
                        problems.append('invalid_evidence_structure')
                        break
                    evidence_ids.add(e['passage_id'])
            gold = (row.get('annotation_reference') or {}).get('gold_answer')
            if not isinstance(gold, str) or not gold.strip():
                problems.append('missing_gold_reference')
            structure[role]['inspected'] += 1
            for issue in set(problems):
                structure[role][issue] += 1
            if problems:
                bad[role].append(row['sample_id'])
        report['original_delivery'] = {'shards': len(paths), 'all_shard_hashes_match': True,
            'raw_role_counts': dict(role_counts), 'structure': {r: dict(v) for r, v in structure.items()},
            'structural_issue_ids': bad,
            'normalized_query_fingerprints_crossing_admitted_roles': sum(len(v) > 1 for v in query_roles.values()),
            'semantic_answer_correctness_verified': False,
            'historical_C1_is_new_context_fallback_baseline': False}
    files = ['baselines/outputs/rag_matched_test_context_fallback/' + name
             for name in ('predictions.jsonl', 'metrics.json', 'run_config.json')]
    report['paired_baseline_files'] = {p: (root / p).is_file() for p in files}
    report['old_local_training_files'] = {p: (root / p).is_file() for p in (
        'data/processed/rerank/train.jsonl', 'data/processed/rerank/validation.jsonl',
        'data/processed/selector/train.jsonl', 'data/processed/selector/validation.jsonl')}
    report['unverified'] = ['AI labels are not human gold; no semantic-clean claim',
        'Exact label counts do not prove every per-ID label matches the absent original Diagnosis records',
        'P60 and previously trained reranker/selector usage IDs not delivered',
        'Near-duplicate question checks against sealed test content were not performed',
        'New matched baseline C1 predictions/recipe alignment is pending; historical C1 remains distinct']
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ids-only', action='store_true', help='skip even admitted-role source structure checks')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError('Preserve the previous audit')
    report = audit(inspect_source=not args.ids_only)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'boundary': report['boundary'], 'original_delivery': report.get('original_delivery'),
                      'known_test_exposure': report['known_test_exposure']}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
