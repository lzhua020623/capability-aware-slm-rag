"""Replay a release with its OWN frozen code; build a deferred human review queue.

Usage: python scripts/audit_answer_repair_release.py --bundle-dir .../gpu_bundle
       --results-dir .../colab_20261014 --out-dir .../audit
No GPU, model loading, inference, or automatic semantic truth labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace


def equal(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    bundle, results = args.bundle_dir.resolve(), args.results_dir.resolve()
    checksums = json.loads((bundle / 'bundle_checksums.json').read_bytes())
    for name, digest in checksums.items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen source checksum mismatch: {name}')
    for name in ('manifest.json', 'bundle_checksums.json'):
        if (bundle / name).read_bytes() != (results / name).read_bytes():
            raise ValueError(f'Export differs from bundle: {name}')
    # Import only after adding the frozen tree. The latest working copy may
    # have changed since inference; its validators must not reinterpret a run.
    if any(n.startswith(('src.', 'scripts.run_answer_repair')) for n in sys.modules):
        raise RuntimeError('Run this audit in a fresh Python process')
    sys.path.insert(0, str(bundle))
    from scripts.prepare_answer_repair_experiment import encode_json, sha256, write_frozen
    from scripts.run_answer_repair import prepare_records, read_jsonl, read_existing, run_fingerprint
    from scripts.summarize_reviewed_answer_repair import result_digest
    from src.compensation.answer_repair import Generation, repair_answer
    from src.diagnosis import DatasetName
    from src.evaluation.answer_repair import summarize_repairs

    class ReplayGenerator:
        def __init__(self, row):
            staged = row.get('repair_trace')
            if staged is not None:
                self.outputs = [Generation(**stage['generation']) for stage in staged['stages']]
                self.calls = 0
                return
            guard = row.get('support_check')
            def cost(field):
                total = row.get(field)
                return total - guard[field] if guard and total is not None and guard[field] is not None else total
            framed = row.get('validation_input')
            self.outputs = [Generation(row['raw_output'] if framed is None else framed,
                                      cost('latency'), cost('input_tokens'), cost('output_tokens_estimate'),
                                      raw_text=row['raw_output'])]
            if guard:
                self.outputs.append(Generation(guard['validation_input'], guard['latency'],
                                               guard['input_tokens'], guard['output_tokens_estimate'],
                                               raw_text=guard['raw_output']))
            self.calls = 0

        def generate(self, prompt, max_new_tokens):
            value = self.outputs[self.calls]
            self.calls += 1
            return value

    manifest = json.loads((bundle / 'manifest.json').read_bytes())
    report = {'version': manifest['version'], 'cpu_replay_only': True,
              'semantic_correctness_verified': False, 'runs': {}, 'replayed_records': 0}
    queue, unique_ids = [], set()
    # (cohort, dataset, manifest section with sample_ids/input_sha256)
    targets = []
    is_v8_nq75 = manifest.get('version') == 'answer_repair_v8_nq75_v1'
    if is_v8_nq75:
        targets.append(('nq75', 'nq', {'sample_ids': manifest['sample_ids'],
                       'input_sha256': manifest['source_input_sha256']['scored_c1']}))
    if manifest.get('cohort') == 'heldout3216':          # single-cohort run-once bundle
        lines = (bundle / 'inputs/nq_heldout3216_c1.jsonl').read_bytes().decode('utf-8').splitlines()
        targets.append(('heldout3216', manifest['dataset'],
                        {'sample_ids': [json.loads(l)['sample_id'] for l in lines if l.strip()],
                         'input_sha256': manifest['input_sha256']}))
    cohort_keys = {'reviewed': 'datasets', 'dev50': 'regression_dev50', 'heldout': 'heldout_prelim'}
    for cohort in [c for c in ('reviewed', 'dev50', 'heldout') if cohort_keys[c] in manifest]:
        section = manifest[cohort_keys[cohort]]
        for ds in ('fever', 'nq'):
            targets.append((cohort, ds, (section['datasets'] if cohort == 'heldout' else section)[ds]))
    for cohort, ds, source_manifest in targets:
        if True:
            source_path = 'inputs/nq75_c1.jsonl' if cohort == 'nq75' else f'inputs/{ds}_{cohort}_c1.jsonl'
            source = (bundle / source_path).read_bytes()
            if sha256(source) != source_manifest['input_sha256']:
                raise ValueError('Input checksum differs from manifest')
            prepared = prepare_records(read_jsonl(source, ds), DatasetName(ds))
            for arm in manifest['protocol']:
                if arm.get('dataset', ds) != ds:
                    continue
                strategy = arm['strategy']
                stem = strategy if cohort == 'nq75' else f'{ds}_{strategy}' + ('' if cohort == 'reviewed' else f'_{cohort}')
                path = results / (stem + '.jsonl')
                rows = read_jsonl(path.read_bytes(), str(path))
                if not rows or [r['sample_id'] for r in rows] != source_manifest['sample_ids']:
                    raise ValueError(f'Incomplete or reordered results: {stem}')
                settings = rows[0]['settings']
                if (settings['strategy'] != strategy or settings['max_new_tokens'] != arm['max_new_tokens']
                        or settings['routing'] != 'all' or settings['support_check_max_new_tokens'] != arm.get('support_check_tokens', 0)
                        or any(r['settings'] != settings for r in rows)):
                    raise ValueError('Result settings differ from frozen protocol')
                read_existing(path, prepared, run_fingerprint(source, b'', settings))
                for row, (original, request) in zip(rows, prepared):
                    if strategy in ('fever_nli', 'fever_nli_anchored'):
                        replay = repair_answer(request, strategy=strategy,
                            verifier=SimpleNamespace(score=lambda data: row['verifier_trace']))
                    else:
                        generator = ReplayGenerator(row)
                        replay = repair_answer(request, generator, strategy=strategy, max_new_tokens=arm['max_new_tokens'])
                        if generator.calls != row['extra_calls']:
                            raise ValueError('Replayed generation call count differs')
                    expected = asdict(replay)
                    if not equal(expected, {k: row[k] for k in expected}):
                        raise ValueError(f'CPU replay mismatch: {stem}/{request.sample_id}')
                    unique_ids.add((ds, request.sample_id))
                    queue.append({'sample_id': request.sample_id, 'dataset': ds, 'strategy': strategy,
                                  'cohort': cohort, 'result_path': str(path), 'result_sha256': result_digest(row),
                                  'query': request.query, 'initial_answer': row['initial_answer'],
                                  'final_answer': row['final_answer'], 'changed': row['changed'],
                                  'model_result': row, 'evidence': original['evidence'],
                                  'frozen_reference': original['gold_answer_or_label'],
                                  'pair_review': {'status': 'pending', 'reviewer': '', 'reason': '',
                                                  'initial_correct': None, 'final_correct': None,
                                                  'initial_supported': None, 'final_supported': None}})
                expected = {'run_fingerprint': rows[0]['run_fingerprint'], 'routing': 'all', 'strategy': strategy,
                            **summarize_repairs(rows)}
                if not equal(expected, json.loads(path.with_suffix('.summary.json').read_bytes())):
                    raise ValueError(f'Summary mismatch: {stem}')
                report['runs'][stem] = expected
                report['replayed_records'] += len(rows)
    if not targets:
        raise ValueError('Unknown release cohort; refusing an empty audit')
    report['unique_dataset_sample_ids'] = len(unique_ids)
    report['manual_review_status'] = 'pending_output_review' if is_v8_nq75 else 'deferred_by_user'
    report['pending_result_pairs'] = len(queue)
    write_frozen(args.out_dir / 'cpu_replay.json', encode_json(report))
    write_frozen(args.out_dir / 'pending_semantic_pairs.jsonl',
                 ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in queue).encode())
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
