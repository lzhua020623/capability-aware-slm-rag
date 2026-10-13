"""Replay saved v5 model outputs on CPU, then compare the four frozen arms."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_answer_repair_experiment import encode_json, sha256, write_frozen
from scripts.run_answer_repair import prepare_records, read_jsonl, read_existing, run_fingerprint
from src.compensation.answer_repair import Generation, repair_answer
from src.diagnosis import DatasetName
from src.evaluation.answer_repair import summarize_repairs


def equal(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


class ReplayGenerator:
    def __init__(self, row):
        guard = row.get('support_check')
        def candidate_cost(field):
            total = row.get(field)
            return total - guard[field] if guard and total is not None and guard[field] is not None else total
        framed = row.get('validation_input')
        self.outputs = [Generation(row['raw_output'] if framed is None else framed,
                                   candidate_cost('latency'), candidate_cost('input_tokens'),
                                   candidate_cost('output_tokens_estimate'), raw_text=row['raw_output'])]
        if guard:
            self.outputs.append(Generation(guard['validation_input'], guard['latency'],
                                           guard['input_tokens'], guard['output_tokens_estimate'],
                                           raw_text=guard['raw_output']))
        self.calls = 0

    def generate(self, prompt, max_new_tokens):
        result = self.outputs[self.calls]
        self.calls += 1
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_raw = (args.bundle_dir / 'manifest.json').read_bytes()
    if (args.results_dir / 'manifest.json').read_bytes() != manifest_raw:
        raise ValueError('Export manifest differs from frozen bundle')
    manifest = json.loads(manifest_raw)
    report = {'cohort_version': manifest['version'], 'cpu_replay_only': True, 'runs': {}, 'candidate_comparison': {}}
    for ds in ('fever','nq'):
        source = (args.bundle_dir / f'inputs/{ds}_reviewed_c1.jsonl').read_bytes()
        assert sha256(source) == manifest['datasets'][ds]['input_sha256']
        prepared = prepare_records(read_jsonl(source, ds), DatasetName(ds))
        all_results = {}
        for arm in manifest['protocol']:
            strategy = arm['strategy']
            path = args.results_dir / f'{ds}_{strategy}.jsonl'
            rows = read_jsonl(path.read_bytes(), str(path))
            if len(rows) != len(prepared): raise ValueError('Incomplete result set')
            settings = rows[0]['settings']
            if (settings['strategy'] != strategy or settings['max_new_tokens'] != arm['max_new_tokens']
                    or settings['routing'] != 'all'
                    or settings['support_check_max_new_tokens'] != arm.get('support_check_tokens', 0)
                    or any(r['settings'] != settings for r in rows)):
                raise ValueError('Result strategy/settings differ from manifest protocol')
            read_existing(path, prepared, run_fingerprint(source, b'', settings))
            for row, (_, request) in zip(rows, prepared):
                generator = ReplayGenerator(row)
                replay = repair_answer(request, generator, strategy=strategy, max_new_tokens=arm['max_new_tokens'])
                if generator.calls != row['extra_calls'] or not equal(asdict(replay), {k:row[k] for k in asdict(replay)}):
                    raise ValueError(f'CPU replay differs: {ds}/{strategy}/{request.sample_id}')
            expected = {'run_fingerprint': rows[0]['run_fingerprint'], 'routing': 'all', 'strategy': strategy,
                        **summarize_repairs(rows)}
            if not equal(expected, json.loads(path.with_suffix('.summary.json').read_bytes())):
                raise ValueError('Saved summary differs from replay')
            groups = {r['sample_id']: r['use'] for r in manifest['included'] if r['dataset']==ds}
            report['runs'][f'{ds}/{strategy}'] = {**expected, 'groups': {
                group: summarize_repairs([r for r in rows if groups[r['sample_id']]==group])
                for group in ('include_e_core','correct_control')}}
            all_results[strategy] = rows
        report['candidate_comparison'][ds] = {'candidate_output_mismatch_ids': [a['sample_id'] for a,b in
            zip(all_results['query_grounded'], all_results['guarded_reanswer'])
            if a['raw_output'] != b['raw_output'] or a['validation_input'] != b['validation_input']]}
    report['replayed_records'] = sum(r['evaluated_samples'] for r in report['runs'].values())
    report['semantic_correctness_verified'] = False
    write_frozen(args.out, encode_json(report))
    print(json.dumps({'replayed_records':report['replayed_records'], 'candidate_comparison':report['candidate_comparison']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
