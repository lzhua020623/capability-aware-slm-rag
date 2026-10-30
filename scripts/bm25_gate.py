"""BM25-only replacement inside P3-20 (configs/controller/bm25_gate_v1.json).

On items route 1 sends to P3-20, the L2 setting bge20_bm2520_ce_only replaces route 1's answer only when its
Top-5 holds a passage that BM25 brought in (one outside the BGE top 20); every other item keeps route 1's answer.
`replay` gives the sweep and confirm numbers from stored outputs. The test-* steps work from the route-1 Test
archive (team-format predictions and judge files): export the registered Test IDs, build the generation
contexts, write the judge inputs, and score the paired comparison with route 1.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from baselines._lib.answer_scoring import score_answer  # noqa: E402
from scripts import l2_confirm, l2_offline  # noqa: E402
from scripts import score_quality_eval as scorer  # noqa: E402
from scripts.quality_sweep_offline import load_judged, read_jsonl  # noqa: E402

REGISTRATION = 'configs/controller/bm25_gate_v1.json'
SETTING = {'k_bge': 20, 'k_bm25': 20, 'fusion': 'ce_only'}
ROUTE_1_TEST = 'artifacts/controller/quality_test_v1/results.tar.gz'
TEAM = ('qwen_only', 'product', 'rag')
P3 = 'retrieval_enhancement'
sha = lambda data: hashlib.sha256(data).hexdigest()


def load_registration(root=ROOT):
    return json.loads((Path(root) / REGISTRATION).read_text(encoding='utf-8'))


def brings_bm25(row, ids):
    bge = {p for p, _ in row['bge'][:SETTING['k_bge']]}
    return any(p not in bge for p in ids)


def bm25_answer(c1, row, base, generations, sid):
    """The L2 answer (P3-20 rules) when its Top-5 holds a BM25-only passage, else None."""
    ids = l2_offline.select_ids(row, SETTING)
    if not brings_bm25(row, ids):
        return None
    return l2_offline.final_answer(c1, ids, base, generations, sid)


def gate(c1, route_1, replacement):
    """Route 1's answers, replaced where the BM25 rule gives an answer (only on route 1's P3 items)."""
    return {sid: replacement[sid] if replacement.get(sid) is not None else route_1[sid] for sid in c1}


# ---- replay on the sweep and confirm ----

def replay(role, args):
    c1, route_1, replacement = {}, {}, {}
    if role == 'sweep':
        records, candidates, generations = l2_offline.load_inputs(args)
        for r in records:
            sid = r['sample_id']
            c1[sid] = r['initial_answer']
            route_1[sid] = r['actions'][P3]['final_answer'] if P3 in r['actions'] else c1[sid]
            if P3 in r['actions']:
                base = tuple(e['passage_id'] for e in r['evidence'])
                replacement[sid] = bm25_answer(c1[sid], candidates[sid], base, generations, sid)
    else:
        results, candidates, generations = l2_confirm.load_inputs(args)
        for r in results:
            sid = r['sample_id']
            c1[sid], route_1[sid] = r['initial_answer'], r['final_answer']
            if r['controller_decision']['action'] == P3:
                replacement[sid] = bm25_answer(c1[sid], candidates[sid], l2_confirm.base_ids(r), generations, sid)
    gold = {r['sample_id']: r['gold'] for r in read_jsonl(args.references)}
    judged = load_judged(args.judged)
    after = gate(c1, route_1, replacement)
    return {'role': role, 'n': len(c1), 'routed': len(replacement),
            'bm25_items': sum(v is not None for v in replacement.values()),
            'against_route_1': l2_confirm.paired(route_1, after, gold, judged),
            'against_c1': {'route_1': l2_confirm.paired(c1, route_1, gold, judged),
                           'gated': l2_confirm.paired(c1, after, gold, judged)}}


# ---- Test ----

def route_1_test(archive, tmp):
    """Route-1 Test predictions plus the judge files in the registered label order (RAG last)."""
    with tarfile.open(archive) as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    paths = {}
    for name, data in members.items():
        if name.endswith('judge_results.jsonl'):
            path = Path(tmp) / name.replace('/', '__')
            path.write_bytes(data)
            paths[name] = path
    predictions = [json.loads(line) for line in io.StringIO(members['scored/predictions.jsonl'].decode('utf-8')) if line.strip()]
    ours = [paths[f'judge/outputs/shard_{i}/judge_results.jsonl'] for i in range(4)]
    team = {k: paths[f'team/{k}/judge_results.jsonl'] for k in TEAM}
    return predictions, ours, team


def routed_ids(predictions):
    return [r['sample_id'] for r in predictions if r['controller_action'] == P3]


def check_ids(ids, registration, ids_path):
    if sha(Path(ids_path).read_bytes()) != registration['test']['ids_sha256'] or len(ids) != registration['test']['n_ids']:
        raise ValueError('Test IDs differ from the registered list')


def test_contexts(predictions, candidates, ids):
    by_id = {r['sample_id']: r for r in predictions}
    rows = []
    for sid in ids:
        selected = l2_offline.select_ids(candidates[sid], SETTING)
        if brings_bm25(candidates[sid], selected):
            rows.append({'sample_id': sid, 'query': by_id[sid]['query'], 'passage_ids': list(selected)})
    return rows


def test_rows(predictions, candidates, generations, ids):
    """Route-1 prediction rows with the BM25 rule applied (team format, scores recomputed)."""
    wanted, out = set(ids), []
    for r in predictions:
        sid = r['sample_id']
        answer = None
        if sid in wanted:
            answer = bm25_answer(r['initial_answer'], candidates[sid], (), generations, sid)
        out.append(r if answer is None else {**r, 'prediction': answer, **score_answer(answer, r['gold'])})
    return out


def paired_rows(before, after, labels):
    fixes = harms = changed = 0
    for old, new in zip(before, after):
        if old['prediction'] == new['prediction']:
            continue
        was, now = labels[(old['sample_id'], old['prediction'])], labels[(new['sample_id'], new['prediction'])]
        changed += 1
        fixes += now and not was
        harms += was and not now
    return {'changed': changed, 'fixes': fixes, 'harms': harms, 'net': fixes - harms, 'mcnemar_p': scorer.mcnemar_p(fixes, harms)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('replay')
    r.add_argument('--role', choices=('sweep', 'confirm'), required=True)
    for name in ('--records', '--coverage', '--results'):
        r.add_argument(name, type=Path)
    r.add_argument('--candidates', type=Path, required=True)
    r.add_argument('--generations', type=Path, nargs='*', default=[])
    r.add_argument('--references', type=Path, required=True)
    r.add_argument('--judged', type=Path, nargs='*', default=[])
    r.add_argument('--output', type=Path, required=True)
    i = sub.add_parser('test-ids')
    i.add_argument('--output', type=Path, required=True)
    for cmd in ('test-contexts', 'test-judge-inputs', 'test-score'):
        s = sub.add_parser(cmd)
        s.add_argument('--candidates', type=Path, required=True, help='l2_candidates.py --role test output dir')
        if cmd != 'test-contexts':
            s.add_argument('--generations', type=Path, nargs='+', required=True)
            s.add_argument('--judged', type=Path, nargs='*', default=[], help='new judge batches for this rule')
        s.add_argument('--output-dir' if cmd == 'test-judge-inputs' else '--output', type=Path, required=True)
        if cmd == 'test-judge-inputs':
            s.add_argument('--shards', type=int, default=4)
    args = p.parse_args(argv)
    if args.command == 'replay':
        out = {'schema_version': 'bm25_gate_replay_v1', **replay(args.role, args)}
        args.output.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(out, indent=2))
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        predictions, ours, team = route_1_test(ROOT / ROUTE_1_TEST, tmp)
        if args.command == 'test-ids':
            ids = routed_ids(predictions)
            args.output.write_bytes(('[\n' + ',\n'.join(json.dumps(s) for s in ids) + '\n]\n').replace('\n', '\r\n').encode('utf-8'))
            print(json.dumps({'ids': len(ids), 'sha256': sha(args.output.read_bytes())}))
            return 0
        registration = load_registration()
        ids = json.loads((ROOT / registration['test']['ids']).read_text(encoding='utf-8'))
        check_ids(ids, registration, ROOT / registration['test']['ids'])
        if ids != routed_ids(predictions):
            raise ValueError('Registered Test IDs are not route 1 Test P3 items')
        manifest = json.loads((args.candidates / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['state'] != 'completed' or manifest['role'] != 'test' or manifest['ids_sha256'] != registration['test']['ids_sha256']:
            raise ValueError('Candidates do not come from a completed run on the registered Test IDs')
        candidates = {row['sample_id']: row for row in read_jsonl(args.candidates / 'candidates.jsonl')}
        if set(candidates) != set(ids):
            raise ValueError('Candidate rows do not cover the registered Test IDs')
        if args.command == 'test-contexts':
            rows = test_contexts(predictions, candidates, ids)
            with args.output.open('x', encoding='utf-8') as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(json.dumps({'routed': len(ids), 'contexts': len(rows)}))
            return 0
        generations = {}
        for path in args.generations:
            for g in read_jsonl(path):
                generations[(g['sample_id'], tuple(g['passage_ids']))] = (g['generated_answer'], g['reject_reason'])
        rows = test_rows(predictions, candidates, generations, ids)
        labels = scorer.prior_labels([team['qwen_only'], team['product'], *ours, *args.judged, team['rag']])
        if args.command == 'test-judge-inputs':
            pending = scorer.judge_rows(rows, labels)
            args.output_dir.mkdir(parents=True, exist_ok=False)
            shards = [[] for _ in range(args.shards)]
            for row in pending:
                shards[int(sha(row['sample_id'].encode()), 16) % args.shards].append(row)
            for k, shard in enumerate(shards):
                with (args.output_dir / f'shard_{k}.jsonl').open('x', encoding='utf-8') as handle:
                    for row in shard:
                        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(json.dumps({'rows': len(pending), 'api_rows': sum(not r['normalized_match'] for r in pending)}))
            return 0
        against = paired_rows(predictions, rows, labels)
        out = {'schema_version': 'bm25_gate_test_v1', 'registration_sha256': sha((ROOT / REGISTRATION).read_bytes()),
               'candidates_sha256': manifest['output_sha256'], 'n': len(rows), 'routed': len(ids),
               'replaced': sum(a['prediction'] != b['prediction'] for a, b in zip(predictions, rows)),
               'against_route_1': against, 'score': scorer.score(rows, labels), 'comparisons': scorer.compare(rows, labels, team),
               'adoption': {'rule': registration['test']['adoption'],
                            'passed': against['net'] > 0 and against['mcnemar_p'] < 0.05}}
        args.output.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({k: out[k] for k in ('routed', 'replaced', 'against_route_1', 'adoption')}, indent=2))
        return 0


if __name__ == '__main__': raise SystemExit(main())
