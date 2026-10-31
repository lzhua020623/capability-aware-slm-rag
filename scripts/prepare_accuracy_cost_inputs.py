"""Archive compact cost receipts; no inference, training, policy selection or judge calls.

The scratch directory is optional historical storage, never required for replay after
this input archive is prepared. Original source hashes and projection checks are saved.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import adaptive_selection as ad, bm25_gate as gate, l2_offline as off

RELEASE = ROOT / 'artifacts/controller/accuracy_cost_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def members(path):
    with tarfile.open(path) as t:
        return {m.name: t.extractfile(m).read() for m in t if m.isfile()}


def rows(data):
    return [json.loads(l) for l in data.splitlines() if l.strip()]


def jsonbytes(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode()


def jsonlines(seq):
    return ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in seq).encode()


def prepare(scratch, out=RELEASE):
    out.mkdir(parents=True, exist_ok=True)
    provenance, payload = {}, {}
    def source(path):
        p = Path(path)
        provenance[str(p)] = {'bytes': p.stat().st_size, 'sha256': digest(p)}
        return p
    ev = members(source(ROOT / 'artifacts/controller/adaptive_selection_eval_v1/results.tar.gz'))
    qt = members(source(ROOT / 'artifacts/controller/quality_test_v1/results.tar.gz'))
    bm = members(source(ROOT / 'artifacts/controller/bm25_gate_test_v1/results.tar.gz'))
    predictions = {r['sample_id']: r for r in rows(qt['scored/predictions.jsonl'])}
    items = {r['sample_id']: r for r in rows(ev['test/items.jsonl'])}
    candidates = {}
    for name in ('cand_test', 'cand_test_unrouted'):
        for r in ad.read_results(source(scratch / f'l2_local/{name}/candidates.jsonl')):
            if r['sample_id'] in candidates:
                raise ValueError('Duplicate candidate ID')
            candidates[r['sample_id']] = r
    if candidates.keys() != items.keys():
        raise ValueError('Candidate coverage differs')
    stored = {}
    for data in [bm['generations/generations.jsonl'],
                 *[ev[f'test/generations/{k}/generations.jsonl'] for k in ('routed', 'unrouted')]]:
        for r in rows(data):
            stored[(r['sample_id'], tuple(r['passage_ids']))] = (r['generated_answer'], r['reject_reason'])
    raw = source(scratch / 'l2_local/ad/test_results.jsonl')
    if digest(raw) != json.loads(qt['run/manifest.json'])['results_sha256']:
        raise ValueError('Raw Test results differ from frozen run')
    projections = []
    for r in ad.read_results(raw):
        sid = r['sample_id']; item = items[sid]; c = candidates[sid]
        assert (r['initial_answer'], r['final_answer']) == (item['c1'], predictions[sid]['prediction'])
        t = ad.trace_of(r)
        if t.get('generated_answer') is not None:
            stored[(sid, tuple(e['passage_id'] for e in t['evidence']))] = (t['generated_answer'], t['reject_reason'])
        c1ids = tuple(e['passage_id'] for e in r['initial_evidence'])
        p3ids = off.select_ids(c, {'k_bge': 20, 'k_bm25': 0, 'fusion': 'rrf'})
        bmids = off.select_ids(c, gate.SETTING)
        brings = gate.brings_bm25(c, bmids)
        p3answer = off.final_answer(item['c1'], p3ids, c1ids, stored, sid)
        basecalls = r['metadata']['enhancement_cost']['llm_calls']
        routed = item['routed']
        if routed:
            assert p3answer == r['final_answer']
            assert basecalls == int(p3ids != c1ids)
        else:
            assert basecalls == 0 and r['final_answer'] == item['c1']
        if routed and brings:
            assert off.final_answer(item['c1'], bmids, c1ids, stored, sid) == item['base']
        else:
            assert item['base'] == r['final_answer']
        projections.append({'sample_id': sid, 'c1': item['c1'], 'b': r['final_answer'],
                            'base': item['base'], 'routed': routed, 'p3_all': p3answer,
                            'c1_ids': c1ids, 'p3_ids': p3ids, 'bm25_ids': bmids, 'bm25_brings': brings,
                            'b_calls': basecalls, 'bm25_calls': int(routed and brings),
                            'p3_all_calls': int(p3ids != c1ids)})
    assert len(projections) == len(items) == 11114
    payload['test/cost_and_p3.jsonl'] = jsonlines(projections)
    for i in range(4):
        p = source(scratch / f'judge_outputs/p60_test/shard_{i}/judge_results.jsonl')
        payload[f'test/p3_judge/shard_{i}.jsonl'] = p.read_bytes()
    raw = source(ROOT / 'data/processed/quality_sweep_v1/adaptive_product_confirm/results.jsonl.gz')
    frozen = members(source(ROOT / 'artifacts/controller/adaptive_product_equivalence_v1/results.tar.gz'))
    expected = {r['sample_id']: r for r in rows(frozen['product.jsonl'])}
    stream_hash = hashlib.sha256(); compact = []
    with gzip.open(raw, 'rb') as f:
        for line in f:
            stream_hash.update(line); r = json.loads(line); sid = r['sample_id']
            s = r['metadata']['answer_selection']; e = r['metadata']['enhancement_cost']
            assert (r['initial_answer'], r['final_answer']) == (expected[sid]['initial_answer'], expected[sid]['final_answer'])
            llm = [c for c in s['calls'] if c['kind'] == 'llm' and c['attempted']]
            assert len(llm) == s['cost']['llm_calls'] and all(c['status'] == 'completed' for c in llm)
            compact.append({'sample_id': sid, 'c1': r['initial_answer'], 'base': r['enhancement_result']['final_answer'],
                            'final': r['final_answer'], 'enhancement_cost': e,
                            'selector_seconds': s['seconds'], 'selector_cost': s['cost'], 'stage': s['stage'],
                            'contexts': {n: {'top5': v['top5'], 'source': v['source']} for n, v in s['contexts'].items()},
                            'llm_events': [{'seconds': c['seconds'], 'input_tokens': c['output']['input_tokens'],
                                            'output_tokens_estimate': c['output']['output_tokens_estimate']} for c in llm]})
    assert stream_hash.hexdigest() == json.loads(frozen['run/manifest.json'])['results_sha256']
    assert len(compact) == 2440
    payload['confirm/costs.jsonl'] = jsonlines(compact)
    for name in ('latency.jsonl', 'latency.manifest.json', 'ids.json', 'time_pipeline.py'):
        p = source(scratch / 'final_eval/timing' / name)
        payload['timing/' + name] = p.read_bytes()
    payload['projection_provenance.json'] = jsonbytes({'sources': provenance, 'checks': {
        'test_raw_matches_frozen_sha256': True, 'confirm_raw_matches_frozen_sha256': True,
        'test_p3_reconstructed_from_candidates_and_generations': 11114,
        'test_routed_p3_equals_b': sum(r['routed'] for r in projections)}})
    # Deterministic tar/gzip; no filesystem timestamps or user names embedded.
    with (out / 'inputs.tar.gz').open('wb') as f, gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0) as g:
        with tarfile.open(fileobj=g, mode='w') as t:
            for name, data in sorted(payload.items()):
                info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o644
                t.addfile(info, io.BytesIO(data))
    print(json.dumps({'inputs': len(payload), 'test': len(projections), 'confirm': len(compact),
                      'sha256': digest(out / 'inputs.tar.gz')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scratch', type=Path, required=True)
    a = p.parse_args(); prepare(a.scratch)
