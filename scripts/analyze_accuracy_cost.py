"""Replay frozen accuracy/cost policies from Git archives (CPU only, no inference).

Run: python scripts/analyze_accuracy_cost.py
The optional figure script consumes summary.json. Inputs are checked before replay.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import adaptive_selection as ad
from scripts.score_quality_eval import mcnemar_p

RELEASE = ROOT / 'artifacts/controller/accuracy_cost_v1'
ARCHIVES = {
    'evaluation': 'adaptive_selection_eval_v1', 'quality_test': 'quality_test_v1',
    'quality_confirm': 'quality_confirm_v1', 'l2_confirm': 'l2_confirm_v1',
    'bm25_test': 'bm25_gate_test_v1', 'equivalence': 'adaptive_product_equivalence_v1'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def source_digest(path):
    # Source code has identical semantics across LF/CRLF checkouts; data receipts stay byte-exact.
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def read_archive(path, wanted=None):
    with tarfile.open(path) as t:
        return {m.name: t.extractfile(m).read() for m in t if m.isfile() and (wanted is None or wanted(m.name))}


def jsonl(data):
    return [json.loads(l) for l in data.splitlines() if l.strip()]


def indexed(rows):
    out = {str(r['sample_id']): r for r in rows}
    if len(out) != len(rows): raise ValueError('Duplicate sample IDs')
    return out


def add_labels(labels, data):
    for r in jsonl(data):
        if r.get('judge_status') in ('llm_ok', 'auto_correct'):
            labels[(str(r['sample_id']).split('::')[0], r['prediction'])] = bool(r['auto_correct']) or r['judge_label'] == 'CORRECT'


def load_inputs(root=ROOT, release=RELEASE):
    paths = {k: root / f'artifacts/controller/{v}/results.tar.gz' for k, v in ARCHIVES.items()}
    paths.update(inputs=release / 'inputs.tar.gz', policy=root / ad.POLICY, method=release / 'method.json')
    paths.update({name:root / name for name in ('scripts/adaptive_selection.py',
        'src/compensation/answer_selection.py', 'scripts/l2_offline.py',
        'src/compensation/retrieval_enhancement/p3_l2.py',
        'baselines/_lib/answer_scoring.py', 'scripts/score_quality_eval.py')})
    seal = json.loads((release / 'input_hashes.json').read_text())
    for k, path in paths.items():
        hasher = source_digest if seal[k].get('hash_mode') == 'source_lf_normalized' else digest
        if hasher(path) != seal[k]['sha256']: raise ValueError(f'Input changed: {k}')
    # Skip 105MB old run/results: its latency was confounded by CPU oversubscription.
    data = {k: read_archive(p, lambda n: n != 'run/results.jsonl') for k, p in paths.items() if str(p).endswith('tar.gz')}
    return data, json.loads(paths['policy'].read_text()), json.loads(paths['method'].read_text())


def labels_for(data, role):
    ev = data['evaluation']; labels = {}
    def batch(archive, prefix):
        for i in range(4): add_labels(labels, archive[f'{prefix}/shard_{i}/judge_results.jsonl'])
    if role == 'confirm':
        batch(data['quality_confirm'], 'judge/outputs')
        for i in range(4): add_labels(labels, ev[f'confirm/prior_labels/round2_paired/shard_{i}.jsonl'])
        batch(data['l2_confirm'], 'judge/l2_confirm/outputs')
        batch(ev, 'confirm/judge/outputs')
    else:
        for k in ('qwen_only', 'product'): add_labels(labels, data['quality_test'][f'team/{k}/judge_results.jsonl'])
        batch(data['quality_test'], 'judge/outputs'); batch(data['bm25_test'], 'judge/outputs')
        batch(ev, 'test/judge/outputs')
        # New P3-all judgments fill only missing answer pairs; preserve every frozen label.
        extra = {}
        for i in range(4): add_labels(extra, data['inputs'][f'test/p3_judge/shard_{i}.jsonl'])
        for key, value in extra.items(): labels.setdefault(key, value)
        add_labels(labels, data['quality_test']['team/rag/judge_results.jsonl'])
    return labels


def paired(before, after, before_ok, after_ok):
    """Unknown labels cancel only for byte-identical answers; never silently score missing as wrong."""
    if len({len(before), len(after), len(before_ok), len(after_ok)}) != 1:
        raise ValueError('Paired arrays have different lengths')
    out = []
    for b, a, x, y in zip(before, after, before_ok, after_ok):
        if b == a: out.append(0)
        elif x is None or y is None: raise ValueError('Unjudged changed answer')
        else: out.append(int(y) - int(x))
    return np.asarray(out, dtype=np.int8)


def paired_stats(delta, seed, replicates=10000):
    d = np.asarray(delta); n = len(d)
    counts = np.array([(d == k).sum() for k in (-1, 0, 1)])
    if n == 0 or counts.sum() != n: raise ValueError('Expected nonempty paired binary differences')
    rng = np.random.default_rng(seed)
    samples = rng.multinomial(n, counts/n, size=replicates)
    effect = (samples[:, 2] - samples[:, 0]) * 100/n
    harms, _, fixes = (int(x) for x in counts)
    return {'n': n, 'fixes': fixes, 'harms': harms, 'net': fixes-harms,
            'delta_pp': float(d.mean()*100), 'ci95_pp': np.quantile(effect, [.025,.975]).tolist(),
            'lower95_one_sided_pp': float(np.quantile(effect,.05)),
            'mcnemar_p': mcnemar_p(fixes, harms)}


def stable_seed(seed, name):
    return seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def holm(ps):
    order = sorted(ps, key=ps.get); adjusted = {}; last = 0.
    for i, name in enumerate(order):
        last = max(last, min(1., ps[name] * (len(order)-i))); adjusted[name] = last
    return adjusted


def pareto(table):
    """Point-estimate frontier, no significance or promotion implied."""
    return [name for name, p in table.items() if not any(
        q['calls_mean'] <= p['calls_mean'] and q['gain_vs_b_bm25']['delta_pp'] >= p['gain_vs_b_bm25']['delta_pp']
        and (q['calls_mean'] < p['calls_mean'] or q['gain_vs_b_bm25']['delta_pp'] > p['gain_vs_b_bm25']['delta_pp'])
        for other, q in table.items() if other != name)]


def selector_prefix(receipt, stage, order):
    if stage == 0: return {'calls': 0, 'seconds': 0., 'input_tokens': 0}
    if stage > receipt['stage']: raise ValueError('Counterfactual requires unmeasured contexts')
    count = sum(receipt['contexts'][n]['source'] == 'generated' for n in order[:stage])
    events = receipt['llm_events']
    assert len(events) == receipt['selector_cost']['llm_calls']
    overhead = receipt['selector_seconds'] - sum(e['seconds'] for e in events)
    if overhead < -1e-8: raise ValueError('Nested timers cannot be summed')
    return {'calls': count, 'seconds': overhead + sum(e['seconds'] for e in events[:count]),
            'input_tokens': sum(e['input_tokens'] for e in events[:count])}


def analyze(root=ROOT, release=RELEASE):
    data, policy, method = load_inputs(root, release)
    projections = {r: indexed(jsonl(data['inputs'][p])) for r,p in
                   [('test','test/cost_and_p3.jsonl'), ('confirm','confirm/costs.jsonl')]}
    all_runs, results = {}, {'schema_version':'accuracy_cost_v1', 'roles':{}, 'comparisons':{}, 'policy_selection':'none'}
    records = []
    for role in ('confirm','test'):
        rows = jsonl(data['evaluation'][f'{role}/items.jsonl']); ids = [r['sample_id'] for r in rows]
        if len(set(ids)) != len(ids) or set(ids) != projections[role].keys(): raise ValueError('Unmatched IDs')
        gold = {r['sample_id']: r['gold'] for r in jsonl(data[f'quality_{role}']['scored/predictions.jsonl'])}
        labels = labels_for(data, role); proj = projections[role]
        outcomes = ad.answers_of(rows, policy)
        runs = {name: {'answers':[o[sid][0] for sid in ids], 'extra':[o[sid][2] for sid in ids],
                       'stage':[o[sid][1] for sid in ids]} for name,o in outcomes.items()}
        route = {r['sample_id']:r['prediction'] for r in jsonl(data[f'quality_{role}']['scored/predictions.jsonl'])}
        runs.update({name: {'answers': vals, 'extra':[0]*len(ids), 'stage':[0]*len(ids)} for name,vals in
                     [('c1',[r['c1'] for r in rows]),('b',[route[sid] for sid in ids]),('b_bm25',[r['base'] for r in rows])]})
        if role == 'test': runs['p3_all'] = {'answers':[proj[s]['p3_all'] for s in ids], 'extra':[0]*len(ids), 'stage':[0]*len(ids)}
        base_calls = np.array([1 + (proj[s]['b_calls'] + proj[s]['bm25_calls'] if role=='test'
                                   else proj[s]['enhancement_cost']['llm_calls']) for s in ids])
        for name, run in runs.items():
            run['ok'] = [ad.label_of(labels, gold, sid, a) for sid,a in zip(ids,run['answers'])]
            if name == 'c1': run['calls'] = np.ones(len(ids),dtype=int)
            elif name == 'b':
                if role == 'test': run['calls'] = np.array([1+proj[s]['b_calls'] for s in ids])
                else:
                    # B's standalone call receipts are unavailable in the compact confirm run.
                    # B is omitted from this confirm cost frontier, not assigned BM25's costs.
                    continue
            elif name == 'p3_all': run['calls'] = np.array([1+proj[s]['p3_all_calls'] for s in ids])
            else: run['calls'] = base_calls + np.asarray(run['extra'])
        if role == 'confirm': del runs['b']
        table = {}
        for name, run in runs.items():
            base = runs['b_bm25']; c1 = runs['c1']
            d = paired(base['answers'],run['answers'],base['ok'],run['ok'])
            dc1 = paired(c1['answers'],run['answers'],c1['ok'],run['ok'])
            known = sum(x is not None for x in run['ok'])
            s = paired_stats(d,stable_seed(method['statistics']['seed'],role+name),method['statistics']['bootstrap_replicates'])
            table[name] = {'n':len(ids),'known_labels':known,'correct':sum(x is True for x in run['ok']) if known==len(ids) else None,
                           'accuracy_pct':100*sum(x is True for x in run['ok'])/len(ids) if known==len(ids) else None,
                           'gain_vs_b_bm25':s,'gain_vs_c1':paired_stats(dc1,stable_seed(20261031,role+name+'c1')),
                           'calls_mean':float(run['calls'].mean()),'calls_total':int(run['calls'].sum()),
                           'extra_calls_mean':float(np.mean(run['extra'])), 'stages':dict(Counter(run['stage']))}
            # Validate the replay against the already published per-policy report.
            if name in outcomes:
                old = json.loads(data['evaluation'][f'{role}/report.json'])['policies'][name]
                assert (s['fixes'],s['harms']) == (old['against_b_bm25']['fixes'],old['against_b_bm25']['harms'])
                assert abs(table[name]['extra_calls_mean']-old['extra_generations_per_item'])<1e-12
            for i,sid in enumerate(ids):
                records.append({'role':role,'sample_id':sid,'strategy':name,'answer':run['answers'][i],
                                'judge_correct':run['ok'][i],'delta_vs_b_bm25':int(d[i]),'delta_vs_c1':int(dc1[i]),
                                'total_calls':int(run['calls'][i]),'extra_calls':run['extra'][i],'stage':run['stage'][i]})
        comp = {}
        for a,b in method['comparisons']:
            if a not in runs or b not in runs: continue
            ra, rb = runs[a], runs[b]
            d = paired(rb['answers'],ra['answers'],rb['ok'],ra['ok']); key=a+'__vs__'+b
            s=paired_stats(d,stable_seed(20261031,role+key)); s.update(
                total_call_saving_fraction=float(1-ra['calls'].sum()/rb['calls'].sum()),
                extra_call_saving_fraction=float(1-sum(ra['extra'])/sum(rb['extra'])) if sum(rb['extra']) else None)
            comp[key]=s
        for name,p in holm({k:v['mcnemar_p'] for k,v in comp.items()}).items(): comp[name]['holm_p']=p
        results['comparisons'][role]=comp
        results['roles'][role]={'n':len(ids),'strategies':table,'point_estimate_pareto':pareto(table)}
        all_runs[role]=(ids,rows,runs)
    # Original online timing subset; same answers as the full confirm run.
    times = jsonl(data['inputs']['timing/latency.jsonl']); indexed(times)
    timed_ids=json.loads(data['inputs']['timing/ids.json'])
    assert [t['sample_id'] for t in times]==timed_ids['timed']
    assert not set(timed_ids['warmup']) & set(timed_ids['timed'])
    idx={s:i for i,s in enumerate(all_runs['confirm'][0])}; runs=all_runs['confirm'][2]
    for t in times:
        i=idx[t['sample_id']]
        assert (t['c1'],t['controller_answer'],t['final_answer']) == tuple(runs[n]['answers'][i] for n in ('c1','b_bm25','chosen'))
    latency={}; timing_records=[]
    def timing_summary(name, secs, calls, tokens, kind):
        inds=[idx[t['sample_id']] for t in times]; run=runs[name.removesuffix('_prefix_estimate')]; base=runs['b_bm25']
        delta=paired([base['answers'][i] for i in inds],[run['answers'][i] for i in inds],
                     [base['ok'][i] for i in inds],[run['ok'][i] for i in inds])
        oks = [run['ok'][i] for i in inds]
        accuracy = 100*sum(x is True for x in oks)/len(oks) if all(x is not None for x in oks) else None
        latency[name]={'n':len(times),'accuracy_pct':accuracy,'kind':kind,'seconds_mean':float(np.mean(secs)),
                       'seconds_p50':float(np.median(secs)),'seconds_p95':float(np.quantile(secs,.95)),
                       'calls_mean':float(np.mean(calls)),'input_tokens_mean':float(np.mean(tokens)),
                       'gain_vs_b_bm25':paired_stats(delta,stable_seed(20261031,'timing'+name))}
        for t,s,c,tok in zip(times,secs,calls,tokens):
            timing_records.append({'sample_id':t['sample_id'],'strategy':name,'seconds':s,'calls':c,'input_tokens':tok,'kind':kind})
    base_s=[t['c1_s']+t['diagnosis_classifier_s']+t['diagnosis_features_s']+t['controller_s'] for t in times]
    for name in ('c1','b_bm25','chosen'):
        secs=[]; calls=[]; tokens=[]
        for t,b in zip(times,base_s):
            e=t['enhancement_cost']; sc=t['selection_cost']; full=name=='chosen'; base=name!='c1'
            secs.append(t['total_s'] if full else b if base else t['c1_s'])
            calls.append(1+(e['llm_calls'] if base else 0)+(sc['llm_calls'] if full else 0))
            tokens.append(t['c1_input_tokens']+(e['input_tokens'] if base else 0)+(sc['input_tokens'] if full else 0))
        timing_summary(name,secs,calls,tokens,'measured_total' if name=='chosen' else 'measured_component_prefix')
    for name in method['latency_replays']:
        secs=[];calls=[];tokens=[]
        for t,b in zip(times,base_s):
            sid=t['sample_id'];i=idx[sid];r=projections['confirm'][sid]
            prefix=selector_prefix(r,runs[name]['stage'][i],policy['order'])
            e=t['enhancement_cost']
            secs.append(b+prefix['seconds']);calls.append(1+e['llm_calls']+prefix['calls'])
            tokens.append(t['c1_input_tokens']+e['input_tokens']+prefix['input_tokens'])
        label = name + '_prefix_estimate' if name == 'chosen' else name
        timing_summary(label,secs,calls,tokens,'two_run_prefix_estimate')
    results['timing']=latency
    results['timing_components']={k:float(np.mean([t[k] for t in times])) for k in
                                  ('c1_s','diagnosis_classifier_s','diagnosis_features_s','controller_s','answer_selection_s','total_s')}
    results['timing_manifest']=json.loads(data['inputs']['timing/latency.manifest.json'])
    results['validation']={'published_policy_counts_replayed':True,'timing_answer_matches':len(times),
        'p3_route_share_test':sum(projections['test'][s]['routed'] for s in projections['test'])/len(projections['test']),
        'confirm_replay_vs_actual_generation_delta':sum(r['selector_cost']['llm_calls'] for r in projections['confirm'].values())-sum(all_runs['confirm'][2]['chosen']['extra'])}
    return results,records,timing_records


def write_outputs(summary, records, timing, output):
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    with (output/'paired_records.jsonl.gz').open('wb') as f,gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as g:
        for r in records:g.write((json.dumps(r,ensure_ascii=False)+'\n').encode())
    (output/'timing_replay.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in timing))
    with (output/'strategy_table.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['role','strategy','n','accuracy_pct','gain_pp','gain_ci95_low','gain_ci95_high','total_calls','extra_calls','frontier'])
        for role,result in summary['roles'].items():
            for name,r in result['strategies'].items():
                w.writerow([role,name,r['n'],r['accuracy_pct'],r['gain_vs_b_bm25']['delta_pp'],*r['gain_vs_b_bm25']['ci95_pp'],r['calls_mean'],r['extra_calls_mean'],name in result['point_estimate_pareto']])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=RELEASE)
    a=p.parse_args();s,r,t=analyze();write_outputs(s,r,t,a.output)
    print(json.dumps({'test':s['roles']['test']['n'],'confirm':s['roles']['confirm']['n'],'timed':s['timing']['chosen']['n'],'strategies':len(s['roles']['test']['strategies'])}))
