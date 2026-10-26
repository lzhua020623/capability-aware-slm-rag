"""Development-only replay; freeze ONE rule for selection, never promote it.

E is the registered unavailable projection. Retain existing Probe events and
reuse only observed single-action states. Full is deliberately not replayed.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_controller_t5_review import MAPPING
from src.controller.execution import fingerprint
from src.controller.v2a_policy import GRID, route_override, choose_r2_modes
from src.evaluation.answer_repair import score_answer

ARCHIVE = 'artifacts/controller/t4_v1/controller_t4_development_v1_outputs.zip'
AMENDMENT = 'configs/controller/e_v2a_amendment_v1.json'
DECISION = 'artifacts/controller/e_v2a_interface_r2/deployment_decision.json'
HUMAN = 'artifacts/controller/t5_human_review_v1/review/review17.jsonl'
HUMAN_MANIFEST = 'artifacts/controller/t5_human_review_v1/review/manifest.json'
CONFIRMATION = 'artifacts/controller/t5_human_review_v1/confirmation.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def semantic_delta_bounds(before, after, *, exact_same=False):
    if any(v is not None and type(v) is not bool for v in (before, after)):
        raise ValueError('Semantic judgments must be explicit booleans or unknown')
    if exact_same:
        return [0, 0]
    values = [int(b) - int(a) for a in ([False, True] if before is None else [before])
              for b in ([False, True] if after is None else [after])]
    return [min(values), max(values)]


def verify_cache(rows):
    idx = {(r['dataset'], r['sample_id'], r['mode']): r for r in rows}
    if len(rows) != 2800 or len(idx) != 2800:
        raise ValueError('Complete unique T4 cache required')
    checked = 0
    for row in rows:
        if row['mode'] not in ('adaptive', 'adaptive_no_probe'):
            continue
        other = idx[row['dataset'], row['sample_id'], MAPPING[row['routing']['decision']['action']]]
        for key in ('input_sha256', 'final_answer', 'final_evidence'):
            if row[key] != other[key]:
                raise ValueError('Fixed-action cache mismatch: ' + key)
        signature = lambda r: [(e['kind'], e['stage'], e['arguments']) for e in r['calls'] if e['stage'] != 'probe']
        if signature(row) != signature(other):
            raise ValueError('Fixed-action module arguments differ')
        checked += 1
    if checked != 800:
        raise ValueError('Complete 800-route input audit required')
    return idx


def project(row, idx, *, threshold=None):
    original = row['routing']['decision']['action']
    rule = route_override(row, threshold) if threshold is not None else {'action': original, 'overridden': False}
    action = rule['action']
    chosen = idx[row['dataset'], row['sample_id'], MAPPING[action]]
    if row['input_sha256'] != chosen['input_sha256']:
        raise ValueError('Different module entering state cannot be replayed')
    unavailable = action == 'generation_repair'
    events = [] if unavailable else list(chosen['calls'])
    events += [e for e in row['calls'] if e['stage'] == 'probe']
    llm = [e for e in events if e['kind'] == 'llm' and e['attempted']]
    tokens = lambda key: (sum(e['output'][key] for e in llm)
        if all(e.get('output', {}).get(key) is not None for e in llm) else None)
    return {'record_type': 'offline_policy_projection_not_GPU_prediction',
        'dataset': row['dataset'], 'sample_id': row['sample_id'],
        'input_sha256': row['input_sha256'], 'action': action,
        'original_action': original, 'overridden': rule['overridden'],
        'threshold': threshold, 'E_mode': 'unavailable',
        'final_answer': row['initial_answer'] if unavailable else chosen['final_answer'],
        'source_mode': 'registered_unavailable_keep' if unavailable else chosen['mode'],
        'source_T4_fixed_input_sha256': chosen['input_sha256'],
        'source_T4_fixed_calls_sha256': fingerprint(chosen['calls']),
        'existing_probe_calls_sha256': fingerprint([e for e in row['calls'] if e['stage'] == 'probe']),
        'cost_estimate': {**{kind + '_calls': sum(e['kind'] == kind and e['attempted'] for e in events)
            for kind in ('llm', 'retrieval', 'scoring')}, 'input_tokens': tokens('input_tokens'),
            'output_tokens_estimate': tokens('output_tokens_estimate'),
            'provider_seconds_estimate': sum(e['seconds'] for e in events)},
        'semantic_verdict': 'incomplete' if unavailable else None,
        'scope': 'Reuses old measured events; estimated incremental cost, not new wall time or AC5 certification'}


def analyze(rows, gold, reviews):
    idx = verify_cache(rows)
    adaptive = sorted((r for r in rows if r['mode'] == 'adaptive'), key=lambda r:(r['dataset'], r['sample_id']))
    if {ds: sum(r['dataset'] == ds for r in adaptive) for ds in ('nq','fever')} != {'nq':200,'fever':200}:
        raise ValueError('Common 200-per-dataset development cohorts required')
    by_review = {r['sample_id']: r for r in reviews}
    baseline = {(r['dataset'], r['sample_id']): project(r, idx) for r in adaptive}
    correct = lambda r: score_answer(r['final_answer'], gold[r['dataset'], r['sample_id']], r['dataset'])['correct']
    records = [dict(r, replay_arm='baseline') for r in baseline.values()]
    trials = []
    for t in GRID:
        selected = [project(r, idx, threshold=t) for r in adaptive]
        fixed = harmed = changed = covered = 0
        lower = upper = 0
        changed_pairs = []
        for row, after in zip(adaptive, selected):
            before = baseline[row['dataset'], row['sample_id']]
            if row['dataset'] == 'fever':
                if any(before[k] != after[k] for k in ('final_answer','action','cost_estimate','existing_probe_calls_sha256')):
                    raise ValueError('NQ-only rule changed FEVER')
                continue
            a, b = correct(before), correct(after)
            fixed += int(not a and b); harmed += int(a and not b)
            if before['final_answer'] == after['final_answer']:
                continue
            changed += 1
            review = by_review.get(row['sample_id'])
            h = None
            if review is not None:
                retrieval = next(v for v in review['variants'] if v['mode'] == 'always_retrieve')
                if (review['adaptive_answer'] != before['final_answer'] or retrieval['answer'] != after['final_answer']
                        or review['query'] != row['input']['query']):
                    raise ValueError('Human review does not describe the replayed answers')
                h = review['human_review']
            a_sem = h['adaptive_correct'] if h else None
            b_sem = h['always_retrieve_correct'] if h else None
            lo, hi = semantic_delta_bounds(a_sem, b_sem)
            lower += lo; upper += hi
            covered += int(a_sem is not None and b_sem is not None)
            changed_pairs.append({'sample_id': row['sample_id'], 'review_id': review['review_id'] if review else None,
                'before': before['final_answer'], 'after': after['final_answer'],
                'frozen_EM_before': a, 'frozen_EM_after': b,
                'semantic_before': a_sem, 'semantic_after': b_sem, 'semantic_delta_bounds': [lo,hi]})
        records.extend(dict(r, replay_arm='threshold_' + str(t)) for r in selected)
        cost = lambda batch, key: sum(r['cost_estimate'][key] for r in batch if r['dataset'] == 'nq')
        trials.append({'threshold': t, 'evaluation_n':200,
            'baseline_EM_correct': sum(correct(r) for r in baseline.values() if r['dataset']=='nq'),
            'candidate_EM_correct': sum(correct(r) for r in selected if r['dataset']=='nq'),
            'fixed':fixed, 'harmed':harmed, 'net_EM':fixed-harmed,
            'overridden_routes':sum(r['overridden'] for r in selected), 'changed_answer_pairs': changed,
            'fully_adjudicated_changed_pairs':covered,
            'semantic_net_bounds':[lower,upper], 'changed_pairs':changed_pairs,
            'FEVER_answers_and_costs_unchanged':True,
            'NQ_incremental_cost_delta_estimate':{key:cost(selected,key)-cost(baseline.values(),key)
                for key in ('llm_calls','retrieval_calls','scoring_calls','provider_seconds_estimate')}})
    best = max(trials,key=lambda t:(t['net_EM'],-t['harmed'],-t['overridden_routes'],t['threshold']))
    return {'attempts':len(trials), 'trials':trials,
        'provisional_threshold': best['threshold'] if best['net_EM']>0 else None,
        'baseline_correct':{ds:sum(correct(r) for r in baseline.values() if r['dataset']==ds) for ds in ('nq','fever')},
        'semantic_scope':'Only exact unchanged answers have zero delta automatically; unknown judgments remain interval bounds. Development bounds do not replace the later selection veto.',
        'cache_checks':800,'new_GPU_calls':0,'selection_scored':False,'final_scored':False}, records


def prepare(output, root=ROOT):
    root, output = Path(root), Path(output)
    if output.exists(): raise FileExistsError('Preserve frozen candidate evidence')
    raw = lambda rel: (root / rel).read_bytes()
    policy = json.loads(raw(AMENDMENT))
    for rel, digest in policy['historical_files_unchanged'].items():
        if sha(raw(rel)) != digest: raise ValueError('Historical protocol or archive changed: '+rel)
    decision = json.loads(raw(DECISION))
    if decision['modes'] != choose_r2_modes('failed',None) or decision['engineering_outcome'] != 'failed':
        raise ValueError('This replay requires the frozen shared unavailable mode')
    hm = json.loads(raw(HUMAN_MANIFEST)); confirmation=json.loads(raw(CONFIRMATION))
    if (sha(raw(HUMAN)) != hm['files']['review17.jsonl'] or sha(raw(CONFIRMATION)) != hm['confirmation_sha256']):
        raise ValueError('Human confirmation provenance changed')
    reviews = [json.loads(l) for l in raw(HUMAN).splitlines()]
    if ({r['review_id'] for r in reviews} != {f'R{i:02}' for i in range(1,18)}
            or confirmation['explicit_user_confirmation'] is not True
            or any(r['human_review'].get('assistance_adopted_by_user') is not True for r in reviews)):
        raise ValueError('All original priority cases require explicit user review adoption; unknowns stay unknown')
    with zipfile.ZipFile(root/ARCHIVE) as z:
        checks=json.loads(z.read('output_checksums.json'))
        if set(checks) != set(z.namelist())-{'output_checksums.json'} or any(sha(z.read(n))!=h for n,h in checks.items()):
            raise ValueError('T4 export checksum coverage mismatch')
        rows=[json.loads(l) for l in z.read('predictions.jsonl').splitlines()]
    gold={};sources={rel:sha(raw(rel)) for rel in (ARCHIVE,AMENDMENT,DECISION,HUMAN,HUMAN_MANIFEST,CONFIRMATION)}
    for ds in ('nq','fever'):
        rel=f'artifacts/controller/data_v1/offline/{ds}/development/gold.jsonl'
        sources[rel]=sha(raw(rel))
        for line in raw(rel).splitlines():
            r=json.loads(line); gold[ds,r['sample_id']]=r
    result, records=analyze(rows,gold,reviews)
    candidate={'schema_version':'controller_t5_candidate_v1','stage':'frozen_for_single_selection_not_promoted',
        'threshold':result['provisional_threshold'],'dataset':'nq','eligible_action':'no_action',
        'new_action':'retrieval_enhancement','existing_supported_probe_veto':True,
        'missing_probe':'use_existing_stage1_probability_without_new_Probe',
        'E_mode':{'nq':'unavailable','fever':'unavailable'},'search_grid':list(GRID),'attempts':3,
        'selection_candidates':1 if result['provisional_threshold'] is not None else 0,
        'selection_decision':'positive frozen EM; negative semantic net or negative unresolved lower bound vetoes promotion',
        'development_review':'All 15 priorities adopted by user with six of 17 total cases still not fully adjudicated; no unknown relabeled',
        'runtime_forbidden':['gold','human_correctness','sample_id_based_rules'],
        'selection_scored':False,'final_scored':False,'promoted':False,
        'source_hashes':sources,'analysis_code_sha256':sha(Path(__file__).read_bytes())}
    output.mkdir(parents=True)
    for name,value in [('analysis.json',result),('candidate.json',candidate)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    (output/'replay_records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    return candidate


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True,type=Path)
    print(json.dumps(prepare(p.parse_args().output_dir),ensure_ascii=False,indent=2))
