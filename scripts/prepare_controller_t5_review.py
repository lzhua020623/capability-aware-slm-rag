"""Development-only search receipts and a reference-visible human review page.

No route is frozen here, and no human judgment is filled automatically.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.controller.v2a_policy import GRID, route_override
from src.evaluation.answer_repair import score_answer

MAPPING = {'no_action': 'c1', 'retrieval_enhancement': 'always_retrieve',
           'evidence_refinement': 'always_rerank', 'generation_repair': 'always_repair'}


def search(rows, gold):
    idx = {(r['dataset'], r['sample_id'], r['mode']): r for r in rows}
    trials = []
    for threshold in GRID:
        fixed = harmed = added = 0
        for r in rows:
            if r['dataset'] != 'nq' or r['mode'] != 'adaptive': continue
            policy = route_override(r, threshold)
            if not policy['overridden']: continue
            added += 1
            alt = idx[('nq', r['sample_id'], 'always_retrieve')]
            if alt['input_sha256'] != r['input_sha256']: raise ValueError('Replay input changed')
            a = score_answer(r['final_answer'], gold[r['sample_id']], 'nq')['correct']
            b = score_answer(alt['final_answer'], gold[r['sample_id']], 'nq')['correct']
            fixed += int(not a and b); harmed += int(a and not b)
        trials.append({'threshold': threshold, 'evaluation_n': 200, 'added_retrieval_calls': added,
                       'fixed_vs_adaptive': fixed, 'harmed_vs_adaptive': harmed, 'net_EM': fixed - harmed})
    best = max(trials, key=lambda t: (t['net_EM'], -t['harmed_vs_adaptive'], -t['added_retrieval_calls'], t['threshold']))
    return {'trials': trials, 'attempts': len(trials),
            'provisional_threshold': best['threshold'] if best['net_EM'] > 0 else None,
            'candidate_frozen': False, 'promoted': False,
            'pending': ['15 user-confirmed routing reviews', 'frozen E v2a deployment mode', 'replay on v2a baseline'],
            'scope': 'post-T4 development frozen EM; not selection, human semantics or efficacy evidence'}


def evidence_html(evidence):
    return ''.join('<details><summary>R' + str(e['rank']) + ' · ' + html.escape(e['passage_id']) +
                   '</summary><p>' + html.escape(e['text']) + '</p></details>' for e in evidence)


def page(cases):
    cards = []
    select = lambda key: ('<select data-key="' + key + '"><option value="">未审</option>'
        '<option value="true">正确</option><option value="false">错误</option><option value="unknown">待定</option></select>')
    for c in cases:
        variants = ''.join('<div><b>' + html.escape(v['mode']) + '</b><pre>' + html.escape(v['answer']) +
            '</pre>冻结 EM：' + str(int(v['metrics']['correct'])) + '　语义：' + select(v['mode']) +
            '<details><summary>此动作实际终答证据</summary>' + evidence_html(v['evidence']) + '</details></div>' for v in c['variants'])
        cards.append('<article data-id="' + c['review_id'] + '"><h2>' + c['review_id'] + ' · ' +
            html.escape(c['query']) + '</h2><small>sample_id: ' + html.escape(c['sample_id']) +
            ' · 自动 EM 机会分类，语义待人工确认</small><h3>C1 原答</h3><pre>' + html.escape(c['initial_answer']) +
            '</pre>语义：' + select('initial_correct') + '<h3>Adaptive 原路由终答</h3><pre>' +
            html.escape(c['adaptive_answer']) + '</pre>语义：' + select('adaptive_correct') +
            '<p>离线参考（仅辅助，不自动裁决）：' + html.escape(json.dumps(c['reference'], ensure_ascii=False)) +
            '</p><details><summary>原始 Top-5 证据</summary>' + evidence_html(c['original_evidence']) +
            '</details><h3>固定动作结果</h3><div class="variants">' + variants +
            '</div><p>备注：<textarea data-key="note" rows="2"></textarea></p></article>')
    data = json.dumps([{'review_id': c['review_id'], 'sample_id': c['sample_id']} for c in cases]).replace('<', '\\u003c')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>NQ 路由 15 题审核</title>
<style>body{max-width:1200px;margin:30px auto;padding:0 20px;font:16px/1.65 system-ui;color:#17212b;background:#f4f6f8}article{background:white;padding:24px;margin:24px 0;border:1px solid #d8dee5;border-radius:12px}pre{white-space:pre-wrap;background:#f0f3f7;padding:12px}details{margin:8px 0}summary{cursor:pointer}textarea{width:95%}.variants{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}button,select{font-size:16px;padding:6px}header{position:sticky;top:0;background:#f4f6f8;padding:12px;border-bottom:1px solid #ccc}</style>
<header><b>NQ 路由重点 15 题</b>　<button id="export">导出人工记录 JSON</button><span id="count"></span></header>
<p>参考可见审核。所有选择默认未审；EM 改善不自动等于语义修复。核对问题所问关系、实体和限定条件。可在这里选择并导出，也可按 R01–R15 在聊天中确认。记录保存在本浏览器，不会自动写入仓库。</p>''' + ''.join(cards) + '''
<script>const ids=''' + data + ''';const storage='controller-t4-routing15-v1';let saved={};try{saved=JSON.parse(localStorage.getItem(storage)||'{}')}catch(e){};
for(const a of document.querySelectorAll('article')){for(const x of a.querySelectorAll('[data-key]')){x.value=(saved[a.dataset.id]||{})[x.dataset.key]||'';x.addEventListener('change',()=>{saved[a.dataset.id]=saved[a.dataset.id]||{};saved[a.dataset.id][x.dataset.key]=x.value;localStorage.setItem(storage,JSON.stringify(saved));});}}
document.getElementById('export').onclick=()=>{const rows=ids.map(c=>({...c,human_review:saved[c.review_id]||{},source:'user_browser_input'}));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:'application/json'}));a.download='controller_routing15_human_review.json';a.click();URL.revokeObjectURL(a.href);};</script></html>'''


def prepare(output, root=ROOT):
    root, output = Path(root), Path(output)
    if output.exists(): raise FileExistsError('Preserve existing human review work')
    policy = json.loads((root / 'configs/controller/e_v2a_amendment_v1.json').read_text())
    archive = root / 'artifacts/controller/t4_v1/controller_t4_development_v1_outputs.zip'
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != policy['historical_files_unchanged'][str(archive.relative_to(root))]: raise ValueError('Changed T4')
    with zipfile.ZipFile(archive) as z:
        with z.open('predictions.jsonl') as f: rows = [json.loads(l) for l in f]
    idx = {(r['dataset'], r['sample_id'], r['mode']): r for r in rows}
    audited = 0
    for r in rows:
        if r['mode'] not in ('adaptive', 'adaptive_no_probe'): continue
        other = idx[(r['dataset'], r['sample_id'], MAPPING[r['routing']['decision']['action']])]
        for key in ('input_sha256', 'final_answer', 'final_evidence'):
            if r[key] != other[key]: raise ValueError('Cached fixed action differs: ' + key)
        calls = [(e['kind'], e['stage'], e['arguments']) for e in r['calls'] if e['stage'] != 'probe']
        if calls != [(e['kind'], e['stage'], e['arguments']) for e in other['calls']]: raise ValueError('Module arguments differ')
        audited += 1
    if audited != 800: raise ValueError('Incomplete replay audit')
    gold = {r['sample_id']: r for r in map(json.loads,
        (root / 'artifacts/controller/data_v1/offline/nq/development/gold.jsonl').read_text().splitlines())}
    priority = [r for r in map(json.loads, (root / 'artifacts/controller/t4_v1/routing_priority_review.jsonl').read_text().splitlines()) if r['dataset'] == 'nq']
    if len(priority) != 15: raise ValueError('Expected 15 original NQ priorities')
    cases = []
    for i, r in enumerate(priority, 1):
        variants = []
        for m in ('always_retrieve', 'always_rerank', 'always_repair'):
            v = idx[('nq', r['sample_id'], m)]
            variants.append({'mode': m, 'answer': v['final_answer'], 'evidence': v['final_evidence'],
                             'metrics': score_answer(v['final_answer'], gold[r['sample_id']], 'nq')})
        cases.append({**r, 'review_id': f'R{i:02}', 'original_evidence': idx[('nq', r['sample_id'], 'c1')]['input']['evidence'],
                      'reference': gold[r['sample_id']]['gold_answer_or_label'], 'variants': variants,
                      'human_review': {'status': 'pending', 'initial_correct': None, 'adaptive_correct': None}})
    controls = [r['sample_id'] for r in rows if r['dataset'] == 'nq' and r['mode'] == 'adaptive'
                and r['routing']['decision']['action'] == 'no_action'
                and score_answer(r['initial_answer'], gold[r['sample_id']], 'nq')['correct']]
    if len(controls) != 73: raise ValueError('Correct keep control cohort changed')
    output.mkdir(parents=True)
    write = lambda name, value: (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    write('routing_search_development.json', {**search(rows, gold), 'source_archive_sha256': digest})
    write('replay_cache_audit.json', {'checked': audited, 'mismatches': 0, 'source_archive_sha256': digest})
    write('correct_keep_regression_ids.json', {'sample_ids': controls, 'n': 73,
        'basis': 'frozen EM correct; regression material, not new human efficacy evidence'})
    (output / 'routing15_review.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
    (output / 'routing15_review.html').write_text(page(cases))
    write('review_manifest.json', {'n': 15, 'human_confirmed': 0, 'reference_visible': True,
        'candidate_frozen': False, 'five_user_reported_semantic_harm_IDs': 'pending user identification',
        'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}})
    return {'n': 15, 'human_confirmed': 0, 'output': str(output)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output-dir', required=True, type=Path)
    print(json.dumps(prepare(p.parse_args().output_dir), ensure_ascii=False))
