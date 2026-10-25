"""Attach source-bound agent suggestions; never create human confirmations."""
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_controller_t5_review import page
from src.controller.v2a_policy import route_override
from src.evaluation.answer_repair import score_answer


def bind(cases, suggestions):
    notes = suggestions['notes']
    by_id = {n['review_id']: n for n in notes}
    if len(by_id) != len(notes) or set(by_id) != {c['review_id'] for c in cases}:
        raise ValueError('Agent notes must match unique review IDs exactly')
    result = []
    for original in cases:
        c = deepcopy(original)
        note = by_id[c['review_id']]
        if 'human_review' in note or suggestions.get('human_confirmation') is not False:
            raise ValueError('Agent suggestions cannot fill human judgments')
        for key in ('initial_correct', 'adaptive_correct', 'always_retrieve_correct'):
            if note['agent_judgment'][key] is not None and type(note['agent_judgment'][key]) is not bool:
                raise ValueError('Agent judgment must be true, false or unresolved')
        anchors = []
        for a in note['anchors']:
            evidence = (c['original_evidence'] if a['source'] == 'original' else
                        next(v['evidence'] for v in c['variants'] if v['mode'] == a['source']))
            e = next(e for e in evidence if e['rank'] == a['rank'])
            start = e['text'].find(a['quote'])
            if not a['quote'] or start < 0:
                raise ValueError('Assisted quote is not an exact source substring')
            anchors.append({**a, 'passage_id': e['passage_id'], 'start': start,
                            'end': start + len(a['quote']),
                            'passage_sha256': hashlib.sha256(e['text'].encode()).hexdigest()})
        c['assisted_review'] = {'source': 'agent_only_not_human', **deepcopy(note['agent_judgment']),
                                'anchors': anchors}
        result.append(c)
    return result


def render(cases):
    text = page(cases).replace('NQ 路由 15 题审核', 'NQ 路由辅助审核 17 题')
    text = text.replace('NQ 路由重点 15 题', 'NQ 路由 17 题：原15＋候选误修补充2')
    notice = '<p><b>辅助意见来自代理，尚未获人工确认。此页是 T4 历史输出，非 r2 新结果；gold 不改，所有人审选择仍由用户填写。</b></p>'
    text = text.replace('</header>', '</header>' + notice, 1)
    for c in cases:
        a = c['assisted_review']
        shown = lambda v: '待定' if v is None else '对' if v else '错'
        summary = 'C1：' + shown(a['initial_correct']) + '；Adaptive：' + shown(a['adaptive_correct']) + '；补检索：' + shown(a['always_retrieve_correct'])
        quotes = ''.join('<li>' + html.escape(f"{q['source']} R{q['rank']} · {q['passage_id']}: {q['quote']}") + '</li>' for q in a['anchors'])
        block = '<aside><b>代理辅助意见（待你确认）</b><p>' + html.escape(summary) + '</p><p>' + html.escape(a['reason']) + '</p><ul>' + quotes + '</ul></aside>'
        tag = '<article data-id="' + c['review_id'] + '">'
        text = text.replace(tag, tag + block, 1)
    return text


def prepare(output, root=ROOT):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('Preserve existing review work')
    source = root / 'artifacts/controller/e_v2a_v1/t5_development_review'
    raw = (source / 'routing15_review.jsonl').read_bytes()
    original_manifest = json.loads((source / 'review_manifest.json').read_text())
    if hashlib.sha256(raw).hexdigest() != original_manifest['files']['routing15_review.jsonl']:
        raise ValueError('Original 15-case review changed')
    cases = [json.loads(l) for l in raw.splitlines()]
    archive = root / 'artifacts/controller/t4_v1/controller_t4_development_v1_outputs.zip'
    policy = json.loads((root / 'configs/controller/e_v2a_amendment_v1.json').read_text())
    expected = policy['historical_files_unchanged'][str(archive.relative_to(root))]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError('T4 archive changed')
    wanted = ['-5328057821735888756', '538314753001203199']
    with zipfile.ZipFile(archive) as z:
        with z.open('predictions.jsonl') as f:
            rows = [json.loads(l) for l in f]
    idx = {(r['dataset'], r['sample_id'], r['mode']): r for r in rows if r['sample_id'] in wanted}
    gold = {r['sample_id']: r for r in map(json.loads, (root / 'artifacts/controller/data_v1/offline/nq/development/gold.jsonl').read_text().splitlines())}
    for i, sid in enumerate(wanted, 16):
        c1, adaptive, retrieve = [idx[('nq', sid, mode)] for mode in ('c1', 'adaptive', 'always_retrieve')]
        if not route_override(adaptive, .2)['overridden'] or not score_answer(adaptive['final_answer'], gold[sid], 'nq')['correct'] or score_answer(retrieve['final_answer'], gold[sid], 'nq')['correct']:
            raise ValueError('Supplement is not a frozen threshold-0.2 EM harm')
        cases.append({'review_id': f'R{i:02}', 'sample_id': sid, 'query': c1['input']['query'],
            'initial_answer': c1['final_answer'], 'adaptive_answer': adaptive['final_answer'],
            'reference': gold[sid]['gold_answer_or_label'], 'original_evidence': c1['input']['evidence'],
            'variants': [{'mode': 'always_retrieve', 'answer': retrieve['final_answer'],
                          'evidence': retrieve['final_evidence'], 'metrics': score_answer(retrieve['final_answer'], gold[sid], 'nq')}],
            'human_review': {'status': 'pending', 'initial_correct': None, 'adaptive_correct': None},
            'scope': 'additional threshold-0.2 EM harm; not an original priority-15 case'})
    notes_path = root / 'artifacts/controller/t5_assisted_review_v1/source_notes.json'
    result = bind(cases, json.loads(notes_path.read_text()))
    output.mkdir(parents=True)
    (output / 'review17.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in result))
    (output / 'review17.html').write_text(render(result))
    manifest = {'cases': len(result), 'original_priority_cases': 15, 'additional_candidate_EM_harms': 2,
        'human_confirmed': sum(c['human_review']['status'] == 'confirmed' for c in result),
        'reference_visible': True, 'method_names_visible': True, 'blind_review': False,
        'source_review_sha256': hashlib.sha256(raw).hexdigest(), 'source_T4_sha256': expected,
        'agent_notes_sha256': hashlib.sha256(notes_path.read_bytes()).hexdigest(),
        'runtime_or_gold_changed': False, 'candidate_frozen': False,
        'scope': 'reference-visible development assistance only; no new GPU inference or human labels'}
    (output / 'review_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    print(json.dumps(prepare(p.parse_args().output_dir), ensure_ascii=False, indent=2))
