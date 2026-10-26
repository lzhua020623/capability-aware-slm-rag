"""Record an explicit chat adoption of sealed assistance without resolving unknowns.

The original assistance stays immutable. This is assisted, reference-visible
review, not an independent human relabeling or a route-promotion decision.
"""
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_controller_t5_review import evidence_html

SOURCE = 'artifacts/controller/t5_assisted_review_v1/review/review17.jsonl'
NOTES = 'artifacts/controller/t5_assisted_review_v1/source_notes.json'
KEYS = ('initial_correct', 'adaptive_correct', 'always_retrieve_correct')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def apply_confirmation(cases, confirmation):
    ids = [c['review_id'] for c in cases]
    if (not ids or len(ids) != len(set(ids)) or ids != confirmation['review_ids']
            or confirmation.get('explicit_user_confirmation') is not True
            or confirmation.get('decision') != 'adopt_assistance_preserve_unknowns'
            or confirmation.get('user_message_exact') != '按照你判断的来'
            or confirmation.get('judged_fields') != list(KEYS)
            or confirmation.get('independent_blind_review') is not False):
        raise ValueError('Explicit source-bound adoption with unchanged scope required')
    result = []
    for original in cases:
        if original['human_review']['status'] != 'pending':
            raise ValueError('Existing human decisions must not be overwritten')
        c = deepcopy(original)
        a = c['assisted_review']
        if a['source'] != 'agent_only_not_human' or any(
                a[k] is not None and type(a[k]) is not bool for k in KEYS):
            raise ValueError('Expected sealed agent judgments: boolean or unresolved')
        values = {k: a[k] for k in KEYS}
        n = sum(v is not None for v in values.values())
        c['human_review'] = {
            'status': 'confirmed' if n == len(KEYS) else 'partial' if n else 'unresolved',
            **values, 'assistance_adopted_by_user': True,
            'source': 'explicit_chat_user_adoption_of_agent_assistance',
            'confirmation_id': confirmation['confirmation_id'],
            'independent_blind_review': False, 'category': a['category'],
            'reason': a['reason'], 'other_fixed_actions_adjudicated': False,
        }
        result.append(c)
    return result


def summarize(cases):
    statuses = [c['human_review']['status'] for c in cases]
    complete = [c for c in cases if c['human_review']['status'] == 'confirmed']
    return {
        'cases': len(cases), 'assistance_adopted': len(cases),
        'fully_adjudicated': statuses.count('confirmed'),
        'partially_adjudicated': statuses.count('partial'),
        'unresolved_all_fields': statuses.count('unresolved'),
        'not_fully_adjudicated_ids': [c['review_id'] for c in cases if c['human_review']['status'] != 'confirmed'],
        'both_initial_and_retrieval_correct': sum(c['human_review']['initial_correct'] is True
            and c['human_review']['always_retrieve_correct'] is True for c in complete),
        'initial_wrong_retrieval_correct': sum(c['human_review']['initial_correct'] is False
            and c['human_review']['always_retrieve_correct'] is True for c in complete),
        'initial_correct_retrieval_wrong': sum(c['human_review']['initial_correct'] is True
            and c['human_review']['always_retrieve_correct'] is False for c in complete),
        'blind_review': False, 'candidate_frozen': False, 'route_promoted': False,
        'runtime_or_gold_changed': False,
        'scope': 'Targeted 17-case assisted review only; not all changed answers, not 200-case semantic accuracy or candidate net gain',
    }


def render(cases, summary):
    esc = lambda value: html.escape(str(value))
    shown = lambda value: '待定' if value is None else '对' if value else '错'
    cards = []
    for c in cases:
        h = c['human_review']
        variants = []
        for v in c['variants']:
            judgment = shown(h['always_retrieve_correct']) if v['mode'] == 'always_retrieve' else '未在本次确认范围内'
            variants.append('<section><b>' + esc(v['mode']) + '</b><pre>' + esc(v['answer'])
                + '</pre><p>语义：' + judgment + '</p><details><summary>该动作终答证据</summary>'
                + evidence_html(v['evidence']) + '</details></section>')
        cards.append('<article id="' + esc(c['review_id']) + '"><h2>' + esc(c['review_id']) + ' · '
            + esc(c['query']) + '</h2><p>状态：' + esc(h['status']) + '；用户已采用辅助意见</p>'
            + '<h3>C1 原答：' + shown(h['initial_correct']) + '</h3><pre>' + esc(c['initial_answer'])
            + '</pre><h3>Adaptive 原路由：' + shown(h['adaptive_correct']) + '</h3><pre>'
            + esc(c['adaptive_answer']) + '</pre><p>理由：' + esc(h['reason']) + '</p>'
            + '<p>离线参考：' + esc(json.dumps(c['reference'], ensure_ascii=False)) + '</p>'
            + '<details><summary>原始 Top-5</summary>' + evidence_html(c['original_evidence'])
            + '</details><div class="variants">' + ''.join(variants) + '</div></article>')
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>T5 已确认审核 17 题</title>'
        '<style>body{max-width:1200px;margin:30px auto;padding:0 20px;font:16px/1.65 system-ui;background:#f4f6f8;color:#17212b}'
        'article{background:white;padding:24px;margin:24px 0;border:1px solid #d8dee5;border-radius:12px}'
        'pre{white-space:pre-wrap;background:#f0f3f7;padding:12px}summary{cursor:pointer}'
        '.variants{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}</style>'
        '<h1>T5 已确认审核 17 题</h1><p>用户回复“按照你判断的来”，据此采用原辅助意见。'
        '这是可见参考的辅助审核，不是独立盲审。待定仍为待定；没有自动确认其他固定动作。</p>'
        '<p>完整定案 ' + str(summary['fully_adjudicated']) + ' 题；尚未完整定案 '
        + str(len(summary['not_fully_adjudicated_ids'])) + ' 题。此页为入库快照，后续可按编号在聊天中修订。</p>'
        + ''.join(cards) + '</html>')


def prepare(output, confirmation_path, root=ROOT):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('Preserve existing human review work')
    confirmation_raw = Path(confirmation_path).read_bytes()
    confirmation = json.loads(confirmation_raw)
    source_raw, notes_raw = (root / SOURCE).read_bytes(), (root / NOTES).read_bytes()
    if (digest(source_raw) != confirmation['source_review_sha256']
            or digest(notes_raw) != confirmation['agent_notes_sha256']):
        raise ValueError('Assistance changed after the user adopted it')
    cases = apply_confirmation([json.loads(l) for l in source_raw.splitlines()], confirmation)
    summary = summarize(cases)
    output.mkdir(parents=True)
    (output / 'review17.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
    (output / 'review17.html').write_text(render(cases, summary))
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    manifest = {'confirmation_sha256': digest(confirmation_raw),
        'source_review_sha256': digest(source_raw), 'agent_notes_sha256': digest(notes_raw),
        'original_assistance_preserved': True, 'reference_visible': True,
        'files': {p.name: digest(p.read_bytes()) for p in output.iterdir()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--confirmation', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, args.confirmation), ensure_ascii=False))
