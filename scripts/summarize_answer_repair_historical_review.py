"""Validate and present explicit historical agent opinions; never fill human labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("initial_correct", "final_correct", "initial_supported", "final_supported")
OUTCOMES = {
    "suggested_fix": "建议修复", "suggested_regression": "疑似误修",
    "correct_preserved": "正确保留／同义改写",
    "unresolved_error_or_abstention": "仍未答对",
    "correctness_unresolved": "正确性待定",
    "reference_only_flip_not_evidence_verified": "仅翻到参考标签，证据尚未核实",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def load(directory):
    raw = (directory / "pending_pairs.jsonl").read_bytes()
    assisted_raw = (directory / "assisted_partial.jsonl").read_bytes()
    manifest = json.loads((directory / "manifest.json").read_bytes())
    pairs = [json.loads(x) for x in raw.splitlines() if x.strip()]
    agents = [json.loads(x) for x in assisted_raw.splitlines() if x.strip()]
    if sha(raw) != manifest["queue_sha256"] or len(pairs) != manifest["selected_pairs"]:
        raise ValueError("Historical queue differs from frozen selection")
    by_number = {x["review_number"]: x for x in agents}
    if len(by_number) != len(agents) or set(by_number) != {x["review_number"] for x in pairs}:
        raise ValueError("Every selected pair needs exactly one explicit assisted opinion")
    result = []
    for row in pairs:
        number = row["review_number"]
        note = by_number[number]
        a = note["agent_review"]
        if (note["sample_id"] != row["sample_id"] or note["dataset"] != row["dataset"]
                or note["review_set_id"] != row["review_set_id"]
                or note["source_queue_sha256"] != sha(raw)
                or note["source_result_sha256"] != row["result_sha256"]
                or sha(json.dumps(row["model_result"], sort_keys=True, ensure_ascii=False).encode()) != row["result_sha256"]
                or note["human_verified"] is not False
                or row["human_verified"] is not False or row["pair_review"]["status"] != "pending"
                or any(row["pair_review"][f] is not None for f in FIELDS)
                or a["status"] != "assisted_pending_human" or not a["reason_zh"]
                or not a["reference_visible"] or not a["evidence_anchors"]):
            raise ValueError(f"Source identity or human status mismatch at H{number}")
        if any(a[f] is not None and type(a[f]) is not bool for f in FIELDS):
            raise ValueError(f"Invalid judgment type at H{number}")
        initial, final = a["initial_correct"], a["final_correct"]
        transition = ("correctness_unresolved" if initial is None or final is None
                      else "suggested_fix" if not initial and final
                      else "suggested_regression" if initial and not final
                      else "correct_preserved" if initial
                      else "unresolved_error_or_abstention")
        special = a["outcome"] == "reference_only_flip_not_evidence_verified"
        if (special and (transition != "suggested_fix" or a["correctness_basis"] != "frozen_reference_provisional")) or (not special and a["outcome"] != transition):
            raise ValueError(f"Outcome disagrees with explicit opinion at H{number}")
        for anchor in a["evidence_anchors"]:
            evidence = row["evidence"][anchor["rank"] - 1]
            if (anchor["passage_id"] != evidence["passage_id"]
                    or anchor["evidence_sha256"] != sha(evidence["text"].encode())
                    or evidence["text"][anchor["start"]:anchor["end"]] != anchor["quote"]):
                raise ValueError(f"Evidence anchor mismatch at H{number}")
        result.append({**row, "agent_review": a})
    return result, {"queue": sha(raw), "assisted": sha(assisted_raw)}


def summary(rows, hashes):
    strata = defaultdict(list)
    for row in rows:
        strata[f"{row['cohort']}/{row['dataset']}/{row['strategy']}"].append(row)
    def count(group):
        return {"records": len(group), "outcomes": dict(Counter(x["agent_review"]["outcome"] for x in group))}
    return {"version": "answer_repair_v7_historical_assisted_complete_v1", "sources_sha256": hashes,
            "agent_assistance_complete": True, "human_confirmed": 0, "human_pending": len(rows),
            "reference_visible": True, "human_fields_unchanged": True,
            "sample": count(rows), "by_stratum": {k: count(v) for k, v in strata.items()},
            "unresolved_field_numbers": [x["review_number"] for x in rows if any(x["agent_review"][f] is None for f in FIELDS)],
            "reference_only_numbers": [x["review_number"] for x in rows if x["agent_review"]["outcome"] == "reference_only_flip_not_evidence_verified"],
            "scope": "Agent suggestions for this fixed 45-case sample only; not human gold or a population estimate"}


def render(rows, report):
    esc = lambda x: html.escape(str(x))
    label = lambda x: "待定" if x is None else "是" if x else "否"
    cards = []
    for r in rows:
        a = r["agent_review"]; number = r["review_number"]
        correctness = ("冻结标签匹配（语义正确性未独立核实）"
                       if a["outcome"] == "reference_only_flip_not_evidence_verified" else "正确")
        citations = "".join(f"<blockquote>R{c['rank']} · {esc(c['passage_id'])}<br>{esc(c['quote'])}</blockquote>" for c in r["model_result"]["citations"])
        evidence = "".join(f"<details><summary>R{i} · {esc(e['passage_id'])}</summary><p>{esc(e['text'])}</p></details>" for i, e in enumerate(r["evidence"], 1))
        cards.append(f'''<article id="H{number}" data-outcome="{a['outcome']}">
<div class="meta">H{number} · {esc(r['dataset'].upper())} · {esc(r['cohort'])} · {esc(r['sample_id'])}</div>
<h2>{esc(r['query'])}</h2><div class="answers"><div><b>C1 原答</b><p>{esc(r['initial_answer'])}</p></div><div><b>v7 改答</b><p>{esc(r['final_answer'])}</p></div></div>
<div class="opinion"><b>辅助建议：{OUTCOMES[a['outcome']]}</b><p>{correctness}：{label(a['initial_correct'])} → {label(a['final_correct'])}；Top-5 支持：{label(a['initial_supported'])} → {label(a['final_supported'])}。实际引用：{esc(a.get('final_citation_support', '未单列，见判断理由与引用原文'))}。</p><p>{esc(a['reason_zh'])}</p></div>
<details><summary>实际引用（{len(r['model_result']['citations'])} 条）</summary>{citations or '<p>无引用</p>'}</details>
<details><summary>完整 Top-5 证据</summary>{evidence}</details>
<details><summary>冻结参考与出处</summary><p>{esc(json.dumps(r['frozen_reference'], ensure_ascii=False))}</p><p>结果 SHA-256：{r['result_sha256']}</p><p>依据：{esc(a['correctness_basis'])}；参考可见；人工状态：pending。</p></details>
</article>''')
    options = "".join(f'<option value="{k}">{v}</option>' for k, v in OUTCOMES.items())
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E · v7 历史 45 条改答审核</title><style>
body{margin:0;background:#f1f5f5;color:#17333c;font:16px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}header{background:#143e45;color:white;padding:28px max(22px,calc((100% - 1100px)/2))}h1{margin:0;font-size:27px}header p{margin:10px 0}main{max-width:1100px;margin:auto;padding:20px}.controls{position:sticky;top:0;background:#f1f5f5;padding:12px 0;display:flex;gap:10px;flex-wrap:wrap}input,select{font:inherit;padding:9px;border:1px solid #afc6c7;border-radius:6px}input{flex:1;min-width:220px}article{background:white;padding:24px;border:1px solid #d2dfdf;border-radius:12px;margin:12px 0 22px;scroll-margin-top:90px}h2{font-size:21px;margin:8px 0 16px}.meta{color:#55717b;font-size:13px}.answers{display:grid;grid-template-columns:1fr 1fr;gap:18px}.answers>div{background:#f5f8f7;border-left:3px solid #167a6b;padding:12px}.answers p{margin:5px 0}.opinion{background:#fff6e7;padding:16px;margin:18px 0}details{border-top:1px solid #dae2e2;padding:10px 0}summary{cursor:pointer}p,blockquote{white-space:pre-wrap;overflow-wrap:anywhere}blockquote{border-left:3px solid #9fc5ba;padding:8px 15px;margin:8px 0}.hidden{display:none}#count{align-self:center;font-size:14px}@media(max-width:650px){.answers{grid-template-columns:1fr}article{padding:16px}}
</style><header><h1>E · v7 历史 45 条改答审核</h1><p>代理辅助已完成 45/45，人工确认 0/45。参考答案可见；判断只针对冻结问题和证据。</p><p>建议修复 4 · 疑似误修 9 · 正确保留／同义改写 15 · 仍未答对 10 · 正确性待定 6 · 仅参考翻转 1。</p><p>请在聊天中按 H 编号确认或纠正，例如“H29 同意；H37 暂缓”。这一页只展示材料，不会自动提交人工标签。</p></header><main><div class="controls"><input id="search" aria-label="搜索编号、问题或证据" placeholder="搜索 H 编号、问题、答案或证据"><select id="filter" aria-label="筛选辅助建议"><option value="all">全部建议</option>''' + options + '''</select><span id="count"></span></div>''' + "".join(cards) + '''<p>固定抽样：FEVER 5/5、旧 NQ 20/20、第 4 批 NQ 20/52。不得把这 45 条或辅助意见当作全部 841 条的人工结论。</p></main><script>
const cards=[...document.querySelectorAll('article')],search=document.getElementById('search'),filter=document.getElementById('filter');function update(){const s=search.value.trim().toLowerCase();let n=0;for(const c of cards){const visible=(!s||c.textContent.toLowerCase().includes(s))&&(filter.value==='all'||c.dataset.outcome===filter.value);c.classList.toggle('hidden',!visible);n+=visible;}document.getElementById('count').textContent=`显示 ${n} / ${cards.length}`;}search.addEventListener('input',update);filter.addEventListener('change',update);update();
</script></html>'''


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=ROOT / "artifacts/answer_repair/v7_historical_review_v1")
    args = p.parse_args(argv)
    rows, hashes = load(args.directory); report = summary(rows, hashes)
    (args.directory / "assisted_summary.json").write_bytes(encode(report))
    (args.directory / "H45_review.html").write_text(render(rows, report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
