"""Record material-specific human decisions, preserving confirmed uncertainty."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_answer_repair_experiment import encode_json, write_frozen
from scripts.summarize_answer_repair_historical_review import FIELDS, load, render


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def outcome(values):
    initial, final = values["initial_correct"], values["final_correct"]
    if initial is None or final is None:
        return "correctness_unresolved"
    if not initial and final:
        return "fixed"
    if initial and not final:
        return "regressed"
    return "correct_preserved" if initial else "wrong_preserved"


def workflow(classification, preference, decision):
    """Closing a review task must not silently invent a correctness label."""
    if decision is None:
        return {"status": "pending" if classification == "correctness_unresolved" else "closed_scored"}
    status = decision.get("status")
    if classification != "correctness_unresolved" or not decision.get("reason"):
        raise ValueError("Workflow exceptions need an unscored case and an explicit reason")
    if status == "closed_keep_initial":
        if preference != "initial_better":
            raise ValueError("Keeping the original requires an explicit original-answer preference")
    elif status == "closed_external_check":
        assessment = decision.get("external_assessment", {})
        if assessment.get("basis") != "assistant_external_fact_check" or not assessment.get("sources"):
            raise ValueError("External closure needs separately attributed sources")
    else:
        raise ValueError("Unknown review workflow decision")
    return dict(decision)


def count(rows):
    judgments = [r["human_review"] for r in rows]
    lower = upper = 0
    for j in judgments:
        initials = [False, True] if j["initial_correct"] is None else [j["initial_correct"]]
        finals = [False, True] if j["final_correct"] is None else [j["final_correct"]]
        deltas = [int(f) - int(i) for i in initials for f in finals]
        lower += min(deltas); upper += max(deltas)
    counts = Counter(j["outcome"] for j in judgments)
    return {"reviewed_pairs": len(rows), "correctness_resolved": sum(j["outcome"] != "correctness_unresolved" for j in judgments),
            "correctness_unresolved": counts["correctness_unresolved"], "outcomes": dict(counts),
            "workflow_pending": sum(r["review_workflow"]["status"] == "pending" for r in rows),
            "closed_without_binary_score": sum(r["review_workflow"]["status"].startswith("closed_")
                                                and r["human_review"]["outcome"] == "correctness_unresolved" for r in rows),
            "resolved_pairs_net_change": counts["fixed"] - counts["regressed"],
            "all_changed_pairs_net_change": lower if lower == upper else None,
            "all_changed_pairs_net_change_bounds": [lower, upper],
            "warning": "Resolved-pair counts omit unresolved outcomes. Bounds retain all changed pairs; not whole-cohort accuracy or independent blind evaluation."}


def apply(rows, confirmation):
    by_number = {r["review_number"]: r for r in rows}
    if len(by_number) != len(rows):
        raise ValueError("Duplicate material review numbers")
    accepted = set(confirmation["accept_agent"])
    overridden = {int(k): v for k, v in confirmation["override"].items()}
    workflow_decisions = {int(k): v for k, v in confirmation.get("workflow_decisions", {}).items()}
    if not workflow_decisions.keys() <= by_number.keys():
        raise ValueError("Workflow decisions reference unknown materials")
    if accepted & overridden.keys() or accepted | overridden.keys() != by_number.keys():
        raise ValueError("Confirmations must cover each material exactly once")
    if (set(confirmation["all_review_numbers_addressed"]) != by_number.keys()
            or not confirmation["reviewer"].strip() or not confirmation["user_text"].strip()):
        raise ValueError("Incomplete confirmation identity or scope")
    outputs = []
    for n, row in by_number.items():
        if row["review_set_id"] != confirmation["review_set_id"]:
            raise ValueError("Review-set mismatch")
        agent = row["agent_review"]
        decision = overridden.get(n)
        if decision:
            if not set(FIELDS) <= decision.keys() or not decision["reason"]:
                raise ValueError("Explicit decision needs all fields and a reason")
            values = {f: decision[f] for f in FIELDS}
            reason = decision["reason"]
            preference = decision.get("preference")
            support_confirmed = decision["support_fields_confirmed"]
            basis = "explicit_user_clarification"
        else:
            values = {f: agent[f] for f in FIELDS}
            reason = "用户同意本条辅助意见；其中 null 继续保留待定。代理理由：" + agent["reason_zh"]
            preference, support_confirmed, basis = None, True, "accepted_agent_fields_including_nulls"
        if any(v is not None and type(v) is not bool for v in values.values()):
            raise ValueError("Judgments must be booleans or unresolved nulls")
        human_initial = row.get("initial_human_review")
        if human_initial and values["initial_correct"] != human_initial["initial_correct"]:
            raise ValueError("This confirmation does not authorize changing prior C1 labels")
        classification = outcome(values)
        review_workflow = workflow(classification, preference, workflow_decisions.get(n))
        # Bind each decision to this output material, never copy by sample ID alone.
        record = {k: row[k] for k in ("review_number", "review_set_id", "source_alias", "source_run", "dataset",
                                      "sample_id", "cohort", "strategy", "result_sha256", "review_material_sha256",
                                      "query", "initial_answer", "final_answer", "changed")}
        record.update({"human_reviewed": True, "review_workflow": review_workflow,
                       "human_review": {"status": "reviewed_unresolved" if classification == "correctness_unresolved" else "completed_correctness",
                                        "reviewer": confirmation["reviewer"], "reviewed_at": confirmation["confirmed_at"],
                                        **values, "outcome": classification, "preference": preference, "reason": reason,
                                        "field_basis": basis, "support_fields_confirmed": support_confirmed},
                       "confirmation": {"mode": confirmation["mode"], "independent_blind_annotation": False,
                                        "timezone": confirmation["timezone"], "source_sha256": confirmation["source_sha256"]}})
        outputs.append(record)
    groups = defaultdict(list)
    for row in outputs:
        groups[f"{row['cohort']}/{row['dataset']}/{row['strategy']}"].append(row)
    report = {"version": "answer_repair_final_human_confirmation_v1", "human_reviewed_pairs": len(outputs),
              "unique_dataset_sample_ids_reviewed": len({(x['dataset'], x['sample_id']) for x in outputs}),
              "correctness_resolved_pairs": sum(x["human_review"]["outcome"] != "correctness_unresolved" for x in outputs),
              "fully_resolved_four_fields": sum(all(x["human_review"][f] is not None for f in FIELDS) for x in outputs),
              "correctness_unresolved_numbers": [x["review_number"] for x in outputs if x["human_review"]["outcome"] == "correctness_unresolved"],
              "workflow_closed_pairs": sum(x["review_workflow"]["status"].startswith("closed_") for x in outputs),
              "workflow_pending_numbers": [x["review_number"] for x in outputs if x["review_workflow"]["status"] == "pending"],
              "closed_without_binary_score_numbers": [x["review_number"] for x in outputs
                                                       if x["review_workflow"]["status"].startswith("closed_")
                                                       and x["human_review"]["outcome"] == "correctness_unresolved"],
              "support_unresolved_numbers": [x["review_number"] for x in outputs if any(x["human_review"][f] is None for f in FIELDS[2:])],
              "initial_answer_preferred_numbers": [x["review_number"] for x in outputs if x["human_review"]["preference"] == "initial_better"],
              "by_run": {k: count(v) for k, v in groups.items()},
              "blind": False, "gold_changed": False, "algorithm_or_metrics_changed": False,
              "scope": "F1–F42 only; each output material retained separately. No pooled cross-strategy net gain or whole-challenge accuracy is inferred."}
    return outputs, report


def page(rows, source_rows, report):
    esc = lambda x: html.escape(str(x))
    label = lambda x: "未作二元评分" if x is None else "对" if x else "错"
    names = {"fixed": "修复", "regressed": "误修", "correct_preserved": "原本正确，仍正确",
             "wrong_preserved": "两答都错", "correctness_unresolved": "尚需明确评分"}
    css = re.search(r"<style>(.*?)</style>", render(source_rows[:1], {}), re.S).group(1)
    def run_label(cohort, strategy):
        group = "开发75" if cohort == "nq75" else "挑战125"
        arm = "完整 r3" if strategy == "conservative_repair" else "v7"
        return group + " · " + arm
    table = "".join(f"<tr><td>{esc(run_label(k.split('/')[0], k.split('/')[-1]))}</td><td>{v['outcomes'].get('fixed',0)}</td><td>{v['outcomes'].get('regressed',0)}</td><td>{v['correctness_unresolved']}</td><td>{esc(v['all_changed_pairs_net_change_bounds'])}</td></tr>" for k, v in report["by_run"].items())
    cards = []
    for row in rows:
        n = row["review_number"]; h = row["human_review"]
        w = row["review_workflow"]
        status = {"closed_keep_initial": "已处理：保留C1原答",
                  "closed_external_check": "已处理：已完成外部核查"}.get(w["status"], names[h["outcome"]])
        note = w.get("reason", "")
        sources = "".join(f'<a href="{esc(s["url"])}" target="_blank" rel="noopener noreferrer">{esc(s["title"])}</a> '
                          for s in w.get("external_assessment", {}).get("sources", []))
        preferred = ("保留C1原答；改答未作二元评分，不自动计为误修。" if h["outcome"] == "correctness_unresolved"
                     else "偏好C1原答。") if h["preference"] == "initial_better" else ""
        reason_html = (f'<details><summary>历史评分说明</summary><p>{esc(h["reason"])}</p></details>'
                       if w["status"] in ("closed_keep_initial", "closed_external_check")
                       else f'<p>{esc(h["reason"])}</p>')
        cards.append(f'''<article id="F{n}"><div class="meta">F{n} · {esc(row['source_alias'])} · {esc(run_label(row['cohort'], row['strategy']))}</div>
<h2>{esc(row['query'])}</h2><div class="answers"><div><b>C1 原答</b><p>{esc(row['initial_answer'])}</p></div><div><b>改答</b><p>{esc(row['final_answer'])}</p></div></div>
<div class="opinion"><b>{esc(status)}</b><p>{esc(note)}</p><p>人工二元评分：{label(h['initial_correct'])} → {label(h['final_correct'])}。{preferred}</p>{reason_html}{sources}</div>
<a href="F42_review.html#F{n}" target="_blank">查看冻结辅助意见、实际引用和完整 Top-5</a><p class="meta">结果 SHA-256：{row['result_sha256']}</p></article>''')
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E · F42 人工审核记录</title><style>{css}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;border:1px solid #ccd8d8;text-align:left}}</style>
<header><h1>E · 最终改答人工审核</h1><p>已处理 {report['workflow_closed_pairs']} 条；待处理 {len(report['workflow_pending_numbers'])} 条。</p><p>{report['correctness_resolved_pairs']} 条可二元计分；{len(report['closed_without_binary_score_numbers'])} 条已处理并保留评分说明。已处理的项目无需重复审核。正确性与证据支持分别记录。</p></header>
<main><p>下表净变化范围包含待定项的所有可能结果，不把只统计已定案条目的差值当最终收益；各策略不能相加。</p>
<table><tr><th>批次／策略</th><th>修复</th><th>误修</th><th>未作二元评分</th><th>完整改答净变化范围</th></tr>{table}</table>
<div class="controls"><input id="search" autocomplete="off" spellcheck="false" aria-label="搜索编号或问题" placeholder="输入F编号、问题或答案"><span id="count"></span></div>{''.join(cards)}
<p>参考可见的聊天辅助人审，非独立盲审；不更改 gold 或 EM，不推广为整个 NQ 的准确率。</p></main><script>
const cards=[...document.querySelectorAll('article')],search=document.getElementById('search');function update(){{const s=search.value.trim().toLowerCase();let n=0;for(const c of cards){{const match=/^f[0-9]+$/.test(s)?c.id.toLowerCase()===s:c.textContent.toLowerCase().includes(s);const visible=!s||match;c.classList.toggle('hidden',!visible);n+=visible;}}document.getElementById('count').textContent=`显示 ${{n}} / ${{cards.length}}`;}}search.addEventListener('input',update);update();
</script></html>'''


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirmations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--directory", type=Path, default=ROOT / "artifacts/answer_repair/final_review_v1")
    args = p.parse_args(argv)
    raw = args.confirmations.read_bytes(); confirmation = json.loads(raw)
    for name, expected in confirmation["source_sha256"].items():
        if sha((args.directory / name).read_bytes()) != expected:
            raise ValueError("Confirmation source has changed: " + name)
    source_rows, _ = load(args.directory)
    rows, report = apply(source_rows, confirmation)
    output = b"".join((json.dumps(x, ensure_ascii=False) + "\n").encode() for x in rows)
    report.update({"confirmation_sha256": sha(raw), "output_sha256": sha(output), "source_sha256": confirmation["source_sha256"]})
    write_frozen(args.out, output)
    write_frozen(args.out.with_suffix(".summary.json"), encode_json(report))
    # This is the current derived view. The confirmation and annotation files above remain immutable.
    view = args.directory / "F42_human_review.html"
    if view.exists() and "<title>E · F42 人工审核记录</title>" not in view.read_text():
        raise ValueError("Refusing to replace an unrelated review page")
    view.write_bytes(page(rows, source_rows, report).encode())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
