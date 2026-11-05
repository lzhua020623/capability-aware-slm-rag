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
    label = lambda x: "Not binary-scored" if x is None else "Correct" if x else "Incorrect"
    names = {"fixed": "Fixed", "regressed": "Regression", "correct_preserved": "Already correct, still correct",
             "wrong_preserved": "Both answers wrong", "correctness_unresolved": "Correctness still needs a score"}
    css = re.search(r"<style>(.*?)</style>", render(source_rows[:1], {}), re.S).group(1)
    def run_label(cohort, strategy):
        group = "Dev 75" if cohort == "nq75" else "Challenge 125"
        arm = "Full r3" if strategy == "conservative_repair" else "v7"
        return group + " · " + arm
    table = "".join(f"<tr><td>{esc(run_label(k.split('/')[0], k.split('/')[-1]))}</td><td>{v['outcomes'].get('fixed',0)}</td><td>{v['outcomes'].get('regressed',0)}</td><td>{v['correctness_unresolved']}</td><td>{esc(v['all_changed_pairs_net_change_bounds'])}</td></tr>" for k, v in report["by_run"].items())
    cards = []
    for row in rows:
        n = row["review_number"]; h = row["human_review"]
        w = row["review_workflow"]
        status = {"closed_keep_initial": "Closed: keep C1 original answer",
                  "closed_external_check": "Closed: external check completed"}.get(w["status"], names[h["outcome"]])
        note = w.get("reason", "")
        sources = "".join(f'<a href="{esc(s["url"])}" target="_blank" rel="noopener noreferrer">{esc(s["title"])}</a> '
                          for s in w.get("external_assessment", {}).get("sources", []))
        preferred = ("Keep the C1 original answer; the rewrite is not binary-scored and is not automatically a regression." if h["outcome"] == "correctness_unresolved"
                     else "Prefer the C1 original answer.") if h["preference"] == "initial_better" else ""
        reason_html = (f'<details><summary>Historical scoring note</summary><p>{esc(h["reason"])}</p></details>'
                       if w["status"] in ("closed_keep_initial", "closed_external_check")
                       else f'<p>{esc(h["reason"])}</p>')
        cards.append(f'''<article id="F{n}"><div class="meta">F{n} · {esc(row['source_alias'])} · {esc(run_label(row['cohort'], row['strategy']))}</div>
<h2>{esc(row['query'])}</h2><div class="answers"><div><b>C1 original answer</b><p>{esc(row['initial_answer'])}</p></div><div><b>Rewritten answer</b><p>{esc(row['final_answer'])}</p></div></div>
<div class="opinion"><b>{esc(status)}</b><p>{esc(note)}</p><p>Human binary score: {label(h['initial_correct'])} → {label(h['final_correct'])}. {preferred}</p>{reason_html}{sources}</div>
<a href="F42_review.html#F{n}" target="_blank">View the frozen assisted opinion, actual citations, and full Top-5</a><p class="meta">Result SHA-256: {row['result_sha256']}</p></article>''')
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E · F42 human review record</title><style>{css}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;border:1px solid #ccd8d8;text-align:left}}</style>
<header><h1>E · Final answer-repair human review</h1><p>Closed {report['workflow_closed_pairs']} items; pending {len(report['workflow_pending_numbers'])} items.</p><p>{report['correctness_resolved_pairs']} items can be binary-scored; {len(report['closed_without_binary_score_numbers'])} items are closed with the scoring note kept. Closed items do not need another review. Correctness and evidence support are recorded separately.</p></header>
<main><p>The net-change range in the table includes every possible outcome of unresolved items. The difference among only adjudicated items is not the final gain. Strategies cannot be added together.</p>
<table><tr><th>Cohort / strategy</th><th>Fixed</th><th>Regression</th><th>Not binary-scored</th><th>Full rewrite net-change range</th></tr>{table}</table>
<div class="controls"><input id="search" autocomplete="off" spellcheck="false" aria-label="Search by id or question" placeholder="Enter an F id, question, or answer"><span id="count"></span></div>{''.join(cards)}
<p>Reference-visible chat-assisted human review, not an independent blind review. Gold and EM are unchanged. This is not an accuracy figure for all of NQ.</p></main><script>
const cards=[...document.querySelectorAll('article')],search=document.getElementById('search');function update(){{const s=search.value.trim().toLowerCase();let n=0;for(const c of cards){{const match=/^f[0-9]+$/.test(s)?c.id.toLowerCase()===s:c.textContent.toLowerCase().includes(s);const visible=!s||match;c.classList.toggle('hidden',!visible);n+=visible;}}document.getElementById('count').textContent=`Showing ${{n}} / ${{cards.length}}`;}}search.addEventListener('input',update);update();
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
    if view.exists() and "<title>E · F42 human review record</title>" not in view.read_text():
        raise ValueError("Refusing to replace an unrelated review page")
    view.write_bytes(page(rows, source_rows, report).encode())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
