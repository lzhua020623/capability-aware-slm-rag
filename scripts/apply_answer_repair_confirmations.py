"""Turn the user's per-number confirmations of the assisted review into human pair reviews.

Input: the assisted pairs file (agent_review present, pair_review pending) and a
confirmation JSON written from the chat decisions, for example:

{
  "reviewer": "Qihan Zhu", "confirmed_at": "2026-10-14", "timezone": "Australia/Sydney",
  "user_text": "A/B 全部同意；31、32 初始算正确；113、121 暂缓",
  "accept_agent": [1, 2, 18, 19],            # review numbers whose four agent booleans are confirmed as-is
  "override": {"31": {"initial_correct": true, "reason": "按完整性口径第 2 条"}},
  "skip": [113, 121]                          # stay pending
}

Every confirmed number is copied to its duplicate records (same dataset/sample_id)
so the per-cohort files stay complete. Output records keep the result SHA-256 so a
review cannot be applied to a different model output. Nothing is inferred: numbers
that are neither accepted nor overridden remain pending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIELDS = ("initial_correct", "final_correct", "initial_supported", "final_supported")


def write_frozen(path: Path, raw: bytes) -> None:
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f"Refusing to overwrite a different file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def apply_confirmations(pairs: list[dict], confirmations: dict) -> tuple[list[dict], dict]:
    by_number = {r["review_number"]: r for r in pairs}
    if len(by_number) != len(pairs):
        raise ValueError("Duplicate review numbers in the assisted pairs file")
    accept = {int(n) for n in confirmations.get("accept_agent", [])}
    override = {int(n): v for n, v in confirmations.get("override", {}).items()}
    skip = {int(n) for n in confirmations.get("skip", [])}
    for group_a, group_b in ((accept, override.keys()), (accept, skip), (override.keys(), skip)):
        if set(group_a) & set(group_b):
            raise ValueError("A review number cannot be accepted, overridden and skipped at once")
    unknown = (accept | set(override) | skip) - by_number.keys()
    if unknown:
        raise ValueError(f"Unknown review numbers: {sorted(unknown)}")
    reviewer = confirmations["reviewer"].strip()
    if not reviewer or not confirmations.get("user_text", "").strip():
        raise ValueError("reviewer and user_text are required")
    # Decisions apply to every record that shares the same review material.
    decided: dict[str, tuple[dict, str]] = {}
    for number in sorted(accept | set(override)):
        row = by_number[number]
        agent = row["agent_review"]
        values = {f: agent[f] for f in FIELDS}
        note = f"用户确认第 {number} 条代理建议"
        if number in override:
            changes = {k: v for k, v in override[number].items() if k in FIELDS}
            bad = [k for k, v in changes.items() if not isinstance(v, bool)]
            if bad:
                raise ValueError(f"Override values must be booleans: {number} {bad}")
            values.update(changes)
            note = f"用户改判第 {number} 条：{override[number].get('reason', '').strip() or '无补充理由'}"
        if any(v is None for v in values.values()):
            raise ValueError(f"Review {number} still has an unresolved field; override it or skip it")
        key = row["shared_review_id"]
        if key in decided and decided[key][0] != values:
            raise ValueError(f"Conflicting decisions for duplicate records of {key}")
        decided[key] = (values, note)
    outputs, summary = [], defaultdict(lambda: {"completed": 0, "pending": 0, "fixed": 0, "regressed": 0,
                                                "grounded_fixed": 0, "initial_wrong": 0, "initial_correct": 0})
    for row in pairs:
        run = f"{row['dataset']}/{row['cohort']}/{row['strategy']}"
        record = {k: row[k] for k in ("review_number", "shared_review_id", "dataset", "cohort", "strategy",
                                      "sample_id", "result_sha256", "review_material_sha256", "query",
                                      "initial_answer", "final_answer", "changed")}
        key = row["shared_review_id"]
        if key in decided and row["review_number"] not in skip:
            values, note = decided[key]
            record["pair_review"] = {"status": "completed", "reviewer": reviewer,
                                     "reason": f"{note}；代理理由：{row['agent_review']['reason_zh']}", **values}
            record["confirmation"] = {"user_text": confirmations["user_text"], "client_date": confirmations["confirmed_at"],
                                      "timezone": confirmations.get("timezone"),
                                      "review_mode": "chat_assisted_summary_confirmation",
                                      "note": "User confirmed or overrode the agent-assisted summary; not an independent blind annotation."}
            s = summary[run]
            s["completed"] += 1
            s["initial_wrong" if not values["initial_correct"] else "initial_correct"] += 1
            if not values["initial_correct"] and values["final_correct"]:
                s["fixed"] += 1
                s["grounded_fixed"] += values["final_supported"]
            if values["initial_correct"] and not values["final_correct"]:
                s["regressed"] += 1
        else:
            record["pair_review"] = {"status": "pending", "reviewer": "", "reason": "",
                                     **{f: None for f in FIELDS}}
            summary[run]["pending"] += 1
        outputs.append(record)
    unique = {}
    for r in outputs:
        if r["pair_review"]["status"] == "completed":
            unique.setdefault(r["shared_review_id"], r["pair_review"])
    report = {"human_verified_mode": "chat_assisted_summary_confirmation", "blind": False,
              "by_run": dict(summary),
              "unique_completed": len(unique),
              "unique_fixed": sum(not v["initial_correct"] and v["final_correct"] for v in unique.values()),
              "unique_regressed": sum(v["initial_correct"] and not v["final_correct"] for v in unique.values()),
              "unique_pending": len({r["shared_review_id"] for r in outputs}) - len(unique)}
    return outputs, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=ROOT / "artifacts/answer_repair/v6r1_reviewed/assisted_review_20261014/assisted_pairs.jsonl")
    parser.add_argument("--confirmations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="human pair review JSONL (frozen once written)")
    args = parser.parse_args(argv)
    pairs = [json.loads(line) for line in args.pairs.read_bytes().decode("utf-8").splitlines() if line.strip()]
    confirmations = json.loads(args.confirmations.read_bytes())
    outputs, report = apply_confirmations(pairs, confirmations)
    raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in outputs).encode()
    write_frozen(args.out, raw)
    report["output_sha256"] = hashlib.sha256(raw).hexdigest()
    report["pairs_sha256"] = hashlib.sha256(args.pairs.read_bytes()).hexdigest()
    write_frozen(args.out.with_suffix(".summary.json"), (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
