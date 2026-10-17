"""Build the offline NQ Train review page from frozen sources and explicit confirmations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "scripts/templates"
MARKER = '<meta name="review-builder" content="nq-train-pending-v1">'


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def rows(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def indexed(records: list[dict], name: str) -> dict[int, dict]:
    result = {r["review_number"]: r for r in records}
    if len(result) != len(records):
        raise ValueError(f"Duplicate review numbers in {name}")
    return result


def build_packet(queue_raw: bytes, assisted_raw: bytes, human_raw: bytes,
                 summary_raw: bytes) -> dict:
    queue = indexed(rows(queue_raw), "source queue")
    assisted = indexed(rows(assisted_raw), "assisted judgments")
    humans = indexed(rows(human_raw), "human confirmations")
    summary = json.loads(summary_raw)
    if set(queue) != set(assisted) or summary["human_sha256"] != digest(human_raw):
        raise ValueError("Review sources or human summary disagree")
    if summary["scope_total"] != len(queue) or summary["human_confirmed"] != len(humans):
        raise ValueError("Human counts disagree with source records")
    confirmed = set()
    main_commits = set()
    for n, row in queue.items():
        agent = assisted[n]
        canonical = json.dumps(row, ensure_ascii=False, sort_keys=True).encode()
        if (row["review_set_id"] != summary["review_set_id"] or row["dataset"] != "nq"
                or row["source"]["data_role"] != "development"
                or agent["sample_id"] != row["sample_id"]
                or agent["review_set_id"] != row["review_set_id"]
                or agent["source_record_sha256"] != digest(canonical)
                or agent["source_review_sha256"] != digest(queue_raw)):
            raise ValueError(f"Source identity or hash mismatch: N{n}")
        if [e["rank"] for e in row["evidence"]] != [1, 2, 3, 4, 5]:
            raise ValueError(f"Invalid frozen Top-5: N{n}")
        main_commits.add(agent["source_main_commit"])
    for n, human in humans.items():
        if (n not in queue or human["sample_id"] != queue[n]["sample_id"]
                or human["review_set_id"] != summary["review_set_id"]
                or human["human_verified"] is not True
                or human["human_review"]["status"] != "completed"
                or human["source"]["review_sha256"] != digest(queue_raw)
                or human["source"]["assisted_sha256"] != digest(assisted_raw)):
            raise ValueError(f"Human confirmation does not match source: N{n}")
        confirmed.add(n)
    pending = set(queue) - confirmed
    if (confirmed != set(summary["confirmed_review_numbers"])
            or pending != set(summary["pending_review_numbers"])
            or len(pending) != summary["human_pending"] or len(main_commits) != 1):
        raise ValueError("Pending scope or source main commit is inconsistent")
    order = sorted(pending, key=lambda n: (
        assisted[n]["agent_review"]["suggested_diagnosis_label"] is not None, n))
    records = []
    for n in order:
        row, agent = queue[n], assisted[n]
        records.append({
            "review_id": f"{row['review_set_id']}:nq:{row['sample_id']}",
            "review_number": n, "review_set_id": row["review_set_id"],
            "sample_id": row["sample_id"], "dataset": "nq", "query": row["query"],
            "initial_answer": row["initial_answer"], "evidence": row["evidence"],
            "offline_reference": row["annotation_reference"],
            "frozen_em": row["frozen_em"], "frozen_f1": row["frozen_f1"],
            "source_label": None, "screening_group": row["screening_group"],
            "source_record_sha256": agent["source_record_sha256"],
            "agent_assessment": agent["agent_review"],
        })
    if len({r["sample_id"] for r in records}) != len(records) or not records:
        raise ValueError("Pending records must be nonempty and unique")
    packet = {
        "schema_version": "nq_train_e_review_packet_v1",
        "packet_id": "nq_train_dev_v1_pending_after_batch01",
        "review_set_id": summary["review_set_id"], "source_total": len(queue),
        "already_confirmed_count": len(confirmed), "excluded_confirmed_numbers": sorted(confirmed),
        "source_main_commit": next(iter(main_commits)),
        "sources_sha256": {"queue": digest(queue_raw), "assisted": digest(assisted_raw),
                           "human": digest(human_raw), "human_summary": digest(summary_raw)},
        "counts": {"pending": len(records),
                   "disputed": sum(r["agent_assessment"]["suggested_diagnosis_label"] is None for r in records),
                   "suggested_correct": sum(r["agent_assessment"]["suggested_diagnosis_label"] == "correct" for r in records)},
        "records": records,
    }
    packet["packet_sha256"] = digest(json.dumps(packet, ensure_ascii=False, sort_keys=True,
                                               allow_nan=False).encode())
    return packet


def render(packet: dict) -> str:
    template = (TEMPLATES / "nq_train_pending_review.html").read_text()
    old_template = (TEMPLATES / "answer_repair_review.html").read_text()
    css = re.search(r"<style>(.*?)</style>", old_template, re.S)
    if css is None:
        raise ValueError("Original review-page style is missing")
    encoded = json.dumps(packet, ensure_ascii=False, allow_nan=False)
    for char, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                          ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        encoded = encoded.replace(char, escaped)
    return (template.replace("__BASE_REVIEW_CSS__", css.group(1))
            .replace("__REVIEW_SCRIPT__", (TEMPLATES / "nq_train_pending_review.js").read_text())
            .replace("__REVIEW_PACKET_JSON__", encoded))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=ROOT / "artifacts/answer_repair/nq_train_dev_v1/review/review_queue.jsonl")
    parser.add_argument("--assisted", type=Path, default=ROOT / "data/annotations/assisted/answer_repair_nq_train_dev_v1_20261017.jsonl")
    parser.add_argument("--human", type=Path, default=ROOT / "data/annotations/human/answer_repair_nq_train_dev_v1_batch01_20261017.jsonl")
    parser.add_argument("--human-summary", type=Path, default=ROOT / "data/annotations/human/answer_repair_nq_train_dev_v1_batch01_20261017.summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/answer_repair/nq_train_dev_v1/review/NQ_train_pending_92.html")
    args = parser.parse_args(argv)
    inputs = [args.queue, args.assisted, args.human, args.human_summary]
    out = args.out.resolve()
    manifest = out.with_suffix(".manifest.json")
    if (out.suffix.lower() != ".html" or out in {p.resolve() for p in inputs}
            or manifest in {p.resolve() for p in inputs}):
        raise ValueError("Output must be a separate HTML file")
    if out.exists() and MARKER not in out.read_text():
        raise ValueError("Refusing to replace a different existing page")
    packet = build_packet(*(p.read_bytes() for p in inputs))
    raw = render(packet).encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    metadata = {"packet_id": packet["packet_id"], "packet_sha256": packet["packet_sha256"],
                "counts": packet["counts"], "excluded_confirmed_numbers": packet["excluded_confirmed_numbers"],
                "review_numbers": [r["review_number"] for r in packet["records"]],
                "source_main_commit": packet["source_main_commit"],
                "sources_sha256": packet["sources_sha256"], "html_sha256": digest(raw),
                "build_sources_sha256": {p.name: digest(p.read_bytes()) for p in (
                    Path(__file__), TEMPLATES / "answer_repair_review.html",
                    TEMPLATES / "nq_train_pending_review.html", TEMPLATES / "nq_train_pending_review.js")},
                "human_decisions_prefilled": False, "external_network_requests": False,
                "scope": "Offline development review; browser drafts and exports do not modify source labels or repository files"}
    manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"html": str(out), "counts": packet["counts"],
                      "already_confirmed_excluded": packet["already_confirmed_count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
