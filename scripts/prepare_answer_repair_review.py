"""Prepare a training-only, offline assisted-review packet; never edit source labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C1_COMMIT = "6f08f2fe2dc2dcf39b75177763207682b2890a31"
LABEL_COMMIT = "7c7863f98f87e892119481a79e23ca6dcebdced0"


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def shared_training(splits: list[dict], dataset: str) -> tuple[set[str], set[str]]:
    trains, reserved = [], set()
    for doc in splits:
        part = doc["splits"][dataset]
        values = {k: set(part[k]) for k in ("train", "validation", "test")}
        if any(len(values[k]) != len(part[k]) for k in values):
            raise ValueError("Duplicate split IDs")
        if any(values[a] & values[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError("Overlapping split partitions")
        trains.append(values["train"])
        reserved |= values["validation"] | values["test"]
    return set.intersection(*trains) - reserved, reserved


def index_jsonl(raw: bytes) -> dict[str, dict]:
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    indexed = {str(row["sample_id"]): row for row in rows}
    if len(rows) != len(indexed):
        raise ValueError("Duplicate sample IDs")
    return indexed


def build_packet(notes_path: Path, controls: int = 5) -> dict:
    sources = {}

    def source(commit: str, path: str) -> bytes:
        raw = git_bytes(commit, path)
        sources[f"{commit}:{path}"] = digest(raw)
        return raw

    splits = [json.loads(source(LABEL_COMMIT, f"configs/splits/{name}.json"))
              for name in ("diagnosis_v1_split", "diagnosis_v2_expanded_split")]
    old_path = ROOT / "configs/splits/answer_repair_v1_dev.json"
    old_raw = old_path.read_bytes()
    sources[str(old_path.relative_to(ROOT))] = digest(old_raw)
    old = json.loads(old_raw)
    notes = {}
    if notes_path.exists():
        notes_raw = notes_path.read_bytes()
        sources[str(notes_path.relative_to(ROOT))] = digest(notes_raw)
        for row in map(json.loads, filter(None, notes_raw.decode().splitlines())):
            key = (row["dataset"], str(row["sample_id"]))
            if key in notes:
                raise ValueError("Duplicate assisted-review note")
            notes[key] = row
    rows, counts = [], {}
    for dataset in ("fever", "nq"):
        eligible, reserved = shared_training(splits, dataset)
        labels = index_jsonl(source(LABEL_COMMIT, f"data/annotations/normalized/{dataset}_500_labels.jsonl"))
        c1 = index_jsonl(source(C1_COMMIT, f"results/preliminary/{dataset}_c1_7b.jsonl"))
        seen = set(old["datasets"][dataset]["sample_ids"])
        candidates = {sid for sid in eligible & labels.keys() & c1.keys()
                      if labels[sid]["needs_manual_review"] is False}
        util = sorted(sid for sid in candidates if labels[sid]["primary_label"] == "evidence_utilisation")
        correct = sorted((sid for sid in candidates - seen if labels[sid]["primary_label"] == "correct"),
                         key=lambda sid: (digest(f"E-review-control:42:{dataset}:{sid}".encode()), sid))[:controls]
        if len(correct) != controls:
            raise ValueError("Insufficient fresh correct controls")
        counts[dataset] = {"eligible_training_ids": len(eligible), "utilisation": len(util),
                           "utilisation_new_to_E": len(set(util) - seen), "correct_controls": len(correct),
                           "reserved_overlap": len((set(util) | set(correct)) & reserved)}
        for sid in util + correct:
            record = c1[sid]
            if record["dataset"] != {"nq": "natural_questions", "fever": "fever"}[dataset] or record["condition"] != "C1":
                raise ValueError("Unexpected source condition")
            evidence = [{"rank": i + 1, **item} for i, item in enumerate(record["evidence"])]
            if len(evidence) != 5:
                raise ValueError("Expected archived Top-5")
            assessment = notes.get((dataset, sid))
            if assessment:
                for quote in assessment["quoted_evidence"]:
                    if quote["quote"] not in evidence[quote["rank"] - 1]["text"]:
                        raise ValueError(f"Non-verbatim agent excerpt: {dataset}/{sid}")
            rows.append({"review_id": f"{dataset}:{sid}", "sample_id": sid, "dataset": dataset,
                         "cohort": "utilisation_candidate" if sid in util else "correct_control",
                         "previous_E_dev50": sid in seen, "query": record["question_or_claim"],
                         "initial_answer": record["prediction"], "evidence": evidence,
                         "source_annotation": labels[sid],
                         "offline_reference": {"gold_answer_or_label": record["gold_answer_or_label"],
                                               "correct": record["correct"], "em": record.get("em"), "f1": record.get("f1")},
                         "agent_assessment": assessment, "human_review_status": "pending"})
    if notes.keys() - {(r["dataset"], r["sample_id"]) for r in rows}:
        raise ValueError("Notes include IDs outside the selected training cohort")
    return {"packet_id": "e_assisted_review_20261013_v1", "created_date": "2026-10-13",
            "purpose": "Offline development review only; agent suggestions are not human labels or runtime input",
            "selection": "All trusted utilisation in v1/v2 shared training within archived C1 frozen500; 5 fresh correct controls per dataset by seeded ID hash",
            "source_commits": {"c1": C1_COMMIT, "annotation_and_split": LABEL_COMMIT},
            "source_sha256": sources, "counts": counts,
            "agent_assessment_counts": dict(Counter(r["agent_assessment"]["e_core_suggestion"] for r in rows if r["agent_assessment"])),
            "records": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/answer_repair/review_20261013")
    parser.add_argument("--notes", type=Path, default=ROOT / "data/annotations/assisted/answer_repair_review_20261013.jsonl")
    args = parser.parse_args()
    packet = build_packet(args.notes.resolve())
    packet["packet_sha256"] = digest(json.dumps(packet, ensure_ascii=False, sort_keys=True).encode())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "review_packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
    template = ROOT / "scripts/templates/answer_repair_review.html"
    if template.exists():
        payload = json.dumps(packet, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
        (args.out / "E_assisted_review.html").write_text(template.read_text().replace("__REVIEW_PACKET_JSON__", payload))
    print(json.dumps({"records": len(packet["records"]), "counts": packet["counts"], "agent_assessment_counts": packet["agent_assessment_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
