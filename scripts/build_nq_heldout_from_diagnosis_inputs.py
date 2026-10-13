"""Freeze an NQ held-out cohort from the Diagnosis 3216 pool that E has never seen.

The shared-training pool of the preliminary 500 has no utilisation-labelled
sample left for E (all were used by the 37-sample review packet), so a
repair-rate estimate needs the Diagnosis v2-expanded TRAIN IDs outside the
preliminary 500: 1587 trusted samples, 120 of them labelled utilisation.
Runtime inputs (question, C1 prediction, Top-5 text) are the committed
upstream export; gold short answers are not in the repository and must be
supplied as a JSONL of {"sample_id": ..., "gold_answers": [...]} obtained from
the Diagnosis owner's workbook or regenerated from raw NQ.

Selection is label-stratified ON PURPOSE (all utilisation, all accessibility,
a seeded sample of correct and insufficiency) and must be reported as an
oracle-label stratified cohort, never as the deployment distribution. Labels
are read here only to choose IDs; they are not written into the C1 records.

Without --gold the script freezes the ID manifest only, so gold can be
requested for exactly these IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_answer_repair_experiment import encode_json, write_frozen
from scripts.run_answer_repair import prepare_records
from src.compensation.answer_repair import MODEL_NAME
from src.diagnosis import DatasetName
from src.evaluation.qa import best_em_f1

UPSTREAM = "origin/feature/diagnosis-probe-zheng"
INPUTS = "data/annotations/normalized/nq_3216_diagnosis_inputs.jsonl"
LABELS = "data/annotations/normalized/nq_3216_labels_normalized.jsonl"
SPLIT = "configs/splits/diagnosis_v2_expanded_split.json"
STRATA = {"evidence_utilisation": None, "evidence_accessibility": None, "correct": 100, "evidence_insufficiency": 45}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_read(ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT)


def select_ids(split: dict, prelim_ids: set[str], labels: dict[str, dict], seed: int = 42) -> dict[str, list[str]]:
    pool = [sid for sid in split["splits"]["nq"]["train"] if sid not in prelim_ids]
    if any(sid not in labels for sid in pool):
        raise ValueError("Every pooled ID needs a normalized label")
    if any(labels[sid]["needs_manual_review"] for sid in pool):
        raise ValueError("Pool must contain trusted labels only")
    reserved = set(split["splits"]["nq"]["validation"]) | set(split["splits"]["nq"]["test"])
    if set(pool) & reserved:
        raise ValueError("Pool overlaps reserved Diagnosis validation/test IDs")
    chosen = {}
    for label, count in STRATA.items():
        ids = sorted((sid for sid in pool if labels[sid]["primary_label"] == label),
                     key=lambda sid: (sha256(f"nq-heldout-3216:{seed}:{label}:{sid}".encode()), sid))
        chosen[label] = ids if count is None else ids[:count]
    return chosen


def c1_record(row: dict, golds: list[str]) -> dict:
    prediction = row["c1_prediction"] if isinstance(row["c1_prediction"], str) else str(row["c1_prediction"])
    em, _ = best_em_f1(prediction, golds)
    return {"sample_id": str(row["sample_id"]), "dataset": "natural_questions", "condition": "C1",
            "model_name": MODEL_NAME, "question_or_claim": row["question"],
            "evidence": [{"passage_id": e["passage_id"], "text": e["text"], "score": e["score"]}
                         for e in sorted(row["top5_evidence"], key=lambda e: e["rank"])],
            "prediction": prediction, "gold_answer_or_label": golds, "correct": bool(em),
            "provenance": "diagnosis_3216_inputs_export; retrieval scores rounded to 4 decimals upstream"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-ref", default=UPSTREAM)
    parser.add_argument("--prelim", type=Path, default=ROOT / "configs/splits/preliminary_nq_500.json")
    parser.add_argument("--gold", type=Path, help="JSONL of {sample_id, gold_answers}; omit to freeze IDs only")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/splits/answer_repair_nq_heldout_3216_v1.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/answer_repair/v7_reviewed/inputs_extra/nq_heldout3216_c1.jsonl")
    args = parser.parse_args(argv)
    commit = subprocess.check_output(["git", "rev-parse", "--verify", args.upstream_ref + "^{commit}"], cwd=ROOT, text=True).strip()
    split_raw, labels_raw, inputs_raw = (git_read(commit, p) for p in (SPLIT, LABELS, INPUTS))
    split = json.loads(split_raw)
    labels = {str(r["sample_id"]): r for r in (json.loads(l) for l in labels_raw.decode().splitlines() if l.strip())}
    prelim = {str(i) for i in json.loads(args.prelim.read_bytes())["ids"]}
    chosen = select_ids(split, prelim, labels)
    ordered = [sid for label in STRATA for sid in chosen[label]]
    manifest = {"version": "answer_repair_nq_heldout_3216_v1", "dataset": "nq", "seed": 42,
                "purpose": "E-untouched, oracle-label stratified held-out from Diagnosis v2-expanded TRAIN outside the preliminary 500; run once; report by stratum",
                "source_commit": commit,
                "sources_sha256": {SPLIT: sha256(split_raw), LABELS: sha256(labels_raw), INPUTS: sha256(inputs_raw),
                                   str(args.prelim.relative_to(ROOT)): sha256(args.prelim.read_bytes())},
                "strata": {label: {"requested": count, "selected": len(chosen[label])} for label, count in STRATA.items()},
                "sample_ids": ordered, "sample_ids_by_stratum": chosen,
                "gold_required": "JSONL {sample_id, gold_answers}; not in the repository"}
    write_frozen(args.manifest, encode_json(manifest))
    print(json.dumps({"manifest": str(args.manifest), "selected": {k: len(v) for k, v in chosen.items()}}, indent=2))
    if not args.gold:
        print("No --gold given: ID manifest frozen only; request gold short answers for these IDs.")
        return 0
    golds = {}
    for line in args.gold.read_bytes().decode().splitlines():
        if line.strip():
            row = json.loads(line)
            answers = [str(a) for a in row["gold_answers"] if str(a).strip()]
            if not answers:
                raise ValueError(f"Empty gold for {row['sample_id']}")
            golds[str(row["sample_id"])] = answers
    missing = [sid for sid in ordered if sid not in golds]
    if missing:
        raise ValueError(f"Gold missing for {len(missing)} IDs, e.g. {missing[:5]}")
    inputs = {str(r["sample_id"]): r for r in (json.loads(l) for l in inputs_raw.decode().splitlines() if l.strip())}
    records = [c1_record(inputs[sid], golds[sid]) for sid in ordered]
    prepared = prepare_records(records, DatasetName.NQ)   # validates schema, scorer and IDs
    raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()
    write_frozen(args.out, raw)
    stats = Counter("initial_correct" if r["correct"] else "initial_wrong" for r in records)
    print(json.dumps({"out": str(args.out), "input_sha256": sha256(raw), "samples": len(prepared), **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
