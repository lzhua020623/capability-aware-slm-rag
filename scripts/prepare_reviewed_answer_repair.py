"""Freeze the chat-reviewed training cohort and a portable v5 development bundle."""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_answer_repair_experiment import (
    SPLIT_PATHS, encode_json, git_read, select_c1, sha256, write_frozen,
)
from scripts.prepare_answer_repair_review import shared_training
from scripts.run_answer_repair import read_jsonl, prepare_records
from src.diagnosis import DatasetName

PROTOCOL = [
    {"strategy": "grounded_reanswer", "max_new_tokens": 384, "max_calls": 1},
    {"strategy": "query_grounded", "max_new_tokens": 384, "max_calls": 1},
    {"strategy": "guarded_reanswer", "max_new_tokens": 384, "support_check_tokens": 192, "max_calls": 2},
    {"strategy": "reanswer", "max_new_tokens": 576, "max_calls": 1},
]


def build_cohort(packet: dict, reviews: list[dict], splits: list[dict], sources: dict[str, bytes]):
    unsigned = {k: v for k, v in packet.items() if k != "packet_sha256"}
    if sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True).encode()) != packet["packet_sha256"]:
        raise ValueError("Packet content hash mismatch")
    packet_by_id = {r["review_id"]: r for r in packet["records"]}
    human = {r["review_id"]: r for r in reviews}
    if (len(packet_by_id) != len(packet["records"]) or len(human) != len(reviews)
            or human.keys() != packet_by_id.keys()):
        raise ValueError("Review IDs must cover the packet exactly once")
    items, excluded, inputs = [], [], {}
    for index, p in enumerate(packet["records"], 1):
        row = human[p["review_id"]]
        h = row["human_review"]
        if (row["packet_id"] != packet["packet_id"] or row["packet_sha256"] != packet["packet_sha256"]
                or row["review_number"] != index or row["dataset"] != p["dataset"]
                or row["sample_id"] != p["sample_id"] or h["status"] != "completed"
                or not h["reviewer"].strip() or not h["reason"].strip() or not row.get("confirmation")):
            raise ValueError("Unconfirmed or mismatched human review")
        use = h["use"]
        if use not in ("include_e_core", "correct_control", "boundary_case", "hold"):
            raise ValueError("Unknown reviewed use")
        if use == "include_e_core" and (h["initial_answer_status"], h["evidence_status"], h["primary_label"]) != (
                "incorrect", "sufficient_clear", "evidence_utilisation"):
            raise ValueError("Core inclusion must confirm incorrect / clear / utilisation")
        if use == "correct_control" and (h["initial_answer_status"], h["primary_label"]) != ("correct", "correct"):
            raise ValueError("Correct control requires confirmed correct answer")
        item = {"review_id": p["review_id"], "sample_id": p["sample_id"], "dataset": p["dataset"],
                "review_number": index, "use": use, "previous_E_dev50": p["previous_E_dev50"],
                "human_initial_status": h["initial_answer_status"], "reason": h["reason"]}
        (items if use in ("include_e_core", "correct_control") else excluded).append(item)
    datasets = {}
    for ds in ("fever", "nq"):
        eligible, reserved = shared_training(splits, ds)
        all_packet_ids = {p["sample_id"] for p in packet["records"] if p["dataset"] == ds}
        if not all_packet_ids <= eligible or all_packet_ids & reserved:
            raise ValueError("Reviewed cohort overlaps held-out data or is not shared training")
        c1 = {r.sample_id: (s, r) for s, r in prepare_records(read_jsonl(sources[ds], ds), DatasetName(ds))}
        for p in packet["records"]:
            if p["dataset"] != ds:
                continue
            source, request = c1[p["sample_id"]]
            archived_evidence = [{"rank": i + 1, **e} for i, e in enumerate(source["evidence"])]
            if (p["query"] != request.query or p["initial_answer"] != request.initial_answer
                    or p["evidence"] != archived_evidence
                    or p["offline_reference"]["gold_answer_or_label"] != source["gold_answer_or_label"]):
                raise ValueError("Packet differs from frozen C1")
        selected = [i for i in items if i["dataset"] == ds]
        ids = [i["sample_id"] for i in selected]
        raw, stats = select_c1(sources[ds], ds, ids)
        inputs[f"inputs/{ds}_reviewed_c1.jsonl"] = raw
        disagreement = [i["sample_id"] for i in selected if
                        (i["human_initial_status"] == "correct") != c1[i["sample_id"]][0]["correct"]]
        datasets[ds] = {"sample_ids": ids, "groups": dict(Counter(i["use"] for i in selected)),
                        "frozen_metric_baseline": stats, "human_vs_frozen_metric_disagreement_ids": disagreement,
                        "previous_dev50_count": sum(i["previous_E_dev50"] for i in selected),
                        "reserved_overlap": 0, "source_sha256": sha256(sources[ds]),
                        "input_sha256": sha256(raw)}
    manifest = {"version": "answer_repair_reviewed_dev_v1", "review_date": "2026-10-14",
                "purpose": "label-selected, chat-assisted training development; not held-out or blind annotation",
                "packet_id": packet["packet_id"], "packet_sha256": packet["packet_sha256"],
                "source_commits": packet["source_commits"], "datasets": datasets,
                "included": items, "deferred": excluded, "protocol": PROTOCOL,
                "metric_policy": "Unmodified frozen EM/F1 and FEVER accuracy; separate paired semantic review of BOTH initial/final answers; no automatic semantic truth from citations or same-model check."}
    return manifest, inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=ROOT / "artifacts/answer_repair/review_20261013/review_packet.json")
    parser.add_argument("--reviews", type=Path, default=ROOT / "data/annotations/human/answer_repair_review_20261014.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/splits/answer_repair_reviewed_dev_v1.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/answer_repair/v5_reviewed")
    parser.add_argument("--bundle-name", default="answer_repair_v5_reviewed_gpu")
    parser.add_argument("--experiment", choices=("v5", "v6", "v7"), default="v5")
    args = parser.parse_args(argv)
    if not args.bundle_name.replace("_", "").isalnum():
        parser.error("Bundle name must contain letters, numbers and underscores only")
    packet = json.loads(args.packet.read_bytes())
    reviews_raw = args.reviews.read_bytes()
    reviews = read_jsonl(reviews_raw, str(args.reviews))
    commit = packet["source_commits"]["annotation_and_split"]
    split_raw = {p: git_read(commit, p) for p in SPLIT_PATHS}
    sources = {ds: git_read(packet["source_commits"]["c1"], f"results/preliminary/{ds}_c1_7b.jsonl")
               for ds in ("fever", "nq")}
    for path, raw in split_raw.items():
        if sha256(raw) != packet["source_sha256"][f"{commit}:{path}"]:
            raise ValueError("Split provenance mismatch")
    manifest, inputs = build_cohort(packet, reviews, [json.loads(r) for r in split_raw.values()], sources)
    manifest.update(human_review_sha256=sha256(reviews_raw),
                    split_sources={p: sha256(r) for p, r in split_raw.items()})
    if args.experiment in ("v6", "v7"):
        protocol = {"v6": [{"dataset": "fever", "strategy": "fever_nli", "max_new_tokens": 1, "max_calls": 0},
                           {"dataset": "nq", "strategy": "extractive_reanswer", "max_new_tokens": 384, "max_calls": 1}],
                    "v7": [{"dataset": "fever", "strategy": "fever_nli_anchored", "max_new_tokens": 1, "max_calls": 0},
                           {"dataset": "nq", "strategy": "short_answer_quote", "max_new_tokens": 384, "max_calls": 1}]}
        policy = {"v6": "One v6 follow-up; do not tune on validation/test; semantic review deferred by user",
                  "v7": "v7 frozen before the GPU run (docs/ANSWER_REPAIR_V7.md); reviewed and dev50 are paired with v6r1; heldout is run once and reported regardless; no threshold or prompt tuning on any cohort"}
        manifest.update(version=f"answer_repair_reviewed_{args.experiment}", parent_cohort="answer_repair_reviewed_dev_v1",
                        protocol=protocol[args.experiment],
                        classifier_policy="MNLI-only documented fine-tuning; FP32; threshold .90 fixed before this run; full overlapping windows; costs separate from Qwen"
                        + ("; v7: only windows containing the claim's subject chunk may vote" if args.experiment == "v7" else ""),
                        development_policy=policy[args.experiment])
        old = json.loads((ROOT / "configs/splits/answer_repair_v1_dev.json").read_bytes())
        regression = {}
        for ds in ("fever", "nq"):
            eligible, reserved = shared_training([json.loads(r) for r in split_raw.values()], ds)
            ids = old["datasets"][ds]["sample_ids"]
            if not set(ids) <= eligible or set(ids) & reserved:
                raise ValueError("Regression cohort is not isolated training development data")
            raw, stats = select_c1(sources[ds], ds, ids)
            inputs[f"inputs/{ds}_dev50_c1.jsonl"] = raw
            regression[ds] = {"sample_ids": ids, "input_sha256": sha256(raw), "baseline": stats,
                              "overlap_with_reviewed": sorted(set(ids) & set(manifest["datasets"][ds]["sample_ids"]))}
        manifest["regression_dev50"] = regression
    if args.experiment == "v7":
        # Every shared-training ID E has never touched: no labels, correctness or references are read.
        used_sha = {"answer_repair_v1_dev": sha256((ROOT / "configs/splits/answer_repair_v1_dev.json").read_bytes()),
                    "review_packet": sha256(args.packet.read_bytes())}
        heldout = {}
        for ds in ("fever", "nq"):
            eligible, reserved = shared_training([json.loads(r) for r in split_raw.values()], ds)
            used = set(old["datasets"][ds]["sample_ids"]) | {p["sample_id"] for p in packet["records"] if p["dataset"] == ds}
            ids = sorted(eligible - used, key=lambda sid: (sha256(f"heldout:42:{ds}:{sid}".encode()), sid))
            if set(ids) & reserved or set(ids) & used:
                raise ValueError("Held-out cohort overlaps reserved or previously used IDs")
            raw, stats = select_c1(sources[ds], ds, ids)
            inputs[f"inputs/{ds}_heldout_c1.jsonl"] = raw
            heldout[ds] = {"sample_ids": ids, "input_sha256": sha256(raw), "baseline": stats,
                           "excluded_used_ids": len(used & eligible), "eligible_shared_training": len(eligible)}
        manifest["heldout_prelim"] = {
            "purpose": "E-untouched shared-training IDs; run once, report regardless; harm/abstention evidence, not a repair-rate estimate",
            "selection": "all shared-training IDs minus answer_repair_v1_dev dev50 and the 37-sample review packet; ordered by SHA256(heldout:42:dataset:sample_id)",
            "exclusion_sources_sha256": used_sha, "datasets": heldout}
    write_frozen(args.manifest, encode_json(manifest))
    files = {**inputs, "manifest.json": encode_json(manifest), "offline/human_reviews.jsonl": reviews_raw,
             "offline/review_packet.json": args.packet.read_bytes()}
    source_files = [p for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts]
    source_files += [p for p in (ROOT / "tests").glob("*.py")]
    source_files += [ROOT / p for p in (
        "scripts/run_answer_repair.py", "scripts/check_answer_repair_environment.py",
        "scripts/prepare_answer_repair_experiment.py", "scripts/prepare_answer_repair_review.py",
        "scripts/prepare_reviewed_answer_repair.py", "scripts/summarize_reviewed_answer_repair.py",
        "configs/base.yaml", "requirements-answer-repair.txt", "docs/ANSWER_REPAIR_V5.md",
    )]
    if args.experiment in ("v6", "v7"):
        source_files += [ROOT / "docs/ANSWER_REPAIR_V6.md", ROOT / "scripts/audit_answer_repair_release.py"]
    if args.experiment == "v7":
        source_files += [ROOT / "docs/ANSWER_REPAIR_V7.md", ROOT / "scripts/replay_fever_nli_anchored.py",
                         ROOT / "artifacts/answer_repair/v7_design/fever_anchored_replay.md"]
    files.update({p.relative_to(ROOT).as_posix(): p.read_bytes() for p in source_files})
    files["bundle_checksums.json"] = encode_json({p: sha256(r) for p, r in sorted(files.items())})
    staging = args.out_dir / "gpu_bundle"
    archive = args.out_dir / f"{args.bundle_name}.tar.gz"
    if archive.exists():
        with tarfile.open(archive) as handle:
            if handle.extractfile(f"{args.bundle_name}/bundle_checksums.json").read() != files["bundle_checksums.json"]:
                raise ValueError("Frozen bundle differs; choose a new output directory and bundle name")
    for path, raw in files.items():
        write_frozen(staging / path, raw)
    if not archive.exists():
        temporary = archive.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz") as handle:
            for path in sorted(files):
                handle.add(staging / path, arcname=f"{args.bundle_name}/{path}", recursive=False)
        temporary.replace(archive)
    summary = {"bundle": str(archive), "bundle_sha256": sha256(archive.read_bytes()),
               "datasets": manifest["datasets"], "real_gpu_experiment_completed": False}
    (args.out_dir / "preparation.json").write_bytes(encode_json(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
