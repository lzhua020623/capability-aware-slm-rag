"""Freeze the one-time E-only calibration challenge without previewing its answers.

Requires the selected final NQ75 archive. Reuses the original partition and
screening hashes; no new model run, diagnosis labels, or human sample selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_answer_repair_experiment import encode_json, write_frozen
from scripts.prepare_answer_repair_final_dev import SELECTED_SHA, payload
from scripts.prepare_nq_train_answer_repair import jsonl, offer, partition, query_key, sha
from scripts.run_answer_repair import prepare_records
from src.compensation.nq_train_adapter import c1_record, offline_reference, runtime_record, validate_generation_source
from src.diagnosis import DatasetName

VERSION = "answer_repair_nq_train_calibration_challenge125_v1"
COUNTS = {"screen_candidate": 100, "em_match_control": 25}


def frozen_selection(receipt_path: Path, archive: Path) -> dict:
    receipt = json.loads(receipt_path.read_bytes())
    revision = receipt.get("selected_final_revision")
    if (revision not in SELECTED_SHA or not receipt.get("frozen_inference_source_unchanged")
            or receipt.get("predictions_expected") != 300):
        raise ValueError("A selected frozen final NQ75 package is required")
    files = payload(archive, receipt["bundle_sha256"])
    manifest = json.loads(files["manifest.json"])
    expected_version = f"answer_repair_v8{revision}_nq75_final_v1"
    ids = manifest.get("sample_ids", [])
    if (manifest.get("version") != expected_version or receipt.get("version") != expected_version
            or len(ids) != 75 or len(set(ids)) != 75
            or manifest.get("selected_final_revision") != revision or manifest.get("samples") != 75
            or manifest.get("final_inference_bundle_sha256") != SELECTED_SHA[revision]
            or manifest.get("frozen_final") is not True
            or manifest.get("tuning_allowed_after_run") is not False):
        raise ValueError("Final NQ75 manifest does not attest a frozen r3/r4 selection")
    sources = manifest["runtime_source_sha256"]
    if (not sources or set(sources) != {p for p in files if p.startswith("src/")}
            or any(sha(files[p]) != digest for p, digest in sources.items())):
        raise ValueError("Final inference source fingerprint disagrees")
    return {"selected_final_revision": revision, "final_dev_bundle_sha256": receipt["bundle_sha256"],
            "final_inference_bundle_sha256": SELECTED_SHA[revision],
            "runtime_source_sha256": sources, "final_dev_ids": manifest["sample_ids"]}


def prepare(*, source_dir: Path, split_path: Path, stats_path: Path,
            final_receipt: Path, final_archive: Path, out_dir: Path,
            exposed_ids_path: Path | None = None) -> dict:
    split_raw, stats_raw = split_path.read_bytes(), stats_path.read_bytes()
    split, stats = json.loads(split_raw), json.loads(stats_raw)
    if split["version"] != "answer_repair_nq_train_dev_v1" or split["seed"] != 42:
        raise ValueError("Expected the original frozen NQ Train split and seed")
    if sha(stats_raw) != split["sources_sha256"]["source_stats"]:
        raise ValueError("Source generation statistics checksum mismatch")
    validate_generation_source(stats)
    selected = frozen_selection(final_receipt, final_archive)
    if not set(selected["final_dev_ids"]).issubset(split["sample_ids"]):
        raise ValueError("Final development IDs do not belong to the original reviewed cohort")
    extra_raw = exposed_ids_path.read_bytes() if exposed_ids_path else b"[]"
    extra = json.loads(extra_raw)
    if not isinstance(extra, list) or any(not isinstance(s, str) for s in extra):
        raise ValueError("Additional exposed IDs must be a JSON list of strings")
    policy = split["partition_policy"]
    previewed = set(policy["previously_previewed_ids_forced_to_development"])
    reserved = set(policy["excluded_old_overlap_and_team_ids"])
    seen_before = set(split["sample_ids"]) | previewed | set(selected["final_dev_ids"]) | set(extra)
    shards = split["source_shards"]
    names = [s["file"] for s in shards]
    if len(names) != len(set(names)) or any(Path(n).name != n or not n.endswith(".jsonl") for n in names):
        raise ValueError("Invalid source shard names")
    inputs = [split_path, stats_path, final_receipt, final_archive, *[source_dir / n for n in names]]
    if exposed_ids_path:
        inputs.append(exposed_ids_path)
    outputs = [out_dir / p for p in ("manifest.json", "preparation.json", "inputs/nq125_c1.jsonl",
                                     "inputs/nq125_runtime.jsonl", "offline/strata.jsonl")]
    if any(p.resolve().is_relative_to(source_dir.resolve()) or p.resolve() in
           {i.resolve() for i in inputs} for p in outputs):
        raise ValueError("Output paths must not replace or enter source inputs")
    # Verify every raw shard before deriving the original query-group partitions.
    exposed_queries = set(); seen_queries = set()
    for shard in shards:
        raw = (source_dir / shard["file"]).read_bytes()
        if sha(raw) != shard["sha256"]:
            raise ValueError("Source shard checksum mismatch: " + shard["file"])
        for line in raw.splitlines():
            if line.strip():
                row = json.loads(line)
                if str(row["sample_id"]) in previewed:
                    exposed_queries.add(query_key(row["query"]))
                if str(row["sample_id"]) in seen_before:
                    seen_queries.add(query_key(row["query"]))

    pools = {g: [] for g in COUNTS}; available = Counter(); roles = Counter()
    seen = set(); query_roles = {}; excluded_exposed = 0
    for shard in shards:
        raw = (source_dir / shard["file"]).read_bytes()
        if sha(raw) != shard["sha256"]:
            raise ValueError("Source shard changed during preparation")
        for line in raw.splitlines():
            if not line.strip():
                continue
            row = json.loads(line); sid = str(row["sample_id"])
            if sid in seen or row.get("label") is not None:
                raise ValueError("Expected unique IDs and the original unlabeled delivery")
            seen.add(sid)
            role = partition(row, 42, exposed_queries, reserved); roles[role] += 1
            q = query_key(row["query"])
            if q in query_roles and query_roles[q] != role:
                raise ValueError("A normalized question crosses partitions")
            query_roles[q] = role
            if role != "calibration":
                continue
            if sid in seen_before or q in seen_queries:
                excluded_exposed += 1
                continue
            scored = c1_record(row, stats, source_shard=shard["file"],
                               source_sha256=shard["sha256"], data_role="calibration")
            _, hit = offline_reference(row)
            group = "em_match_control" if scored["correct"] else "screen_candidate" if hit else None
            if group:
                available[group] += 1
                offer(pools[group], COUNTS[group], row, scored, 42, group)
    if len(seen) != split["source_total"] or {g: roles[g] for g in split["partition_counts"]} != split["partition_counts"]:
        raise ValueError("Original partition counts changed")
    if stats["annotation_candidate_count"] != len(seen):
        raise ValueError("Source C1 count changed")
    if any(len(pools[g]) != n for g, n in COUNTS.items()):
        raise ValueError("Insufficient untouched calibration candidates or controls")
    ordered = []; candidates = list(pools["screen_candidate"]); controls = list(pools["em_match_control"])
    while candidates or controls:
        ordered.extend(("screen_candidate", r) for r in candidates[:4]); del candidates[:4]
        if controls:
            ordered.append(("em_match_control", controls.pop(0)))
    scored = [item[3] for _, item in ordered]
    prepare_records(scored, DatasetName.NQ)
    runtime = [runtime_record(item[2]) for _, item in ordered]
    strata = [{"challenge_number": n, "sample_id": item[1], "screening_group": group,
               "human_verified": False, "diagnosis_label": None, "source_role": "calibration"}
              for n, (group, item) in enumerate(ordered, 1)]
    files = {"inputs/nq125_c1.jsonl": jsonl(scored), "inputs/nq125_runtime.jsonl": jsonl(runtime),
             "offline/strata.jsonl": jsonl(strata)}
    manifest = {"version": VERSION, "seed": 42, "samples": 125, "groups": COUNTS,
        "sample_ids": [r["sample_id"] for r in scored], "final_selection": selected,
        "source_split_sha256": sha(split_raw), "source_stats_sha256": sha(stats_raw),
        "source_shards": shards, "source_total": len(seen), "partition_counts": dict(roles),
        "available_calibration_screening_counts": dict(available),
        "previously_exposed_ids_sha256": sha(encode_json(sorted(seen_before))),
        "additional_exposed_ids_sha256": sha(extra_raw), "excluded_exposed_calibration_rows": excluded_exposed,
        "selection": "original SHA256(E-NQ-Train-review-v1:42:group:sample_id), then 4:1 interleave",
        "screening": "offline EM=0 plus reference passage in Top5; EM=1 controls; neither is a human diagnosis",
        "scope": "E-only stratified challenge; no claim of whole-system unseen or population NQ gain",
        "team_registration": "pending; isolate these IDs from future Diagnosis training before team-wide claims",
        "calibration_ids_consumed": True, "model_outputs_previewed": False,
        "human_selection_or_gold_editing": False, "post_result_tuning_allowed": False,
        "source_c1_revision": None, "gpu_inference_completed": False,
        "outputs_sha256": {p: sha(raw) for p, raw in files.items()}}
    receipt = {"version": VERSION, "samples": 125, "groups": COUNTS,
               "manifest_sha256": sha(encode_json(manifest)), "outputs_sha256": manifest["outputs_sha256"],
               "selected_final_revision": selected["selected_final_revision"],
               "available_calibration_screening_counts": dict(available), "gpu_inference_completed": False}
    files.update({"manifest.json": encode_json(manifest), "preparation.json": encode_json(receipt)})
    for name, raw in files.items():
        path = out_dir / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("Frozen challenge differs; do not resample after seeing results")
    for name, raw in files.items():
        write_frozen(out_dir / name, raw)
    # Deliberately omit question, answer, evidence and selected IDs from console output.
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, default=ROOT / "data/processed/external/nq_train_delivery_20261017")
    p.add_argument("--split", type=Path, default=ROOT / "configs/splits/answer_repair_nq_train_dev_v1.json")
    p.add_argument("--source-stats", type=Path, default=ROOT / "data/annotations/nq_train_annotation_candidates_stats.json")
    p.add_argument("--final-receipt", type=Path, default=ROOT / "artifacts/answer_repair/final_nq75/preparation.json")
    p.add_argument("--final-archive", type=Path, required=True)
    p.add_argument("--additional-exposed-ids", type=Path)
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/answer_repair/unseen_nq125")
    a = p.parse_args(argv)
    result = prepare(source_dir=a.source_dir, split_path=a.split, stats_path=a.source_stats,
                     final_receipt=a.final_receipt, final_archive=a.final_archive, out_dir=a.out_dir,
                     exposed_ids_path=a.additional_exposed_ids)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
