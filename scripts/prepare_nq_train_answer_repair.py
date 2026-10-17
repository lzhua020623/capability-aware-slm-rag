"""Verify all NQ Train shards, adapt C1, and freeze a development-only review packet.

No inference or diagnosis labels. Hash partitioning precedes metric screening.
Calibration/evaluation reservations are E-only, pending team-wide coordination.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_answer_repair_experiment import encode_json, write_frozen
from scripts.run_answer_repair import prepare_records
from src.compensation.nq_train_adapter import c1_record, offline_reference, runtime_record, validate_generation_source
from src.diagnosis import DatasetName

VERSION = "answer_repair_nq_train_dev_v1"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def location(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def query_key(query: str) -> str:
    return " ".join(query.strip().lower().split())


def partition(row: dict, seed: int, exposed: set[str], reserved: set[str]) -> str:
    sid = str(row["sample_id"])
    if sid in reserved:
        return "excluded_by_team"
    # A previously previewed question must never be called unseen.
    if query_key(row["query"]) in exposed:
        return "development"
    value = int(sha(f"E-NQ-Train-partition-v1:{seed}:{query_key(row['query'])}".encode()), 16) % 100
    return "development" if value < 80 else "calibration" if value < 90 else "reserved_evaluation"


def offer(pool: list, limit: int, row: dict, c1: dict, seed: int, group: str) -> None:
    if not limit:
        return
    sid = str(row["sample_id"])
    key = sha(f"E-NQ-Train-review-v1:{seed}:{group}:{sid}".encode())
    pool.append((key, sid, row, c1))
    pool.sort(key=lambda x: x[:2])
    del pool[limit:]


def jsonl(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n").encode() for r in rows)


def check_output_paths(source_dir: Path, inputs: list[Path], manifest_out: Path,
                       out_dir: Path, full_out: Path) -> None:
    """Refuse aliases and output locations that could replace delivery inputs."""
    outputs = [manifest_out, full_out, full_out.parent / "partition_ids.json",
               *[out_dir / name for name in (
                   "preparation.json", "inputs/nq_train_review_c1.jsonl",
                   "inputs/nq_train_review_runtime.jsonl", "review/review_queue.jsonl",
                   "review/review_queue.txt")]]
    resolved = [p.resolve() for p in outputs]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Output paths must be distinct")
    protected = {p.resolve() for p in inputs}
    source_root = source_dir.resolve()
    if any(p in protected or p.is_relative_to(source_root) for p in resolved):
        raise ValueError("Output paths must not replace source inputs or enter the source directory")


def prepare(*, source_dir: Path, delivery_manifest: Path, stats_path: Path, audit_path: Path,
            manifest_out: Path, out_dir: Path, full_out: Path, seed: int = 42,
            screen_count: int = 80, control_count: int = 20, reserved_ids: Path | None = None) -> dict:
    if min(screen_count, control_count) < 0 or screen_count + control_count < 1:
        raise ValueError("Review counts must be nonnegative with a positive total")
    source_raw, stats_raw, audit_raw = (p.read_bytes() for p in (delivery_manifest, stats_path, audit_path))
    delivery, stats, audit = map(json.loads, (source_raw, stats_raw, audit_raw))
    validate_generation_source(stats)
    shards = delivery["shards"]
    names = [s["file"] for s in shards]
    if (len(names) != len(set(names)) or any(Path(n).name != n or not n.endswith(".jsonl") for n in names)
            or len(shards) != delivery["shard_count"]
            or sum(s["n_samples"] for s in shards) != delivery["total_samples"]):
        raise ValueError("Invalid delivery manifest")
    inputs = [delivery_manifest, stats_path, audit_path, *[source_dir / name for name in names]]
    if reserved_ids is not None:
        inputs.append(reserved_ids)
    check_output_paths(source_dir, inputs, manifest_out, out_dir, full_out)
    if (audit.get("repository_manifest_sha256") != sha(source_raw)
            or audit.get("total_samples") != delivery["total_samples"]
            or audit.get("schema_issue_counts") or not audit.get("all_shard_hashes_match")):
        raise ValueError("Audit does not attest this exact delivery")
    audited_shards = {r["file"]: r["sha256"] for r in audit["shards"]}
    if audited_shards != {s["file"]: s["sha256"] for s in shards}:
        raise ValueError("Audit and delivery shard hashes disagree")
    if stats.get("annotation_candidate_count") != delivery["total_samples"]:
        raise ValueError("Source generation count disagrees with delivery")
    reserved_raw = reserved_ids.read_bytes() if reserved_ids else b""
    reserved_list = json.loads(reserved_raw) if reserved_raw else []
    if not isinstance(reserved_list, list) or any(not isinstance(s, str) for s in reserved_list):
        raise ValueError("Reserved IDs must be a JSON list of strings")
    excluded = set(reserved_list)
    for old in audit["overlap_checks"].values():
        excluded.update(old["id_overlap_new_ids"])
        excluded.update(old["query_overlap_new_ids"])
    previewed = set(audit.get("semantically_previewed_sample_ids", []))
    # First pass is integrity-only, before exporting any new final artifact.
    exposed_queries = set()
    for shard in shards:
        raw = (source_dir / shard["file"]).read_bytes()
        if sha(raw) != shard["sha256"]:
            raise ValueError(f"Source checksum mismatch: {shard['file']}")
        for line in raw.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if str(r["sample_id"]) in previewed:
                exposed_queries.add(query_key(r["query"]))

    groups = {"screen_candidate": [], "em_match_control": []}
    available = Counter(); roles = {k: [] for k in ("development", "calibration", "reserved_evaluation", "excluded_by_team")}
    seen = set(); query_roles = {}; labels = Counter(); full_digest = hashlib.sha256()
    full_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=full_out.parent, delete=False) as handle:
            temporary = Path(handle.name)
            for shard in shards:
                raw = (source_dir / shard["file"]).read_bytes()
                if sha(raw) != shard["sha256"]:
                    raise ValueError(f"Source changed during preparation: {shard['file']}")
                count = 0; first = last = None
                for line in raw.splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line); sid = str(row["sample_id"]); count += 1
                    first = first or sid; last = sid
                    if sid in seen:
                        raise ValueError(f"Duplicate sample ID: {sid}")
                    seen.add(sid)
                    if row.get("label") is not None:
                        raise ValueError("Use the original unlabeled delivery, not returned annotations")
                    labels["null"] += 1
                    role = partition(row, seed, exposed_queries, excluded)
                    q = query_key(row["query"])
                    if q in query_roles and query_roles[q] != role:
                        raise ValueError("Same normalized query crosses partitions; reserve its full group")
                    query_roles[q] = role; roles[role].append(sid)
                    record = c1_record(row, stats, source_shard=shard["file"],
                                       source_sha256=shard["sha256"], data_role=role)
                    encoded = jsonl([record]); handle.write(encoded); full_digest.update(encoded)
                    _, gold_hit = offline_reference(row)
                    if role == "development":
                        group = "em_match_control" if record["correct"] else "screen_candidate" if gold_hit else None
                        if group:
                            available[group] += 1
                            offer(groups[group], control_count if group == "em_match_control" else screen_count,
                                  row, record, seed, group)
                if count != shard["n_samples"] or first != shard["first_sample_id"] or last != shard["last_sample_id"]:
                    raise ValueError(f"Shard count or ID range mismatch: {shard['file']}")
        if len(seen) != delivery["total_samples"]:
            raise ValueError("Unique sample count disagrees with manifest")
        for name, wanted in (("screen_candidate", screen_count), ("em_match_control", control_count)):
            if len(groups[name]) != wanted:
                raise ValueError(f"Only {len(groups[name])} eligible {name}; requested {wanted}")
        # Interleave four screened rows and one control rather than hide controls at the end.
        ordered = []; candidates = list(groups["screen_candidate"]); controls = list(groups["em_match_control"])
        while candidates or controls:
            ordered.extend(("screen_candidate", item) for item in candidates[:4]); del candidates[:4]
            if controls:
                ordered.append(("em_match_control", controls.pop(0)))
        scored = [item[3] for _, item in ordered]
        prepare_records(scored, DatasetName.NQ)
        pure = [runtime_record(item[2]) for _, item in ordered]
        reviews = []
        for number, (group, item) in enumerate(ordered, 1):
            row, record = item[2:]
            reviews.append({"review_number": number, "review_set_id": VERSION,
                "dataset": "nq", "sample_id": record["sample_id"], "screening_group": group,
                "query": row["query"], "initial_answer": row["initial_answer"], "evidence": row["evidence"],
                "annotation_reference": row["annotation_reference"], "frozen_em": record["em"],
                "frozen_f1": record["f1"], "human_verified": False,
                "human_review": {"status": "pending", "reviewer": "", "initial_correct": None,
                    "evidence_sufficient": None, "evidence_accessible": None, "suggested_diagnosis_label": None,
                    "reference_issue": None, "reason": ""}, "source": record["provenance"]})
        manifest = {"version": VERSION, "seed": seed, "dataset": "nq", "source_total": len(seen),
            "sources_sha256": {"delivery_manifest": sha(source_raw), "source_stats": sha(stats_raw),
                               "drive_audit": sha(audit_raw), "team_reserved_ids": sha(reserved_raw) if reserved_raw else None},
            "source_shards": [{"file": s["file"], "sha256": s["sha256"]} for s in shards],
            "partition_policy": {"group_key": "lowercase/collapsed-whitespace query", "key": "SHA256(E-NQ-Train-partition-v1:seed:query) mod 100",
                "buckets": {"development": [0, 79], "calibration": [80, 89], "reserved_evaluation": [90, 99]},
                "previously_previewed_ids_forced_to_development": sorted(previewed),
                "excluded_old_overlap_and_team_ids": sorted(excluded),
                "scope": "E-only; team-wide unseen status and Wikipedia page grouping are NOT established"},
            "partition_counts": {k: len(v) for k,v in roles.items()},
            "review_selection": "development only; seeded SHA256 per screening group; 4 candidates then 1 control",
            "requested_review_counts": {"screen_candidate": screen_count, "em_match_control": control_count},
            "available_development_screening_counts": dict(available),
            "sample_ids": [r["sample_id"] for r in scored],
            "sample_ids_by_screening_group": {k: [item[1] for item in v] for k,v in groups.items()},
            "human_labels_assigned": False,
            "screening_policy": "EM=0 plus gold passage hit is not utilisation; EM=1 is not human-confirmed correct",
            "source_c1_model_revision": None}
        files = {"inputs/nq_train_review_c1.jsonl": jsonl(scored),
                 "inputs/nq_train_review_runtime.jsonl": jsonl(pure),
                 "review/review_queue.jsonl": jsonl(reviews)}
        lines = ["NQ Train E 首批开发审核", "筛查分组不是诊断标签；所有人工字段待审。", ""]
        for r in reviews:
            lines.extend([f"{r['review_number']}. {r['sample_id']} [{r['screening_group']}]", r['query'],
                          "C1: " + r['initial_answer'], "人工判断：待审", ""])
        files["review/review_queue.txt"] = ("\n".join(lines).rstrip()+"\n").encode()
        summary = {"version": VERSION, "source_samples": len(seen), "source_label_counts": dict(labels),
            "review_samples": len(scored), "review_human_status": "all_pending",
            "partition_counts": manifest["partition_counts"], "full_scored_c1": location(full_out),
            "full_scored_c1_sha256": full_digest.hexdigest(), "full_scored_c1_bytes": temporary.stat().st_size,
            "partition_ids_local": location(full_out.parent / "partition_ids.json"),
            "outputs_sha256": {p: sha(raw) for p,raw in files.items()},
            "manifest": location(manifest_out), "manifest_sha256": sha(encode_json(manifest)),
            "gpu_inference_run": False, "runtime_gold_fields": False, "v8_algorithm_implemented": False}
        outputs = {manifest_out: encode_json(manifest),
                   full_out.parent / "partition_ids.json": encode_json({"version": VERSION, "partitions": roles}),
                   out_dir / "preparation.json": encode_json(summary),
                   **{out_dir / p: raw for p,raw in files.items()}}
        # Check all existing frozen outputs before publishing any result.
        if full_out.exists() and file_sha(full_out) != full_digest.hexdigest():
            raise ValueError("Frozen full C1 differs; use a new output version")
        for path, raw in outputs.items():
            if path.exists() and path.read_bytes() != raw:
                raise ValueError(f"Frozen artifact differs: {path}; use a new version")
        if full_out.exists():
            temporary.unlink()
        else:
            temporary.replace(full_out)
        temporary = None
        for path,raw in outputs.items():
            write_frozen(path, raw)
        return summary
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/processed/external/nq_train_delivery_20261017")
    parser.add_argument("--delivery-manifest", type=Path, default=ROOT / "data/annotations/nq_train_delivery/manifest.json")
    parser.add_argument("--source-stats", type=Path, default=ROOT / "data/annotations/nq_train_annotation_candidates_stats.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "data/annotations/nq_train_delivery/drive_audit_20261017.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/splits/answer_repair_nq_train_dev_v1.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/answer_repair/nq_train_dev_v1")
    parser.add_argument("--full-out", type=Path, default=ROOT / "data/processed/answer_repair/nq_train_v1/c1_scored.jsonl")
    parser.add_argument("--reserved-ids", type=Path, help="Optional JSON list of team-reserved sample IDs to exclude")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--screen-count", type=int, default=80)
    parser.add_argument("--control-count", type=int, default=20)
    args = parser.parse_args(argv)
    result = prepare(source_dir=args.source_dir, delivery_manifest=args.delivery_manifest,
        stats_path=args.source_stats, audit_path=args.audit, manifest_out=args.manifest,
        out_dir=args.out_dir, full_out=args.full_out, seed=args.seed,
        screen_count=args.screen_count, control_count=args.control_count, reserved_ids=args.reserved_ids)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
