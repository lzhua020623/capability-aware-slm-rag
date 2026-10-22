"""Freeze a development-only, label-stratified audit without selecting by EM/gold."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.import_nq_train_preliminary import jsonl, sha, write_new
from scripts.build_nq_train_pending_review import render
from src.compensation.nq_train_adapter import runtime_record

VERSION = "nq_train_label_audit_200_v1"
LABELS = ("correct", "evidence_accessibility", "evidence_insufficiency", "evidence_utilisation")
HUMAN_FIELDS = {"status": "pending", "initial_correct": None,
                "answer_supported_by_top5": None, "evidence_sufficient": None,
                "evidence_accessible": None, "primary_label": None,
                "supporting_ranks": [], "supporting_quote": "", "reference_issue": None, "reason": ""}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def query_key(value):
    return " ".join(value.lower().split())


def select(labels, partitions, excluded, count=50, seed=42):
    """Read only ID, existing role and preliminary stratum; never answer metrics."""
    pools = {key: [] for key in LABELS}
    seen = set()
    for row in labels:
        sid, label = row["sample_id"], row["label"]
        if sid in seen or sid not in partitions or label not in LABELS:
            raise ValueError("Duplicate/unknown ID or label")
        seen.add(sid)
        if row["data_role"] != partitions[sid]:
            raise ValueError("Overlay role differs from frozen partition")
        if partitions[sid] == "development" and sid not in excluded:
            order = digest(f"E-label-audit-v1:{seed}:{label}:{sid}".encode())
            pools[label].append((order, sid))
    if seen != set(partitions):
        raise ValueError("Overlay does not cover exact frozen partition")
    if count < 1 or any(len(pool) < count for pool in pools.values()):
        raise ValueError("Insufficient rows for requested strata")
    chosen = {sid: key for key, pool in pools.items() for _, sid in sorted(pool)[:count]}
    order = sorted(chosen, key=lambda sid: (digest(f"E-label-audit-display-v1:{seed}:{sid}".encode()), sid))
    return order, chosen, {key: len(pool) for key, pool in pools.items()}


def build_page(packet):
    # Reuse the existing offline review UI, in a new versioned template/script.
    base = render(json.loads(json.dumps(packet, ensure_ascii=False, sort_keys=True)))
    old_js = (ROOT / "scripts/templates/nq_train_pending_review.js").read_text()
    new_js = (ROOT / "scripts/templates/nq_label_audit.js").read_text()
    base = base.replace(old_js, new_js)
    base = base.replace("nq-train-pending-v1", "nq-label-audit-v1")
    base = base.replace("E · NQ Train 待审核", "E · NQ 初标分层审核 200")
    base = base.replace("原编号，如 N63", "本批编号，如 A12")
    base = base.replace("问题、N 编号", "问题、A 编号")
    base = base.replace("展开 Codex 辅助判断 · 供你核对", "展开辅助判断 · 未审核处保持留空")
    base = base.replace('<label for="label">主标签</label>',
        '<label for="supported">Top-5 是否支持 C1 原答（与正确性分开）</label>'
        '<select id="supported"><option value="">尚未判断</option><option value="true">支持</option>'
        '<option value="false">不支持</option><option value="uncertain">无法确定</option></select>'
        '<label for="supportQuote">支持位置的原文摘录（可选）</label>'
        '<textarea id="supportQuote" placeholder="保留原文；不能只凭词面重合判支持"></textarea>'
        '<label for="label">主标签</label>')
    return base.encode()


def prepare(*, source_dir, overlay, import_audit, delivery_manifest, partition_path,
            development_manifest, challenge_manifest, out_dir, count=50, seed=42):
    paths = [overlay, import_audit, delivery_manifest, partition_path, development_manifest, challenge_manifest]
    audit = json.loads(import_audit.read_bytes())
    if not audit["ok"] or sha(overlay) != audit["overlay"]["sha256"]:
        raise ValueError("Overlay checksum does not match import audit")
    if sha(partition_path) != audit["inputs_sha256"]["partition_ids"]:
        raise ValueError("Partition checksum does not match import audit")
    delivery = json.loads(delivery_manifest.read_bytes())
    if sha(delivery_manifest) != audit["inputs_sha256"]["delivery_manifest"]:
        raise ValueError("Delivery checksum does not match import audit")
    roles = {}
    for role, ids in json.loads(partition_path.read_bytes())["partitions"].items():
        for sid in ids:
            if sid in roles:
                raise ValueError("ID crosses frozen roles")
            roles[sid] = role
    dev, challenge = (json.loads(p.read_bytes()) for p in (development_manifest, challenge_manifest))
    exposed = set(dev["sample_ids"]) | set(challenge["sample_ids"])
    exposed.update(dev["partition_policy"]["previously_previewed_ids_forced_to_development"])
    # Existing normalized question grouping has no duplicates, but explicitly
    # exclude all siblings of known exposures if future deliveries add any.
    exposed_queries, source_checks = set(), {}
    for shard in delivery["shards"]:
        source = source_dir / shard["file"]
        if sha(source) != shard["sha256"]:
            raise ValueError("Original shard checksum mismatch")
        source_checks[shard["file"]] = shard["sha256"]
        for row in jsonl(source):
            if row["sample_id"] in exposed:
                exposed_queries.add(query_key(row["query"]))
    originals, query_excluded = {}, set()
    for shard in delivery["shards"]:
        for row in jsonl(source_dir / shard["file"]):
            sid = row["sample_id"]
            if query_key(row["query"]) in exposed_queries:
                query_excluded.add(sid)
    excluded = exposed | query_excluded
    order, strata, available = select(jsonl(overlay), roles, excluded, count=count, seed=seed)
    for shard in delivery["shards"]:
        for row in jsonl(source_dir / shard["file"]):
            if row["sample_id"] in strata:
                originals[row["sample_id"]] = row
    if set(originals) != set(order):
        raise ValueError("Selected IDs missing from original delivery")
    label_sources = {row["sample_id"]: row for row in jsonl(overlay) if row["sample_id"] in strata}
    records, runtime = [], []
    for number, sid in enumerate(order, 1):
        original = originals[sid]
        pure = runtime_record(original)
        runtime.append(pure)
        record = {"review_id": f"{VERSION}:nq:{sid}", "review_number": number,
            "review_set_id": VERSION, **pure, "offline_reference": original["annotation_reference"],
            "source_label": strata[sid], "screening_group": "preliminary_stratum_" + strata[sid],
            "human_verified": False, "human_review": dict(HUMAN_FIELDS),
            "source_record_sha256": digest(json.dumps(original, sort_keys=True, ensure_ascii=False).encode()),
            "preliminary_source": label_sources[sid]["source"],
            "agent_assessment": {"suggested_diagnosis_label": None, "confidence": "未审核",
                                 "reason_zh": "尚未做独立语义辅助审核；原初标仅用于分层抽样。",
                                 "quoted_evidence": []}, "frozen_em": None, "frozen_f1": None}
        records.append(record)
    manifest = {"version": VERSION, "seed": seed, "samples": len(order),
        "stratum_counts": dict(Counter(strata.values())), "available_development_by_preliminary_label": available,
        "sample_ids": order, "excluded_known_exposed_ids": sorted(excluded),
        "exposure_sources": [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
                             for p in (development_manifest, challenge_manifest)],
        "selection_fields": ["sample_id", "frozen_data_role", "preliminary_label"],
        "selection_uses_gold_em_or_answer": False, "all_development": True,
        "newly_exposed_for": "label_quality_audit_and_development_only_not_final_test",
        "population_error_rate_estimator": False, "new_human_labels_assigned": 0,
        "team_training_and_page_semantic_dedup_confirmed": False,
        "source_files_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in paths},
        "source_shards_sha256": source_checks}
    packet = {"schema_version": "nq_label_audit_packet_v1", "packet_id": VERSION, "review_set_id": VERSION,
        "source_main_commit": "058f10baa47083220f95edb39db8f106aa7e85d6", "source_total": len(roles),
        "already_confirmed_count": 0, "sources_sha256": manifest["source_files_sha256"],
        "counts": {"pending": len(records), "disputed": len(records), "suggested_correct": 0}, "records": records}
    packet["packet_sha256"] = digest(json.dumps(packet, ensure_ascii=False, sort_keys=True).encode())
    files = {"manifest.json": encoded(manifest), "review_packet.json": encoded(packet),
             "review_queue.jsonl": b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in records),
             "runtime.jsonl": b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in runtime),
             "A200_review.html": build_page(packet)}
    source_paths = {p.resolve() for p in paths}
    for name, raw in files.items():
        path = out_dir / name
        if path.resolve() in source_paths or path.resolve().is_relative_to(source_dir.resolve()):
            raise ValueError("Output would replace a source")
        if path.exists() and path.read_bytes() != raw:
            raise FileExistsError(f"Frozen output differs: {path}")
    for name, raw in files.items():
        write_new(out_dir / name, raw)
    return {"samples": len(order), "stratum_counts": manifest["stratum_counts"],
            "known_exposures_excluded": len(excluded), "human_confirmed": 0,
            "outputs_sha256": {name: digest(raw) for name, raw in files.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    defaults = {
        "source_dir": "data/processed/external/nq_train_delivery_20261017",
        "overlay": "data/processed/answer_repair/nq_train_preliminary_20261022/labels.jsonl",
        "import_audit": "data/annotations/nq_train_preliminary/import_audit_20261022.json",
        "delivery_manifest": "data/annotations/nq_train_delivery/manifest.json",
        "partition_path": "data/processed/answer_repair/nq_train_v1/partition_ids.json",
        "development_manifest": "configs/splits/answer_repair_nq_train_dev_v1.json",
        "challenge_manifest": "artifacts/answer_repair/unseen_nq125/manifest.json",
        "out_dir": "artifacts/answer_repair/nq_label_audit_200_v1"}
    for name, value in defaults.items():
        p.add_argument("--" + name.replace("_", "-"), type=Path, default=ROOT / value)
    p.add_argument("--count", type=int, default=50); p.add_argument("--seed", type=int, default=42)
    print(json.dumps(prepare(**vars(p.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
