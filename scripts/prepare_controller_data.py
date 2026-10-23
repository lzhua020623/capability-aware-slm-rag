"""Freeze T2 cohorts without inference, scoring predictions or printing questions.

NQ exports paired C1 runtime/gold. FEVER exports separate query-only C1 tasks
and gold until its new C1 has actually been generated. Missing assets are
reported explicitly; this preparation never calls an LLM or downloads a corpus.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compensation.nq_train_adapter import validate_generation_source
from src.controller.data_preparation import (
    Candidate, OLD_ROLES, ROLES, fever_outputs, fever_role, jsonl_bytes,
    nq_outputs, precision_notes, query_key, select_cohorts, sha,
)

POLICY = "configs/controller/data_policy_v1.json"
ASSETS = {
    "nq": ["data/processed/retrieval/nq_train_corpus.jsonl",
           "data/processed/indexes/nq_train_bge_base.index",
           "data/processed/indexes/nq_train_bge_base_ids.jsonl",
           "data/processed/indexes/nq_train_bge_base_offsets.npy",
           "data/processed/indexes/nq_train_bge_metadata.json"],
    "fever": ["data/processed/retrieval/fever_corpus.jsonl",
              "data/processed/indexes/fever_bge_ivfpq.index",
              "data/processed/indexes/fever_bge_ivfpq_ids.jsonl",
              "results/retrieval/fever_bge_ivfpq_meta.json"],
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_hint(value):
    import re
    if value in ("nq", "natural_questions"):
        return "nq"
    if value == "fever":
        return "fever"
    tokens = re.split(r"[^a-z0-9]+", str(value).lower())
    found = [d for d in ("nq", "fever") if d in tokens]
    return found[0] if len(found) == 1 else None


ID_LIST_KEYS = {"sample_ids", "known_exposed_ids", "new_development_audit_ids",
                "previously_previewed_ids_forced_to_development", "train", "validation", "test"}
QUERY_KEYS = ("query", "question_or_claim", "question", "claim")


def extract_exposure(value, hint=None):
    """Collect only sample identifiers and question text, never passage IDs.

    Explicit dataset metadata wins over filename hints. Scalar IDs in lists
    need a known dataset; rows with ambiguous metadata are not guessed by ID
    magnitude. The audit states its coverage and includes explicit split files.
    """
    ids, queries = defaultdict(set), defaultdict(set)

    def walk(node, dataset, field=None):
        if isinstance(node, dict):
            current = dataset_hint(node.get("dataset")) or dataset
            question = next((node[k] for k in QUERY_KEYS if isinstance(node.get(k), str)), None)
            identifier = node.get("sample_id")
            if identifier is None and (question is not None or "label" in node):
                identifier = node.get("id")
            if current in ("nq", "fever"):
                if identifier is not None and isinstance(identifier, (str, int)) and not isinstance(identifier, bool):
                    ids[current].add(str(identifier))
                if question and question.strip():
                    queries[current].add(question)
            for key, child in node.items():
                # Dataset-keyed split tables propagate their own namespace.
                walk(child, dataset_hint(key) or current, key)
        elif isinstance(node, list):
            for child in node:
                if field in ID_LIST_KEYS and dataset and isinstance(child, (str, int)) and not isinstance(child, bool):
                    ids[dataset].add(str(child))
                else:
                    walk(child, dataset, field)

    walk(value, hint)
    return ids, queries


def exposure_audit(root, commit):
    prefixes = ["artifacts/answer_repair", "results", "data/annotations/assisted",
                "data/annotations/human", "data/annotations/normalized", "configs/splits", "configs/probe"]
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit, *prefixes],
                                    cwd=root, text=True).splitlines()
    ids, queries, inventory = defaultdict(set), defaultdict(set), []
    for path in sorted(p for p in files if p.endswith((".json", ".jsonl"))):
        # Original dataset/label pools alone are not evidence of optimization.
        # The included normalized Diagnosis pools are conservative exclusions;
        # the new full NQ Train preliminary label export is outside this scope.
        values = rows(root / path) if path.endswith(".jsonl") else [read_json(root / path)]
        file_ids, file_queries = defaultdict(set), defaultdict(set)
        for value in values:
            found_ids, found_queries = extract_exposure(value, dataset_hint(path))
            for ds in ("nq", "fever"):
                file_ids[ds].update(found_ids[ds])
                file_queries[ds].update(found_queries[ds])
        for ds in ("nq", "fever"):
            ids[ds].update(file_ids[ds])
            queries[ds].update(file_queries[ds])
        inventory.append({"path": path, "sha256": file_sha(root / path),
                          "sample_ids": {d: len(file_ids[d]) for d in ("nq", "fever")},
                          "query_counts": {d: len(file_queries[d]) for d in ("nq", "fever")}})
    refs = subprocess.check_output(["git", "for-each-ref", "--format=%(refname:short) %(objectname)",
                                    "refs/remotes/origin"], cwd=root, text=True).splitlines()
    return ids, queries, {"source_commit": commit, "files": inventory, "remote_refs_observed": refs,
        "known_ids": {ds: sorted(ids[ds]) for ds in ("nq", "fever")},
        "normalized_query_sha256": {ds: sorted({query_key(q) for q in queries[ds]}) for ds in ("nq", "fever")},
        "coverage": "Tracked JSON/JSONL in explicit paths at source commit plus existing E exposure registry; archives, unpublished work, prose-only examples and model pretraining are not exhaustively audited"}


def asset_inventory(root):
    result = {}
    for dataset, paths in ASSETS.items():
        result[dataset] = []
        for path in paths:
            target = root / path
            stat = target.stat() if target.exists() else None
            # macOS SF_DATALESS: stat alone must not trigger multi-GB hydration.
            dataless = bool(stat and getattr(stat, "st_flags", 0) & 0x40000000)
            result[dataset].append({"path": path, "exists": bool(stat),
                "bytes": stat.st_size if stat else None, "dataless": dataless,
                "sha256": file_sha(target) if stat and not dataless and stat.st_size < 10_000_000 else None,
                "verification": "missing" if not stat else "cloud_placeholder" if dataless else "present_not_semantically_audited"})
    return {"datasets": result, "index_c1_match_verified": False, "bge_revision": None,
            "nq_adapter_requirement": "Preserve official IndexFlatIP + IDs + offsets + corpus. Current controller provider expects full-text JSONL metadata; T3 must add the supplied-format adapter or a verified lossless conversion.",
            "full_corpus_cleanliness_claim": False}


def prepare(root, output_dir, policy_path=POLICY):
    root, output_dir = Path(root), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a new output directory; frozen cohorts cannot be overwritten")
    policy = read_json(root / policy_path)
    if policy["schema_version"] != "controller_data_policy_v1":
        raise ValueError("Unknown policy")
    counts, seed = policy["counts_per_dataset"], policy["seed"]
    protocol = read_json(root / policy["protocol_path"])
    protocol_sha = file_sha(root / policy["protocol_path"])
    expected_protocol_sha = (root / Path(policy["protocol_path"]).with_suffix(".sha256")).read_text().split()[0]
    if protocol_sha != expected_protocol_sha or protocol["protocol_id"] != "controller_experiment_v1":
        raise ValueError("T1 protocol changed")
    partitions = read_json(root / policy["nq_partition_source"])["partitions"]
    roles = {}
    for old_role, sample_ids in partitions.items():
        for sample_id in sample_ids:
            if sample_id in roles:
                raise ValueError("An NQ ID occurs in multiple original partitions")
            roles[sample_id] = OLD_ROLES.get(old_role, "excluded_by_team")
    delivery = read_json(root / policy["nq_delivery_manifest"])
    stats = read_json(root / policy["nq_generation_stats"])
    validate_generation_source(stats)
    sources, candidates, invalid, nq_seen = [], [], [], set()
    source_keys = ["nq_partition_source", "nq_delivery_manifest", "nq_generation_stats", "fever_source",
                   "preexisting_variant_quarantine"]
    for key in source_keys:
        path = policy[key]
        sources.append({"path": path, "sha256": file_sha(root / path)})
    for shard in delivery["shards"]:
        path = Path(policy["nq_delivery_directory"]) / shard["file"]
        if file_sha(root / path) != shard["sha256"]:
            raise ValueError(f"Delivery SHA mismatch: {path}")
        count = 0
        for row in rows(root / path):
            count += 1
            sample_id = row["sample_id"]
            if sample_id in nq_seen:
                raise ValueError("Duplicate NQ source ID")
            nq_seen.add(sample_id)
            if sample_id not in roles:
                raise ValueError("NQ ID absent from original partitions")
            try:
                nq_outputs(row)  # structural/reference validation, no correctness scoring
                if roles[sample_id] not in ROLES:
                    raise ValueError("Source ID reserved outside this experiment")
                candidates.append(Candidate("nq", sample_id, row["query"], roles[sample_id]))
            except (ValueError, TypeError, KeyError) as exc:
                invalid.append({"dataset": "nq", "sample_id": sample_id, "reason": str(exc)})
        if count != shard["n_samples"]:
            raise ValueError("Delivery shard count mismatch")
        sources.append({"path": str(path), "sha256": shard["sha256"], "rows": count})
    if len(nq_seen) != delivery["total_samples"] or set(roles) != nq_seen:
        raise ValueError("NQ source/partition coverage mismatch")
    fever_seen = set()
    for row in rows(root / policy["fever_source"]):
        sample_id = str(row["id"])
        if sample_id in fever_seen:
            raise ValueError("Duplicate FEVER source ID")
        fever_seen.add(sample_id)
        try:
            task, _ = fever_outputs(row)
            candidates.append(Candidate("fever", sample_id, task["query"], fever_role(task["query"], seed)))
        except (ValueError, TypeError, KeyError) as exc:
            invalid.append({"dataset": "fever", "sample_id": sample_id, "reason": str(exc)})
    ids, queries, exposure = exposure_audit(root, policy["source_commit"])
    # IDs without query text are still matched to the source pool here so their
    # normalized/lexical variants are not allowed through under another ID.
    for c in candidates:
        if c.sample_id in ids[c.dataset]:
            queries[c.dataset].add(c.query)
    exposure["normalized_query_sha256"] = {ds: sorted({query_key(q) for q in queries[ds]}) for ds in ("nq", "fever")}
    exposure["team_attestation"] = policy["team_exposure_attestation"]
    old_variants = read_json(root / policy["preexisting_variant_quarantine"])
    quarantine = {"nq": {str(m["sample_id"]) for g in old_variants["groups"] for m in g["members"]}}
    selected, selection_summary, exclusions = select_cohorts(candidates, ids, queries, counts, seed=seed,
                                                            quarantined_ids=quarantine)
    protected = {
        "nq": sorted(partitions["reserved_evaluation"]),
        "fever": sorted(c.sample_id for c in candidates if c.dataset == "fever"
                        and c.role == "final_candidate" and c.sample_id not in ids["fever"]),
    }
    populations = {"nq": len(nq_seen), "fever": sum(c.dataset == "fever" for c in candidates)}
    reserve_summary = {}
    for ds, sample_ids in protected.items():
        fraction = len(sample_ids) / populations[ds]
        if not policy["evaluation_reserve"]["minimum_fraction"] <= fraction <= policy["evaluation_reserve"]["maximum_fraction"]:
            raise ValueError(f"{ds} protected evaluation pool must retain 10%-15% before inference")
        if not set(selected[ds]["final_candidate"]) <= set(sample_ids):
            raise ValueError("Final subset is outside the protected pool")
        if set(sample_ids) & set(selected[ds]["development"] + selected[ds]["selection"]):
            raise ValueError("Protected evaluation IDs leaked into optimization cohorts")
        reserve_summary[ds] = {"population_count": populations[ds], "protected_count": len(sample_ids),
            "fraction": fraction, "active_final_subset_count": counts["final_candidate"],
            "additional_reserved_count": len(sample_ids) - counts["final_candidate"],
            "scope": "Do not use any protected ID for training, prompt design, tuning or selection; active subset receives the additional lexical exclusion checks"}
    keys = {(ds, sid): role for ds, parts in selected.items() for role, sample_ids in parts.items() for sid in sample_ids}
    batches, references = defaultdict(dict), defaultdict(dict)
    for shard in delivery["shards"]:
        for row in rows(root / policy["nq_delivery_directory"] / shard["file"]):
            key = ("nq", row["sample_id"])
            if key in keys:
                runtime, gold = nq_outputs(row)
                batches[("nq", keys[key])][key[1]] = runtime
                references[("nq", keys[key])][key[1]] = gold
    for row in rows(root / policy["fever_source"]):
        key = ("fever", str(row["id"]))
        if key in keys:
            task, gold = fever_outputs(row)
            batches[("fever", keys[key])][key[1]] = task
            references[("fever", keys[key])][key[1]] = gold
    # Write only after selection/source validation has succeeded. A partial I/O
    # failure cannot produce a completed manifest, and reruns never overwrite.
    output_dir.mkdir(parents=True)
    artifacts = {}

    def save(path, raw):
        target = output_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(raw)
        artifacts[path] = {"sha256": sha(raw), "bytes": len(raw)}

    def save_json(path, value):
        save(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())

    for ds, parts in selected.items():
        for role, sample_ids in parts.items():
            if set(batches[(ds, role)]) != set(sample_ids) or set(references[(ds, role)]) != set(sample_ids):
                raise ValueError("Selected input/reference join is incomplete")
            filename = "runtime.jsonl" if ds == "nq" else "c1_tasks.jsonl"
            save(f"{ds}/{role}/{filename}", jsonl_bytes(batches[(ds, role)][i] for i in sample_ids))
            save(f"offline/{ds}/{role}/gold.jsonl", jsonl_bytes(references[(ds, role)][i] for i in sample_ids))
    save_json("selection_ids.json", selected)
    save_json("evaluation_reserve_ids.json", protected)
    save_json("evaluation_reserve_summary.json", reserve_summary)
    save_json("exposure_audit.json", exposure)
    save_json("asset_inventory.json", asset_inventory(root))
    save("exclusions.jsonl", jsonl_bytes(invalid + exclusions))
    save_json("policy.json", policy)
    source_code = {p: file_sha(root / p) for p in ["scripts/prepare_controller_data.py",
                    "src/controller/data_preparation.py", "src/compensation/nq_train_adapter.py",
                    "src/diagnosis/features.py", "src/diagnosis/schema.py"]}
    manifest = {"schema_version": "controller_data_manifest_v1", "protocol_sha256": protocol_sha,
        "policy_sha256": file_sha(root / policy_path), "source_commit": policy["source_commit"],
        "source_files": sources, "source_code_sha256": source_code,
        "source_counts": {"nq": len(nq_seen), "fever": len(fever_seen)},
        "structural_exclusions": dict(Counter(r["dataset"] for r in invalid)),
        "selection": selection_summary, "counts_per_dataset": counts,
        "evaluation_reserve": reserve_summary,
        "precision": {role: precision_notes(n) for role, n in counts.items()},
        "artifacts": dict(artifacts), "prediction_scores_computed": False,
        "new_human_labels_created": 0, "gpu_inference_performed": False,
        "nq": {"paired_runtime_and_gold_ready": True, "c1_revision": None,
               "baseline_scope": policy["historical_c1_decision"]},
        "fever": {"paired_runtime_and_gold_ready": False, "c1_tasks_and_offline_gold_ready": True,
                  "missing_c1_count": sum(counts.values()), "decision": policy["fever_c1_decision"]},
        "full_T2_ready": False,
        "remaining_requirements": [f"Generate and seal FEVER C1 Top5/answers for the fixed {sum(counts.values())} tasks without outcome-dependent removal",
            "Obtain and hash matching NQ Train and FEVER index/IDs/corpus/sidecars and pin verified BGE weights",
            "T3 must implement the NQ Train source-format adapter and FEVER retrieval; all formal arms require an execution seal"],
        "unseen_scope": "No recorded ID/canonical/direct-Jaccard exposure plus user-confirmed absence of extra team use; no semantic, Wikipedia-page or pretraining isolation guarantee"}
    save_json("manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = prepare(ROOT, args.output_dir)
    print(json.dumps({"output": str(args.output_dir), "counts_per_dataset": manifest["counts_per_dataset"],
                      "source_counts": manifest["source_counts"], "full_T2_ready": manifest["full_T2_ready"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
