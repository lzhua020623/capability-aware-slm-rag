"""Archive a final run, replay its frozen code, and keep all human judgments pending.

Use --python with the inference Python major/minor (Kaggle: 3.13). Synthetic
integration exports are rejected unless explicitly allowed into a /tmp output.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_answer_repair_experiment import encode_json, write_frozen
from scripts.prepare_answer_repair_final_dev import payload


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def collect(*, run, zip_path, python, out_dir=None, allow_synthetic=False):
    if run not in ("dev75", "challenge125"):
        raise ValueError("Unknown final run")
    name = "final_nq75" if run == "dev75" else "unseen_nq125"
    source_dir = ROOT / "artifacts/answer_repair" / name
    prep = json.loads((source_dir / ("preparation.json" if run == "dev75" else "gpu_preparation.json")).read_bytes())
    n = 75 if run == "dev75" else 125; revision = prep["selected_final_revision"]
    archive = source_dir / f"answer_repair_final_{revision}_nq{n}_gpu.tar.gz"
    files = payload(archive, prep["bundle_sha256"])
    manifest = json.loads(files["manifest.json"])
    if manifest["samples"] != n or manifest["tuning_allowed_after_run"] is not False:
        raise ValueError("Expected a frozen final run")
    destination = (out_dir or source_dir).resolve()
    if allow_synthetic and not destination.resolve().is_relative_to(Path("/tmp").resolve()):
        raise ValueError("Synthetic exports may only be collected under /tmp")
    zip_raw = zip_path.read_bytes(); exported = {}
    with zipfile.ZipFile(zip_path) as z:
        if len(z.namelist()) != len(set(z.namelist())) or sum(i.file_size for i in z.infolist()) > 100_000_000:
            raise ValueError("Duplicate or oversized result ZIP")
        if z.testzip() is not None:
            raise ValueError("Result ZIP CRC failed")
        for info in z.infolist():
            p = Path(info.filename)
            if p.is_absolute() or ".." in p.parts or info.file_size > 50_000_000:
                raise ValueError("Unsafe result ZIP entry")
            if not info.is_dir():
                exported[info.filename] = z.read(info)
    for key in ("manifest.json", "bundle_checksums.json"):
        expected = files[key] if key in files else None
        if expected is None:
            with tarfile.open(archive) as z:
                expected = z.extractfile(archive.name.removesuffix(".tar.gz") + "/" + key).read()
        if exported.get(key) != expected:
            raise ValueError("Export differs from frozen bundle: " + key)
    all_rows = []
    for arm in manifest["protocol"]:
        rows = [json.loads(s) for s in exported[arm["strategy"] + ".jsonl"].splitlines() if s.strip()]
        if [x["sample_id"] for x in rows] != manifest["sample_ids"]:
            raise ValueError("Missing or reordered result rows")
        if any(x["settings"]["model_revision"] != manifest["model_revision"] for x in rows):
            raise ValueError("Model revision differs from freeze")
        all_rows.extend(rows)
    synthetic = any(x["settings"].get("synthetic_preflight_only") for x in all_rows)
    if synthetic and not allow_synthetic:
        raise ValueError("Synthetic integration export cannot be archived as a model result")
    if not synthetic:
        required = {"bootstrap.json", "environment.json", "tests.log", "packages.log", "preflight_ast_compat.py"}
        if not required <= exported.keys():
            raise ValueError("Missing real execution provenance files")
        bootstrap = json.loads(exported["bootstrap.json"]); environment = json.loads(exported["environment.json"])
        if (bootstrap["expected_archive_sha256"] != prep["bundle_sha256"]
                or bootstrap["checksums_sha256"] != prep["checksums_sha256"]
                or bootstrap["algorithm_or_input_modified"] is not False
                or environment["cuda_available"] is not True or environment["errors"]
                or b"\nOK\n" not in exported["tests.log"] or b"FAILED" in exported["tests.log"]
                or sha(exported["preflight_ast_compat.py"]) != manifest["preflight_adapter_sha256"]):
            raise ValueError("Execution provenance or frozen preflight identity disagrees")
    raw_dir = destination / "kaggle_20261020"
    if raw_dir.exists() and {p.relative_to(raw_dir).as_posix() for p in raw_dir.rglob("*") if p.is_file()} - exported.keys():
        raise ValueError("Existing raw directory contains files outside this export")
    for path, raw in exported.items():
        write_frozen(raw_dir / path, raw)
    write_frozen(destination / zip_path.name, zip_raw)
    raw_hashes = {p: sha(raw) for p, raw in sorted(exported.items())}
    with tempfile.TemporaryDirectory(prefix="e-final-collect-") as temp:
        bundle = Path(temp) / "bundle"; bundle.mkdir()
        for path, raw in files.items():
            rel = Path(path)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe frozen source path")
            target = bundle / rel; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
        (bundle / "bundle_checksums.json").write_bytes(exported["bundle_checksums.json"])
        env = {**os.environ, "PYTHONPATH": str(bundle), "PYTHONPYCACHEPREFIX": str(Path(temp) / "pycache")}
        def execute(script, *args):
            subprocess.run([str(python), script, *map(str, args)], cwd=bundle, env=env, check=True)
        execute("scripts/audit_answer_repair_release.py", "--bundle-dir", bundle,
                "--results-dir", raw_dir, "--out-dir", destination / "release_audit")
        audit = json.loads((destination / "release_audit/cpu_replay.json").read_bytes())
        expected = manifest["expected_predictions"]
        if audit["replayed_records"] != expected:
            raise ValueError("Frozen replay did not cover every result")
        # Recompute derived reports separately; the original export is immutable.
        derived = Path(temp) / "derived"; shutil.copytree(raw_dir, derived)
        (derived / "comparison.json").unlink()
        execute("scripts/summarize_answer_repair_v8.py" if run == "dev75" else "scripts/summarize_answer_repair_challenge.py",
                "--bundle", bundle, "--results", derived)
        spec = importlib.util.spec_from_file_location("frozen_final_audit", bundle / "scripts/audit_answer_repair_release.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        if not module.equal(json.loads((raw_dir / "comparison.json").read_bytes()),
                            json.loads((derived / "comparison.json").read_bytes())):
            raise ValueError("Derived report differs beyond the original frozen numeric tolerance")
        if run == "dev75" and (raw_dir / "output_review_queue.jsonl").read_bytes() != (derived / "output_review_queue.jsonl").read_bytes():
            raise ValueError("Derived output review queue differs")
        human_path = bundle / "offline/initial_reviews.jsonl"
        initial = {x["sample_id"]: x for x in map(json.loads, human_path.read_bytes().splitlines())} if human_path.exists() else {}
        strata_path = bundle / "offline/strata.jsonl"
        strata = {x["sample_id"]: x for x in map(json.loads, strata_path.read_bytes().splitlines())} if strata_path.exists() else {}
        primary = []
        for line in (destination / "release_audit/pending_semantic_pairs.jsonl").read_bytes().splitlines():
            row = json.loads(line)
            if row["changed"] and row["strategy"] in ("short_answer_quote", "conservative_repair"):
                sid = row["sample_id"]
                primary.append({**row, "initial_human_review": initial.get(sid), "screening_stratum": strata.get(sid),
                                "human_verified": False, "agent_assistance_status": "pending"})
        write_frozen(destination / "review/primary_changed_pairs.jsonl",
                     b"".join((json.dumps(x, ensure_ascii=False) + "\n").encode() for x in primary))
    if {p: sha((raw_dir / p).read_bytes()) for p in raw_hashes} != raw_hashes:
        raise ValueError("Raw export changed during replay")
    receipt = {"version": "answer_repair_final_collection_v1", "run": run, "result_directory": raw_dir.name,
               "source_bundle_sha256": prep["bundle_sha256"], "result_zip_sha256": sha(zip_raw),
               "raw_file_sha256": raw_hashes, "zip_crc_ok": True, "replayed_records": expected,
               "frozen_comparison_recalculation_matches": True,
               "pending_primary_changed_pairs": len(primary), "human_final_gain": None,
               "semantic_correctness_verified": False, "synthetic_preflight_only": synthetic,
               "research_result": not synthetic, "selected_final_revision": revision,
               "version_changed_after_evaluation": False}
    write_frozen(destination / "result_validation.json", encode_json(receipt))
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", choices=("dev75", "challenge125"), required=True)
    p.add_argument("--zip", type=Path, required=True); p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--out-dir", type=Path); p.add_argument("--allow-synthetic", action="store_true")
    a = p.parse_args(argv)
    report = collect(run=a.run, zip_path=a.zip, python=a.python, out_dir=a.out_dir, allow_synthetic=a.allow_synthetic)
    print(json.dumps({k: v for k, v in report.items() if k != "raw_file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
