"""Collect the original Kaggle ZIP and replay only hash-verified frozen code.

Run with Python 3.13. This does not load a model or regenerate predictions.
Raw files are immutable; unpacked source copies remain under ignored gpu_bundle/.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile

FOLDER = Path(__file__).resolve().parents[1]
EXPECTED_ZIP_SHA256 = "864f9f3ca01b305be3434accb06c4b01ee55bdad320f5b88b88fa6aec687b1bd"
EXPECTED_BUNDLE_SHA256 = "8dced5b6c02e1295393dcfe00bfc73e4c612ca39c7c03d98bb77a62e4d722159"
EXPECTED_FINGERPRINT = "559e48fc46f5ec6d2e63dc018a4ac4d5e5669768468be035859305d506dc12e2"
PREFIX = "e_contract_v1_8dced5b6c02e/contract/"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def write_once(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Refusing to replace frozen bytes: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=FOLDER / "kaggle_outputs_v1_361060970.zip")
    parser.add_argument("--log", type=Path, default=FOLDER / "kaggle_20261022/kaggle_full.log")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Replay must use inference Python major/minor: 3.13")

    zip_raw = args.zip.read_bytes()
    if sha(zip_raw) != EXPECTED_ZIP_SHA256:
        raise ValueError("Unexpected original Kaggle output archive")
    with zipfile.ZipFile(io.BytesIO(zip_raw)) as archive:
        infos = archive.infolist()
        if len(infos) != len({v.filename for v in infos}) or sum(v.file_size for v in infos) > 20_000_000:
            raise ValueError("Duplicate or oversized ZIP")
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC error")
        exported = {}
        for info in infos:
            path = Path(info.filename)
            if path.is_absolute() or ".." in path.parts or info.is_dir():
                raise ValueError("Unexpected ZIP member")
            exported[info.filename] = archive.read(info)

    bundle_raw = (FOLDER / "answer_repair_contract_v1_gpu.tar.gz").read_bytes()
    if sha(bundle_raw) != EXPECTED_BUNDLE_SHA256:
        raise ValueError("Frozen GPU input archive changed")
    frozen = {}
    with tarfile.open(fileobj=io.BytesIO(bundle_raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or path.parts[0] != "contract" or member.size > 5_000_000):
                raise ValueError("Unsafe frozen archive")
            relative = path.relative_to("contract").as_posix()
            if relative in frozen:
                raise ValueError("Duplicate frozen member")
            frozen[relative] = archive.extractfile(member).read()
    checksums = json.loads(frozen["bundle_checksums.json"])
    if len(checksums) != 40 or set(frozen) != set(checksums) | {"bundle_checksums.json"}:
        raise ValueError("Incomplete frozen source list")
    for name, raw in frozen.items():
        if exported.get(PREFIX + name) != raw:
            raise ValueError(f"Returned source differs from frozen input: {name}")
        if name in checksums and sha(raw) != checksums[name]:
            raise ValueError("Frozen source checksum mismatch")

    raw_dir = FOLDER / "kaggle_20261022"
    for name, raw in exported.items():
        if name.startswith(PREFIX + "results/"):
            write_once(raw_dir / name.removeprefix(PREFIX + "results/"), raw)
    write_once(FOLDER / "kaggle_outputs_v1_361060970.zip", zip_raw)
    full_log = args.log.read_bytes()
    write_once(raw_dir / "kaggle_full.log", full_log)
    if (b'"replay": "passed", "predictions": 48' not in full_log
            or b"ZIP does not support timestamps before 1980" not in full_log):
        raise ValueError("Full cloud log does not attest replay then export failure")
    with zipfile.ZipFile(io.BytesIO(exported["e_contract_v1_outputs.zip"])) as inner:
        if inner.namelist():
            raise ValueError("Expected the failed, empty inner export")

    source = FOLDER / "gpu_bundle/contract"
    for name, raw in frozen.items():
        write_once(source / name, raw)
    prediction_raw = (raw_dir / "predictions.jsonl").read_bytes()
    rows = [json.loads(line) for line in prediction_raw.splitlines() if line.strip()]
    cases = [json.loads(line) for line in frozen["artifacts/answer_repair/v9_contract_v1/synthetic_cases.jsonl"].splitlines()]
    expected = [(c["case_id"], s, d) for c in cases for s in ("evidence_id", "factored_evidence_id")
                for d in ("prompt_only", "xgrammar")]
    if (len(rows) != 48 or [(r["case_id"], r["strategy"], r["decoder"]) for r in rows] != expected
            or any(r["synthetic"] is not True or r["fingerprint"] != EXPECTED_FINGERPRINT for r in rows)):
        raise ValueError("Incomplete, reordered or non-frozen synthetic results")

    environment = rows[0]["runtime_environment"]
    if any(r["runtime_environment"] != environment for r in rows):
        raise ValueError("Model environment differs across rows")
    quant = environment["quantization_config"]
    if (environment["model_revision"] != "a09a35458c702b33eeacc393d103063234e8bc28"
            or environment["xgrammar"] != "0.2.8" or environment["fourbit_module_count"] <= 0
            or environment["fourbit_quant_types"] != ["nf4"] or not quant["load_in_4bit"]
            or quant["bnb_4bit_compute_dtype"] != "float16" or not quant["bnb_4bit_use_double_quant"]
            or not environment["parameter_devices"] or any(not d.startswith("cuda:") for d in environment["parameter_devices"])):
        raise ValueError("Recorded runtime does not attest frozen CUDA NF4 settings")

    calls = []
    for row in rows:
        result = row["result"]; trace = result["repair_trace"]
        current = trace["calls"]
        if len(current) != result["extra_calls"] or len(current) > (1 if row["strategy"] == "evidence_id" else 5):
            raise ValueError("Call budget or accounting mismatch")
        if len(trace["constraints"]) != len(current):
            raise ValueError("Missing decoder accounting")
        for call, constraint in zip(current, trace["constraints"]):
            if (call["max_new_tokens"] != 384 or call["generation"]["output_tokens_estimate"] > 384
                    or constraint["backend"]["decoder"] != row["decoder"]
                    or constraint["backend"].get("output_tokens_including_stop", 0) > 384):
                raise ValueError("Token budget or decoder mismatch")
            calls.append(call)
    if len(calls) > 144:
        raise ValueError("Total call budget exceeded")

    # The frozen runner replays prompt/schema hashes, raw calls and every result
    # field, then recomputes the complete summary without loading any backend.
    with tempfile.TemporaryDirectory(prefix="e-contract-local-replay-") as temp:
        predictions = Path(temp) / "predictions.jsonl"
        predictions.write_bytes(prediction_raw)
        command = [sys.executable, "-B", str(source / "scripts/run_answer_repair_contract.py"),
                   "--out", str(predictions)]
        proc = subprocess.run(command, cwd=source, text=True, capture_output=True, check=True)
        recomputed = predictions.with_suffix(".summary.json").read_bytes()
        if json.loads(recomputed) != json.loads((raw_dir / "predictions.summary.json").read_bytes()):
            raise ValueError("Recomputed summary differs from raw GPU summary")
        replay_proc = subprocess.run(command + ["--replay"], cwd=source, text=True, capture_output=True, check=True)
        local_replay = json.loads(replay_proc.stdout)
    summary = json.loads(recomputed)
    if (local_replay["replay"] != "passed" or local_replay["predictions"] != 48
            or local_replay["fingerprint"] != EXPECTED_FINGERPRINT
            or summary["constrained_valid"] != 22 or not summary["engineering_gate_passed"]):
        raise ValueError("Engineering gate does not pass frozen checks")
    write_once(FOLDER / "release_audit/recomputed_summary.json", recomputed)

    report = {
        "date": "2026-10-22", "script_version_id": 361060970,
        "source_bundle_sha256": EXPECTED_BUNDLE_SHA256, "original_zip_sha256": sha(zip_raw),
        "original_zip_bytes": len(zip_raw), "original_zip_members": len(exported),
        "cloud_full_log_sha256": sha(full_log), "frozen_source_files_verified": len(checksums),
        "returned_checksums_and_all_frozen_sources_match": True,
        "zip_member_sha256": {name: sha(raw) for name, raw in sorted(exported.items())},
        "fingerprint": EXPECTED_FINGERPRINT, "predictions": len(rows), "synthetic_cases": len(cases),
        "local_python": sys.version.split()[0], "inference_python": environment["python"],
        "local_replay": local_replay, "summary_recomputed_matches": True,
        "budget": {"calls": len(calls), "maximum_calls": 144,
                   "maximum_output_tokens_observed": max(c["generation"]["output_tokens_estimate"] for c in calls),
                   "max_new_tokens_per_call": 384, "respected": True},
        "cuda_nf4_attested_in_recorded_environment": True,
        "constrained_valid": 22, "constrained_total": 24,
        "unknown_ids": summary["unknown_ids"], "keep_rewrites": summary["keep_rewrites"],
        "engineering_gate_passed": True,
        "platform_job_status": "failed", "gpu_inference_and_replay_completed": True,
        "failure_stage": "notebook_final_zip_export_after_successful_inference_and_replay",
        "failure_cause": "Frozen tar members have epoch-zero mtime; default ZipFile rejects years before 1980.",
        "failed_inner_zip_bytes": len(exported["e_contract_v1_outputs.zip"]),
        "gpu_rerun_required": False, "new_model_calls_during_collection": 0,
        "benchmark_gain_claim": False, "default_promoted": False, "human_labels_changed": False,
        "groups": summary["groups"],
        "invalid_results": [{"case_id": r["case_id"], "strategy": r["strategy"], "decoder": r["decoder"],
                             "status": r["result"]["status"], "error": r["result"]["validation_error"]}
                            for r in rows if r["result"]["status"].startswith("invalid_")],
        "raw_result_sha256": {p.name: sha(p.read_bytes()) for p in sorted(raw_dir.iterdir()) if p.is_file()},
    }
    write_once(FOLDER / "result_validation.json", encode(report))
    print(json.dumps({k: report[k] for k in ("predictions", "frozen_source_files_verified", "local_replay", "budget", "constrained_valid", "engineering_gate_passed", "failure_stage")}, indent=2))


if __name__ == "__main__":
    main()
