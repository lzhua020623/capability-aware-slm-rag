"""Package already-exposed development inputs; never select unseen records."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_answer_repair_experiment import encode_json, write_frozen
from scripts.run_answer_repair import read_jsonl, prepare_records
from src.compensation.evidence_id_repair import STRATEGIES
from src.diagnosis import DatasetName


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/answer_repair/v9_dev_v1")
    args = p.parse_args(argv)
    sources = {
        "nq": ("artifacts/answer_repair/final_nq75/answer_repair_final_r3_nq75_gpu.tar.gz",
               "answer_repair_final_r3_nq75_gpu/inputs/nq75_c1.jsonl"),
        "fever": ("artifacts/answer_repair/v7_reviewed/answer_repair_v7_reviewed_gpu.tar.gz",
                  "answer_repair_v7_reviewed_gpu/inputs/fever_dev50_c1.jsonl"),
    }
    known_nq = json.loads((ROOT / "configs/splits/answer_repair_nq_train_confirmed_dev_v6.json").read_text())
    old_smoke = json.loads((ROOT / "configs/answer_repair_v8r4_smoke8_v1.json").read_text())
    old_ids = {sid for number, sid in zip(known_nq["review_numbers"], known_nq["sample_ids"])
               if number in old_smoke["review_numbers"]}
    human_nq = {r["sample_id"]: r for r in read_jsonl(
        (ROOT / "data/annotations/human/answer_repair_nq_train_dev_v1_through_batch06_20261019.jsonl").read_bytes(), "NQ human")}
    human_fever = {r["sample_id"]: r for r in read_jsonl(
        (ROOT / "data/annotations/human/answer_repair_v6r1_pairs_20261014.jsonl").read_bytes(), "FEVER human")
                   if r["dataset"] == "fever" and r["cohort"] == "dev50"}
    files, manifest = {}, {"version": "v9_dev_v1", "purpose": "exposed development only, not unseen",
                          "datasets": {}, "gpu_inference_completed": False,
                          "smoke_policy": "5 per dataset by SHA256; exclude retired NQ smoke8 IDs; engineering only, no EM selection",
                          "selection_gate": "GPU smoke: full output count, finite costs, successful frozen replay; log all format/source failures. No performance-based prompt changes on smoke.",
                          "full_dev_gate": "Before full development submit, document semantic scoring and acceptable false-reject/cost tradeoff from development only. No unseen data included.",
                          "not_compute_matched": True}
    for dataset, (archive_path, member) in sources.items():
        archive = ROOT / archive_path
        with tarfile.open(archive) as tar:
            raw = tar.extractfile(member).read()
        records = read_jsonl(raw, member)
        prepared = prepare_records(records, DatasetName(dataset))
        if dataset == "nq" and {r.sample_id for _, r in prepared} != set(known_nq["sample_ids"]):
            raise ValueError("NQ input is not the frozen exposed development75")
        if len(records) != (75 if dataset == "nq" else 50):
            raise ValueError("Unexpected development cohort")
        ordered = sorted(prepared, key=lambda x: digest(f"v9-smoke:42:{dataset}:{x[1].sample_id}".encode()))
        smoke = [(r, q) for r, q in ordered if dataset != "nq" or q.sample_id not in old_ids][:5]
        prefix = f"inputs/{dataset}"
        files[prefix + "_dev_c1.jsonl"] = raw
        files[prefix + "_smoke5_c1.jsonl"] = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r, _ in smoke)
        files[prefix + "_dev_runtime.jsonl"] = b"".join((json.dumps(asdict(q), ensure_ascii=False) + "\n").encode() for _, q in prepared)
        labels = []
        for _, q in prepared:
            h = (human_nq if dataset == "nq" else human_fever).get(q.sample_id)
            initial = h.get("human_review", {}).get("initial_correct") if dataset == "nq" and h else (
                h.get("pair_review", {}).get("initial_correct") if h else None)
            if h and (h["query"] != q.query or h["initial_answer"] != q.initial_answer):
                raise ValueError("Human label is bound to another question/answer material")
            labels.append({"sample_id": q.sample_id, "initial_correct": initial,
                           "basis": "reference_visible_chat_assisted_confirmation" if initial is not None else "not_human_scored",
                           "runtime_allowed": False})
        files[f"offline/{dataset}_initial_human_labels.jsonl"] = b"".join((json.dumps(r) + "\n").encode() for r in labels)
        manifest["datasets"][dataset] = {"samples": len(records), "input_sha256": digest(raw),
                                          "source_archive": archive_path, "source_archive_sha256": digest(archive.read_bytes()),
                                          "member": member, "sample_ids": [q.sample_id for _, q in prepared],
                                          "smoke_ids": [q.sample_id for _, q in smoke],
                                          "human_initial_labels": sum(r["initial_correct"] is not None for r in labels)}
    source_files = list((ROOT / "src").rglob("*.py")) + [ROOT / f for f in (
        "scripts/run_answer_repair.py", "scripts/run_answer_repair_v9.py", "scripts/prepare_answer_repair_experiment.py",
        "scripts/prepare_answer_repair_v9.py", "scripts/check_answer_repair_environment.py",
        "tests/__init__.py", "tests/test_answer_repair_v9.py", "configs/base.yaml",
        "configs/answer_repair_v9_protocol_v1.json", "requirements-answer-repair.txt")]
    files.update({p.relative_to(ROOT).as_posix(): p.read_bytes() for p in source_files})
    commands = ["python -m unittest tests.test_answer_repair_v9 -v",
                "python scripts/check_answer_repair_environment.py --require-cuda --out environment.json"]
    for dataset in sources:
        for strategy in STRATEGIES:
            command = (f"python scripts/run_answer_repair_v9.py --dataset {dataset} --strategy {strategy} --all "
                       f"--c1 inputs/{dataset}_smoke5_c1.jsonl --out results/{dataset}_smoke5_{strategy}.jsonl")
            commands += [command + " --check-only", command, command + " --replay"]
    files["RUN_SMOKE.sh"] = ("#!/bin/sh\nset -eu\nmkdir -p results\n" + "\n".join(commands) + "\n").encode()
    files["RUN_GPU.txt"] = (
        "Python 3.11+ and CUDA required. Install requirements-answer-repair.txt with CUDA PyTorch.\n"
        "Run: sh RUN_SMOKE.sh (30 predictions total, at most 70 generation calls).\n"
        "All sources are already-exposed development. This is engineering smoke, not a gain claim.\n"
        "Full development uses *_dev_c1.jsonl via the same runner; do not start it before its scoring gate is fixed.\n"
        "Return results/, environment.json, manifest.json and bundle_checksums.json.\n"
        "Gold and human labels are offline only; the model adapter receives runtime fields only.\n"
        "No challenge125 or unseen inputs are included. Factored calls share Qwen weights.\n"
    ).encode()
    files["manifest.json"] = encode_json(manifest)
    files["bundle_checksums.json"] = encode_json({name: digest(raw) for name, raw in sorted(files.items())})
    for name, raw in files.items():
        write_frozen(args.out_dir / "gpu_bundle" / name, raw)
    archive_out = args.out_dir / "answer_repair_v9_dev_v1_gpu.tar.gz"
    if not archive_out.exists():
        temporary = archive_out.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz") as tar:
            for name, raw in sorted(files.items()):
                info = tarfile.TarInfo("answer_repair_v9_dev_v1_gpu/" + name)
                info.size = len(raw); info.mtime = 0
                tar.addfile(info, io.BytesIO(raw))
        temporary.replace(archive_out)
    else:
        with tarfile.open(archive_out) as tar:
            if tar.extractfile("answer_repair_v9_dev_v1_gpu/bundle_checksums.json").read() != files["bundle_checksums.json"]:
                raise ValueError("Existing frozen bundle differs")
    write_frozen(args.out_dir / "manifest.json", encode_json(manifest))
    write_frozen(args.out_dir / "preparation.json", encode_json({"archive_sha256": digest(archive_out.read_bytes()),
        "files": len(files), "gpu_inference_completed": False, "smoke_predictions": 30, "smoke_max_calls": 70}))
    print(json.dumps({"archive": str(archive_out), "samples": {k: v["samples"] for k, v in manifest["datasets"].items()},
                      "human_initial_labels": {k: v["human_initial_labels"] for k, v in manifest["datasets"].items()},
                      "gpu_inference_completed": False}, indent=2))


if __name__ == "__main__":
    main()
