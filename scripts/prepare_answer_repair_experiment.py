"""Freeze training-only E development cohorts and make a portable GPU bundle.

Uses archived C1 evidence, so no raw datasets, retrieval index or Git checkout
is needed on the GPU host. Selection never reads answers, correctness or labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_answer_repair import prepare_records, read_jsonl
from src.diagnosis import DatasetName
from src.compensation.answer_repair import (
    MODEL_NAME, MODEL_REVISION, REPAIR_PROMPT_VERSION, GROUNDED_REANSWER_PROMPT_VERSION,
)

SPLIT_PATHS = (
    "configs/splits/diagnosis_v1_split.json",
    "configs/splits/diagnosis_v2_expanded_split.json",
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_read(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def select_development_ids(splits: list[dict], dataset: str, count: int, seed: int = 42) -> list[str]:
    """Intersect training partitions and exclude every validation/test partition."""
    if not splits or count < 1:
        raise ValueError("At least one split and a positive sample count are required")
    train_sets, reserved = [], set()
    for document in splits:
        partitions = document["splits"][dataset]
        parsed = {}
        for name in ("train", "validation", "test"):
            ids = partitions[name]
            if (not isinstance(ids, list) or any(not isinstance(s, str) or not s.strip() for s in ids)
                    or len(ids) != len(set(ids))):
                raise ValueError(f"Invalid or duplicate IDs in {dataset}/{name}")
            parsed[name] = set(ids)
        if any(parsed[a] & parsed[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError(f"Overlapping partitions in {dataset}")
        train_sets.append(parsed["train"])
        reserved.update(parsed["validation"] | parsed["test"])
    candidates = set.intersection(*train_sets) - reserved
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} shared training IDs for {dataset}; requested {count}")
    return sorted(candidates, key=lambda sid: (sha256(f"{seed}:{dataset}:{sid}".encode()), sid))[:count]


def encode_json(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def write_frozen(path: Path, raw: bytes) -> None:
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f"Frozen artifact differs: {path}; use a new experiment path")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)


def select_c1(raw: bytes, dataset: str, ids: list[str]) -> tuple[bytes, dict]:
    records = read_jsonl(raw, dataset)
    prepared = prepare_records(records, DatasetName(dataset))
    by_id = {request.sample_id: record for record, request in prepared}
    missing = set(ids) - by_id.keys()
    if missing:
        raise ValueError(f"C1 is missing {len(missing)} selected IDs for {dataset}")
    selected = [by_id[sid] for sid in ids]
    correct = sum(row["correct"] for row in selected)
    return ("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode(),
            {"samples": len(selected), "initial_correct": correct, "initial_wrong": len(selected) - correct})


def bundle_readme(count: int, smoke_count: int) -> bytes:
    return f"""# E GPU experiment

This is a snapshot of local E code plus archived C1 evidence. No Git clone,
Wikipedia download or retrieval index is required. The bundle includes offline
reference answers for scoring; the runtime adapter excludes them from prompts.

Use Python 3.11+ and a CUDA-enabled PyTorch installation on an NVIDIA GPU host.
Install the remaining E dependencies:

```sh
python -m pip install -r requirements-answer-repair.txt
python scripts/check_answer_repair_environment.py --require-cuda --out environment.json
python -m unittest discover -s tests -v
```

First inspect and run the FEVER {smoke_count}-sample engineering smoke:

```sh
python scripts/run_answer_repair.py --dataset fever --c1 inputs/fever_smoke{smoke_count}_c1.jsonl --all --check-only --out results/answer_repair/fever_smoke{smoke_count}.jsonl
python scripts/run_answer_repair.py --dataset fever --c1 inputs/fever_smoke{smoke_count}_c1.jsonl --all --out results/answer_repair/fever_smoke{smoke_count}.jsonl
```

Inspect every smoke output for JSON errors, fabricated or irrelevant quotes,
truncated JSON, changed answers and actual evidence support. This stage checks
execution, not algorithm effectiveness. No automatic semantic verifier is active.

After that review, run the fixed {count}-sample development cohorts:

```sh
python scripts/run_answer_repair.py --dataset fever --c1 inputs/fever_dev{count}_c1.jsonl --all --out results/answer_repair/fever_dev{count}_verify.jsonl
python scripts/run_answer_repair.py --dataset nq --c1 inputs/nq_dev{count}_c1.jsonl --all --out results/answer_repair/nq_dev{count}_verify.jsonl
python scripts/run_answer_repair.py --dataset fever --c1 inputs/fever_dev{count}_c1.jsonl --all --strategy reanswer --out results/answer_repair/fever_dev{count}_reanswer.jsonl
python scripts/run_answer_repair.py --dataset nq --c1 inputs/nq_dev{count}_c1.jsonl --all --strategy reanswer --out results/answer_repair/nq_dev{count}_reanswer.jsonl
```

Both strategies use the same 384-token ceiling. The reanswer control receives
the original C1 instruction and evidence; it receives no initial answer.
The optional `--strategy grounded_reanswer` creates a cited answer without
seeing the initial answer and derives keep/repair afterward. Compare it on the
same frozen IDs; it uses the same literal citation checks, not an NLI verifier.
Report fixed, regressed, net gain, invalid outputs, calls, tokens and latency.
Review all changed answers and quote/answer support, including kept answers.
NQ EM/F1 and FEVER accuracy remain the frozen offline metrics.

Samples come only from the intersection of the Diagnosis v1/v2 training sets,
with all validation/test IDs excluded. IDs are ranked by a seeded SHA-256 key
without consulting correctness or labels. Smoke IDs are the first development
IDs, so smoke and development results overlap and must not be pooled as
independent samples. These are exploratory development results, not held-out
final results. GPU verification has not yet been performed in this bundle.

Return environment.json, all result JSONL and .summary.json files, and manual
review notes. manifest.json records frozen IDs, source commits, checksums and
initial metric counts; bundle_checksums.json records every packaged file hash.
See docs/ANSWER_REPAIR.md for the full E protocol and resume rules.
""".encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1-ref", default="origin/feature/-ce-70B")
    parser.add_argument("--split-ref", default="origin/feature/diagnosis-probe-zheng")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--smoke-count", type=int, default=10)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/splits/answer_repair_v1_dev.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/answer_repair/v1")
    parser.add_argument("--bundle-name", default="answer_repair_v1_gpu")
    args = parser.parse_args(argv)
    if not 0 < args.smoke_count <= args.count:
        parser.error("Require 0 < smoke-count <= count")
    if not args.bundle_name.replace("_", "").isalnum():
        parser.error("Bundle name must contain only letters, numbers and underscores")
    commits = {name: subprocess.check_output(["git", "rev-parse", "--verify", ref + "^{commit}"],
                                             cwd=ROOT, text=True).strip()
               for name, ref in (("c1", args.c1_ref), ("split", args.split_ref))}
    split_raw = {path: git_read(commits["split"], path) for path in SPLIT_PATHS}
    split_docs = [json.loads(raw) for raw in split_raw.values()]
    manifest = {
        "version": "answer_repair_v1_dev", "seed": 42,
        "selection": "SHA256(42:dataset:sample_id), shared training IDs only; no labels or correctness",
        "purpose": "exploratory development; smoke is a subset; not held-out evaluation",
        "source_commits": commits,
        "split_sources": {path: sha256(raw) for path, raw in split_raw.items()},
        "datasets": {},
    }
    inputs = {}
    for dataset in ("fever", "nq"):
        ids = select_development_ids(split_docs, dataset, args.count)
        source_path = f"results/preliminary/{dataset}_c1_7b.jsonl"
        raw = git_read(commits["c1"], source_path)
        dev, dev_stats = select_c1(raw, dataset, ids)
        smoke, smoke_stats = select_c1(raw, dataset, ids[:args.smoke_count])
        inputs[f"inputs/{dataset}_dev{args.count}_c1.jsonl"] = dev
        inputs[f"inputs/{dataset}_smoke{args.smoke_count}_c1.jsonl"] = smoke
        manifest["datasets"][dataset] = {
            "sample_ids": ids, "smoke_sample_ids": ids[:args.smoke_count],
            "source_path": source_path, "source_sha256": sha256(raw),
            "development": dev_stats, "smoke": smoke_stats,
        }
    write_frozen(args.manifest, encode_json(manifest))
    staging = args.out_dir / "gpu_bundle"
    files = {**inputs, "manifest.json": encode_json(manifest),
             "bundle_metadata.json": encode_json({"bundle_name": args.bundle_name,
                "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
                "repair_prompt_version": REPAIR_PROMPT_VERSION,
                "grounded_reanswer_prompt_version": GROUNDED_REANSWER_PROMPT_VERSION,
                "json_stop": "complete_object_final_token_framing"}),
             "README_GPU.md": bundle_readme(args.count, args.smoke_count)}
    source_files = [p for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts]
    source_files += [ROOT / path for path in (
        "configs/base.yaml", "requirements-answer-repair.txt", "docs/ANSWER_REPAIR.md",
        "scripts/run_answer_repair.py", "scripts/check_answer_repair_environment.py",
        "scripts/prepare_answer_repair_experiment.py", "tests/test_answer_repair_experiment.py",
        "tests/test_answer_repair.py", "tests/test_diagnosis_schema.py", "tests/__init__.py",
    )]
    files.update({p.relative_to(ROOT).as_posix(): p.read_bytes() for p in source_files})
    files["bundle_checksums.json"] = encode_json({path: sha256(raw) for path, raw in sorted(files.items())})
    archive = args.out_dir / f"{args.bundle_name}.tar.gz"
    if archive.exists():
        with tarfile.open(archive) as handle:
            existing = handle.extractfile(f"{args.bundle_name}/bundle_checksums.json")
            if existing is None or existing.read() != files["bundle_checksums.json"]:
                raise ValueError("Frozen GPU bundle differs; use a new --out-dir and --bundle-name")
    for path, raw in files.items():
        target = staging / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    if not archive.exists():
        temporary = archive.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz") as handle:
            for path in sorted(files):
                handle.add(staging / path, arcname=f"{args.bundle_name}/{path}", recursive=False)
        temporary.replace(archive)
    summary = {"manifest": str(args.manifest), "bundle": str(archive),
               "bundle_sha256": sha256(archive.read_bytes()), "source_commits": commits,
               "cohorts": {ds: {k: v for k, v in item.items() if k in ("development", "smoke")}
                           for ds, item in manifest["datasets"].items()},
               "real_gpu_experiment_completed": False}
    (args.out_dir / "preparation.json").write_bytes(encode_json(summary))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
