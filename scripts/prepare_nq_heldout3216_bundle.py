"""Freeze the run-once NQ held-out (Diagnosis 3216 pool) as its own GPU bundle and notebook.

The cohort is label-stratified from Diagnosis v2-expanded TRAIN IDs outside the
preliminary 500 (configs/splits/answer_repair_nq_heldout_3216_v1.json). E has
never seen these samples; the v7 strategy and prompt were frozen before this
bundle existed. Run once, report by stratum, no tuning afterwards.

Usage:
  python scripts/prepare_nq_heldout3216_bundle.py \
      --c1 artifacts/answer_repair/v7_reviewed/inputs_extra/nq_heldout3216_c1.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_reviewed_answer_repair_notebook import cell
from scripts.prepare_answer_repair_experiment import encode_json, sha256, write_frozen
from scripts.run_answer_repair import prepare_records, read_jsonl
from src.compensation.short_answer_repair import SHORT_ANSWER_PROMPT_VERSION
from src.diagnosis import DatasetName

PROTOCOL = [{"dataset": "nq", "strategy": "short_answer_quote", "max_new_tokens": 384, "max_calls": 1}]


def build_notebook(bundle: Path, out: Path) -> None:
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    name = bundle.name.removesuffix(".tar.gz")
    cells = [cell("markdown", """# E v7 held-out: NQ Diagnosis-3216 pool, run once

300 NQ samples E has never seen (120 utilisation, 35 accessibility, 100 correct, 45 insufficiency
by offline annotation; labels are not in the bundle). Strategy `short_answer_quote`, frozen in
docs/ANSWER_REPAIR_V7.md before this cohort existed. One Qwen NF4 call per sample, 384 tokens.
Run every cell once, download the ZIP, and report the result whatever it is. No reruns, no tuning.
"""), cell("code", f"""import json, os, sys, subprocess, hashlib, tarfile, shutil
from pathlib import Path
import torch
assert sys.version_info >= (3, 11), 'Python 3.11+ required'
assert torch.cuda.is_available(), 'Select a GPU runtime'
WORK = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/content')
EXPECTED_SHA256 = {digest!r}
BUNDLE_NAME = {name!r}
OUTPUT = WORK / (BUNDLE_NAME.removesuffix('_gpu') + '_outputs')
OUTPUT.mkdir(parents=True, exist_ok=True)
print('GPU:', torch.cuda.get_device_name(0), 'CUDA:', torch.version.cuda)
"""), cell("code", """archive = WORK / (BUNDLE_NAME + '.tar.gz')
if not archive.exists() and Path('/kaggle/input').exists():
    candidates = list(Path('/kaggle/input').rglob(archive.name))
    assert len(candidates) == 1, 'Attach exactly one bundle input'
    archive = candidates[0]
elif not archive.exists():
    from google.colab import files
    uploaded = files.upload()
    assert archive.name in uploaded, 'Upload the matching bundle'
    archive.write_bytes(uploaded[archive.name])
assert hashlib.sha256(archive.read_bytes()).hexdigest() == EXPECTED_SHA256, 'Archive hash mismatch'
extract_root = WORK / ('e_heldout_' + EXPECTED_SHA256[:12])
BUNDLE = extract_root / BUNDLE_NAME
if not BUNDLE.exists():
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        handle.extractall(extract_root, filter='data')
checksums = json.loads((BUNDLE / 'bundle_checksums.json').read_text())
assert all(hashlib.sha256((BUNDLE / p).read_bytes()).hexdigest() == h for p, h in checksums.items())
manifest = json.loads((BUNDLE / 'manifest.json').read_text())
print('Bundle verified:', len(checksums), 'files;', manifest['samples'], 'samples')
"""), cell("code", """def run_logged(argv, name):
    with (OUTPUT / name).open('a', encoding='utf-8') as log:
        log.write('COMMAND ' + json.dumps(argv) + '\\n'); log.flush()
        process = subprocess.Popen(argv, cwd=BUNDLE, env={**os.environ, 'PYTHONUNBUFFERED': '1'},
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            status = process.wait()
        except BaseException:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
    if status: raise subprocess.CalledProcessError(status, argv)

run_logged([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-answer-repair.txt'], 'install.log')
run_logged([sys.executable, 'scripts/check_answer_repair_environment.py', '--require-cuda',
            '--out', str(OUTPUT / 'environment.json')], 'environment.log')
run_logged([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'], 'tests.log')
for path in ('manifest.json', 'bundle_checksums.json'):
    shutil.copy2(BUNDLE / path, OUTPUT / path)
arm = manifest['protocol'][0]
RESULT = OUTPUT / 'nq_short_answer_quote_heldout3216.jsonl'
ARGV = [sys.executable, '-u', 'scripts/run_answer_repair.py', '--dataset', 'nq',
        '--c1', str(BUNDLE / 'inputs/nq_heldout3216_c1.jsonl'), '--all',
        '--strategy', arm['strategy'], '--max-new-tokens', str(arm['max_new_tokens']), '--out', str(RESULT)]
run_logged(ARGV + ['--check-only'], 'nq_short_answer_quote_heldout3216_preflight.log')
"""), cell("markdown", "## Run once\nResumable within the same runtime if interrupted; never rerun after completion.\n"),
        cell("code", """run_logged(ARGV, 'nq_short_answer_quote_heldout3216_run.log')
"""), cell("code", """export = shutil.make_archive(str(WORK / (OUTPUT.name + '_export')), 'zip', OUTPUT)
print('Result archive:', export)
print('SHA256:', hashlib.sha256(Path(export).read_bytes()).hexdigest())
if not Path('/kaggle/working').exists():
    from google.colab import files
    files.download(export)
""")]
    for i, c in enumerate(cells):
        c["id"] = f"e-heldout3216-{i:02d}"
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                             "language_info": {"name": "python"}, "accelerator": "GPU",
                             "colab": {"name": out.name}}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1", type=Path, required=True)
    parser.add_argument("--ids", type=Path, default=ROOT / "configs/splits/answer_repair_nq_heldout_3216_v1.json")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/annotations/gold/nq_heldout_3216_v1_gold.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/answer_repair/v7_heldout3216")
    parser.add_argument("--bundle-name", default="answer_repair_v7_heldout3216_gpu")
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/answer_repair_colab_v7_heldout3216.ipynb")
    args = parser.parse_args(argv)
    ids_doc = json.loads(args.ids.read_bytes())
    raw = args.c1.read_bytes()
    prepared = prepare_records(read_jsonl(raw, str(args.c1)), DatasetName.NQ)
    if [r.sample_id for _, r in prepared] != ids_doc["sample_ids"]:
        raise ValueError("C1 input order differs from the frozen ID manifest")
    correct = sum(s["correct"] for s, _ in prepared)
    manifest = {"version": "answer_repair_v7_heldout3216", "cohort": "heldout3216", "dataset": "nq",
                "purpose": ids_doc["purpose"], "ids_manifest": {"path": str(args.ids.relative_to(ROOT)), "sha256": sha256(args.ids.read_bytes())},
                "gold_source": {"path": str(args.gold.relative_to(ROOT)), "sha256": sha256(args.gold.read_bytes())},
                "samples": len(prepared), "frozen_metric_baseline": {"initial_correct": correct, "initial_wrong": len(prepared) - correct},
                "strata_sizes": {k: v["selected"] for k, v in ids_doc["strata"].items()},
                "protocol": PROTOCOL, "prompt_version": SHORT_ANSWER_PROMPT_VERSION,
                "run_policy": "run once; report by stratum using offline labels after inference; no tuning",
                "input_sha256": sha256(raw)}
    files = {"inputs/nq_heldout3216_c1.jsonl": raw, "manifest.json": encode_json(manifest)}
    source_files = [p for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts]
    source_files += [p for p in (ROOT / "tests").glob("*.py")]
    source_files += [ROOT / p for p in (
        "scripts/run_answer_repair.py", "scripts/check_answer_repair_environment.py",
        "scripts/prepare_answer_repair_experiment.py", "scripts/prepare_answer_repair_review.py",
        "scripts/prepare_reviewed_answer_repair.py", "scripts/summarize_reviewed_answer_repair.py",
        "scripts/audit_answer_repair_release.py", "scripts/build_nq_heldout_from_diagnosis_inputs.py",
        "scripts/prepare_nq_heldout3216_bundle.py", "scripts/build_reviewed_answer_repair_notebook.py",
        "scripts/apply_answer_repair_confirmations.py",
        "configs/base.yaml", "requirements-answer-repair.txt", "docs/ANSWER_REPAIR_V7.md",
    )]
    files.update({p.relative_to(ROOT).as_posix(): p.read_bytes() for p in source_files})
    files["bundle_checksums.json"] = encode_json({p: sha256(r) for p, r in sorted(files.items())})
    staging = args.out_dir / "gpu_bundle"
    archive = args.out_dir / f"{args.bundle_name}.tar.gz"
    if archive.exists():
        with tarfile.open(archive) as handle:
            if handle.extractfile(f"{args.bundle_name}/bundle_checksums.json").read() != files["bundle_checksums.json"]:
                raise ValueError("Frozen bundle differs; choose a new output directory and bundle name")
    for path, content in files.items():
        write_frozen(staging / path, content)
    if not archive.exists():
        temporary = archive.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz") as handle:
            for path in sorted(files):
                handle.add(staging / path, arcname=f"{args.bundle_name}/{path}", recursive=False)
        temporary.replace(archive)
    build_notebook(archive, args.notebook)
    summary = {"bundle": str(archive), "bundle_sha256": sha256(archive.read_bytes()), "notebook": str(args.notebook),
               "samples": len(prepared), "baseline": manifest["frozen_metric_baseline"], "real_gpu_experiment_completed": False}
    write_frozen(args.out_dir / "preparation.json", encode_json(summary))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
