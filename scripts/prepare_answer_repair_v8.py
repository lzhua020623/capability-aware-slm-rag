"""Freeze 75 resolved NQ Train development cases, four arms and a Kaggle/Colab notebook."""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_reviewed_answer_repair_notebook import cell
from scripts.prepare_answer_repair_experiment import encode_json, sha256, write_frozen
from scripts.run_answer_repair import prepare_records, read_jsonl
from src.compensation.answer_repair import MODEL_NAME, MODEL_REVISION
from src.compensation.conservative_repair import V8_VERSION
from src.diagnosis import DatasetName

PORTABLE_TESTS = ("tests.test_answer_repair", "tests.test_answer_repair_experiment",
                  "tests.test_answer_repair_v5", "tests.test_answer_repair_v6",
                  "tests.test_answer_repair_v7", "tests.test_answer_repair_v8")


def build_notebook(archive: Path, out: Path):
    digest, name = sha256(archive.read_bytes()), archive.name.removesuffix(".tar.gz")
    cells = [cell("markdown", """# E v8：75 条已定案 NQ Train 开发样本

选择 Kaggle T4 GPU，打开联网，添加配套 tar.gz 为输入，依次运行全部单元。
也支持 Colab T4 上传。四组：v7 对照、v8 完整流程、去原答案核验、去候选核验。
每组同 75 条，共 300 条输出，最多 600 次 Qwen 生成。C1 直接复用原始结果，零新增调用。
11 条 E 目标与 64 条正确对照只用于开发；25 条待定不在包内。同模型核验不是独立裁判。
结束后下载导出的 ZIP；语义收益留待人工审核，不由 EM 或核验通过自动认定。
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
    matches = list(Path('/kaggle/input').rglob(archive.name))
    assert len(matches) == 1, 'Attach exactly one matching tar.gz input'
    archive = matches[0]
elif not archive.exists():
    from google.colab import files
    uploaded = files.upload()
    assert archive.name in uploaded, 'Upload the matching tar.gz'
    archive.write_bytes(uploaded[archive.name])
assert hashlib.sha256(archive.read_bytes()).hexdigest() == EXPECTED_SHA256, 'Archive hash mismatch'
extract_root = WORK / ('e_v8_' + EXPECTED_SHA256[:12])
BUNDLE = extract_root / BUNDLE_NAME
if not BUNDLE.exists():
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        handle.extractall(extract_root, filter='data')
checksums = json.loads((BUNDLE / 'bundle_checksums.json').read_text())
assert all(hashlib.sha256((BUNDLE / p).read_bytes()).hexdigest() == h for p, h in checksums.items())
manifest = json.loads((BUNDLE / 'manifest.json').read_text())
assert manifest['samples'] == 75
for source in ('manifest.json', 'bundle_checksums.json'):
    shutil.copy2(BUNDLE / source, OUTPUT / source)
shutil.copy2(archive, OUTPUT / archive.name)
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

run_logged([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-answer-repair-v8.txt'], 'install.log')
run_logged([sys.executable, '-m', 'pip', 'freeze'], 'packages.log')
run_logged([sys.executable, 'scripts/check_answer_repair_environment.py', '--require-cuda',
            '--out', str(OUTPUT / 'environment.json')], 'environment.log')
run_logged([sys.executable, '-m', 'unittest', *manifest['portable_tests']], 'tests.log')
RUNS = []
for arm in manifest['protocol']:
    strategy = arm['strategy']
    argv = [sys.executable, '-u', 'scripts/run_answer_repair.py', '--dataset', 'nq',
            '--c1', str(BUNDLE / 'inputs/nq75_c1.jsonl'), '--all', '--strategy', strategy,
            '--max-new-tokens', str(arm['max_new_tokens']), '--out', str(OUTPUT / (strategy + '.jsonl'))]
    RUNS.append((strategy, argv))
    run_logged(argv + ['--check-only'], strategy + '_preflight.log')
"""), cell("markdown", """依次完成四组，不根据中途结果改提示或挑选样本。
中断后可在相同环境续跑；已完成的预测会校验并跳过。更换环境需保留旧输出，另开新实验目录。
"""), cell("code", """try:
    for strategy, argv in RUNS:
        run_logged(argv, strategy + '_run.log')
finally:
    # Preserve partial progress even if a backend fails; errors still stop the experiment.
    partial = shutil.make_archive(str(WORK / (OUTPUT.name + '_partial')), 'zip', OUTPUT)
    print('Recovery archive:', partial)
"""), cell("code", """run_logged([sys.executable, 'scripts/summarize_answer_repair_v8.py',
            '--bundle', str(BUNDLE), '--results', str(OUTPUT)], 'comparison.log')
report = json.loads((OUTPUT / 'comparison.json').read_text())
assert report['all_arms_complete'], 'Incomplete experiment; preserve partial output'
export = shutil.make_archive(str(WORK / (OUTPUT.name + '_export')), 'zip', OUTPUT)
print('Result archive:', export)
print('SHA256:', hashlib.sha256(Path(export).read_bytes()).hexdigest())
print('Changed outputs pending review:', report['pending_changed_outputs'])
if not Path('/kaggle/working').exists():
    from google.colab import files
    files.download(export)
""")]
    for index, item in enumerate(cells):
        item["id"] = f"e-v8-nq75-{index:02d}"
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                             "language_info": {"name": "python"}, "accelerator": "GPU"}}
    write_frozen(out, encode_json(notebook))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/answer_repair_v8_nq75_v1.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/answer_repair/v8_nq75")
    parser.add_argument("--bundle-name", default="answer_repair_v8_nq75_gpu")
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/answer_repair_v8_nq75.ipynb")
    args = parser.parse_args(argv)
    if not args.bundle_name.replace("_", "").isalnum():
        raise ValueError("Bundle name must contain letters, numbers and underscores only")
    protocol = json.loads(args.protocol.read_bytes())
    cohort_path = ROOT / protocol["cohort"]["path"]
    if sha256(cohort_path.read_bytes()) != protocol["cohort"]["sha256"]:
        raise ValueError("Frozen cohort hash mismatch")
    cohort = json.loads(cohort_path.read_bytes())
    payload = {}
    for kind, spec in cohort["inputs"].items():
        raw = (ROOT / spec["path"]).read_bytes()
        if sha256(raw) != spec["sha256"]:
            raise ValueError("Frozen input hash mismatch")
        payload[kind] = raw
    prepared = prepare_records(read_jsonl(payload["scored_c1"], "scored input"), DatasetName.NQ)
    runtime = read_jsonl(payload["runtime"], "runtime input")
    if len(prepared) != 75 or [r.sample_id for _, r in prepared] != cohort["sample_ids"]:
        raise ValueError("Expected the ordered, frozen 75-sample cohort")
    if runtime != [json.loads(json.dumps(asdict(r))) for _, r in prepared]:
        raise ValueError("Pure runtime input differs from C1 allowlist")
    if (cohort["calibration_overlap"] or cohort["reserved_evaluation_overlap"] or
            len(cohort["e_core_review_numbers"]) != 11 or len(cohort["correct_control_review_numbers"]) != 64):
        raise ValueError("Unexpected cohort grouping/isolation")
    review_spec = cohort["sources"]["human_reviews"]
    review_raw = (ROOT / review_spec["path"]).read_bytes()
    if sha256(review_raw) != review_spec["sha256"]:
        raise ValueError("Initial human review source hash mismatch")
    reviews = {r["sample_id"]: r for r in read_jsonl(review_raw, "reviews")}
    selected = []
    for number, (_, request) in zip(cohort["review_numbers"], prepared):
        row = reviews[request.sample_id]
        h = row["human_review"]
        group = "include_e_core" if number in cohort["e_core_review_numbers"] else "correct_control"
        if (row["review_number"] != number or not row["human_verified"] or h["status"] != "completed"
                or h["e_development_use"] != group):
            raise ValueError("Cohort differs from completed initial human review")
        selected.append({"sample_id": request.sample_id, "review_number": number, "group": group,
                         "initial_correct": h["initial_correct"], "initial_answer_status": h.get("initial_answer_status"),
                         "reason": h["reason"], "mode": "chat_assisted_not_blind"})
    manifest = {**protocol, "samples": 75, "sample_ids": cohort["sample_ids"],
                "groups": {"include_e_core": 11, "correct_control": 64},
                "excluded_deferred_review_numbers": cohort["excluded_deferred_review_numbers"],
                "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
                "v8_prompt_version": V8_VERSION, "portable_tests": PORTABLE_TESTS,
                "source_input_sha256": {k: sha256(raw) for k, raw in payload.items()},
                "initial_human_review_sha256": sha256(review_raw), "gpu_inference_completed": False}
    files = {"inputs/nq75_c1.jsonl": payload["scored_c1"], "inputs/nq75_runtime.jsonl": payload["runtime"],
             "offline/initial_reviews.jsonl": b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in selected),
             "manifest.json": encode_json(manifest)}
    source_files = [p for folder in ("src", "scripts") for p in (ROOT / folder).rglob("*.py") if "__pycache__" not in p.parts]
    source_files += [ROOT / (name.replace(".", "/") + ".py") for name in PORTABLE_TESTS]
    source_files += [ROOT / p for p in ("tests/__init__.py", "configs/base.yaml", "configs/preliminary.yaml",
                     "requirements-answer-repair.txt", "requirements-answer-repair-v8.txt", "docs/ANSWER_REPAIR_V8.md")]
    files.update({p.relative_to(ROOT).as_posix(): p.read_bytes() for p in source_files})
    files["bundle_checksums.json"] = encode_json({path: sha256(raw) for path, raw in sorted(files.items())})
    archive = args.out_dir / (args.bundle_name + ".tar.gz")
    if archive.exists():
        with tarfile.open(archive) as handle:
            if handle.extractfile(args.bundle_name + "/bundle_checksums.json").read() != files["bundle_checksums.json"]:
                raise ValueError("Frozen bundle differs; choose a new output directory and bundle name")
    for path, raw in files.items():
        write_frozen(args.out_dir / "gpu_bundle" / path, raw)
    if not archive.exists():
        temporary = archive.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz") as handle:
            for path in sorted(files):
                handle.add(args.out_dir / "gpu_bundle" / path, arcname=args.bundle_name + "/" + path, recursive=False)
        temporary.replace(archive)
    build_notebook(archive, args.notebook)
    write_frozen(args.out_dir / "manifest.json", encode_json(manifest))
    summary = {"bundle": str(archive.relative_to(ROOT)) if archive.is_relative_to(ROOT) else str(archive),
               "bundle_sha256": sha256(archive.read_bytes()), "files": len(files),
               "samples_per_arm": 75, "predictions_expected": 300, "generation_call_upper_bound": 600,
               "notebook": str(args.notebook.relative_to(ROOT)) if args.notebook.is_relative_to(ROOT) else str(args.notebook),
               "gpu_inference_completed": False}
    write_frozen(args.out_dir / "preparation.json", encode_json(summary))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
