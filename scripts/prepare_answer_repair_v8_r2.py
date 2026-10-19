"""Freeze a smoke revision, preserving every previous package and result."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_answer_repair_experiment import encode_json, write_frozen
from scripts.run_answer_repair import prepare_records, read_jsonl
from src.compensation.conservative_repair import V8_VERSION
from src.diagnosis import DatasetName

OVERLAYS = (
    "src/compensation/answer_repair.py", "src/compensation/qwen_repair.py",
    "src/compensation/conservative_repair.py", "src/compensation/line_contract.py",
    "scripts/run_answer_repair.py", "scripts/summarize_answer_repair_v8.py",
    "scripts/audit_answer_repair_release.py", "scripts/check_answer_repair_v8_smoke.py",
    "scripts/prepare_answer_repair_v8_r2.py", "scripts/load_answer_repair_bundle.py",
    "tests/test_answer_repair_v8_r2.py", "tests/fixtures/answer_repair_v8r1_contract_cases.json",
    "docs/ANSWER_REPAIR_V8_R2.md", "configs/answer_repair_v8r2_smoke8_v1.json",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def cell(kind, text):
    result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        compile(text, "<v8r2-notebook-cell>", "exec")
        result.update(execution_count=None, outputs=[])
    return result


def notebook(archive: Path, files: dict[str, bytes], *, archive_raw: bytes | None = None,
             revision: str = "r2") -> bytes:
    digest, name = sha(archive.read_bytes() if archive_raw is None else archive_raw), archive.name.removesuffix(".tar.gz")
    loader = files["scripts/load_answer_repair_bundle.py"].decode()
    version = f"answer_repair_v8{revision}_smoke8_v1"
    cells = [cell("markdown", "# E v8" + revision + "：8 条真实模型工程验收\n" + """

Kaggle 选择 T4 GPU 并打开联网，添加配套私有数据包后运行全部单元。
四组各 8 条，32 条输出，最多 64 次生成。模型和 Top-5 来源保持原实验设置。
这是已用于开发的样本；smoke_acceptance 只验收输出格式、来源和阶段覆盖，不证明语义提升。
原 75 条失败实验保留；本 Notebook 不会自动启动新的 75 条完整实验。
"""), cell("code", f"""import json, os, sys, subprocess, hashlib, shutil
from pathlib import Path
import torch
assert sys.version_info >= (3, 11)
assert torch.cuda.is_available(), 'Select a GPU runtime'
WORK = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/content')
BUNDLE_NAME = {name!r}
EXPECTED_SHA256 = {digest!r}
EXPECTED_CHECKSUMS_SHA256 = {sha(files['bundle_checksums.json'])!r}
OUTPUT = WORK / (BUNDLE_NAME.removesuffix('_gpu') + '_outputs')
OUTPUT.mkdir(parents=True, exist_ok=True)
print('GPU:', torch.cuda.get_device_name(0), 'CUDA:', torch.version.cuda)
"""), cell("code", loader + r"""
if not Path('/kaggle/input').exists() and not (WORK / (BUNDLE_NAME + '.tar.gz')).exists():
    from google.colab import files
    uploaded = files.upload()
    upload_name = BUNDLE_NAME + '.tar.gz'
    assert upload_name in uploaded
    (WORK / upload_name).write_bytes(uploaded[upload_name])
BUNDLE, archive, bootstrap = load_frozen_bundle(
    work=WORK, input_root=Path('/kaggle/input'), bundle_name=BUNDLE_NAME,
    archive_sha256=EXPECTED_SHA256, checksums_sha256=EXPECTED_CHECKSUMS_SHA256)
(OUTPUT / 'bootstrap.json').write_text(json.dumps(bootstrap, indent=2) + '\n')
manifest = json.loads((BUNDLE / 'manifest.json').read_text())
assert manifest['version'] == 'EXPECTED_VERSION_LITERAL' and manifest['samples'] == 8
for name in ('manifest.json', 'bundle_checksums.json'):
    shutil.copy2(BUNDLE / name, OUTPUT / name)
print('Verified bundle:', bootstrap, manifest['review_numbers'])
""".replace("EXPECTED_VERSION_LITERAL", version)), cell("code", r"""def run_logged(argv, name):
    with (OUTPUT / name).open('a', encoding='utf-8') as log:
        log.write('COMMAND ' + json.dumps(argv) + '\n'); log.flush()
        process = subprocess.Popen(argv, cwd=BUNDLE, env={**os.environ, 'PYTHONUNBUFFERED': '1'},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
    if code: raise subprocess.CalledProcessError(code, argv)

run_logged([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-answer-repair-v8.txt'], 'install.log')
run_logged([sys.executable, '-m', 'pip', 'freeze'], 'packages.log')
run_logged([sys.executable, 'scripts/check_answer_repair_environment.py', '--require-cuda',
            '--out', str(OUTPUT / 'environment.json')], 'environment.log')
run_logged([sys.executable, '-m', 'unittest', *manifest['portable_tests']], 'tests.log')
RUNS = []
for arm in manifest['protocol']:
    strategy = arm['strategy']
    argv = [sys.executable, '-u', 'scripts/run_answer_repair.py', '--dataset', 'nq',
        '--c1', str(BUNDLE / manifest['scored_input']), '--all', '--strategy', strategy,
        '--max-new-tokens', str(arm['max_new_tokens']), '--out', str(OUTPUT / (strategy + '.jsonl'))]
    run_logged(argv + ['--check-only'], strategy + '_preflight.log')
    RUNS.append((strategy, argv))
"""), cell("code", """try:
    for strategy, argv in RUNS:
        run_logged(argv, strategy + '_run.log')
finally:
    partial = shutil.make_archive(str(WORK / (OUTPUT.name + '_partial')), 'zip', OUTPUT)
    print('Recovery archive:', partial)
"""), cell("code", """run_logged([sys.executable, 'scripts/check_answer_repair_v8_smoke.py',
            '--bundle', str(BUNDLE), '--results', str(OUTPUT)], 'smoke_check.log')
report = json.loads((OUTPUT / 'smoke_acceptance.json').read_text())
export = shutil.make_archive(str(WORK / (OUTPUT.name + '_export')), 'zip', OUTPUT)
print('Result ZIP:', export)
print('SHA256:', hashlib.sha256(Path(export).read_bytes()).hexdigest())
print('Engineering gate passed:', report['engineering_gate_passed'])
print('Failures:', report['failures'])
print('Semantic acceptance: pending human review. No full run launched.')
if not Path('/kaggle/working').exists():
    from google.colab import files
    files.download(export)
""")]
    for i, c in enumerate(cells):
        c["id"] = f"e-v8{revision}-smoke-{i:02d}"
    return encode_json({"nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU"}})


def main(argv=None, *, revision="r2"):
    if revision not in ("r2", "r3"):
        raise ValueError("Unknown engineering revision")
    if not V8_VERSION.startswith("v8" + revision + "_"):
        raise ValueError("Use the builder matching the current source revision; old packages remain frozen")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-archive", type=Path,
        default=ROOT / "artifacts/answer_repair/v8_nq75/answer_repair_v8_nq75_gpu.tar.gz")
    parser.add_argument("--protocol", type=Path, default=ROOT / f"configs/answer_repair_v8{revision}_smoke8_v1.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / f"artifacts/answer_repair/v8{revision}_smoke8")
    parser.add_argument("--notebook", type=Path, default=ROOT / f"notebooks/answer_repair_v8{revision}_smoke8.ipynb")
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_bytes())
    if protocol["version"] != f"answer_repair_v8{revision}_smoke8_v1":
        raise ValueError("This builder freezes only the predefined smoke protocol")
    if args.out_dir.resolve() == args.parent_archive.parent.resolve():
        raise ValueError("Use a new revision directory")
    parent_archive_raw = args.parent_archive.read_bytes()
    if sha(parent_archive_raw) != protocol["parent_bundle_sha256"]:
        raise ValueError("Parent frozen archive hash mismatch")
    base_name = args.parent_archive.name.removesuffix(".tar.gz")
    with tarfile.open(fileobj=io.BytesIO(parent_archive_raw)) as handle:
        checks = json.loads(handle.extractfile(base_name + "/bundle_checksums.json").read())
        files = {p: handle.extractfile(base_name + "/" + p).read() for p in checks}
    if any(sha(files[p]) != h for p, h in checks.items()):
        raise ValueError("Parent payload hash mismatch")
    parent = json.loads(files["manifest.json"])
    def lines(path):
        return {json.loads(raw)["sample_id"]: raw for raw in files[path].splitlines(keepends=True)}
    source_lines, runtime_lines, human_lines = (lines(p) for p in
        ("inputs/nq75_c1.jsonl", "inputs/nq75_runtime.jsonl", "offline/initial_reviews.jsonl"))
    by_number = {json.loads(raw)["review_number"]: sample_id for sample_id, raw in human_lines.items()}
    ids = [by_number[n] for n in protocol["review_numbers"]]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError("Expected eight unique smoke cases")
    source_raw = b"".join(source_lines[i] for i in ids)
    runtime_raw = b"".join(runtime_lines[i] for i in ids)
    human_raw = b"".join(human_lines[i] for i in ids)
    prepared = prepare_records(read_jsonl(source_raw, "smoke C1"), DatasetName.NQ)
    if read_jsonl(runtime_raw, "runtime") != [json.loads(json.dumps(asdict(r))) for _, r in prepared]:
        raise ValueError("Runtime input does not match the C1 allowlist")
    human = read_jsonl(human_raw, "initial reviews")
    groups = {g: sum(h["group"] == g for h in human) for g in ("include_e_core", "correct_control")}
    if groups != {"include_e_core": 4, "correct_control": 4}:
        raise ValueError("Expected four targets and four correct controls")
    if protocol["protocol"] != parent["protocol"]:
        raise ValueError("The four arm configurations must remain unchanged")
    portable_tests = list(parent["portable_tests"]) + ["tests.test_answer_repair_v8_r2"]
    overlays = OVERLAYS
    if revision == "r3":
        portable_tests += ["tests.test_answer_repair_v8_r3"]
        overlays += ("scripts/prepare_answer_repair_v8_r3.py", "tests/test_answer_repair_v8_r3.py",
                     "tests/fixtures/answer_repair_v8r2_candidate_cases.json", "docs/ANSWER_REPAIR_V8_R3.md",
                     "configs/answer_repair_v8r3_smoke8_v1.json")
    manifest = {**protocol, "samples": 8, "sample_ids": ids, "groups": groups,
        "scored_input": "inputs/nq_smoke8_c1.jsonl", "runtime_input": "inputs/nq_smoke8_runtime.jsonl",
        "model_name": parent["model_name"], "model_revision": parent["model_revision"],
        "v8_prompt_version": V8_VERSION, "portable_tests": portable_tests,
        "source_input_sha256": {"scored_c1": sha(source_raw), "runtime": sha(runtime_raw)},
        "initial_human_review_sha256": sha(human_raw), "parent_source_input_sha256": parent["source_input_sha256"],
        "source_row_sha256": {i: sha(source_lines[i]) for i in ids},
        "payload_policy": "Verified original frozen source plus explicitly listed revision overlays",
        "overlay_paths": list(overlays), "gpu_inference_completed": False}
    for p in ("inputs/nq75_c1.jsonl", "inputs/nq75_runtime.jsonl"):
        del files[p]
    files.update({p: (ROOT / p).read_bytes() for p in overlays})
    files.update({manifest["scored_input"]: source_raw, manifest["runtime_input"]: runtime_raw,
                  "offline/initial_reviews.jsonl": human_raw, "manifest.json": encode_json(manifest)})
    files["bundle_checksums.json"] = encode_json({p: sha(raw) for p, raw in sorted(files.items())})
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as handle:
            for p, raw in sorted(files.items()):
                item = tarfile.TarInfo(f"answer_repair_v8{revision}_smoke8_gpu/" + p)
                item.size, item.mode = len(raw), 0o644
                handle.addfile(item, io.BytesIO(raw))
    archive = args.out_dir / f"answer_repair_v8{revision}_smoke8_gpu.tar.gz"
    archive_raw = buffer.getvalue()
    notebook_raw = notebook(archive, files, archive_raw=archive_raw, revision=revision)
    write_frozen(archive, archive_raw)
    write_frozen(args.notebook, notebook_raw)
    write_frozen(args.out_dir / "manifest.json", encode_json(manifest))
    preparation = {"version": protocol["version"], "parent_bundle_sha256": sha(parent_archive_raw),
        "bundle_sha256": sha(archive.read_bytes()), "checksums_sha256": sha(files["bundle_checksums.json"]),
        "notebook_sha256": sha(args.notebook.read_bytes()), "samples_per_arm": 8,
        "predictions_expected": 32, "generation_call_upper_bound": 64, "files": len(files),
        "review_numbers": protocol["review_numbers"], "gpu_inference_completed": False}
    write_frozen(args.out_dir / "preparation.json", encode_json(preparation))
    print(json.dumps(preparation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
