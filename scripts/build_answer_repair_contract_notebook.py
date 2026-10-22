"""Freeze a self-contained synthetic-only Kaggle notebook and source archive."""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.import_nq_train_preliminary import write_new


def main():
    folder = ROOT / "artifacts/answer_repair/v9_contract_v1"
    paths = [*sorted((ROOT / "src").rglob("*.py")), ROOT / "configs/base.yaml",
             ROOT / "configs/answer_repair_contract_v1.json", ROOT / "requirements-answer-repair-v8.txt",
             ROOT / "requirements-answer-repair-contract.txt", ROOT / "scripts/import_nq_train_preliminary.py",
             ROOT / "scripts/prepare_answer_repair_contract.py", ROOT / "scripts/run_answer_repair_contract.py",
             folder / "synthetic_cases.jsonl"]
    files = {str(p.relative_to(ROOT)): p.read_bytes() for p in paths}
    checksums = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
    files["bundle_checksums.json"] = json.dumps(checksums, sort_keys=True, indent=2).encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, raw in sorted(files.items()):
                info = tarfile.TarInfo("contract/" + name); info.size = len(raw); info.mtime = 0
                archive.addfile(info, io.BytesIO(raw))
    raw = buffer.getvalue(); digest = hashlib.sha256(raw).hexdigest()
    archive_path = folder / "answer_repair_contract_v1_gpu.tar.gz"
    write_new(archive_path, raw)
    code = f'''from pathlib import Path
import base64, hashlib, io, json, os, subprocess, sys, tarfile, zipfile
raw = base64.b64decode({base64.b64encode(raw).decode()!r}, validate=True)
assert hashlib.sha256(raw).hexdigest() == {digest!r}
base = Path('/kaggle/working/e_contract_v1_' + {digest[:12]!r})
base.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
    for member in archive.getmembers():
        assert member.isfile() and (base / member.name).resolve().is_relative_to(base.resolve())
    archive.extractall(base)
root = base / 'contract'
for name, expected in json.loads((root / 'bundle_checksums.json').read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
os.chdir(root)
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
result = root / 'results'
result.mkdir(exist_ok=True)
(result / 'bootstrap.json').write_text(json.dumps({{'archive_sha256':{digest!r},'synthetic_only':True,'real_benchmark_count':0}}, indent=2))
def run(args):
    with (result / 'run.log').open('a') as log:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line); log.flush(); print(line, end='', flush=True)
        if proc.wait():
            raise RuntimeError('Frozen command failed: ' + repr(args))
try:
    run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-answer-repair-contract.txt'])
    (result / 'pip_freeze.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    run([sys.executable,'scripts/run_answer_repair_contract.py','--check-only','--out',str(result / 'preflight.json')])
    run([sys.executable,'scripts/run_answer_repair_contract.py','--compile-only','--out',str(result / 'cpu_grammar_check.json')])
    run([sys.executable,'scripts/run_answer_repair_contract.py','--out',str(result / 'predictions.jsonl')])
    run([sys.executable,'scripts/run_answer_repair_contract.py','--replay','--out',str(result / 'predictions.jsonl')])
finally:
    with zipfile.ZipFile('/kaggle/working/e_contract_v1_outputs.zip','w',zipfile.ZIP_DEFLATED) as export:
        for path in sorted(root.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                export.write(path,path.relative_to(root))
print('SYNTHETIC CONTRACT RUN COMPLETE. No benchmark gain claim; inspect the predeclared gate.')
'''
    compile(code, "contract_notebook", "exec")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [],
                "isGpuEnabled": True, "isInternetEnabled": True, "language": "python", "sourceType": "notebook"}},
        "cells": [{"id": "protocol", "cell_type": "markdown", "metadata": {}, "source": [
            "# E contract v1: synthetic interface experiment\n",
            "12 fictional cases, 2 structured strategies, prompt-only vs XGrammar. Same schema text/model/budget. "
            "48 predictions; frozen gate requires >=22/24 constrained valid results, zero unknown IDs/keep rewrites and full replay. "
            "No training, retired smoke, real benchmark or human labels. Keep private."]},
            {"id": "run", "cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": code.splitlines(keepends=True)}]}
    notebook_raw = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode()
    write_new(ROOT / "notebooks/answer_repair_contract_v1.ipynb", notebook_raw)
    receipt = {"archive_sha256": digest, "notebook_sha256": hashlib.sha256(notebook_raw).hexdigest(),
               "source_files": len(checksums), "synthetic_cases": 12, "predictions": 48,
               "gpu_run_completed": False, "default_promoted": False}
    write_new(folder / "bundle_manifest.json", (json.dumps(receipt, indent=2) + "\n").encode())
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
