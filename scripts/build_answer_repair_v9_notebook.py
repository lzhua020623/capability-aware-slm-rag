"""Embed a frozen small GPU package in a private Kaggle engineering notebook."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.archive.read_bytes()
    archive_hash = hashlib.sha256(raw).hexdigest()
    with tarfile.open(args.archive) as tar:
        members = [m for m in tar.getmembers() if m.name.endswith("/bundle_checksums.json")]
        if len(members) != 1:
            raise ValueError("Expected exactly one frozen package")
        checksums_hash = hashlib.sha256(tar.extractfile(members[0]).read()).hexdigest()
    code = f'''from pathlib import Path
import base64, hashlib, io, json, os, subprocess, sys, tarfile
EXPECTED_ARCHIVE = {archive_hash!r}
EXPECTED_CHECKSUMS = {checksums_hash!r}
raw = base64.b64decode({base64.b64encode(raw).decode()!r}, validate=True)
assert hashlib.sha256(raw).hexdigest() == EXPECTED_ARCHIVE
extract = Path("/kaggle/working/v9_embedded_" + EXPECTED_CHECKSUMS[:12])
extract.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
    for member in tar.getmembers():
        assert member.isfile() and (extract / member.name).resolve().is_relative_to(extract.resolve())
    tar.extractall(extract)
paths = list(extract.rglob("bundle_checksums.json"))
assert len(paths) == 1
assert hashlib.sha256(paths[0].read_bytes()).hexdigest() == EXPECTED_CHECKSUMS
root = paths[0].parent
for name, expected in json.loads(paths[0].read_text()).items():
    path = (root / name).resolve()
    assert path.is_relative_to(root.resolve()) and hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
os.chdir(root)
(root / "bootstrap.json").write_text(json.dumps({{"expected_archive_sha256":EXPECTED_ARCHIVE,"expected_checksums_sha256":EXPECTED_CHECKSUMS,"purpose":"10 exposed development samples; engineering only, no EM selection"}}, indent=2))
subprocess.run([sys.executable,"-m","pip","install","-r","requirements-answer-repair.txt"],check=True)
subprocess.run(["sh","RUN_SMOKE.sh"],check=True)
outputs = list((root / "results").glob("*.jsonl"))
assert len(outputs) == 6 and all(len(p.read_text().splitlines()) == 5 for p in outputs)
print("V9 ENGINEERING SMOKE COMPLETE: 30 predictions, 6 replay checks; no effectiveness claim.")
'''
    compile(code, "v9_smoke", "exec")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "dockerImageVersionId": None,
                   "isGpuEnabled": True, "isInternetEnabled": True, "language": "python", "sourceType": "notebook"}},
        "cells": [
            {"id": "protocol", "cell_type": "markdown", "metadata": {}, "source": [
                "# E v9 compact context engineering smoke\n",
                "Private frozen input embedded in this notebook. Full Top-5 retained. Same Qwen, calls and samples; compact evidence serialization only. No effectiveness or prompt selection claim."]},
            {"id": "run", "cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": code.splitlines(keepends=True)}]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode()
    if args.out.exists() and args.out.read_bytes() != encoded:
        raise ValueError("Refusing to overwrite another notebook snapshot")
    args.out.write_bytes(encoded)
    print(json.dumps({"archive_sha256": archive_hash, "checksums_sha256": checksums_hash,
                      "notebook_bytes": len(encoded), "out": str(args.out)}))


if __name__ == "__main__":
    main()
