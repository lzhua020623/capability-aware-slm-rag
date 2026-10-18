"""Repair the v8 notebook's transport only; preserve the original frozen GPU bundle."""
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

from scripts.prepare_answer_repair_experiment import encode_json, write_frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "artifacts/answer_repair/v8_nq75/answer_repair_v8_nq75_gpu.tar.gz")
    parser.add_argument("--source-notebook", type=Path, default=ROOT / "notebooks/answer_repair_v8_nq75.ipynb")
    parser.add_argument("--out", type=Path, default=ROOT / "notebooks/answer_repair_v8_nq75_transport_r1.ipynb")
    args = parser.parse_args(argv)
    notebook = json.loads(args.source_notebook.read_bytes())
    digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    name = args.archive.name.removesuffix(".tar.gz")
    with tarfile.open(args.archive) as handle:
        checksum_raw = handle.extractfile(name + "/bundle_checksums.json").read()
    checksum_sha = hashlib.sha256(checksum_raw).hexdigest()
    loader_raw = (ROOT / "scripts/load_answer_repair_bundle.py").read_text()
    bootstrap = loader_raw + f'''\n# Transport r1; original frozen payload and protocol are unchanged.
EXPECTED_CHECKSUMS_SHA256 = {checksum_sha!r}
if not Path('/kaggle/input').exists() and not (WORK / (BUNDLE_NAME + '.tar.gz')).exists():
    from google.colab import files
    uploaded = files.upload()
    upload_name = BUNDLE_NAME + '.tar.gz'
    assert upload_name in uploaded, 'Upload the matching tar.gz'
    (WORK / upload_name).write_bytes(uploaded[upload_name])
try:
    BUNDLE, archive, bootstrap_report = load_frozen_bundle(
        work=WORK, input_root=Path('/kaggle/input'), bundle_name=BUNDLE_NAME,
        archive_sha256=EXPECTED_SHA256, checksums_sha256=EXPECTED_CHECKSUMS_SHA256)
except Exception as exc:
    (OUTPUT / 'bootstrap_error.json').write_text(json.dumps({{'error': str(exc)}}) + '\\n')
    raise
(OUTPUT / 'bootstrap.json').write_text(json.dumps(bootstrap_report, indent=2) + '\\n')
manifest = json.loads((BUNDLE / 'manifest.json').read_text())
assert manifest['samples'] == 75
for source in ('manifest.json', 'bundle_checksums.json'):
    shutil.copy2(BUNDLE / source, OUTPUT / source)
if archive is not None:
    shutil.copy2(archive, OUTPUT / archive.name)
print('Bundle verified:', bootstrap_report)
'''
    notebook["cells"][2]["source"] = bootstrap.splitlines(keepends=True)
    notebook["cells"][0]["source"] += ["\n\nTransport r1：兼容 Kaggle 自动解压输入；原 906f096f 冻结包、75 条样本和四组算法完全相同。\n"]
    notebook["metadata"]["v8_transport_revision"] = "r1_auto_extracted_input"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<v8-transport-r1-cell>", "exec")
            cell.update(execution_count=None, outputs=[])
    write_frozen(args.out, encode_json(notebook))
    report = {"version": "v8_transport_r1", "archive_sha256": digest,
              "frozen_checksums_sha256": checksum_sha,
              "bootstrap_source_sha256": hashlib.sha256(loader_raw.encode()).hexdigest(),
              "notebook": str(args.out.relative_to(ROOT)),
              "notebook_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
              "unchanged": ["75 sample IDs", "runtime evidence", "initial answers", "gold and scorer", "all four algorithm arms", "Qwen revision and generation budgets"],
              "reason": "Kaggle automatically extracted the tar.gz; version 1 failed before inference while searching for the original filename."}
    write_frozen(ROOT / "artifacts/answer_repair/v8_nq75/transport_r1.json", encode_json(report))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
