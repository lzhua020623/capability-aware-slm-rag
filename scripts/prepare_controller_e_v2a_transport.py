"""Dataset-backed transport; preserve the exact sealed inference archive."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'artifacts/controller/e_v2a_v1/gpu_release_v1'
V1_NAME = 'controller_e_v2a_development_v1'


def dataset_variant(notebook, name=V1_NAME):
    nb = json.loads(json.dumps(notebook))
    code = ''.join(nb['cells'][1]['source'])
    lines = code.splitlines(keepends=True)
    indexes = [i for i, line in enumerate(lines) if line.startswith('raw=base64.b64decode(')]
    if len(indexes) != 1: raise ValueError('Expected exactly one embedded archive loader')
    i = indexes[0]
    lines[i] = (f"matches=[p for p in Path('/kaggle/input').rglob('{name}.tar.gz.bin') if p.is_file()]\n"
                f"if not matches: matches=[p for p in Path('/kaggle/input').rglob('{name}.tar.gz') if p.is_file()]\n"
                "assert len(matches)==1, 'Attach the private frozen E v2a archive dataset'\n"
                "raw=matches[0].read_bytes()\n")
    changed = ''.join(lines)
    if len(changed.encode()) >= 1_000_000: raise ValueError('Kaggle source limit still exceeded')
    compile(changed, '<dataset-transport>', 'exec')
    nb['cells'][1]['source'] = changed.splitlines(keepends=True)
    return nb


def prepare(output, release=RELEASE):
    output = Path(output)
    release = Path(release)
    if output.exists(): raise FileExistsError('Preserve transport receipts')
    receipt = json.loads((release / 'bundle_receipt.json').read_text())
    name = receipt.get('bundle_name', V1_NAME)
    original = release / (name + '.ipynb')
    archive = release / (name + '.tar.gz')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != receipt['archive_sha256']:
        raise ValueError('Frozen archive changed')
    if hashlib.sha256(original.read_bytes()).hexdigest() != receipt['notebook_sha256']:
        raise ValueError('Frozen original notebook changed')
    nb = dataset_variant(json.loads(original.read_text()), name)
    raw = (json.dumps(nb, ensure_ascii=False, indent=1) + '\n').encode()
    output.mkdir(parents=True)
    (output / (name + '_dataset.ipynb')).write_bytes(raw)
    # Opaque extension prevents the dataset uploader from automatically
    # extracting the archive and discarding the original compressed bytes.
    filename = name + '.tar.gz.bin'
    (output / filename).write_bytes(archive.read_bytes())
    info = {'variant': 'dataset_transport_r2_opaque_archive', 'archive_sha256': receipt['archive_sha256'],
            'dataset_filename': filename,
            'original_notebook_sha256': receipt['notebook_sha256'],
            'notebook_sha256': hashlib.sha256(raw).hexdigest(), 'notebook_bytes': len(raw),
            'inference_source_commit': receipt['source_commit'],
            'change': 'only archive loader; same inference archive/model/prompt/engineering gate/budget',
            'reason': 'Kaggle rejected embedded notebook source >1 MB before GPU allocation',
            'GPU_run_before_transport_change': False, 'semantic_retry': False}
    (output / 'transport_receipt.json').write_text(json.dumps(info, indent=2) + '\n')
    return info


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--release-dir', type=Path, default=RELEASE)
    args = p.parse_args()
    print(json.dumps(prepare(args.output_dir, args.release_dir), indent=2))
