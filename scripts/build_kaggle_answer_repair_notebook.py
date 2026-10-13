"""Adapt notebook transport for Kaggle's automatic archive extraction.

The experiment code and inputs remain in the original frozen bundle. Only
notebook loading changes; no model, strategy, cohort or threshold changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    notebook = json.loads(args.notebook.read_text())
    bundle_name = args.bundle.name.removesuffix('.tar.gz')
    archive_digest = hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    code_cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    assert archive_digest in ''.join(code_cells[0]['source'])
    assert bundle_name in ''.join(code_cells[0]['source'])
    with tarfile.open(args.bundle) as archive:
        index_bytes = archive.extractfile(f'{bundle_name}/bundle_checksums.json').read()
        for name, digest in json.loads(index_bytes).items():
            content = archive.extractfile(f'{bundle_name}/{name}').read()
            assert hashlib.sha256(content).hexdigest() == digest, name
    index_digest = hashlib.sha256(index_bytes).hexdigest()
    loader = f'''# Kaggle may unpack the uploaded archive before attaching it.
EXPECTED_INDEX_SHA256 = {index_digest!r}
archive = WORK / (BUNDLE_NAME + '.tar.gz')
extract_root = WORK / ('e_reviewed_' + EXPECTED_SHA256[:12])
BUNDLE = extract_root / BUNDLE_NAME
transport = 'archive_sha256'
if not archive.exists() and Path('/kaggle/input').exists():
    candidates = list(Path('/kaggle/input').rglob(archive.name))
    if candidates:
        assert len(candidates) == 1, '实验归档必须唯一'
        archive = candidates[0]
    else:
        indexes = [p for p in Path('/kaggle/input').rglob('bundle_checksums.json')
                   if p.parent.name == BUNDLE_NAME]
        assert len(indexes) == 1, '请附加唯一的冻结实验包输入'
        assert hashlib.sha256(indexes[0].read_bytes()).hexdigest() == EXPECTED_INDEX_SHA256
        source = indexes[0].parent
        source_checksums = json.loads(indexes[0].read_text())
        assert all(hashlib.sha256((source / p).read_bytes()).hexdigest() == h
                   for p,h in source_checksums.items()), '已解压输入文件哈希不一致'
        if not BUNDLE.exists():
            shutil.copytree(source, BUNDLE)
        transport = 'pinned_checksum_index_and_all_files'
elif not archive.exists():
    from google.colab import files
    uploaded = files.upload()
    assert archive.name in uploaded, '请选择配套实验包'
    archive.write_bytes(uploaded[archive.name])
if transport == 'archive_sha256':
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == EXPECTED_SHA256
    if not BUNDLE.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as handle:
            handle.extractall(extract_root, filter='data')
assert hashlib.sha256((BUNDLE / 'bundle_checksums.json').read_bytes()).hexdigest() == EXPECTED_INDEX_SHA256
checksums = json.loads((BUNDLE / 'bundle_checksums.json').read_text())
assert all(hashlib.sha256((BUNDLE / p).read_bytes()).hexdigest() == h for p,h in checksums.items())
manifest = json.loads((BUNDLE / 'manifest.json').read_text())
integrity = {{'transport': transport, 'expected_archive_sha256': EXPECTED_SHA256,
             'checksum_index_sha256': EXPECTED_INDEX_SHA256,
             'verified_files': len(checksums), 'all_files_verified': True}}
(OUTPUT / 'input_integrity.json').write_text(json.dumps(integrity, indent=2))
print('输入校验通过:', integrity)
print({{ds: r['groups'] for ds,r in manifest['datasets'].items()}})
'''
    code_cells[1]['source'] = loader.splitlines(keepends=True)
    notebook['metadata']['colab']['name'] = args.out.name
    for cell in code_cells:
        compile(''.join(cell['source']), '<kaggle-cell>', 'exec')
        cell['execution_count'] = None
        cell['outputs'] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'notebook': str(args.out), 'archive_sha256': archive_digest,
                      'checksum_index_sha256': index_digest,
                      'code_cells_syntax_checked': len(code_cells)}, indent=2))


if __name__ == '__main__':
    main()
