"""Register the pre-generation import fix without changing frozen C1 inputs."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controller.fever_c1 import ROLES, verify_collection
from src.retrieval.offset_corpus import file_sha

IMPORT_FIX = 'src/retrieval/__init__.py'


def prepare(parent, output, *, root=ROOT):
    parent, output, root = map(Path, (parent, output, root))
    if output.exists():
        raise FileExistsError('Preserve existing collection and retry registration')
    original = json.loads((parent/'manifest.json').read_text())
    changed = {p: {'before': digest, 'after': file_sha(root/p)}
               for p, digest in original['source_files'].items()
               if file_sha(root/p) != digest}
    if set(changed) != {IMPORT_FIX}:
        raise ValueError('Transport r1 permits only the retrieval package import fix')
    for role in ROLES:
        info = original['files'][role]
        if info['path'] != f'{role}/retrieval.jsonl' or file_sha(parent/info['path']) != info['sha256']:
            raise ValueError('Original frozen retrieval changed')
    revised = copy.deepcopy(original)
    revised['source_files'][IMPORT_FIX] = changed[IMPORT_FIX]['after']
    revised['transport_revision'] = {
        'name': 'import_isolation_r1',
        'parent_manifest_sha256': file_sha(parent/'manifest.json'),
        'source_delta': changed,
        'reason': 'Kaggle v1 failed importing faiss during CPU preflight; no Qwen call or C1 answer',
        'semantic_inputs_unchanged': True,
        'retry_number': 1,
    }
    output.mkdir(parents=True)
    for role in ROLES:
        relative = original['files'][role]['path']
        (output/relative).parent.mkdir(parents=True)
        shutil.copyfile(parent/relative, output/relative)
    (output/'manifest.json').write_text(json.dumps(revised, ensure_ascii=False, indent=2)+'\n')
    verify_collection(output, root=root)
    return revised['transport_revision']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.parent, args.output), indent=2))


if __name__ == '__main__':
    main()
