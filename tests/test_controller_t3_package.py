"""Development-only transport, complete input seal and no QA references in GPU payload."""
from copy import deepcopy
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import subprocess
from unittest.mock import patch

from scripts.prepare_controller_t3_gpu import prepare, notebook, read_source_commit, ROOT
from scripts.run_controller_t4 import verify_manifest


class T3PackageTests(unittest.TestCase):
    def test_real_package_preserves_development_inputs_excludes_references_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'package'
            with patch('scripts.prepare_controller_t3_gpu.read_source_commit', return_value='0' * 40):
                receipt = prepare(out)
            manifest = json.loads((out / 'execution_manifest.json').read_text())
            self.assertEqual(manifest['datasets']['nq']['n'], 200)
            self.assertEqual(manifest['datasets']['fever']['n'], 200)
            self.assertFalse(manifest['contains_gold'] or manifest['contains_selection_or_final_inputs'])
            archive = Path(receipt['archive'])
            with tarfile.open(archive) as handle:
                names = handle.getnames()
                self.assertFalse(any('/offline/' in n or n.endswith('/gold.jsonl') for n in names))
                self.assertFalse(any('selection_runtime' in n or 'final_candidate_runtime' in n for n in names))
                handle.extractall(Path(directory) / 'unpack', filter='data')
            root = Path(directory) / 'unpack/controller_t3_gpu_v1'
            inputs, _ = verify_manifest(root, manifest)
            self.assertEqual(set(inputs), {'nq', 'fever'})
            altered = deepcopy(manifest)
            altered['datasets']['nq']['input']['path'] = '../unregistered'
            with self.assertRaises(ValueError):
                verify_manifest(root, altered)
            (root / 'configs/base.yaml').write_text('changed')
            with self.assertRaises(ValueError):
                verify_manifest(root, manifest)
            with self.assertRaises(FileExistsError), patch(
                    'scripts.prepare_controller_t3_gpu.read_source_commit', return_value='0' * 40):
                prepare(out)

    def test_notebook_cells_compile_and_T4_requires_explicit_receipt_and_resources(self):
        receipt = {'checksum_index_sha256': 'a' * 64, 'archive_sha256': 'b' * 64}
        for value in (notebook(receipt), notebook(receipt, engineering_only=False,
                     engineering_sha='c' * 64, resource_receipt={'sha256': 'd' * 64, 'files': {}})):
            for cell in value['cells']:
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']), '<frozen-notebook>', 'exec')
        with self.assertRaises(ValueError):
            notebook(receipt, engineering_only=False)

    def test_sealer_rejects_staged_code_but_allows_unrelated_annotation_drafts(self):
        name = subprocess.check_output(['git', 'config', 'user.name'], cwd=ROOT, text=True).strip()
        email = subprocess.check_output(['git', 'config', 'user.email'], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.run(['git', '-c', 'user.name=' + name, '-c', 'user.email=' + email, *args],
                                      cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            git('init'); (root / 'src').mkdir(); source = root / 'src/model.py'; source.write_text('old')
            (root / 'tests').mkdir(); test_source = root / 'tests/check.py'; test_source.write_text('old test')
            git('add', 'src/model.py', 'tests/check.py'); git('commit', '-m', 'Fixture for seal validation')
            self.assertEqual(len(read_source_commit(root)), 40)
            source.write_text('staged new implementation'); git('add', 'src/model.py')
            with self.assertRaises(subprocess.CalledProcessError):
                read_source_commit(root)
            git('reset', '--hard', 'HEAD')  # disposable test repository only
            test_source.write_text('staged test change'); git('add', 'tests/check.py')
            with self.assertRaises(subprocess.CalledProcessError):
                read_source_commit(root)
            git('reset', '--hard', 'HEAD')
            (root / 'data/annotations').mkdir(parents=True)
            (root / 'data/annotations/user_draft.jsonl').write_text('preserve')
            self.assertEqual(len(read_source_commit(root)), 40)
            (root / 'src/new.py').write_text('uncommitted')
            with self.assertRaises(ValueError):
                read_source_commit(root)


if __name__ == '__main__':
    unittest.main()
