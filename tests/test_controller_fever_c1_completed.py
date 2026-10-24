"""Accept the downloaded run, without scoring or displaying final answers."""
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
import zipfile

from scripts.audit_controller_fever_c1_outputs import audit
from src.compensation.answer_repair import MODEL_REVISION

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/controller/fever_c1_v1'
RELEASE=BASE/'prefill_r2/completed_v3'


def digest(content):return hashlib.sha256(content).hexdigest()


class CompletedC1Tests(unittest.TestCase):
    def test_registered_900_outputs_match_cloud_zip_and_all_submitted_inputs(self):
        report=audit(BASE/'prefill_r2/collection',RELEASE/'raw_outputs')
        self.assertTrue(report['complete'] and report['runtime_gold_isolated'])
        self.assertEqual(report['counts'],{'development':200,'selection':200,'final_candidate':500})
        self.assertFalse(report['quality_scored'] or report['answers_printed'])
        registration=json.loads((ROOT/'configs/controller/fever_c1_completed_v1.json').read_text())
        for item in [registration[k] for k in ('data_manifest','collection_manifest','completion_receipt','coverage_audit')]+list(registration['runtime_files'].values()):
            path=Path(item['path'])
            self.assertFalse(path.is_absolute() or '..' in path.parts)
            self.assertEqual(digest((ROOT/path).read_bytes()),item['sha256'])
        receipt=json.loads((RELEASE/'completion_receipt.json').read_text())
        archive=RELEASE/'kaggle_version3_outputs.zip'
        self.assertEqual(digest(archive.read_bytes()),receipt['outer_archive_sha256'])
        logged=re.search(r'SHA256: ([0-9a-f]{64})',(RELEASE/'kaggle_version3.log').read_text()).group(1)
        with zipfile.ZipFile(archive) as outer:
            content=outer.read('controller_fever_c1_v1_outputs.zip')
            self.assertEqual(digest(content),logged)
            self.assertEqual(digest(content),receipt['inner_output_zip_sha256'])
            with zipfile.ZipFile(io.BytesIO(content)) as inner:
                checks=json.loads(inner.read('output_checksums.json'))
                self.assertEqual(len(checks),11)
                self.assertEqual({p.filename for p in inner.infolist() if not p.is_dir()},set(checks)|{'output_checksums.json'})
                for path,sha in checks.items():
                    self.assertEqual(digest(inner.read(path)),sha)
                    self.assertEqual(inner.read(path),(RELEASE/'raw_outputs'/path).read_bytes())
            prefix='input_601b7070eab5/controller_fever_c1_gpu_v1/'
            index=outer.read(prefix+'bundle_checksums.json')
            bundle=json.loads((BASE/'prefill_r2/bundle_receipt.json').read_text())
            self.assertEqual(digest(index),bundle['checksum_index_sha256'])
            checks=json.loads(index);self.assertEqual(len(checks),77)
            for path,sha in checks.items():self.assertEqual(digest(outer.read(prefix+path)),sha)
        self.assertEqual(receipt['n_completed'],900)
        self.assertFalse(registration['formal_controller_comparison_ready'])

    def test_long_fixture_nf4_runtime_and_every_real_input_length_are_observed(self):
        raw=RELEASE/'raw_outputs'
        engineering=json.loads((raw/'engineering_smoke.json').read_text())
        self.assertTrue(engineering['passed']);self.assertEqual(engineering['llm_calls'],3)
        long=engineering['traces'][-1]
        self.assertEqual(long['sample_id'],'engineering-long-context')
        self.assertGreaterEqual(long['generation']['input_tokens'],13000)
        env=json.loads((raw/'environment.json').read_text())
        self.assertEqual(env['model_revision'],MODEL_REVISION)
        self.assertEqual(env['tokenizer_revision'],MODEL_REVISION)
        self.assertFalse(env['json_output'])
        self.assertEqual(env['generation_config']['prefill_chunk_size'],256)
        self.assertEqual(env['quantization_config']['bnb_4bit_compute_dtype'],'float16')
        self.assertTrue(env['quantization_config']['bnb_4bit_use_double_quant'])
        self.assertEqual(env['fourbit_quant_types'],['nf4'])
        self.assertEqual(env['fourbit_module_count'],196)
        profile=json.loads((BASE/'attempt2_oom/input_token_lengths.json').read_text())
        observed=[]
        for role,info in profile['roles'].items():
            traces=[json.loads(line) for line in (raw/role/'generations.jsonl').read_text().splitlines()]
            self.assertEqual([(r['sample_id'],r['generation']['input_tokens']) for r in traces],
                             [(r['sample_id'],r['input_tokens']) for r in info['lengths']])
            observed.extend(traces)
        formerly_failed=next(r for r in observed if r['sample_id']=='81950')
        self.assertEqual(formerly_failed['generation']['input_tokens'],12911)

    def test_dropping_one_prediction_trace_cannot_pass_completed_acceptance(self):
        source=RELEASE/'raw_outputs'
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            for path in source.rglob('*'):
                target=output/path.relative_to(source)
                if path.is_dir():target.mkdir(exist_ok=True)
                else:target.hardlink_to(path)
            target=output/'development/generations.jsonl'
            lines=target.read_bytes().splitlines(keepends=True);target.unlink()
            target.write_bytes(b''.join(lines[:-1]))
            with self.assertRaisesRegex(ValueError,'Unpaired/excess'):
                audit(BASE/'prefill_r2/collection',output)


if __name__=='__main__':unittest.main()
