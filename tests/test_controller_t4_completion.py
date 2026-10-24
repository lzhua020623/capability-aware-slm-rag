"""Exercise the full 400-query runner through final summary, using fictional tools."""
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_controller_t4 as runner
from scripts.run_controller_integration import smoke_inputs, smoke_pipeline
from src.controller.execution import DEFAULT_MODES, ExecutionConfig


class T4CompletionTests(unittest.TestCase):
    def test_all_2800_saved_predictions_reach_completed_summary(self):
        # No CUDA, real resources or QA benchmark: use the real Controller and
        # frozen Diagnosis with fictional generator/scorer/retriever providers.
        config = ExecutionConfig(max_scoring_calls=2)
        pipeline = smoke_pipeline(config)
        historical = smoke_inputs()
        inputs = {item.dataset.value: [replace(item, sample_id=f'fiction-{item.dataset.value}-{n}')
                  for n in range(200)] for item in historical}

        class Retriever:
            def retrieve(self, query, *, dataset, top_k):
                item = next(i for i in historical if i.dataset.value == dataset)
                rows = [asdict(e) for e in item.evidence]
                rows += [{'rank': n, 'passage_id': f'extra-{n}',
                          'text': item.evidence[0].text, 'score': .1} for n in range(6, 21)]
                return rows[:top_k]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'runtime.json').write_text(json.dumps(asdict(config)))
            history = root / 'historical.jsonl'
            history.write_text(''.join(json.dumps(asdict(i)) + '\n' for i in historical))
            manifest = {'source_files': {}, 'required_modes': list(DEFAULT_MODES),
                        'historical_checks': {'path': history.name, 'sha256': runner.sha(history)}}
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            receipt_path = root / 'receipt.json'; receipt_path.write_text('{}')
            policy = {'runtime_config': 'runtime.json', 'datasets': ['nq', 'fever'],
                      'frozen_components': {'diagnosis': 'unused-fiction-classifier-path'}}
            output = root / 'outputs'
            # Resource and CUDA receipt gates have their own strict tests. These
            # stubs isolate the expensive-run finalization boundary that failed.
            with (patch.object(runner, 'ROOT', root),
                  patch.object(runner, 'verify_manifest', return_value=(inputs, policy)),
                  patch.object(runner, 'accept_engineering'),
                  patch('src.controller.providers.load_pipeline', return_value=(pipeline, {'packages': {}})),
                  patch('src.controller.nq_train_resources.load_backend', return_value=(Retriever(), {})),
                  patch('src.controller.fever_resources.load_backend', return_value=(Retriever(), {})),
                  redirect_stdout(io.StringIO())):
                result = runner.main(['--manifest', str(manifest_path), '--output-dir', str(output),
                                      '--engineering-receipt', str(receipt_path)])
            self.assertEqual(result, 0)
            state = json.loads((output / 'manifest.json').read_text())
            self.assertEqual((state['state'], state['n_predictions']), ('completed', 2800))
            self.assertEqual(state['failed_records'], 0)
            with (output / 'predictions.jsonl').open() as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(len(records), 2800)
            self.assertTrue((output / 'execution_seal.json').is_file())
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(len(summary['groups']), 14)
            for key, group in summary['groups'].items():
                dataset, mode = key.split('/')
                self.assertEqual(group['n'], 200)
                self.assertAlmostEqual(group['total_seconds'], sum(r['total_seconds'] for r in records
                                       if r['dataset'] == dataset and r['mode'] == mode))


if __name__ == '__main__':
    unittest.main()
