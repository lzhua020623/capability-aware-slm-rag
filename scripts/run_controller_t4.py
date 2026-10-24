"""Real CUDA engineering gate followed by ONE sealed development comparison.

No gold/selection/final inputs are read. Engineering-only is a distinct launch
mode; benchmark inference requires real resources and an accepted CUDA receipt.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_controller_integration import read_inputs, summarize
from src.controller.execution import ExecutionConfig, fingerprint
from src.controller.runtime import canonical_input


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def verify_manifest(root, manifest):
    root = Path(root)
    if (manifest['schema_version'] != 'controller_t3_execution_manifest_v1'
            or manifest['role'] != 'development' or manifest['contains_gold'] is not False
            or manifest['contains_selection_or_final_inputs'] is not False
            or set(manifest['datasets']) != {'nq', 'fever'}):
        raise ValueError('Unregistered execution scope')
    for rel, digest in manifest['source_files'].items():
        path = Path(rel)
        if path.is_absolute() or '..' in path.parts or sha(root / path) != digest:
            raise ValueError(f'Frozen source changed: {rel}')
    policy = json.loads((root / 'configs/controller/t3_execution_policy_v1.json').read_text())
    if (sha(root / 'configs/controller/t3_execution_policy_v1.json') != manifest['policy_sha256']
            or manifest['required_modes'] != policy['required_modes']):
        raise ValueError('Execution policy changed')
    inputs = {}
    for ds, info in manifest['datasets'].items():
        path = Path(info['input']['path'])
        if path.is_absolute() or '..' in path.parts or sha(root / path) != info['input']['sha256']:
            raise ValueError('Frozen input changed')
        batch = read_inputs(root / path)
        if len(batch) != 200 or [i.sample_id for i in batch] != info['sample_ids'] or any(i.dataset.value != ds for i in batch):
            raise ValueError('Input count/dataset/order mismatch')
        if {i.sample_id: fingerprint(asdict(canonical_input(i))) for i in batch} != info['effective_input_hashes']:
            raise ValueError('Effective input adapter changed')
        inputs[ds] = batch
    return inputs, policy


def engineering_environment(environment):
    return {k: environment.get(k) for k in ('model_revision', 'tokenizer_revision', 'packages', 'cuda_version',
            'gpus', 'fourbit_module_count', 'fourbit_quant_types', 'quantization_config', 'parameter_devices',
            'controller_generation_runtime', 'reranker_model', 'reranker_revision', 'reranker_device', 'reranker_dtype',
            'installed_packages', 'CPU_runtime')}


def accept_engineering(receipt, manifest, environment):
    if (receipt.get('passed') is not True or receipt.get('minimum_input_tokens', 0) < 14000
            or receipt.get('source_files') != manifest['source_files']
            or receipt.get('policy_sha256') != manifest['policy_sha256']
            or receipt.get('engineering_environment') != engineering_environment(environment)):
        raise ValueError('CUDA receipt does not match this source/policy/environment')
    coverage = {(c['dataset'], c['path']) for c in receipt.get('checks', []) if c.get('passed') is True
                and c.get('minimum_actual_input_tokens', 0) >= 14000}
    if coverage != {(ds, mode) for ds in ('nq', 'fever')
                    for mode in ('probe', 'always_retrieve', 'always_rerank', 'always_repair')}:
        raise ValueError('Incomplete eight-path long-input CUDA coverage')
    from src.controller.engineering import inspect_engineering_calls
    if len(receipt['checks']) != 8:
        raise ValueError('Duplicate or additional engineering checks')
    for check in receipt['checks']:
        actual = inspect_engineering_calls(check['record']['calls'], 14000)
        if actual['minimum_actual_input_tokens'] != check['minimum_actual_input_tokens']:
            raise ValueError('Engineering summary differs from raw call receipts')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'execution_manifest.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--engineering-only', action='store_true')
    parser.add_argument('--engineering-receipt', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error('Preserve prior execution; choose a new output directory')
    manifest = json.loads(args.manifest.read_text())
    inputs, policy = verify_manifest(ROOT, manifest)
    if args.preflight:
        print(json.dumps({'preflight': 'passed', 'nq': 200, 'fever': 200, 'gold_read': False}))
        return 0
    if args.engineering_only and args.engineering_receipt:
        parser.error('Engineering-only creates its own receipt')
    if not args.engineering_only and not args.engineering_receipt:
        parser.error('T4 requires a separately accepted real CUDA engineering receipt')
    args.output_dir.mkdir(parents=True)
    state = {'state': 'starting', 'engineering_only': args.engineering_only, 'T4_started': False,
             'manifest_sha256': sha(args.manifest), 'n_predictions': 0, 'quality_scored': False}
    write(args.output_dir / 'manifest.json', state)
    try:
        from src.controller.providers import load_pipeline
        config = ExecutionConfig(**json.loads((ROOT / policy['runtime_config']).read_text()))
        started = perf_counter()
        pipeline, environment = load_pipeline(config, ROOT / policy['frozen_components']['diagnosis'])
        environment['model_loading_seconds'] = perf_counter() - started
        for name in ('sentence-transformers', 'scikit-learn', 'faiss-cpu', 'numpy', 'scipy'):
            environment['packages'][name] = importlib.metadata.version(name)
        # Full version lock without pip direct-URL metadata or environment
        # variables that could contain credentials. No hostnames are recorded.
        environment['installed_packages'] = sorted({d.metadata['Name'] + '==' + d.version
            for d in importlib.metadata.distributions() if d.metadata.get('Name')})
        import torch
        environment['CPU_runtime'] = {'machine': platform.machine(), 'logical_cpus': os.cpu_count(),
                                      'torch_threads': torch.get_num_threads(), 'reranker_batch_size': 4}
        write(args.output_dir / 'environment.json', environment)
        if args.engineering_only:
            from src.controller.engineering import engineering_checks
            partial = args.output_dir / 'engineering_checks'; partial.mkdir()
            receipt = engineering_checks(pipeline, minimum=policy['engineering_gate']['minimum_input_tokens'],
                on_check=lambda c: write(partial / f'{c["dataset"]}_{c["path"]}.json', c))
            receipt.update(source_files=manifest['source_files'], policy_sha256=manifest['policy_sha256'],
                           engineering_environment=engineering_environment(environment))
            accept_engineering(receipt, manifest, environment)
            write(args.output_dir / 'engineering_receipt.json', receipt)
            state.update(state='engineering_completed', engineering_passed=True)
            write(args.output_dir / 'manifest.json', state)
            return 0
        receipt = json.loads(args.engineering_receipt.read_text())
        accept_engineering(receipt, manifest, environment)
        # Actual tools stay live on the GPU host; no precomputed answer/scoring
        # cache is substituted for the measured fixed-action or Full groups.
        from src.controller.nq_train_resources import load_backend as load_nq
        from src.controller.fever_resources import load_backend as load_fever, DatasetRetriever
        before = perf_counter()
        nq, nq_receipt = load_nq(ROOT / 'configs/controller/nq_train_resources_v1.json', root=ROOT)
        fever, fever_receipt = load_fever(ROOT / 'configs/controller/fever_resources_v1.json', root=ROOT)
        pipeline.retriever = DatasetRetriever({'nq': nq, 'fever': fever})
        historical_info = manifest['historical_checks']
        historical_path = ROOT / historical_info['path']
        if sha(historical_path) != historical_info['sha256']:
            raise ValueError('Historical engineering input changed')
        lookup = []
        for item in read_inputs(historical_path):
            hits = pipeline.retriever.retrieve(item.query, dataset=item.dataset.value, top_k=20)
            passed = [h['passage_id'] for h in hits[:5]] == [e.passage_id for e in item.evidence]
            lookup.append({'dataset': item.dataset.value, 'sample_id': item.sample_id,
                           'same_ordered_top5': passed})
            if not passed:
                raise ValueError('Live GPU-host BGE/index lookup differs from historical C1')
        write(args.output_dir / 'historical_lookup.json', lookup)
        environment.update(retrieval_resources_loading_seconds=perf_counter() - before,
                           nq_train_resources=nq_receipt, fever_resources=fever_receipt)
        write(args.output_dir / 'environment.json', environment)
        verify_manifest(ROOT, manifest)
        seal = {**manifest, 'GPU_engineering_passed': True, 'engineering_receipt_sha256': sha(args.engineering_receipt),
                'environment_sha256': sha(args.output_dir / 'environment.json'),
                'environment': environment, 'T4_started': True}
        # Saved before the first real benchmark prediction; never rewritten.
        with (args.output_dir / 'execution_seal.json').open('x') as handle:
            json.dump(seal, handle, indent=2); handle.write('\n')
        state.update(state='running', T4_started=True)
        write(args.output_dir / 'manifest.json', state)
        lean = []
        with (args.output_dir / 'predictions.jsonl').open('x') as handle:
            for ds in policy['datasets']:
                for ordinal, item in enumerate(inputs[ds], 1):
                    for mode in manifest['required_modes']:
                        record = pipeline.run(item, mode=mode)
                        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'); handle.flush()
                        state['n_predictions'] += 1
                        lean.append({k: record[k] for k in ('mode', 'dataset', 'cost', 'routing', 'answer_changed',
                                                           'status', 'unresolved_failure')})
                        print(json.dumps({'dataset': ds, 'ordinal': ordinal, 'mode': mode,
                                          'n_predictions': state['n_predictions'], 'status': record['status']}), flush=True)
                        write(args.output_dir / 'manifest.json', state)
                        if any('out of memory' in str(c.get('error', {})).lower() for c in record['calls']):
                            raise RuntimeError('Infrastructure OOM: preserving prefix, refusing repeated failing calls')
        errors = sum(bool(r['cost']['failed_calls'] or r['cost']['budget_blocks']) for r in lean)
        write(args.output_dir / 'summary.json', summarize(lean))
        state.update(state='completed_with_errors' if errors else 'completed', expected_predictions=2800,
                     failed_records=errors)
        write(args.output_dir / 'manifest.json', state)
        return 2 if errors else 0
    except Exception as exc:
        state.update(state='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        write(args.output_dir / 'manifest.json', state)
        raise


if __name__ == '__main__':
    sys.exit(main())
