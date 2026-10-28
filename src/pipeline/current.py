"""Current 2026-10-24 Diagnosis/P60/P64 integration, without legacy routing.

The metered module executor is reused only to execute a resolved action.
It never runs its old Diagnosis/Probe/controller loop on this path.
"""
from dataclasses import asdict, replace
import hashlib
import importlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from src.compensation.executor import EnhancementExecutor
from src.compensation.schema import EnhancementResult, EnhancementStatus
from src.controller.execution import CallLog, ExecutionConfig, fingerprint
from src.controller.runtime import canonical_input, unresolved_assessment
from src.controller.schema import ControllerAction
from src.controller.selection import FrozenOnlineSelectionAdapter
from src.diagnosis.schema import DiagnosisInput, EvidenceItem
from src.pipeline.adaptive_rag import build_adaptive_pipeline
from src.data.team_nq import BoundaryError, TeamNQBoundary

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = 'configs/controller/current_baseline.json'


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance_issues(root, spec, config):
    """Structural/hash integrity for P60 training provenance.

    Checks file existence, registered sha, model sha map, and that each
    referenced HGB artifact exists on disk. Does not audit Train/Test ID
    overlap or data-boundary membership of usage records.
    """
    root = Path(root)
    path = root / spec['path']
    if not path.is_file() or file_sha(path) != spec['sha256']:
        return ['P60_provenance_missing_or_changed']
    payload = json.loads(path.read_text(encoding='utf-8'))
    expected = {p: h for p, h in config['frozen_assets'].items()
                if p.startswith('models/controller/p60_online_adapter/')}
    records = payload.get('records')
    if (payload.get('complete') is not True or payload.get('model_sha256') != expected
            or not isinstance(records, list) or not records):
        return ['P60_provenance_incomplete_or_wrong_models']
    for rel, digest in expected.items():
        asset = root / rel
        if not asset.is_file() or file_sha(asset) != digest:
            return ['P60_provenance_model_artifact_missing_or_changed']
    covered = set()
    for row in records:
        if (set(row) != {'model', 'dataset', 'sample_id', 'role'} or row['model'] not in expected
                or row['dataset'] not in ('nq', 'fever')
                or row['role'] not in ('training', 'tuning', 'calibration', 'error_analysis')
                or not isinstance(row['sample_id'], str) or not row['sample_id']):
            return ['P60_provenance_invalid_usage_record']
        covered.add(row['model'])
    return [] if covered == set(expected) else ['P60_provenance_does_not_cover_all_three_models']


def readiness(root=ROOT):
    """No model/GPU load and no benchmark data read, including final inputs."""
    root = Path(root)
    config = json.loads((root / REGISTRATION).read_text(encoding='utf-8'))
    missing = []
    try:
        data_boundary = TeamNQBoundary(root).receipt()
    except BoundaryError as exc:
        data_boundary = {'valid': False, 'error': str(exc)}
        missing.append('team_data_boundary_missing_or_changed')
    for rel, expected in config['frozen_assets'].items():
        if not (root / rel).is_file() or file_sha(root / rel) != expected:
            missing.append('asset_missing_or_changed:' + rel)
    provider = config['feature_provider']
    if provider is None:
        missing.append('runtime_78_feature_provider_not_delivered')
    else:
        path = root / provider['source_path']
        if not path.is_file() or file_sha(path) != provider['source_sha256']:
            missing.append('runtime_feature_provider_source_mismatch')
        if provider.get('metered_runtime') is not True:
            missing.append('runtime_feature_provider_cost_contract_unregistered')
    provenance = config['training_provenance']
    if provenance is None:
        missing.append('P60_training_tuning_calibration_ID_provenance_not_delivered')
    else:
        missing.extend(provenance_issues(root, provenance, config))
    return {'baseline': config['baseline'], 'upstream_commit': config['upstream_commit'],
            'runtime_entry': 'src.pipeline.current.load_current_pipeline',
            'routing': 'Diagnosis -> P60 -> Controller -> EnhancementExecutor',
            'probe_enabled': False, 'ready_for_gpu': not missing, 'missing': missing,
            'legacy_fallback': False, 'benchmark_quality_evaluated': False,
            'E_mode': config['E_mode'], 'data_boundary': data_boundary}


class MeteredEnhancementAdapter:
    """Bridge the actual P3-20, refiner and E implementations to P64."""
    def __init__(self, executor):
        self.executor = executor

    def enhance(self, request):
        item = DiagnosisInput(request.sample_id, request.dataset, request.query,
                              list(request.evidence), request.initial_answer)
        calls = CallLog(self.executor.config)
        started = perf_counter()
        native = self.executor.execute(item, request.controller_decision, calls=calls)
        operational = (native.get('error') or calls.summary()['failed_calls']
                       or calls.summary()['budget_blocks'])
        unavailable = native['status'] in ('verification_unavailable', 'unsupported_dataset')
        status = (EnhancementStatus.UNAVAILABLE if unavailable else
                  EnhancementStatus.ERROR if operational else EnhancementStatus.SUCCESS)
        success = status is EnhancementStatus.SUCCESS
        # The public executor requires a non-success response to retain its
        # entering state. Keep the attempted native state in the audit trace.
        answer = native['final_answer'] if success else request.initial_answer
        evidence = tuple(EvidenceItem(**e) for e in native['final_evidence']) if success else request.evidence
        warning = unresolved_assessment({**native, 'cost': calls.summary(), 'routing': {}})
        return EnhancementResult(action=request.controller_decision.action,
            final_answer=answer, evidence=evidence,
            changed_answer=answer != request.initial_answer, status=status,
            metadata={'adapter': 'metered_real_modules_1024_v1', 'native': native,
                      'calls': calls.events, 'enhancement_cost': calls.summary(),
                      'unresolved_failure': warning},
            latency_ms=(perf_counter() - started) * 1000,
            error=json.dumps(native['error']) if native.get('error') else None)


class CheckedFeatures:
    """Preserve all 78 positions, including repeated names; never pad values."""
    def __init__(self, provider, names, *, require_costs=False):
        if provider is None or not callable(getattr(provider, 'features_for', None)):
            raise ValueError('The original online 78-feature provider is required')
        if tuple(getattr(provider, 'feature_names', ())) != tuple(names):
            raise ValueError('Feature names/order must match all 78 frozen positions')
        self.provider, self.names = provider, tuple(names)
        self.require_costs = require_costs
        self.last_receipt = None
        self.last_receipts = None

    def _validate_receipt(self, receipt, item):
        if (not isinstance(receipt, dict) or set(receipt) != {'input_sha256', 'calls'}
                or receipt['input_sha256'] != fingerprint(asdict(item))
                or not isinstance(receipt['calls'], list)):
            raise ValueError('Production features need an input-bound runtime cost receipt')
        for event in receipt['calls']:
            if (not isinstance(event, dict) or type(event.get('attempted')) is not bool
                    or type(event.get('seconds')) not in (int, float)
                    or not math.isfinite(event['seconds']) or event['seconds'] < 0
                    or event.get('status') not in ('completed', 'failed', 'budget_exceeded')
                    or not isinstance(event.get('kind'), str) or not event['kind']):
                raise ValueError('Production feature call accounting is incomplete')

    def features_for(self, item, diagnosis):
        rows = self.features_for_batch([item], [diagnosis])
        return rows[0]

    def features_for_batch(self, items, diagnoses):
        self.last_receipt = None
        self.last_receipts = None
        started = perf_counter()
        batch_fn = getattr(self.provider, 'features_for_batch', None)
        if callable(batch_fn):
            values_list = batch_fn(items, diagnoses)
        else:
            values_list = [self.provider.features_for(item, diagnosis)
                           for item, diagnosis in zip(items, diagnoses)]
        if len(values_list) != len(items):
            raise ValueError('Feature batch size mismatch')
        receipts_fn = getattr(self.provider, 'runtime_receipts', None)
        provider_receipts = receipts_fn() if callable(receipts_fn) else None
        if provider_receipts is None:
            receipt_fn = getattr(self.provider, 'runtime_receipt', None)
            single = receipt_fn() if callable(receipt_fn) else None
            provider_receipts = [single] * len(items) if single is not None else [None] * len(items)
        if len(provider_receipts) != len(items):
            raise ValueError('Feature receipt batch size mismatch')
        out = []
        wrapped = []
        elapsed = perf_counter() - started
        for item, values, receipt in zip(items, values_list, provider_receipts):
            arr = np.asarray(values, dtype=float)
            if arr.shape not in ((len(self.names),), (1, len(self.names))) or not np.isfinite(arr).all():
                raise ValueError('Runtime features must contain 78 finite original values')
            if self.require_costs:
                self._validate_receipt(receipt, item)
            wrapped.append({
                'input_sha256': fingerprint(asdict(item)),
                'feature_names_sha256': fingerprint(list(self.names)),
                'seconds': elapsed / max(len(items), 1),
                'provider_cost_receipt': receipt,
                'cost_state': 'reported_by_provider' if receipt is not None else 'not_accounted',
            })
            out.append(arr.reshape(-1))
        self.last_receipts = wrapped
        self.last_receipt = wrapped[-1] if wrapped else None
        return out


class CurrentPipeline:
    def __init__(self, host, features, config, *, data_boundary=None, data_role=None, e_mode=None):
        self.host, self.features, self.config = host, features, config
        self.data_boundary, self.data_role = data_boundary, data_role
        self.e_mode = e_mode if e_mode is not None else {'nq': 'allow_repair', 'fever': 'allow_repair'}

    def run(self, query, *, sample_id=None, dataset='nq'):
        """Raw query → formal Reference RAG → Diagnosis / Selection / Controller."""
        return self.run_batch([(query, sample_id)], dataset=dataset)[0]

    def _boundary_flags(self, item):
        enforce_boundary = True
        if self.data_boundary is not None:
            if item.dataset.value != 'nq':
                raise BoundaryError('FEVER production data roles need a separate team registration')
            # Ad-hoc runtime IDs are allowed for online queries; registered IDs stay gated.
            if item.sample_id not in self.data_boundary.membership:
                enforce_boundary = False
        return enforce_boundary

    def _finalize_result(self, item, result, *, enforce_boundary=True, feature_receipt=None):
        cost = result.enhancement_result.metadata.get('enhancement_cost')
        if cost is None:
            cost = CallLog(self.config).summary()  # no_action invokes no module
        boundary_meta = {'enforced': False, 'scope': 'engineering_injection'}
        if self.data_boundary is not None:
            if enforce_boundary:
                boundary_meta = {
                    'enforced': True,
                    'policy_sha256': self.data_boundary.receipt()['policy_sha256'],
                    'split_sha256': self.data_boundary.policy['team_split']['sha256'],
                }
            else:
                boundary_meta = {
                    'enforced': False,
                    'scope': 'ad_hoc_query',
                    'policy_sha256': self.data_boundary.receipt()['policy_sha256'],
                    'split_sha256': self.data_boundary.policy['team_split']['sha256'],
                }
        receipt = feature_receipt if feature_receipt is not None else self.features.last_receipt
        return replace(result, metadata={**result.metadata,
            'baseline': 'diagnosis_20261024_p60_p64_v1',
            'dataset': item.dataset.value, 'initial_answer_raw': item.initial_answer,
            'feature_provider': receipt,
            'data_role': self.data_role,
            'data_boundary': boundary_meta,
            'enhancement_cost': cost,
            'cost_scope': 'enhancement calls measured; feature cost separately reported, cached C1 excluded',
            'E_mode': self.e_mode,
            'quality_evaluated': False})

    def run_from_item(self, item, *, enforce_boundary=True):
        if enforce_boundary and self.data_boundary is not None:
            if item.dataset.value != 'nq':
                raise BoundaryError('FEVER production data roles need a separate team registration')
            self.data_boundary.require([item.sample_id], self.data_role)
        canonical = canonical_input(item)
        result = self.host.run_from_item(canonical)
        return self._finalize_result(canonical, result, enforce_boundary=enforce_boundary)

    def run_batch(self, queries, *, dataset='nq', feature_batch_size=None):
        """Micro-batch runtime path: serial RAG/Qwen, batched P26, serial control.

        ``queries``: sequence of ``(query, sample_id)``. Qwen stays serial to preserve
        decoding behavior; P26 BGE/CE are batched for throughput.
        """
        if self.host.reference_rag is None:
            raise ValueError('reference_rag is required for CurrentPipeline.run_batch')
        if not queries:
            return []
        batch_size = int(feature_batch_size) if feature_batch_size else len(queries)
        if batch_size < 1:
            raise ValueError('feature_batch_size must be >= 1')

        prepared = []
        for query, sample_id in queries:
            item = self.host.reference_rag.run(query, sample_id=sample_id, dataset=dataset)
            enforce_boundary = self._boundary_flags(item)
            if enforce_boundary and self.data_boundary is not None:
                self.data_boundary.require([item.sample_id], self.data_role)
            prepared.append((canonical_input(item), enforce_boundary, item.initial_answer))

        results = []
        for offset in range(0, len(prepared), batch_size):
            chunk = prepared[offset:offset + batch_size]
            items = [row[0] for row in chunk]
            diagnoses = [self.host.classifier.diagnose(item) for item in items]
            features = self.features.features_for_batch(items, diagnoses)
            feature_receipts = list(self.features.last_receipts or [None] * len(items))
            for (item, enforce_boundary, raw_answer), diagnosis, feat, receipt in zip(
                    chunk, diagnoses, features, feature_receipts):
                result = self.host.run_from_prepared(item, diagnosis, feat)
                finalized = self._finalize_result(
                    item, result, enforce_boundary=enforce_boundary, feature_receipt=receipt)
                meta = dict(finalized.metadata)
                meta['initial_answer_raw'] = raw_answer
                results.append(replace(finalized, metadata=meta))
        return results


def wire_latest_pipeline(execution_backend, feature_provider, *, selection_adapter=None, root=ROOT,
                         require_feature_costs=False, data_boundary=None, data_role=None,
                         reference_rag=None):
    """Wire real modules; injections also support explicitly fictional tests.

    Production callers use load_current_pipeline, which additionally checks
    delivery/provenance before loading any GPU model.
    """
    config = json.loads((Path(root) / REGISTRATION).read_text(encoding='utf-8'))
    expected = ExecutionConfig(**config['execution_config'])
    if execution_backend.config != expected:
        raise ValueError('Use the registered latest execution config from current_baseline.json')
    selector = (FrozenOnlineSelectionAdapter.from_config_path(config['selector_config'], repo_root=Path(root))
                if selection_adapter is None else selection_adapter)
    names = json.loads((Path(root) / config['selector_config']).read_text(encoding='utf-8'))['feature_names']
    features = CheckedFeatures(feature_provider, names, require_costs=require_feature_costs)
    executor = EnhancementExecutor()
    adapter = MeteredEnhancementAdapter(execution_backend)
    for action in (ControllerAction.RETRIEVAL_ENHANCEMENT, ControllerAction.EVIDENCE_REFINEMENT,
                   ControllerAction.GENERATION_REPAIR):
        executor.register(action, adapter)
    host = build_adaptive_pipeline(classifier=execution_backend.classifier,
        selection_adapter=selector, feature_provider=features, executor=executor,
        reference_rag=reference_rag)
    return CurrentPipeline(host, features, expected, data_boundary=data_boundary, data_role=data_role,
                           e_mode=config.get('E_mode'))


def load_current_pipeline(*, root=ROOT, nq_resources=None, fever_resources=None, data_role='validation',
                          attach_reference_rag=True):
    """Current production entry. Missing inputs stop; never run old routing."""
    root = Path(root)
    if data_role not in ('train', 'validation'):
        raise BoundaryError('Final test execution requires a separately sealed T7 protocol')
    receipt = readiness(root)
    if not receipt['ready_for_gpu']:
        raise RuntimeError('Latest Diagnosis integration not ready: ' + '; '.join(receipt['missing']))
    config = json.loads((root / REGISTRATION).read_text(encoding='utf-8'))
    spec = config['feature_provider']
    module = importlib.import_module(spec['module'])
    if Path(module.__file__).resolve() != (root / spec['source_path']).resolve():
        raise ValueError('Imported feature provider differs from its registered source')
    from src.controller.providers import load_pipeline
    backend, environment = load_pipeline(ExecutionConfig(**config['execution_config']),
        root / config['classifier'], nq_resources=nq_resources, fever_resources=fever_resources)
    provider = getattr(module, spec['factory'])(execution_backend=backend)
    if not callable(getattr(provider, 'runtime_receipt', None)):
        raise ValueError('Production feature provider must report its runtime costs')
    reference_rag = None
    if attach_reference_rag:
        from src.pipeline.reference_rag import MatchedReferenceRAG
        reference_rag = MatchedReferenceRAG.from_shared_generator(backend.raw_generator, root=root)
        # Same BAAI/bge-base-en-v1.5 weights as P26; share the instance to free ~0.4GB VRAM.
        # P26 still encodes at max_seq_length=128 (restored after each encode).
        shared_bge = getattr(getattr(reference_rag, 'retriever', None), 'model', None)
        runtime = getattr(provider, '_runtime', None)
        attach = getattr(runtime, 'attach_shared_bge', None)
        shared_ok = False
        if shared_bge is not None and callable(attach):
            attach(shared_bge)
            shared_ok = True
        environment = {
            **environment,
            'reference_rag': {
                'config': getattr(reference_rag, 'config', {}).get('_config_path'),
                'prompt_style': reference_rag.prompt_style,
                'embedding_model': reference_rag.embedding_model,
                'top_k': reference_rag.top_k,
                'p26_shared_bge': shared_ok,
            },
        }
    return wire_latest_pipeline(backend, provider, root=root, require_feature_costs=True,
        data_boundary=TeamNQBoundary(root), data_role=data_role, reference_rag=reference_rag), {
        **environment, 'current_baseline': receipt, 'probe_enabled': False}
