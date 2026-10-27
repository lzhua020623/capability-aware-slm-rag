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
    """Check declared per-model usage IDs, never infer a clean split from labels."""
    root = Path(root)
    path = root / spec['path']
    if not path.is_file() or file_sha(path) != spec['sha256']:
        return ['P60_provenance_missing_or_changed']
    payload = json.loads(path.read_text())
    expected = {p: h for p, h in config['frozen_assets'].items()
                if p.startswith('models/controller/p60_online_adapter/')}
    records = payload.get('records')
    if (payload.get('complete') is not True or payload.get('model_sha256') != expected
            or not isinstance(records, list) or not records):
        return ['P60_provenance_incomplete_or_wrong_models']
    protected = json.loads((root / 'artifacts/controller/data_v1/evaluation_reserve_ids.json').read_text())
    selected = json.loads((root / 'artifacts/controller/data_v1/selection_ids.json').read_text())
    forbidden = {(ds, str(sid)) for ds in ('nq', 'fever')
                 for sid in protected[ds] + selected[ds]['selection']}
    covered = set()
    try:
        boundary = TeamNQBoundary(root)
    except BoundaryError:
        return ['team_data_boundary_missing_or_changed']
    for row in records:
        if (set(row) != {'model', 'dataset', 'sample_id', 'role'} or row['model'] not in expected
                or row['dataset'] not in ('nq', 'fever')
                or row['role'] not in ('training', 'tuning', 'calibration', 'error_analysis')):
            return ['P60_provenance_invalid_usage_record']
        if (row['dataset'], str(row['sample_id'])) in forbidden:
            return ['P60_training_or_tuning_overlaps_protected_or_selection_IDs']
        if row['dataset'] == 'fever':
            return ['P60_FEVER_source_needs_separate_team_registration']
        if row['dataset'] == 'nq':
            try:
                if not isinstance(row['sample_id'], str):
                    raise BoundaryError('P60 IDs must be exact strings')
                roles = ('train',) if row['role'] == 'training' else ('train', 'validation')
                if not any(row['sample_id'] in boundary.eligible[r] for r in roles):
                    raise BoundaryError('P60 data outside registered role')
                boundary.require([row['sample_id']], next(r for r in roles if row['sample_id'] in boundary.eligible[r]))
            except BoundaryError:
                return ['P60_IDs_outside_registered_team_roles']
        covered.add(row['model'])
    return [] if covered == set(expected) else ['P60_provenance_does_not_cover_all_three_models']


def readiness(root=ROOT):
    """No model/GPU load and no benchmark data read, including final inputs."""
    root = Path(root)
    config = json.loads((root / REGISTRATION).read_text())
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

    def features_for(self, item, diagnosis):
        self.last_receipt = None
        started = perf_counter()
        values = np.asarray(self.provider.features_for(item, diagnosis), dtype=float)
        if values.shape not in ((len(self.names),), (1, len(self.names))) or not np.isfinite(values).all():
            raise ValueError('Runtime features must contain 78 finite original values')
        receipt_fn = getattr(self.provider, 'runtime_receipt', None)
        receipt = receipt_fn() if callable(receipt_fn) else None
        if self.require_costs:
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
        self.last_receipt = {'input_sha256': fingerprint(asdict(item)),
            'feature_names_sha256': fingerprint(list(self.names)),
            'seconds': perf_counter() - started,
            'provider_cost_receipt': receipt,
            'cost_state': 'reported_by_provider' if receipt is not None else 'not_accounted'}
        return values.reshape(-1)


class CurrentPipeline:
    def __init__(self, host, features, config, *, data_boundary=None, data_role=None):
        self.host, self.features, self.config = host, features, config
        self.data_boundary, self.data_role = data_boundary, data_role

    def run_from_item(self, item):
        if self.data_boundary is not None:
            if item.dataset.value != 'nq':
                raise BoundaryError('FEVER production data roles need a separate team registration')
            self.data_boundary.require([item.sample_id], self.data_role)
        canonical = canonical_input(item)
        result = self.host.run_from_item(canonical)
        cost = result.enhancement_result.metadata.get('enhancement_cost')
        if cost is None:
            cost = CallLog(self.config).summary()  # no_action invokes no module
        return replace(result, metadata={**result.metadata,
            'baseline': 'diagnosis_20261024_p60_p64_v1',
            'dataset': canonical.dataset.value, 'initial_answer_raw': item.initial_answer,
            'feature_provider': self.features.last_receipt,
            'data_role': self.data_role,
            'data_boundary': ({'enforced': True,
                'policy_sha256': self.data_boundary.receipt()['policy_sha256'],
                'split_sha256': self.data_boundary.policy['team_split']['sha256']}
                if self.data_boundary is not None else {'enforced': False, 'scope': 'engineering_injection'}),
            'enhancement_cost': cost,
            'cost_scope': 'enhancement calls measured; feature cost separately reported, cached C1 excluded',
            'E_mode': 'unavailable', 'quality_evaluated': False})


def wire_latest_pipeline(execution_backend, feature_provider, *, selection_adapter=None, root=ROOT,
                         require_feature_costs=False, data_boundary=None, data_role=None):
    """Wire real modules; injections also support explicitly fictional tests.

    Production callers use load_current_pipeline, which additionally checks
    delivery/provenance before loading any GPU model.
    """
    config = json.loads((Path(root) / REGISTRATION).read_text())
    expected = ExecutionConfig(**config['execution_config'])
    if execution_backend.config != expected:
        raise ValueError('Use the registered latest execution config; E stays unavailable')
    selector = (FrozenOnlineSelectionAdapter.from_config_path(config['selector_config'], repo_root=Path(root))
                if selection_adapter is None else selection_adapter)
    names = json.loads((Path(root) / config['selector_config']).read_text())['feature_names']
    features = CheckedFeatures(feature_provider, names, require_costs=require_feature_costs)
    executor = EnhancementExecutor()
    adapter = MeteredEnhancementAdapter(execution_backend)
    for action in (ControllerAction.RETRIEVAL_ENHANCEMENT, ControllerAction.EVIDENCE_REFINEMENT,
                   ControllerAction.GENERATION_REPAIR):
        executor.register(action, adapter)
    host = build_adaptive_pipeline(classifier=execution_backend.classifier,
        selection_adapter=selector, feature_provider=features, executor=executor)
    return CurrentPipeline(host, features, expected, data_boundary=data_boundary, data_role=data_role)


def load_current_pipeline(*, root=ROOT, nq_resources=None, fever_resources=None, data_role='validation'):
    """Current production entry. Missing inputs stop; never run old routing."""
    root = Path(root)
    if data_role not in ('train', 'validation'):
        raise BoundaryError('Final test execution requires a separately sealed T7 protocol')
    receipt = readiness(root)
    if not receipt['ready_for_gpu']:
        raise RuntimeError('Latest Diagnosis integration not ready: ' + '; '.join(receipt['missing']))
    config = json.loads((root / REGISTRATION).read_text())
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
    return wire_latest_pipeline(backend, provider, root=root, require_feature_costs=True,
        data_boundary=TeamNQBoundary(root), data_role=data_role), {
        **environment, 'current_baseline': receipt, 'probe_enabled': False}
