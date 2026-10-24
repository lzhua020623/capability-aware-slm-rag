"""Mechanical runtime adaptation; no references, diagnosis labels or tuning."""
from dataclasses import replace

from src.diagnosis.schema import DatasetName
from src.evaluation.fever import parse_fever_prediction

ADAPTER_VERSION = 'controller_fever_label_v1'
PREFILL_CHUNK_SIZE = 256


def canonical_input(item):
    if item.dataset is DatasetName.FEVER:
        label = parse_fever_prediction(item.initial_answer)
        if label != 'UNKNOWN':
            return replace(item, initial_answer=label)
    return item


def configure_generation_runtime(generator, chunk_size=PREFILL_CHUNK_SIZE):
    if type(chunk_size) is not int or chunk_size != PREFILL_CHUNK_SIZE:
        raise ValueError('Register a new execution revision for a different prefill size')
    config = generator.model.generation_config
    if not hasattr(config, 'prefill_chunk_size'):
        raise RuntimeError('The sealed Transformers environment must support native chunked prefill')
    config.prefill_chunk_size = chunk_size
    settings = {'prefill_chunk_size': chunk_size, 'do_sample': False,
                'scope': 'shared Qwen: Probe, retrieval/rerank regeneration and E'}
    generator.runtime_environment['controller_generation_runtime'] = settings
    return settings


def unresolved_assessment(record):
    """A runtime warning, never an offline correctness verdict or a success claim.

    Operational failures and explicit lack-of-support signals are review
    candidates. Keeping or changing an answer alone never establishes support.
    No additional model call is introduced by this reporting layer.
    """
    reasons = []
    cost = record['cost']
    if cost['failed_calls'] or cost['budget_blocks']:
        reasons.append('execution_or_budget_failure')
    trace = record.get('module_trace') or {}
    traces = [trace]
    traces.extend(s.get('module_trace') or {} for s in trace.get('stages', []))
    for native in traces:
        if native.get('verdict') == 'insufficient':
            reasons.append('module_reports_insufficient_evidence')
        if native.get('validation_error'):
            reasons.append('candidate_validation_failed')
        if native.get('reject_reason'):
            reasons.append('candidate_rejected')
    reasons = sorted(set(reasons))
    return {'flagged': bool(reasons), 'state': 'flagged_for_review' if reasons else 'unverified',
            'reasons': reasons, 'semantic_correctness': 'not_checked',
            'evidence_support': 'not_independently_checked',
            'criterion': 'operational signals only; AC6 requires independent flagged-sample review'}
