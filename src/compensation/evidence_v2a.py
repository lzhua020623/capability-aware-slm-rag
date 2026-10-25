"""Single-call E interface. Provenance and structure are not semantic truth.

The original verdict survives an invalid candidate. Deployment protection is
separate from the diagnostic candidate, so a saved run can be gated offline.
"""
from dataclasses import asdict, replace
import hashlib
import json

from src.compensation.answer_repair import Citation, Generation, RepairResult
from src.compensation.evidence_id_repair import (
    RULES, context, evidence_units, reconstruct_citations, selected_units, strict_object,
)
from src.compensation.short_answer_repair import locate_excerpt
from src.diagnosis import DatasetName
from src.evaluation.qa import normalize_answer
from src.evaluation.fever import parse_fever_prediction

VERSION = 'evidence_id_v2a_v1'
MODES = ('allow_repair', 'verify_only', 'unavailable')
LIMIT = 384
FIELDS = ('original_verification', 'original_evidence_ids', 'candidate_answer',
          'candidate_evidence_ids', 'reason')


def schema(units):
    ids = {'type': 'array', 'items': {'type': 'string', 'enum': [u.id for u in units]},
           'maxItems': 5}
    return {'type': 'object', 'properties': {
        'original_verification': {'enum': ['supported', 'unsupported', 'insufficient']},
        'original_evidence_ids': ids, 'candidate_answer': {'type': 'string'},
        'candidate_evidence_ids': ids, 'reason': {'type': 'string'}},
        'required': list(FIELDS), 'additionalProperties': False}


def prompt(request, units):
    task = (
        'FEVER: the query is the ORIGINAL claim; the entering answer is a SUPPORTS/REFUTES '
        'label. Verify whether that label is justified. unsupported means the entering '
        'label is demonstrated wrong, NOT merely that the claim lacks support. A replacement '
        'REFUTES requires explicit contradictory evidence; missing/conflicting evidence is '
        'insufficient. A candidate is only SUPPORTS or REFUTES.'
        if request.dataset == DatasetName.FEVER else
        'NQ: verify whether the entering answer answers the requested fact. A replacement '
        'must be a continuous exact span in selected original evidence, at most 20 words.'
    )
    return (RULES + task + '\nIn ONE response separately verify the entering answer and '
        'optionally propose a minimal replacement. original_verification is supported, '
        'unsupported or insufficient. supported: keep even if a longer wording is possible; '
        'candidate_answer is empty. unsupported: demonstrate the answer error and provide '
        'a candidate only if sufficient evidence exists. Without a qualified candidate leave '
        'it empty; this does not turn unsupported into supported. insufficient: leave the '
        'candidate empty; explain relevant evidence conflicts explicitly. For supported and '
        'unsupported select 1-5 unique original_evidence_ids supporting the verdict. For '
        'a candidate select 1-5 candidate_evidence_ids. Empty candidates have no IDs. '
        'Use only exact sentence keys, never generate quotations. Return JSON only.\n'
        f'Query: {json.dumps(request.query, ensure_ascii=False)}\n'
        f'Entering answer: {json.dumps(request.initial_answer, ensure_ascii=False)}\n'
        f'Evidence: {context(request, units)}\n'
        'Output JSON Schema: ' + json.dumps(schema(units), separators=(',', ':')))


def apply_mode(result, mode):
    if mode not in MODES:
        raise ValueError('Unknown E v2a mode')
    saved = result.support_check or {}
    if saved.get('version') != VERSION:
        raise ValueError('Not a v2a result')
    diagnostic = saved.get('diagnostic_check', saved)
    check = dict(diagnostic)
    check.update(deployment_mode=mode, diagnostic_check=dict(diagnostic),
                 diagnostic_final_answer=saved.get('diagnostic_final_answer', result.final_answer),
                 diagnostic_status=saved.get('diagnostic_status', result.status),
                 diagnostic_citations=saved.get('diagnostic_citations', [asdict(c) for c in result.citations]),
                 candidate_check_before_policy=diagnostic['candidate_check'])
    final, status = check['diagnostic_final_answer'], check['diagnostic_status']
    citations = [Citation(**c) for c in check['diagnostic_citations']]
    if mode != 'allow_repair':
        final = result.initial_answer
        citations = [Citation(**c) for c in check.get('original_citations', [])]
        check['final_action'] = 'keep'
        if check['candidate_check_before_policy'] == 'passed':
            check['candidate_check'] = 'protected'
        if check['original_verification'] != 'supported':
            check['unresolved'] = True
            check['unresolved_reasons'] = sorted(set(check['unresolved_reasons'] + ['replacement_disabled']))
        status = 'verify_only' if mode == 'verify_only' else 'verification_unavailable'
    if mode == 'unavailable':
        citations = []
        check.update(original_verification='incomplete', contract_passed=False,
                     unresolved=True, unresolved_reasons=['engineering_gate_failed'])
    return replace(result, final_answer=final, status=status, citations=citations,
                   verdict=check['original_verification'], support_check=check)


def repair(request, backend, *, mode='allow_repair', max_new_tokens=LIMIT):
    if mode not in MODES or type(max_new_tokens) is not int or max_new_tokens != LIMIT:
        raise ValueError('E v2a requires a registered mode and 384-token allocation')
    units = evidence_units(request)
    check = {'version': VERSION, 'original_verification': 'incomplete',
             'candidate_check': 'absent', 'final_action': 'keep', 'contract_passed': False,
             'unresolved': True, 'unresolved_reasons': ['verification_incomplete']}
    trace = {'version': VERSION, 'units': [asdict(u) for u in units], 'calls': [],
             'semantic_verifier': 'same_Qwen_not_independent'}
    generated = None

    def finish(status, final=None, citations=None, error=None, candidate=None):
        result = RepairResult(request.initial_answer, request.initial_answer if final is None else final,
            status, verdict=check['original_verification'], citations=citations or [],
            raw_output=generated.text if generated else '', validation_error=error,
            extra_calls=len(trace['calls']), latency=generated.latency if generated else 0,
            input_tokens=generated.input_tokens if generated else None,
            output_tokens_estimate=generated.output_tokens_estimate if generated else None,
            candidate_answer=candidate, candidate_status=check['candidate_check'],
            support_check=check, repair_trace=trace)
        return apply_mode(result, mode)

    if mode == 'unavailable':
        return finish('verification_unavailable')
    if not units or not any(u.text.strip() for u in units):
        return finish('verification_incomplete', error='No evidence units')
    text = prompt(request, units)
    call = {'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'schema_sha256': hashlib.sha256(json.dumps(schema(units), sort_keys=True).encode()).hexdigest(),
            'max_new_tokens': LIMIT}
    trace['calls'].append(call)
    try:
        observed, metadata = backend.generate_schema(text, LIMIT, schema(units))
        if not isinstance(observed, Generation):
            raise TypeError('Structured backend must return Generation')
        generated = observed
        call.update(generation=asdict(generated), decoder=metadata)
        if metadata.get('hit_token_budget'):
            raise ValueError('Output reached the token budget; verification not accepted')
        value = strict_object(generated.text, FIELDS)
        verdict = value['original_verification']
        if verdict not in ('supported', 'unsupported', 'insufficient'):
            raise ValueError('Invalid original verification')
        if not isinstance(value['reason'], str) or not value['reason'].strip():
            raise ValueError('Nonempty verification reason required')
        original = selected_units(value['original_evidence_ids'], units, required=verdict != 'insufficient')
        citations = reconstruct_citations(request, original)
        check.update(original_verification=verdict, original_evidence_ids=value['original_evidence_ids'],
                     original_citations=[asdict(c) for c in citations],
                     reason=value['reason'], unresolved=verdict != 'supported',
                     unresolved_reasons=[] if verdict == 'supported' else [
                         'evidence_insufficient_or_conflicting' if verdict == 'insufficient' else 'unsupported_original'])
    except Exception as exc:
        call.setdefault('error', {'type': type(exc).__name__, 'message': str(exc)})
        return finish('verification_incomplete', error=str(exc))

    candidate = value['candidate_answer']
    try:
        chosen = selected_units(value['candidate_evidence_ids'], units, required=bool(candidate))
        if not candidate and chosen:
            raise ValueError('An absent candidate must not carry source IDs')
    except (ValueError, TypeError) as exc:
        check['candidate_check'] = 'invalid_source'
        return finish('candidate_rejected', citations=citations, error=str(exc), candidate=candidate if isinstance(candidate, str) else None)
    if not isinstance(candidate, str):
        check['candidate_check'] = 'invalid_span'
        return finish('candidate_rejected', citations=citations, error='Candidate must be text')
    if not candidate:
        check['contract_passed'] = True
        return finish('kept' if verdict == 'supported' else 'insufficient_evidence' if verdict == 'insufficient'
                      else 'candidate_unavailable', citations=citations)
    if verdict != 'unsupported':
        check.update(candidate_check='protected', contract_passed=True)
        return finish('kept' if verdict == 'supported' else 'insufficient_evidence', citations=citations, candidate=candidate)
    candidate_citations = reconstruct_citations(request, chosen)
    try:
        if request.dataset == DatasetName.FEVER:
            if candidate not in ('SUPPORTS', 'REFUTES'):
                raise ValueError('Candidate must be a legal FEVER label')
        else:
            if not candidate.strip() or len(candidate.split()) > 20:
                raise ValueError('Candidate must contain 1-20 words')
            for citation in candidate_citations:
                try:
                    candidate, _, _ = locate_excerpt(candidate, citation.quote)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError('Candidate is not a continuous original source span')
    except ValueError as exc:
        check['candidate_check'] = 'invalid_span'
        return finish('candidate_rejected', citations=citations, error=str(exc), candidate=candidate)
    same_label = request.dataset == DatasetName.FEVER and candidate == parse_fever_prediction(request.initial_answer)
    if same_label or candidate == request.initial_answer or (normalize_answer(candidate) and
                                               normalize_answer(candidate) == normalize_answer(request.initial_answer)):
        check.update(candidate_check='protected', contract_passed=True)
        check['unresolved_reasons'].append('unchanged_or_equivalent_candidate')
        return finish('kept', citations=citations, candidate=candidate)
    check.update(candidate_check='passed', contract_passed=True, final_action='replace',
                 unresolved=False, unresolved_reasons=[])
    return finish('repaired', final=candidate, citations=candidate_citations, candidate=candidate)
