"""Single-call E interface. Provenance and structure are not semantic truth.

The original verdict survives an invalid candidate. Deployment protection is
separate from the diagnostic candidate, so a saved run can be gated offline.
r2: one label definition for both datasets; the model says contradicted and
code maps it to the internal unsupported state, keeping the raw label.
"""
from dataclasses import asdict, replace
import hashlib
import json
import re

from src.compensation.answer_repair import Citation, Generation, RepairResult
from src.compensation.evidence_id_repair import (
    RULES, context, evidence_units, reconstruct_citations, selected_units, strict_object,
)
from src.compensation.short_answer_repair import locate_excerpt
from src.diagnosis import DatasetName
from src.evaluation.qa import normalize_answer
from src.evaluation.fever import parse_fever_prediction

VERSION = 'evidence_id_v2a_v3'
MODES = ('allow_repair', 'verify_only', 'unavailable')
LIMIT = 384
FIELDS = ('original_verification', 'verdict_evidence_ids', 'replacement_answer',
          'replacement_evidence_ids', 'reason')
MODEL_LABELS = {'supported': 'supported', 'contradicted': 'unsupported', 'insufficient': 'insufficient'}
DEFINITIONS = (
    'Judge the ENTERING ANSWER against the evidence with exactly one label; the labels mean the same '
    'for every dataset. supported: the evidence establishes that the entering answer is correct. '
    'contradicted: the evidence explicitly establishes that the entering answer is wrong, because it '
    'gives a different value for the exact requested fact or rules the entering answer out. '
    'insufficient: the evidence neither establishes nor rules out the entering answer; the requested '
    'fact is absent and only related facts appear, or relevant passages give incompatible answers and '
    'the evidence does not state which is right. Missing information is insufficient, NEVER '
    'contradicted. In a conflict, source names, authority, recency or rank do not decide, and unstated '
    'events must not be invented to reconcile passages. '
)
_NUMBER_WORDS = ('one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen '
                 'sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety '
                 'hundred thousand million billion dozen').split()
_MONTHS = ('january february march april may june july august september october november december '
           'century decade spring summer autumn fall winter').split()
_FUNCTION_WORDS = set('a an the in on at of to by for from with and or it its this that these those he she they'.split())


def type_warnings(query, answer):
    """Surface-form hints only; recorded, never used to block or change an answer."""
    q, words = query.lower().strip(), re.findall(r"[A-Za-z]+|\d+", answer)
    lower = {w.lower() for w in words}
    has_number = any(w.isdigit() for w in words) or bool(lower & set(_NUMBER_WORDS))
    has_name = any(w[0].isupper() and w.lower() not in _FUNCTION_WORDS for w in words if w.isalpha())
    if q.startswith('when') or re.search(r'\bwhat (year|date)\b|\bwhich year\b', q):
        return [] if has_number or lower & set(_MONTHS) else ['time_question_without_time_expression']
    if re.search(r'\bhow (many|much)\b', q):
        return [] if has_number else ['quantity_question_without_number']
    if q.startswith('who') or q.startswith('where'):
        return [] if has_name else [('person' if q.startswith('who') else 'place') + '_question_without_name']
    return []


def schema(units, request=None):
    def ids(minimum, maximum=5):
        return {'type': 'array', 'items': {'$ref': '#/$defs/source_id'},
                'minItems': minimum, 'maxItems': maximum}
    branches = []
    for status, replacement in (('supported', False), ('contradicted', False),
                                ('contradicted', True), ('insufficient', False)):
        answer = {'type': 'string', 'minLength': 1} if replacement else {'const': ''}
        if replacement and request is not None and request.dataset == DatasetName.FEVER:
            entering = parse_fever_prediction(request.initial_answer)
            answer = {'enum': [v for v in ('SUPPORTS', 'REFUTES') if v != entering]}
        props = {'original_verification': {'const': status},
                 'verdict_evidence_ids': ids(0 if status == 'insufficient' else 1),
                 'replacement_answer': answer,
                 'replacement_evidence_ids': ids(1) if replacement else ids(0, 0),
                 'reason': {'type': 'string', 'minLength': 1}}
        branches.append({'type': 'object', 'properties': props,
                         'required': list(FIELDS), 'additionalProperties': False})
    return {'$defs': {'source_id': {'type': 'string', 'enum': [u.id for u in units]}}, 'anyOf': branches}


def prompt(request, units):
    task = (
        'FEVER: the query is the ORIGINAL claim and the entering answer is its SUPPORTS/REFUTES '
        'label. The label is supported when it is SUPPORTS and the evidence establishes the claim, '
        'or when it is REFUTES and the evidence explicitly contradicts the claim. The label is '
        'contradicted when the evidence establishes the opposite label. A replacement is only the '
        'opposite label.'
        if request.dataset == DatasetName.FEVER else
        'NQ: the entering answer is a short answer to the question. A replacement must be the '
        'shortest continuous exact span in selected original evidence that answers the question, '
        'at most 20 words.'
    )
    return (RULES + DEFINITIONS + task + '\nIn ONE response verify the entering answer and '
        'optionally propose a minimal replacement. original_verification is supported, '
        'contradicted or insufficient. supported: keep even if a longer wording is possible; '
        'replacement_answer is empty. contradicted: provide a replacement only if the evidence '
        'establishes the correct answer; otherwise leave it empty, which does not turn '
        'contradicted into supported. insufficient: leave the replacement empty; name any '
        'conflict in reason. For supported and '
        'contradicted select 1-5 unique verdict_evidence_ids for the JUDGMENT: these facts '
        'may support OR contradict the entering answer. replacement_answer is the NEW '
        'correct answer, not the entering answer being checked. For a replacement select '
        '1-5 replacement_evidence_ids. Empty replacements have no IDs. '
        'Use only exact sentence keys, never generate quotations. Return JSON only.\n'
        f'Query: {json.dumps(request.query, ensure_ascii=False)}\n'
        f'Entering answer: {json.dumps(request.initial_answer, ensure_ascii=False)}\n'
        f'Evidence: {context(request, units)}\n'
        'Output JSON Schema: ' + json.dumps(schema(units, request), separators=(',', ':')))


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
    check = {'version': VERSION, 'original_verification': 'incomplete', 'model_label': None,
             'candidate_check': 'absent', 'final_action': 'keep', 'contract_passed': False,
             'unresolved': True, 'unresolved_reasons': ['verification_incomplete'], 'type_warnings': []}
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
            'schema_sha256': hashlib.sha256(json.dumps(schema(units, request), sort_keys=True).encode()).hexdigest(),
            'max_new_tokens': LIMIT}
    trace['calls'].append(call)
    try:
        observed, metadata = backend.generate_schema(text, LIMIT, schema(units, request))
        if not isinstance(observed, Generation):
            raise TypeError('Structured backend must return Generation')
        generated = observed
        call.update(generation=asdict(generated), decoder=metadata)
        if metadata.get('hit_token_budget'):
            raise ValueError('Output reached the token budget; verification not accepted')
        value = strict_object(generated.text, FIELDS)
        if value['original_verification'] not in MODEL_LABELS:
            raise ValueError('Invalid original verification')
        check['model_label'] = value['original_verification']
        verdict = MODEL_LABELS[value['original_verification']]
        if not isinstance(value['reason'], str) or not value['reason'].strip():
            raise ValueError('Nonempty verification reason required')
        original = selected_units(value['verdict_evidence_ids'], units, required=verdict != 'insufficient')
        citations = reconstruct_citations(request, original)
        check.update(original_verification=verdict, verdict_evidence_ids=value['verdict_evidence_ids'],
                     original_citations=[asdict(c) for c in citations],
                     reason=value['reason'], unresolved=verdict != 'supported',
                     unresolved_reasons=[] if verdict == 'supported' else [
                         'evidence_insufficient_or_conflicting' if verdict == 'insufficient' else 'unsupported_original'])
    except Exception as exc:
        call.setdefault('error', {'type': type(exc).__name__, 'message': str(exc)})
        return finish('verification_incomplete', error=str(exc))

    candidate = value['replacement_answer']
    if request.dataset == DatasetName.NQ and isinstance(candidate, str) and candidate.strip():
        check['type_warnings'] = type_warnings(request.query, candidate)
    try:
        chosen = selected_units(value['replacement_evidence_ids'], units, required=bool(candidate))
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
        check.update(candidate_check='protected', contract_passed=False)
        return finish('kept' if verdict == 'supported' else 'insufficient_evidence', citations=citations,
                      error='Replacement forbidden for supported/insufficient verdict', candidate=candidate)
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
