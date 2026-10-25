"""Registered runtime action mapping and offline E activation, kept separate.

Only choose_deployment_modes consumes offline scores. route_override consumes
runtime Diagnosis/Probe records and never sees references or correctness.
"""
import math

GRID = (.2, .3, .4)


def route_override(record, threshold):
    if isinstance(threshold, bool) or threshold not in GRID:
        raise ValueError('Threshold outside the registered three-point grid')
    decision = record['routing']['decision']
    original = decision['action']
    result = {'action': original, 'overridden': False, 'reason': 'outside_NQ_keep_scope',
              'original_decision': decision, 'threshold': threshold}
    if record['dataset'] != 'nq' or original != 'no_action':
        return result
    probe = record['routing'].get('probe')
    if probe and probe.get('answer_supported') == 'supported':
        result['reason'] = 'existing_supported_probe_veto'
        return result
    probability = record['routing']['diagnosis']['stage1_failure_probability']
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Invalid recorded Stage-1 probability')
    if probability >= threshold:
        result.update(action='retrieval_enhancement', overridden=True,
                      reason='NQ_low_confidence_keep_to_retrieve')
    else:
        result['reason'] = 'below_registered_threshold'
    return result


def choose_deployment_modes(dataset_stats):
    """Stats are offline; the zero-sized FEVER E route is NOT zero-risk proof.

    n is the complete development denominator; routed contains original and
    diagnostic-final correctness of the fixed T4 E subset, never selected IDs.
    """
    if set(dataset_stats) != {'nq', 'fever'}:
        raise ValueError('Both dataset gate results required')
    for s in dataset_stats.values():
        n = s['n']
        if type(n) is not int or n < 200 or any(type(s[k]) is not int or not 0 <= s[k] <= n
                                               for k in ('contract_passed', 'unresolved_or_insufficient')):
            raise ValueError('Invalid complete development denominator')
    if any(s['contract_passed'] / s['n'] < .9 for s in dataset_stats.values()):
        return {ds: {'mode': 'unavailable', 'reason': 'development_engineering_gate_failed'} for ds in dataset_stats}
    result = {}
    for ds, s in dataset_stats.items():
        routed = s['routed']
        if len(routed) != (16 if ds == 'nq' else 0):
            raise ValueError('Wrong fixed T4 E reference count')
        if s['unresolved_or_insufficient'] / s['n'] > .9:
            result[ds] = {'mode': 'verify_only', 'reason': 'review_line_exceeded'}
        elif not routed:
            result[ds] = {'mode': 'verify_only', 'reason': 'routed_net_unestimable'}
        else:
            if any(type(r['initial_correct']) is not bool or type(r['final_correct']) is not bool for r in routed):
                raise ValueError('Offline correctness must be explicit booleans')
            net = sum(int(r['final_correct']) - int(r['initial_correct']) for r in routed)
            result[ds] = {'mode': 'allow_repair' if net >= 0 else 'verify_only',
                          'reason': 'routed_net_nonnegative' if net >= 0 else 'routed_net_negative',
                          'net_correct': net, 'weighted_net': net / len(routed), 'reference_n': len(routed)}
    return result
