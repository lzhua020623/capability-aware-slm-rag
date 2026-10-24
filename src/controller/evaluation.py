"""Offline T3 acceptance and paired analysis. Never imported by online routing."""
from collections import Counter
import math

import numpy as np
from scipy.stats import beta, binomtest

from src.controller.execution import ARM_BUDGETS, MODE_ACTION, fingerprint, record_action
from src.evaluation.answer_repair import score_answer

FIXED = ('c1', 'always_retrieve', 'always_rerank', 'always_repair')


def paired_statistics(candidate, baseline, *, seed=42, replicates=10000):
    a, b = np.asarray(candidate, dtype=int), np.asarray(baseline, dtype=int)
    if a.shape != b.shape or a.ndim != 1 or not len(a) or not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError('Paired nonempty binary outcomes required')
    delta = a - b
    counts = np.array([(delta == -1).sum(), (delta == 0).sum(), (delta == 1).sum()])
    # Resampling these three paired-outcome counts is exactly the distribution
    # of resampling sample IDs for the accuracy difference; never unpair arms.
    draws = np.random.default_rng(seed).multinomial(len(a), counts / len(a), size=replicates)
    differences = (draws[:, 2] - draws[:, 0]) / len(a)
    gains, losses = int(counts[2]), int(counts[0])
    return {'n': len(a), 'gain': gains, 'loss': losses, 'difference': float(delta.mean()),
            'bootstrap_two_sided_95': np.quantile(differences, [.025, .975]).tolist(),
            'bootstrap_one_sided_95_lower': float(np.quantile(differences, .05)),
            'mcnemar_two_sided_p': float(binomtest(min(gains, losses), gains + losses, .5).pvalue)
            if gains + losses else 1.0, 'replicates': replicates, 'seed': seed}


def paired_noninferiority(candidate, baseline, *, margin=.02):
    """Conservative finite-sample test using simultaneous exact binomial bounds.

    Gain and loss probabilities have binomial marginals even though they are
    dependent. Two one-sided Clopper-Pearson bounds, each at alpha/2, give a
    lower bound on P(gain)-P(loss) by the union bound. Inverting this nested
    bound gives a conservative p-value for delta <= -margin, suitable for
    Holm. This deliberately trades power for finite-sample validity. It is
    NOT a bootstrap tail probability or an independent-proportions test.
    """
    a, b = np.asarray(candidate, dtype=int), np.asarray(baseline, dtype=int)
    paired_statistics(a, b, replicates=1)  # validation only
    n = len(a); gain = int(((a == 1) & (b == 0)).sum()); loss = int(((a == 0) & (b == 1)).sum())
    def lower(alpha):
        alpha = max(float(alpha), 1e-15)
        lo = float(beta.ppf(alpha / 2, gain, n - gain + 1)) if gain else 0.0
        hi = float(beta.ppf(1 - alpha / 2, loss + 1, n - loss)) if loss < n else 1.0
        return lo - hi
    left, right = 0.0, 1.0
    if lower(right) <= -margin:
        pvalue = 1.0
    else:
        for _ in range(55):
            mid = (left + right) / 2
            if lower(mid) > -margin:
                right = mid
            else:
                left = mid
        pvalue = right
    return {'null': f'candidate_minus_baseline <= {-margin}', 'margin': margin,
            'method': 'conservative_exact_gain_loss_CP_union_bound',
            'pvalue': pvalue, 'one_sided_95_lower': lower(.05),
            'limitation': 'Conservative; bootstrap CI separately descriptive; non-establishment is not inferiority'}


def holm(pvalues):
    if not pvalues or any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError('Finite p-values in [0,1] required')
    ordered = sorted(pvalues, key=lambda k: (pvalues[k], k)); adjusted = {}; running = 0.0
    for index, name in enumerate(ordered):
        running = max(running, (len(ordered) - index) * pvalues[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def validate_predictions(predictions, registration):
    expected = {(ds, str(sid), mode) for ds, info in registration['datasets'].items()
                for sid in info['sample_ids'] for mode in registration['required_modes']}
    seen = set(); base_hashes = {}
    for row in predictions:
        key = (row['dataset'], str(row['sample_id']), row['mode'])
        if key in seen or key not in expected:
            raise ValueError(f'Duplicate or unregistered prediction: {key}')
        seen.add(key)
        item = row['input']
        if set(item) != {'sample_id', 'dataset', 'query', 'initial_answer', 'evidence'}:
            raise ValueError('Runtime input includes unregistered/offline fields')
        if row['input_sha256'] != fingerprint(item) or row['initial_answer'] != item['initial_answer']:
            raise ValueError('Forged or mismatched runtime input hash')
        if (item['dataset'], item['sample_id']) != key[:2]:
            raise ValueError('Output and input IDs differ')
        expected_hash = registration['datasets'][key[0]]['effective_input_hashes'][key[1]]
        if row['input_sha256'] != expected_hash:
            raise ValueError('Runtime input differs from frozen effective input')
        base_key = key[:2]
        if base_key in base_hashes and base_hashes[base_key] != row['input_sha256']:
            raise ValueError('Different C1 inputs across arms')
        base_hashes[base_key] = row['input_sha256']
        counts = {kind: sum(e['kind'] == kind and e['attempted'] for e in row['calls'])
                  for kind in ('llm', 'retrieval', 'scoring')}
        limits = ARM_BUDGETS[row['mode']]
        for index, kind in enumerate(('llm', 'retrieval', 'scoring')):
            if counts[kind] != row['cost'][kind + '_calls'] or counts[kind] > limits[index]:
                raise ValueError('Invalid call accounting or per-arm budget')
        allocated = sum(e['arguments']['max_new_tokens'] for e in row['calls'] if e['kind'] == 'llm' and e['attempted'])
        if allocated > limits[3]:
            raise ValueError('Aggregate output-token budget exceeded')
        for event in row['calls']:
            if event['kind'] == 'llm' and event['arguments']['prompt_sha256'] != __import__('hashlib').sha256(
                    event['arguments']['prompt'].encode()).hexdigest():
                raise ValueError('Prompt hash mismatch')
        if row['answer_changed'] != (row['initial_answer'] != row['final_answer']):
            raise ValueError('Incorrect answer-changed receipt')
    if seen != expected:
        raise ValueError(f'Incomplete frozen comparison: missing {len(expected - seen)} rows')
    return {'complete': True, 'n_records': len(predictions), 'n_expected': len(expected),
            'required_modes': registration['required_modes'],
            'execution_errors': sum(bool(r['cost']['failed_calls'] or r['cost']['budget_blocks']) for r in predictions)}


def transition(initial, final):
    n = len(initial); fixed = sum(not a and b for a, b in zip(initial, final)); harmed = sum(a and not b for a, b in zip(initial, final))
    n_correct = sum(initial); n_wrong = n - n_correct
    return {'n': n, 'initial_correct': n_correct, 'initial_wrong': n_wrong,
            'fixed': fixed, 'regressed': harmed, 'net_correct': fixed - harmed,
            'repair_rate': fixed / n_wrong if n_wrong else None,
            'regression_rate': harmed / n_correct if n_correct else None}


def dataset_analysis(rows, references, dataset, *, replicates=10000):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['mode'], {})[str(row['sample_id'])] = row
    ids = sorted(grouped['c1'])
    def outcomes(mode):
        return np.array([score_answer(grouped[mode][sid]['final_answer'], references[(dataset, sid)], dataset)['correct']
                         for sid in ids], dtype=int)
    base = outcomes('c1'); correct = {mode: outcomes(mode) for mode in grouped}
    adaptive = correct['adaptive']; oracle = np.maximum.reduce([correct[m] for m in FIXED])
    comparisons = {f'{mode}_vs_c1': paired_statistics(values, base, replicates=replicates)
                   for mode, values in correct.items() if mode != 'c1'}
    comparisons['adaptive_vs_full'] = paired_statistics(adaptive, correct['full_enhancement'], replicates=replicates)
    if 'adaptive_no_probe' in correct:
        comparisons['adaptive_vs_no_probe'] = paired_statistics(adaptive, correct['adaptive_no_probe'], replicates=replicates)
    comparisons['adaptive_vs_full']['noninferiority'] = paired_noninferiority(adaptive, correct['full_enhancement'])
    changes = {mode: transition(base.tolist(), values.tolist()) for mode, values in correct.items()}
    routed = {}; replay_agreement = 0
    action_modes = {action.value: mode for mode, action in MODE_ACTION.items()}
    for action, mode in action_modes.items():
        indexes = [i for i, sid in enumerate(ids) if record_action(grouped['adaptive'][sid]) == action]
        routed[action] = {'actual_adaptive': transition(base[indexes].tolist(), adaptive[indexes].tolist()),
                          'fixed_action_on_same_subset': transition(base[indexes].tolist(), correct[mode][indexes].tolist())}
        replay_agreement += sum(grouped['adaptive'][ids[i]]['final_answer'] == grouped[mode][ids[i]]['final_answer']
                                for i in indexes)
    opportunities = {'oracle_correct': int(oracle.sum()), 'adaptive_correct': int(adaptive.sum()),
        'avoidable_harm': int(((base == 1) & (adaptive == 0)).sum()),
        'missed_repair_opportunities': int(((base == 0) & (oracle == 1) & (adaptive == 0)).sum()),
        'fixed_action_residual_errors': int((oracle == 0).sum()),
        'adaptive_outside_observed_oracle': int(((adaptive == 1) & (oracle == 0)).sum()),
        'selected_action_answer_agreement': replay_agreement, 'n': len(ids),
        'scope': 'Offline observed action set including keep; not deployable, not intrinsic model limit or pure Diagnosis causality'}
    cost = {}
    for mode in ('adaptive', 'full_enhancement'):
        batch = list(grouped[mode].values())
        missing = sum(r['cost']['input_tokens'] is None or r['cost']['output_tokens_estimate'] is None for r in batch)
        total = None if missing else sum(r['cost']['input_tokens'] + r['cost']['output_tokens_estimate'] for r in batch)
        cost[mode] = {'tokens_input_plus_estimated_output': total, 'missing_token_rows': missing,
                      'llm_calls': sum(r['cost']['llm_calls'] for r in batch),
                      'seconds': sum(r['total_seconds'] for r in batch)}
    a, f = (cost[m]['tokens_input_plus_estimated_output'] for m in ('adaptive', 'full_enhancement'))
    reduction = 1 - a / f if a is not None and f else None
    cost.update(incremental_token_reduction_estimate=reduction,
                incremental_token_point_target_met=reduction >= .20 if reduction is not None else None,
                full_pipeline_AC5='not_certified_without_common_C1_cost_and_complete_token_accounting',
                scope='after cached C1; includes Probe; output tokens are retokenized estimates')
    return {'comparisons': comparisons, 'transitions': changes, 'routed_subsets': routed,
            'routing_opportunities': opportunities, 'cost_comparison': cost}


def analyze(predictions, gold, registration, *, replicates=10000):
    acceptance = validate_predictions(predictions, registration)
    references = {}
    for row in gold:
        key = (row['dataset'], str(row['sample_id']))
        if key in references:
            raise ValueError('Duplicate reference')
        references[key] = row
    expected = {(ds, str(sid)) for ds, info in registration['datasets'].items() for sid in info['sample_ids']}
    if set(references) != expected:
        raise ValueError('References must exactly cover frozen evaluated inputs')
    analyses = {ds: dataset_analysis([r for r in predictions if r['dataset'] == ds], references, ds,
                                   replicates=replicates) for ds in registration['datasets']}
    pvalues = {}
    for ds, report in analyses.items():
        pvalues[ds + ':adaptive_vs_c1'] = report['comparisons']['adaptive_vs_c1']['mcnemar_two_sided_p']
        pvalues[ds + ':adaptive_vs_full_NI'] = report['comparisons']['adaptive_vs_full']['noninferiority']['pvalue']
    # Both datasets must be present for the predeclared four-hypothesis family.
    family_complete = set(analyses) == {'nq', 'fever'}
    corrected = holm(pvalues) if family_complete else None
    return {'schema_version': 'controller_t3_analysis_v1', 'acceptance': acceptance, 'datasets': analyses,
            'holm_family_complete': family_complete, 'holm_adjusted_pvalues': corrected,
            'inference_role': 'development_descriptive_not_final_confirmation' if registration['role'] == 'development'
            else 'frozen_final_comparison',
            'human_type_comparator': 'not_executed_no_reliable_new_cohort_types',
            'semantic_review': 'NQ changed-answer union and ambiguous bounds separate; FEVER support/AC6 still requires review'}
