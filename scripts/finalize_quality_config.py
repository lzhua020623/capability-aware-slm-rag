"""Turn the offline sweep choice into the production P60 config and registration.

Before writing, the production FrozenOnlineSelectionAdapter is replayed on the stored sweep
records (Diagnosis, 78 features) with the new config, and must select exactly the items the
offline calibration selected; the production correct gate must open on exactly the items the
offline grid sent to P3-20; and the original config must reproduce the stored frozen
selections. The new registration keeps the original one untouched.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.precompute_quality_sweep import LABEL_FOR_ACTION, gated_correct, load_registration
from scripts.quality_sweep_offline import SELECTOR, calibrate, in_pool, read_records

BASELINE = 'configs/controller/current_baseline.json'
P60_OUT = 'configs/controller/p60_quality_v1.json'
REGISTRATION_OUT = 'configs/controller/quality_baseline_v1.json'
SWEEP = 'configs/controller/quality_sweep_v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_crlf(path, payload):
    """CRLF bytes, as a core.autocrlf=true checkout writes them, so registered hashes hold after deploy."""
    Path(path).write_bytes((json.dumps(payload, ensure_ascii=False, indent=1) + '\n').replace('\n', '\r\n').encode())


def diagnosis_of(record):
    from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
    from src.diagnosis.labels import DiagnosisLabel
    d = record['diagnosis']
    labels = lambda scores: None if scores is None else {DiagnosisLabel(k): v for k, v in scores.items()}
    return HierarchicalDiagnosis(d['stage1_failure_probability'], d['stage1_label'], labels(d['stage2_class_scores']),
                                 DiagnosisLabel(d['final_label']), d['confidence'], labels(d['class_scores']))


def replay(records, config):
    """IDs the production adapter selects with `config`, from stored Diagnosis and features."""
    from src.controller.selection import FrozenOnlineSelectionAdapter, _load_models
    adapter = FrozenOnlineSelectionAdapter(config, _load_models(config['models'], ROOT))
    return {r['sample_id'] for r in records
            if adapter.select(diagnosis_of(r), r['features'], sample_id=r['sample_id']).needs_enhancement}


def gate_for(chosen, sweep):
    """Registered correct gate: the cut sends correct-labelled items to P3-20 (the insufficiency route)."""
    if chosen['correct_gate'] == 'closed':
        return None
    return {'stage1_p_fail_at_least': sweep['precompute']['correct_p_fail_cut'], 'route': 'evidence_insufficiency'}


def gate_replay(records, gate):
    from src.pipeline.adaptive_rag import correct_gate_opens
    return {r['sample_id'] for r in records if correct_gate_opens(diagnosis_of(r), gate)}


def routes_for(action_map):
    """Failure label -> label whose action it takes (or no_action), from a registered action map."""
    return {label: 'no_action' if action == 'no_action' else LABEL_FOR_ACTION[action]
            for label, action in action_map.items()}


def p60_config(base, chosen, thresholds, pool_n):
    config = copy.deepcopy(base)
    config.update(version='p60_quality_v1', source='EXP-P60 models; thresholds and z-stats recalibrated on the '
                  'quality sweep failure pool (configs/controller/quality_sweep_v1.json)',
                  coverage_target=chosen['coverage'], lane1_frac=chosen['insuff_share'],
                  thresholds={'t_insuff': thresholds['t_insuff'], 't_combo': thresholds['t_combo']},
                  z_stats=thresholds['z_stats'], calibration={'role': 'Validation sweep (6,000 IDs)',
                                                              'failure_pool_n': pool_n})
    return config


def registration_for(base, chosen, report_sha):
    sweep = json.loads((ROOT / SWEEP).read_text(encoding='utf-8'))
    out = copy.deepcopy(base)
    out.update(baseline='quality_round1_v1', registered_date='2026-10-30', upstream_registration=BASELINE,
               upstream_registration_sha256=sha(ROOT / BASELINE), selector_config=P60_OUT,
               failure_type_routes=routes_for(sweep['grid']['action_maps'][chosen['action_map']]),
               correct_gate=gate_for(chosen, sweep), c1_prefill='full', quality_sweep={'registration': SWEEP, 'sha256': sha(ROOT / SWEEP),
                                                 'offline_report_sha256': report_sha, 'chosen': chosen})
    out['frozen_assets'] = {**base['frozen_assets'], P60_OUT: None}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True, help='Offline sweep report on the full sweep')
    p.add_argument('--records', type=Path, required=True)
    args = p.parse_args(argv)
    report = json.loads(args.report.read_text(encoding='utf-8'))
    if report['partial'] or report['metric'] != 'net':
        raise ValueError('Finalize only from the complete, fully judged sweep')
    chosen = report['chosen']
    sweep, ids = load_registration()
    records = read_records(args.records, ids)
    pool = [r for r in records if in_pool(r)]
    base = json.loads((ROOT / SELECTOR).read_text(encoding='utf-8'))
    frozen = {r['sample_id'] for r in records if r['frozen_selection']['needs_enhancement']}
    if replay(records, base) != frozen:
        raise RuntimeError('Stored records do not replay under the original P60 config')
    cal = calibrate(pool, chosen['coverage'], chosen['insuff_share'], base['d_combo']['alpha'], base['d_combo']['lam'])
    new = p60_config(base, chosen, cal, len(pool))
    if replay(records, new) != cal['selected'] or any(gated_correct(r['diagnosis']) for r in records
                                                      if r['sample_id'] in cal['selected']):
        raise RuntimeError('Production selection differs from the offline calibration')
    gate = gate_for(chosen, sweep)
    offline_gate = {r['sample_id'] for r in records if gated_correct(r['diagnosis']) and gate is not None
                    and r['diagnosis']['stage1_failure_probability'] >= gate['stage1_p_fail_at_least']}
    if gate_replay(records, gate) != offline_gate:
        raise RuntimeError('Production correct gate differs from the offline grid')
    write_crlf(ROOT / P60_OUT, new)
    registration = registration_for(json.loads((ROOT / BASELINE).read_text(encoding='utf-8')), chosen,
                                    sha(args.report))
    registration['frozen_assets'][P60_OUT] = sha(ROOT / P60_OUT)
    write_crlf(ROOT / REGISTRATION_OUT, registration)
    print(json.dumps({'selected': len(cal['selected']), 'pool': len(pool), 'gate_opened': len(offline_gate),
                      'chosen': chosen, 'routes': registration['failure_type_routes'],
                      'correct_gate': registration['correct_gate']}, indent=1))
    return 0


if __name__ == '__main__': raise SystemExit(main())
