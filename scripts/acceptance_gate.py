"""Post-call acceptance gate (configs/controller/acceptance_gate_v1.json): keep C1 when a second context returns to it.

On items where route 1's P3-20 answer differs from C1, the voter (L2 bge20_bm2520_ce_only, answered as P3-20
answers) decides: if its answer normalizes to C1's and not to route 1's, C1 is kept. `score --role sweep` replays
the in-sample sweep number from the stored route-1 records and the L2 sweep generations; `score --role confirm`
applies the gate to the route-1 confirm results with the L2 confirm generations. Both compare the gated answers
with route 1 and with C1 under the judge-assisted metric.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from baselines._lib.answer_scoring import normalize_standard  # noqa: E402
from scripts import l2_confirm, l2_offline  # noqa: E402
from scripts.quality_sweep_offline import load_judged, read_jsonl  # noqa: E402

REGISTRATION = 'configs/controller/acceptance_gate_v1.json'
VOTER = {'k_bge': 20, 'k_bm25': 20, 'fusion': 'ce_only'}


def load_registration(root=ROOT):
    return json.loads((Path(root) / REGISTRATION).read_text(encoding='utf-8'))


def keep_c1(c1, route_1, voter):
    return normalize_standard(voter) == normalize_standard(c1) != normalize_standard(route_1)


def gated(c1, route_1, voter):
    """Route 1's answers, with C1 restored where route 1 changed C1 and the voter returns to C1."""
    return {sid: c1[sid] if route_1[sid] != c1[sid] and keep_c1(c1[sid], route_1[sid], voter[sid]) else route_1[sid]
            for sid in c1}


def sweep_answers(records, candidates, generations):
    c1, route_1, voter = {}, {}, {}
    for r in records:
        sid = r['sample_id']
        c1[sid] = r['initial_answer']
        route_1[sid] = r['actions'][l2_offline.P3]['final_answer'] if l2_offline.P3 in r['actions'] else c1[sid]
        if route_1[sid] != c1[sid]:
            base = tuple(e['passage_id'] for e in r['evidence'])
            voter[sid] = l2_offline.final_answer(c1[sid], l2_offline.select_ids(candidates[sid], VOTER), base, generations, sid)
    return c1, route_1, voter


def confirm_answers(results, candidates, generations):
    c1, route_1, voter = {}, {}, {}
    for r in results:
        sid = r['sample_id']
        c1[sid], route_1[sid] = r['initial_answer'], r['final_answer']
        if route_1[sid] != c1[sid]:
            voter[sid] = l2_offline.final_answer(c1[sid], l2_offline.select_ids(candidates[sid], VOTER),
                                                 l2_confirm.base_ids(r), generations, sid)
    return c1, route_1, voter


def report(role, c1, route_1, voter, gold, judged, registration):
    after = gated(c1, route_1, voter)
    against = l2_confirm.paired(route_1, after, gold, judged)
    vs_c1 = {k: l2_confirm.paired(c1, v, gold, judged) for k, v in (('route_1', route_1), ('gated', after))}
    complete = not against['unjudged'] and not any(v['unjudged'] for v in vs_c1.values())
    out = {'schema_version': 'acceptance_gate_score_v1', 'role': role,
           'registration_sha256': hashlib.sha256((ROOT / REGISTRATION).read_bytes()).hexdigest(),
           'n': len(c1), 'in_scope': len(voter), 'kept_c1': sum(after[s] != route_1[s] for s in c1),
           'complete': complete, 'against_route_1': against, 'against_c1': vs_c1}
    if role == 'confirm':
        out['rule'] = registration['confirm']['rule']
        out['proceed_to_test'] = complete and against['net'] > 0
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('score')
    s.add_argument('--role', choices=('sweep', 'confirm'), required=True)
    s.add_argument('--records', type=Path, help='sweep: route-1 sweep records.jsonl')
    s.add_argument('--coverage', type=Path, help='sweep: step-1 p3_records.jsonl')
    s.add_argument('--results', type=Path, help='confirm: route-1 confirm results.jsonl')
    s.add_argument('--candidates', type=Path, required=True)
    s.add_argument('--generations', type=Path, nargs='*', default=[])
    s.add_argument('--references', type=Path, required=True)
    s.add_argument('--judged', type=Path, nargs='*', default=[])
    s.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    registration = load_registration()
    if args.role == 'sweep':
        records, candidates, generations = l2_offline.load_inputs(args)
        answers = sweep_answers(records, candidates, generations)
    else:
        results, candidates, generations = l2_confirm.load_inputs(SimpleNamespace(
            results=args.results, candidates=args.candidates, generations=args.generations))
        answers = confirm_answers(results, candidates, generations)
    gold = {r['sample_id']: r['gold'] for r in read_jsonl(args.references)}
    out = report(args.role, *answers, gold, load_judged(args.judged), registration)
    args.output.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
