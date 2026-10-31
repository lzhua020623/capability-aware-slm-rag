"""Adaptive multi-context answer selection on top of B+BM25 (configs/controller/adaptive_selection_v1.json).

After route 1 and P3-BM25 have given the product answer, an item enters when route 1 sent it to P3 or its
stage-1 failure probability is at least theta. One BGE top-100 and one BM25 top-50 search feed one GPU
cross-encoder pass over their union; each registered context then picks its Top-5 from those lists, and
its answer follows P3-20 rules: C1's answer when the Top-5 equals C1's, otherwise build_c1_prompt and the
shared generator, with answer_rejection keeping C1. Top-5 lists the item already answered (P3-20, P3-BM25)
are reused. Stages add contexts in the registered order (3, 5, 8) until the largest answer group among C1
and the stage's answers holds at least tau; that stage's logistic selector keeps the product answer unless
another group has a higher predicted probability. Errors keep the product answer, with the error traced.

The pure functions below are the ones the offline evaluation (scripts/adaptive_selection.py) used.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import math
from time import perf_counter

from .retrieval_enhancement.p3 import answer_rejection, build_c1_prompt
from .retrieval_enhancement.p3_l2 import select_l2, union_candidates

FEATURES = ('votes', 'share', 'is_c1', 'is_base', 'n_support', 'max_ce', 'best_bge', 'best_bm25', 'c1_support',
            'tokens', 'p_fail', 'routed', 'n_groups', 'top_votes', 'margin')
STAGES = (3, 5, 8)
BGE_DEPTH, BM25_DEPTH = 100, 50


def setting(name):
    """'bge20_bm2520_ce_only' -> (20, 20, 'ce_only')."""
    bge, rest = name[3:].split('_bm25')
    bm25, fusion = rest.split('_', 1)
    return int(bge), int(bm25), fusion


def select_context(bge_ids, bm25_ids, ce, name):
    """Top-5 passage IDs of one context from the ranked ID lists and the cross-encoder scores."""
    k_bge, k_bm25, fusion = setting(name)
    pool = union_candidates([{'passage_id': p, 'text': p} for p in bge_ids[:k_bge]],
                            [{'passage_id': p, 'text': p} for p in bm25_ids[:k_bm25]])
    return tuple(r['passage_id'] for r in select_l2(pool, [ce[r['passage_id']] for r in pool], fusion))


def context_answer(c1, top5, c1_top5, answered):
    """P3-20's rule for one Top-5: C1 when it equals C1's Top-5, else the answer unless rejected."""
    if tuple(top5) == tuple(c1_top5):
        return c1
    answer, reason = answered[tuple(top5)]
    return c1 if reason is not None else answer.strip()


def norm(answer):
    from baselines._lib.answer_scoring import normalize_standard
    return normalize_standard(answer)


def supports(answer, text):
    """The answer_rejection grounding test: the normalized answer is a token span of the normalized passage."""
    from src.evaluation.qa import normalize_answer
    needle = normalize_answer(answer)
    return bool(needle) and f' {needle} ' in f' {normalize_answer(text)} '


def groups(item, names, texts):
    """Candidate answer groups of C1 plus the named contexts, with FEATURES; the product answer's group is always present."""
    pool = [item['c1'], *(item['contexts'][n]['answer'] for n in names)]
    counts = Counter(norm(a) for a in pool)
    rep = {}
    for a in pool:
        rep.setdefault(norm(a), a)
    rep.setdefault(norm(item['base']), item['base'])
    counts.setdefault(norm(item['base']), 0)
    allowed = list(dict.fromkeys(item['c1_top5'] + [p for n in names for p in item['contexts'][n]['top5']]))
    ranked = sorted(counts.values(), reverse=True) + [0]
    out = []
    for g, a in rep.items():
        sup = [p for p in allowed if supports(a, texts[p])]
        ce = [item['ce'][p] for p in sup if p in item['ce']]
        out.append({'answer': a, 'x': [
            counts[g], counts[g] / len(pool), float(g == norm(item['c1'])), float(g == norm(item['base'])),
            len(sup), max(ce) if ce else -12.0, min([item['bge_rank'].get(p, 101) for p in sup] or [101]),
            min([item['bm25_rank'].get(p, 51) for p in sup] or [51]), float(any(p in item['c1_top5'] for p in sup)),
            len(g.split()), item['p_fail'], float(item['routed']), len(counts), ranked[0], ranked[0] - ranked[1]]})
    return out


def probability(model, x):
    """Standardized logistic regression from its stored parameters (sklearn StandardScaler + LogisticRegression)."""
    z = sum(w * (v - m) / s for w, v, m, s in zip(model['coef'], x, model['mean'], model['scale'])) + model['intercept']
    return 1.0 / (1.0 + math.exp(-z))


def decide_groups(gs, base, decision, model=None):
    if len(gs) == 1:
        return base
    if decision == 'vote':
        top = max(g['x'][0] for g in gs)
        winners = [g for g in gs if g['x'][0] == top]
        return base if any(g['x'][3] for g in winners) or len(winners) > 1 else winners[0]['answer']
    p = [probability(model, g['x']) for g in gs]
    at_base = next(i for i, g in enumerate(gs) if g['x'][3] == 1.0)
    best = max(range(len(gs)), key=lambda i: (p[i], -i))
    return gs[best]['answer'] if p[best] > p[at_base] else base


class AdaptiveSelection:
    """policy: artifacts/controller/adaptive_selection_v1/policy.json.

    bge(query, k) -> [{'passage_id', 'text', ...}] in BGE order; bm25(query, k) -> [[corpus position, score]];
    record(position) -> (passage_id, text); position(passage_id) -> corpus position; scorer(query, texts) ->
    scores; generate_fn(prompt) -> answer text.
    """

    def __init__(self, policy, bge: Callable, bm25: Callable, record: Callable, position: Callable,
                 scorer: Callable, generate_fn: Callable[[str], str]):
        if tuple(policy['stages']) != STAGES or tuple(policy['features']) != FEATURES:
            raise ValueError('Policy stages or features differ from this module')
        self.order, self.chosen, self.models = policy['order'], policy['chosen'], policy['models']
        self.bge, self.bm25, self.record, self.position = bge, bm25, record, position
        self.scorer, self.generate_fn = scorer, generate_fn

    def run(self, item: dict) -> dict:
        """item: query, c1, c1_top5, base, routed, p_fail, answered {Top-5 tuple: (answer, reject reason)}.

        Returns the trace; its final_answer is the product answer unless the selection chose another.
        """
        started = perf_counter()
        base = item['base']
        trace = {'policy': 'adaptive_selection_v1', **self.chosen, 'entered': False, 'stage': 0, 'contexts': {},
                 'groups': [], 'generations': 0, 'final_answer': base, 'changed': False, 'evidence_from': None,
                 'evidence': None,
                 'reason': None, 'error': None, 'seconds': 0.0}
        done = lambda reason: {**trace, 'reason': reason, 'seconds': perf_counter() - started}
        if not (item['routed'] or item['p_fail'] >= self.chosen['theta']):
            return done('below_gate')
        trace['entered'] = True
        query, c1, c1_top5 = item['query'], item['c1'], tuple(item['c1_top5'])
        stage = 'retrieval'
        try:
            bge_hits = self.bge(query, BGE_DEPTH)
            bm25_hits = self.bm25(query, BM25_DEPTH)
            texts = {h['passage_id']: h['text'] for h in bge_hits}
            bm25_ids = []
            for position, _ in bm25_hits:
                pid, text = self.record(position)
                texts.setdefault(pid, text)
                bm25_ids.append(pid)
            union = list(texts)
            stage = 'scoring'
            scores = [float(s) for s in self.scorer(query, [texts[p] for p in union])]
            if len(scores) != len(union):
                raise ValueError('scorer must return one score per pooled passage')
            ce = dict(zip(union, scores))
            for pid in c1_top5:
                if pid not in texts:
                    texts[pid] = self.record(self.position(pid))[1]
            bge_ids = [h['passage_id'] for h in bge_hits]
            view = {'c1': c1, 'base': base, 'c1_top5': list(c1_top5), 'routed': item['routed'], 'p_fail': item['p_fail'],
                    'ce': ce, 'bge_rank': {p: i + 1 for i, p in enumerate(bge_ids)},
                    'bm25_rank': {p: i + 1 for i, p in enumerate(bm25_ids)}, 'contexts': {}}
            answered = {tuple(k): v for k, v in item['answered'].items()}
            for k in STAGES:
                for name in self.order[len(view['contexts']):k]:
                    stage = 'selection'
                    top5 = select_context(bge_ids, bm25_ids, ce, name)
                    source = 'c1' if top5 == c1_top5 else 'reused' if top5 in answered else 'generated'
                    if source == 'generated':
                        stage = 'generation'
                        passages = [{'passage_id': p, 'text': texts[p]} for p in top5]
                        answer = self.generate_fn(build_c1_prompt(query, passages))
                        if not isinstance(answer, str):
                            raise TypeError('generate_fn must return a string')
                        answered[top5] = (answer, answer_rejection(answer, passages))
                        trace['generations'] += 1
                    view['contexts'][name] = {'top5': list(top5), 'answer': context_answer(c1, top5, c1_top5, answered),
                                              'source': source}
                stage = 'decision'
                gs = groups(view, self.order[:k], texts)
                if max(g['x'][1] for g in gs) >= self.chosen['tau'] or k == STAGES[-1]:
                    break
            model = self.models[str(k)]
            final = decide_groups(gs, base, self.chosen['decision'], model)
        except Exception as exc:
            trace['error'] = {'stage': stage, 'type': type(exc).__name__, 'message': str(exc)}
            return done(f'{stage}_failed')
        trace.update(stage=k, contexts=view['contexts'], final_answer=final, changed=final != base,
                     groups=[{**g, 'p': probability(model, g['x'])} for g in gs])
        if final != base:
            # The chosen string is C1's or the first context answer of its group (groups keep the first spelling).
            source = 'c1' if final == c1 else next(n for n in self.order[:k] if view['contexts'][n]['answer'] == final)
            trace['evidence_from'] = source
            if source != 'c1':
                trace['evidence'] = [{'rank': i, 'passage_id': p, 'text': texts[p], 'score': ce[p]}
                                     for i, p in enumerate(view['contexts'][source]['top5'], 1)]
        return done('selector_kept_product' if final == base else 'selector_switched')


def deep_backend(backend, depth=BGE_DEPTH):
    """The product NQ backend with its top-k cap raised (as scripts/l2_candidates.py); retrieval and order unchanged."""
    from .retrieval_enhancement.adapters import NQFaissBackend

    class DeepNQFaissBackend(NQFaissBackend):
        def __init__(self, retriever, max_top_k):
            self.max_top_k, self.retriever, self._resource_signature = max_top_k, retriever, None
            self._check_resources()

    return DeepNQFaissBackend(backend.retriever, depth)
