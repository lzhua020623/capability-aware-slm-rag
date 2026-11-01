"""Write the current product's NQ Test results in the team baseline output format, replayed from archived answers.

Outputs go where the team runner and judge preset put the current product (baselines/configs/adaptive_product_v2.json):
baselines/outputs/adaptive_product_v2_test/ (predictions.jsonl, metrics.json, run_config.json) and
baselines/outputs/judge/adaptive_product_v2_full/ (judge_results.jsonl, judge_summary.json, run_config.json).

No GPU and no judge call. Answers are the archived Test run of the frozen product (route 1, P3-BM25, adaptive
selection; artifacts/controller/final_evaluation_v1), whose production wiring matched the offline answers 2440/2440
on confirm (artifacts/controller/adaptive_product_equivalence_v1). Rows are built field for field like
baselines/run_product_evaluation.py's build_product_row; the final evidence follows the product rule in
final_evidence_ids, which equals the confirm production run on all 2440 items. Judge labels are reused by
(sample_id, answer) in the order of final_evaluation_v1, normalized matches auto-correct as in the team judge.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines._lib.answer_scoring import answer_in_context, score_answer  # noqa: E402
from baselines._lib.judge_io import build_result_row, summarize_results  # noqa: E402
from baselines.run_product_evaluation import MODE, aggregate_product_metrics  # noqa: E402
from scripts import adaptive_selection as ad  # noqa: E402
from scripts import bm25_gate as gate  # noqa: E402
from scripts import final_evaluation as fe  # noqa: E402
from scripts import l2_offline  # noqa: E402
from scripts import score_quality_eval as scorer  # noqa: E402

CONFIG = 'baselines/configs/adaptive_product_v2.json'
PRED_DIR = 'baselines/outputs/adaptive_product_v2_test'
JUDGE_DIR = 'baselines/outputs/judge/adaptive_product_v2_full'
GENERATED_AT = '2026-11-01T00:00:00+00:00'
P3_CONTEXT = 'bge20_bm250_rrf'
BM25_CONTEXT = l2_offline.name(gate.SETTING)
ROW_KEYS = ('sample_id', 'query', 'prediction', 'gold', 'mode', 'em', 'f1', 'normalized_match', 'normalized_match_type',
            'initial_answer', 'diagnosis_label', 'selected', 'controller_action', 'enhancement_status',
            'retrieved_passage_ids', 'retrieved_ranks', 'answer_in_context', 'answer_in_context_final', 'rag_prompt_style')
RAW = {name: (env, default) for name, env, default in (
    ('test_results', 'ROUTE_1_TEST_RESULTS', 'data/processed/quality_sweep_v1/test/results.jsonl.gz'),
    ('test_candidates', 'BM25_TEST_CANDIDATES', 'data/processed/quality_sweep_v1/bm25_test/candidates.jsonl'),
    ('test_unrouted_candidates', 'ADAPTIVE_TEST_CANDIDATES', 'data/processed/quality_sweep_v1/adaptive_test/candidates_unrouted.jsonl'),
    ('corpus', 'NQ_TRAIN_CORPUS', 'data/processed/retrieval/nq_train_corpus.jsonl'))}


def raw_paths(root=ROOT):
    return {name: Path(os.environ.get(env, root / default)) for name, (env, default) in RAW.items()}


def final_evidence_ids(c1, base, final, c1_top5, base_top5, contexts, order, stage):
    """The product's final Top-5: B+BM25's when selection kept its answer, C1's for C1's answer, else the first
    context of the stage whose answer is the chosen string (MeteredAnswerSelection in src/pipeline/current.py)."""
    if final == base:
        return list(base_top5)
    if final == c1:
        return list(c1_top5)
    return list(next(contexts[n]['top5'] for n in order[:stage] if contexts[n]['answer'] == final))


def base_evidence_ids(c1, base, c1_top5, routed, applied, p3_top5, bm25_top5):
    """B+BM25's Top-5 (src/controller/execution.py): C1's unless its answer changed, then P3-BM25's when it ran,
    else P3-20's, each in the order its own run recorded."""
    if not routed or base == c1:
        return list(c1_top5)
    return list(bm25_top5 if applied else p3_top5)


def label_rows(tmp, root=ROOT):
    """(sample_id, prediction) -> (judge row, source) in final_evaluation_v1's order, later files winning."""
    rel, bm = fe.untar(root / fe.AD_RELEASE), fe.untar(root / fe.BM25_RELEASE)
    members = fe.untar(root / fe.RELEASE / 'results.tar.gz')
    _, ours, team = gate.route_1_test(root / gate.ROUTE_1_TEST, tmp)
    sources = [('team qwen_only judge', [team['qwen_only']]), ('team product judge', [team['product']]),
               ('route-1 Test judge (quality_test_v1)', ours),
               ('BM25 Test judge (bm25_gate_test_v1)', fe.batch_files(tmp, 'bm25', bm, 'judge/outputs')),
               ('adaptive selection Test judge (adaptive_selection_eval_v1)', fe.batch_files(tmp, 'ad', rel, 'test/judge/outputs')),
               ('P60 Test judge (final_evaluation_v1)', fe.batch_files(tmp, 'p60', members, 'judge/outputs')),
               ('team RAG judge', [team['rag']])]
    out = {}
    for source, paths in sources:
        for path in paths:
            for row in scorer.read_jsonl(path):
                if row.get('judge_status') in ('llm_ok', 'auto_correct'):
                    out[(str(row['sample_id']).split('::')[0], row['prediction'])] = (row, source)
    return out, [s for s, _ in sources]


def build(root=ROOT, raw=None):
    """(prediction rows, judge rows, label sources) for all 11,114 Test items in the team's sample_id order."""
    raw = raw or raw_paths(root)
    policy = json.loads((root / ad.POLICY).read_text(encoding='utf-8'))
    order = policy['order']
    configs = {l2_offline.name(c): c for c in ad.settings({'contexts': {'order': order}})}
    rel, bm = fe.untar(root / fe.AD_RELEASE), fe.untar(root / fe.BM25_RELEASE)
    items = fe.jsonl(rel['test/items.jsonl'])
    chosen = ad.answers_of(items, policy)['chosen']
    base = {it['sample_id']: it['base'] for it in items}
    with tempfile.TemporaryDirectory() as tmp:
        preds, _, _ = gate.route_1_test(root / gate.ROUTE_1_TEST, tmp)
        labels, sources = label_rows(tmp, root)
        files = [fe.write(tmp, 'bm25_gen', bm['generations/generations.jsonl'])]
        files += [fe.write(tmp, f'ad_{p}', rel[f'test/generations/{p}/generations.jsonl']) for p in ('routed', 'unrouted')]
        results = list(ad.read_results(raw['test_results']))
        stored = ad.stored_generations(results, files)
    candidates = {r['sample_id']: r for p in (raw['test_candidates'], raw['test_unrouted_candidates'])
                  for r in scorer.read_jsonl(p)}
    texts, rows, finals = {}, [], {}
    for r in results:
        sid, c1 = r['sample_id'], r['initial_answer']
        for e in r['initial_evidence']:
            texts[e['passage_id']] = e['text']
        for e in ad.trace_of(r).get('candidates') or ():
            texts[e['passage_id']] = e['text']
        c1_top5 = tuple(e['passage_id'] for e in r['initial_evidence'])
        cand = candidates[sid]
        contexts = {}
        for name, config in configs.items():
            top5 = l2_offline.select_ids(cand, config)
            contexts[name] = {'top5': top5, 'answer': l2_offline.final_answer(c1, top5, c1_top5, stored, sid)}
        routed = r['controller_decision']['action'] == gate.P3
        applied = routed and gate.brings_bm25(cand, l2_offline.select_ids(cand, gate.SETTING))
        expected = c1 if not routed else contexts[BM25_CONTEXT if applied else P3_CONTEXT]['answer']
        if base[sid] != expected:
            raise ValueError(f'B+BM25 answer of {sid} does not follow route 1 and the BM25 rule')
        answer, stage, _ = chosen[sid]
        p3_top5 = [e['passage_id'] for e in ad.trace_of(r).get('evidence') or ()] if routed else None
        base_top5 = base_evidence_ids(c1, base[sid], c1_top5, routed, applied, p3_top5, contexts[BM25_CONTEXT]['top5'])
        finals[sid] = (answer, final_evidence_ids(c1, base[sid], answer, c1_top5, base_top5, contexts, order, stage))
    need = {p for _, ids in finals.values() for p in ids} - set(texts)
    prefix = '{"passage_id": "'
    with Path(raw['corpus']).open(encoding='utf-8') as handle:
        for line in handle:
            if not need:
                break
            pid = line[len(prefix):line.index('"', len(prefix))] if line.startswith(prefix) else json.loads(line)['passage_id']
            if pid in need:
                texts[pid] = json.loads(line)['text']
                need.discard(pid)
    if need:
        raise ValueError(f'Corpus lacks {len(need)} final-evidence passages')
    gold = {p['sample_id']: p['gold'] for p in preds}
    judge = []
    for p in sorted(preds, key=lambda x: x['sample_id']):
        sid = p['sample_id']
        answer, ids = finals[sid]
        if p['answer_in_context'] != answer_in_context(p['gold'], [{'text': texts[i]} for i in p['retrieved_passage_ids']]):
            raise ValueError(f'Initial evidence of {sid} does not match route 1')
        row = {'sample_id': sid, 'query': p['query'], 'prediction': answer, 'gold': p['gold'], 'mode': MODE,
               **score_answer(answer, p['gold'])}
        row.update({k: p[k] for k in ROW_KEYS[9:16]})
        row.update(answer_in_context=p['answer_in_context'],
                   answer_in_context_final=answer_in_context(p['gold'], [{'text': texts[i]} for i in ids]),
                   rag_prompt_style=p['rag_prompt_style'])
        rows.append({k: row[k] for k in ROW_KEYS})
        if score_answer(answer, gold[sid])['normalized_match']:
            judge.append(build_result_row(row, judge_label='CORRECT', judge_reason='normalized_match', judge_model=None,
                                          judge_status='auto_correct', auto_correct=True))
            continue
        source = labels.get((sid, answer))
        if source is None:
            raise ValueError(f'Unlabelled Test answer {sid}')
        j, _ = source
        judge.append(build_result_row(row, judge_label=j['judge_label'], judge_reason=j.get('judge_reason') or '',
                                      judge_model=j.get('judge_model'), judge_status=j['judge_status'],
                                      auto_correct=False, api_attempts=j.get('api_attempts', 1), retry_n=j.get('retry_n', 0)))
    return rows, judge, sources


def run_config(cfg, root=ROOT):
    """build_run_config's fields for the product config, marked as a replay of the archived Test answers."""
    gen, ret = cfg['generator'], cfg['retrieval']
    split = root / 'baselines/splits/diagnosis_nq_train_74097_split.json'
    return {
        'mode': MODE, 'smoke': False, 'generated_at': GENERATED_AT, 'config_path': CONFIG,
        'split_file': split.relative_to(root).as_posix(), 'split_sha256': fe_sha(split),
        'dataset_file': cfg['dataset']['records_path'], 'eval_split': 'test', 'eval_n': 11114, 'allow_sealed_test': True,
        'rag_prompt_style': cfg['rag_prompt']['default_style'],
        'generator': {k: gen[k] for k in ('model_name', 'quantization', 'do_sample', 'max_new_tokens')},
        'retrieval': {k: ret[k] for k in ('embedding_model', 'top_k', 'backend', 'index_path', 'ids_path', 'offsets_path',
                                          'corpus_path', 'expected_ntotal')},
        'production': dict(cfg['production']), 'scoring': dict(cfg['scoring']),
        'aic_definition': 'answer_in_context on AdaptiveRAGResult.initial_evidence (Reference RAG Top-5)',
        'output_dir': PRED_DIR, 'status': 'REPLAYED_FROM_ARCHIVED_TEST_RUN',
        'production_env_reference_rag': None, 'allow_sealed_test_effective': True,
        'replay': {
            'producer': 'scripts/export_product_v2_outputs.py (no model load, no GPU)',
            'answers': 'Archived Test run of the frozen product (configs/controller/quality_adaptive_v1.json): route-1 results '
                       '(artifacts/controller/quality_test_v1), P3-BM25 (artifacts/controller/bm25_gate_test_v1) and adaptive '
                       'selection (artifacts/controller/adaptive_selection_eval_v1), as reported in artifacts/controller/final_evaluation_v1.',
            'equivalence': 'Production wiring equals these offline answers on confirm, 2440/2440 '
                           '(artifacts/controller/adaptive_product_equivalence_v1); the final-evidence rule equals the confirm production run, 2440/2440.',
            'archives_sha256': {p: fe_sha(root / p) for p in (gate.ROUTE_1_TEST, fe.BM25_RELEASE, fe.AD_RELEASE,
                                                               f'{fe.RELEASE}/results.tar.gz', ad.POLICY)},
            'note': 'Not produced by baselines/run_product_evaluation.py; running it on Test would be a further Test run.'}}


def fe_sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def judge_outputs(judge, sources):
    summary = summarize_results(judge)
    auto = summary['auto_correct_n']
    summary.update({
        'input': f'{PRED_DIR}/predictions.jsonl', 'output_dir': JUDGE_DIR, 'dry_run': False, 'pilot_sample': None,
        'seed': None, 'pilot_group_counts': None, 'this_run_processed': len(judge), 'this_run_auto_correct': auto,
        'this_run_api_called': 0, 'this_run_api_succeeded': 0, 'this_run_api_attempted': 0, 'this_run_retry': 0,
        'this_run_failed': 0, 'labels': ['CORRECT', 'INCORRECT', 'UNCERTAIN'],
        'label_reuse': {'rule': 'Normalized matches auto-correct as in baselines/judge/run_llm_judge.py; every other answer '
                                'reuses the earlier judge row of the same (sample_id, answer), later sources winning.',
                        'sources_in_order': sources,
                        'api_counts': 'api_* fields count the earlier calls behind the reused rows; this export made none.'}})
    config = {'generated_at': GENERATED_AT, 'input': f'{PRED_DIR}/predictions.jsonl', 'output_dir': JUDGE_DIR,
              'limit': None, 'pilot_sample': None, 'seed': None, 'pilot_group_counts': None, 'pilot_manifest': None,
              'resume': False, 'dry_run': False, 'live': False, 'judge_config': 'baselines/judge/judge_config.json',
              'labels': ['CORRECT', 'INCORRECT', 'UNCERTAIN'],
              'note': 'No API call: labels reused from earlier batches of the same team judge (see judge_summary.json label_reuse).'}
    return summary, config


def write_outputs(out_root, rows, judge, sources, root=ROOT):
    cfg = json.loads((root / CONFIG).read_text(encoding='utf-8'))
    metrics = aggregate_product_metrics(rows)
    metrics.update(eval_split='test', rag_prompt_style=cfg['rag_prompt']['default_style'], smoke=False)
    summary, jconfig = judge_outputs(judge, sources)
    pred_dir, judge_dir = Path(out_root) / PRED_DIR, Path(out_root) / JUDGE_DIR
    for d in (pred_dir, judge_dir):
        d.mkdir(parents=True, exist_ok=True)
    lines = lambda rs: ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rs)
    (pred_dir / 'predictions.jsonl').write_text(lines(rows), encoding='utf-8')
    (pred_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    (pred_dir / 'run_config.json').write_text(json.dumps(run_config(cfg, root), indent=2, ensure_ascii=False), encoding='utf-8')
    (judge_dir / 'judge_results.jsonl').write_text(lines(judge), encoding='utf-8')
    (judge_dir / 'judge_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (judge_dir / 'run_config.json').write_text(json.dumps(jconfig, indent=2, ensure_ascii=False), encoding='utf-8')
    return metrics, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out-root', type=Path, default=ROOT, help='Repository root to write baselines/outputs under')
    args = p.parse_args(argv)
    rows, judge, sources = build()
    metrics, summary = write_outputs(args.out_root, rows, judge, sources)
    print(json.dumps({'n': len(rows), 'em': metrics['em'], 'f1': metrics['f1'],
                      'judge_assisted_accuracy': summary['judge_assisted_accuracy']}, indent=2))


if __name__ == '__main__':
    main()
