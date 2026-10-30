"""L2 candidate stage (configs/controller/l2_round_v1.json step_2_l2); no gold is read.

bm25-index: BM25 over the NQ Train corpus (document_title + text), in corpus line order.
check-gpu:  the GPU reranker re-scores candidate sets stored in P3-20 traces; reports score
            gaps and how often select_p3_20 picks the same Top-5 as the stored CPU scores.
candidates: per registered ID, BGE top 100 and BM25 top 50 (zero-score hits dropped) and GPU
            cross-encoder scores for their union. One JSON line per ID, so a run resumes.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

REGISTRATION = 'configs/controller/l2_round_v1.json'
NQ_RESOURCES = 'configs/controller/nq_train_resources_v1.json'
BGE_DEPTH, BM25_DEPTH = 100, 50
SCORER_BATCH = 32


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_registration(root=ROOT):
    return json.loads((Path(root) / REGISTRATION).read_text(encoding='utf-8'))


def bm25_spec(registration):
    spec = registration['step_2_l2']['bm25']
    import bm25s
    if spec['library'] != f'bm25s=={bm25s.__version__}':
        raise RuntimeError(f"Registered {spec['library']}, installed bm25s {bm25s.__version__}")
    return spec


def tokenize(texts):
    import bm25s
    return bm25s.tokenize(list(texts), stopwords='en', return_ids=False, show_progress=False)


class Bm25Index:
    """A saved bm25s index whose document order is the corpus line order."""

    def __init__(self, directory):
        import bm25s
        directory = Path(directory)
        self.receipt = json.loads((directory / 'receipt.json').read_text(encoding='utf-8'))
        self.passage_ids = json.loads((directory / 'passage_ids.json').read_text(encoding='utf-8'))
        self.model = bm25s.BM25.load(str(directory / 'index'))

    def search(self, query, k):
        """[(corpus position, score)] best first; zero scores (no shared term) are dropped."""
        docs, scores = self.model.retrieve(tokenize([query]), k=k, show_progress=False)
        return [(int(d), float(s)) for d, s in zip(docs[0], scores[0]) if s > 0]


def build_bm25(corpus, output_dir, registration):
    import bm25s
    spec = bm25_spec(registration)
    ids, texts = [], []
    with Path(corpus).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                ids.append(row['passage_id'])
                texts.append(f"{row.get('document_title') or ''} {row['text']}")
    started = time.perf_counter()
    model = bm25s.BM25(k1=spec['k1'], b=spec['b'])
    model.index(bm25s.tokenize(texts, stopwords='en', show_progress=False), show_progress=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save(str(output_dir / 'index'))
    (output_dir / 'passage_ids.json').write_text(json.dumps(ids), encoding='utf-8')
    receipt = {'corpus_sha256': file_sha(corpus), 'n': len(ids), 'k1': spec['k1'], 'b': spec['b'],
               'library': spec['library'], 'fields': spec['fields'], 'tokenizer': spec['tokenizer'],
               'seconds': round(time.perf_counter() - started, 1), 'built_UTC': now()}
    (output_dir / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    return receipt


def gpu_scorer():
    """bge-reranker-large on GPU: FP32, raw logits, max_length 512, TF32 off."""
    import torch
    from sentence_transformers import CrossEncoder
    from src.compensation.retrieval_enhancement.p3 import RERANKER_MODEL, RERANKER_REVISION
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = CrossEncoder(RERANKER_MODEL, revision=RERANKER_REVISION, device='cuda', max_length=512,
                           activation_fn=torch.nn.Identity(), model_kwargs={'torch_dtype': torch.float32})
    return lambda query, texts: [float(s) for s in encoder.predict([(query, t) for t in texts],
                                                                   batch_size=SCORER_BATCH, show_progress_bar=False)]


def compare_trace(trace, gpu_scores):
    """Score gap and Top-5 agreement of GPU scores against one stored P3-20 trace."""
    from src.compensation.retrieval_enhancement.p3 import select_p3_20
    stored = [s['cross_encoder_score'] for s in trace['scores']]
    picked = [r['passage_id'] for r in select_p3_20(trace['candidates'], gpu_scores)['evidence']]
    return {'max_gap': max(abs(a - b) for a, b in zip(stored, gpu_scores)),
            'same_top5': picked == [r['passage_id'] for r in trace['evidence']]}


def check_gpu(records, output, limit=None):
    score, rows = gpu_scorer(), []
    for line in Path(records).open(encoding='utf-8'):
        record = json.loads(line)
        action = record['actions'].get('retrieval_enhancement') or {}
        trace = (action.get('native') or {}).get('module_trace') or {}
        if not trace.get('candidates'):
            continue
        rows.append(compare_trace(trace, score(record['query'], [c['text'] for c in trace['candidates']])))
        if limit and len(rows) >= limit:
            break
    gaps = sorted(r['max_gap'] for r in rows)
    report = {'records_sha256': file_sha(records), 'traces': len(rows), 'same_top5': sum(r['same_top5'] for r in rows),
              'max_gap': gaps[-1] if gaps else None, 'median_max_gap': gaps[len(gaps) // 2] if gaps else None,
              'batch_size': SCORER_BATCH, 'checked_UTC': now()}
    report['same_top5_rate'] = report['same_top5'] / max(1, report['traces'])
    report['passes_99'] = report['same_top5_rate'] >= 0.99
    Path(output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def deep_backend(backend, depth=BGE_DEPTH):
    """The product NQ backend with its top-k cap raised; retrieval code and ordering are unchanged."""
    from src.compensation.retrieval_enhancement.adapters import NQFaissBackend

    class DeepNQFaissBackend(NQFaissBackend):
        def __init__(self, retriever, max_top_k):
            self.max_top_k, self.retriever, self._resource_signature = max_top_k, retriever, None
            self._check_resources()

    return DeepNQFaissBackend(backend.retriever, depth)


def candidate_row(sample_id, bge_hits, bm25_hits, passage_ids, ce):
    return {'sample_id': sample_id, 'bge': [[h['passage_id'], h['score']] for h in bge_hits],
            'bm25': [[passage_ids[p], s] for p, s in bm25_hits], 'ce': ce}


def role_ids(role, root=ROOT):
    if role == 'sweep':
        from scripts.precompute_quality_sweep import load_registration as load_sweep
        return load_sweep(root)[1]
    sweep = json.loads((Path(root) / 'configs/controller/quality_sweep_v1.json').read_text(encoding='utf-8'))
    return json.loads((Path(root) / sweep['split']['confirm']['path']).read_text(encoding='utf-8'))


def test_ids(path):
    """Test runs take the registered ID list (configs/controller/bm25_gate_v1_test_ids.json)."""
    if path is None:
        raise ValueError('--role test needs --ids')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def done_ids(path):
    if not path.exists():
        return set()
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        path.write_bytes(data[:data.rfind(b'\n') + 1])
    return {json.loads(line)['sample_id'] for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}


def candidates(args, registration):
    from scripts.precompute_quality_sweep import read_queries
    from src.controller.nq_train_resources import load_backend
    ids = test_ids(args.ids) if args.role == 'test' else role_ids(args.role)
    ids = ids[:args.limit] if args.limit else ids
    queries = read_queries(args.queries, ids)
    bm25 = Bm25Index(args.bm25_dir)
    backend, receipt = load_backend(ROOT / NQ_RESOURCES, root=ROOT)
    if bm25.passage_ids != list(backend.retriever.passage_ids):
        raise ValueError('BM25 document order differs from the BGE corpus order')
    deep, score = deep_backend(backend), gpu_scorer()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / 'candidates.jsonl'
    finished = done_ids(out)
    manifest = {'schema_version': 'l2_candidates_v1', 'state': 'running', 'started_UTC': now(), 'role': args.role,
                'registration_sha256': file_sha(ROOT / REGISTRATION), 'query_sha256': file_sha(args.queries),
                'bm25_receipt': bm25.receipt, 'nq_resources': receipt, 'bge_depth': BGE_DEPTH,
                'bm25_depth': BM25_DEPTH, 'scorer_batch': SCORER_BATCH, 'engineering_run': bool(args.limit),
                'n_ids': len(ids), 'ids_sha256': file_sha(args.ids) if args.ids else None,
                'resumed_with': len(finished)}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n', encoding='utf-8')
    todo, started = [q for q in queries if q[0] not in finished], time.perf_counter()
    with out.open('a', encoding='utf-8') as handle:
        for i, (sid, query) in enumerate(todo, 1):
            bge_hits = deep.retrieve(query, dataset='nq', top_k=BGE_DEPTH)
            bm25_hits = bm25.search(query, BM25_DEPTH)
            texts = {h['passage_id']: h['text'] for h in bge_hits}
            for position, _ in bm25_hits:
                pid, text = deep._record(position)
                texts.setdefault(pid, text)
            union = list(texts)
            ce = dict(zip(union, score(query, [texts[p] for p in union])))
            handle.write(json.dumps(candidate_row(sid, bge_hits, bm25_hits, bm25.passage_ids, ce)) + '\n')
            handle.flush()
            if i % 100 == 0 or i == len(todo):
                print(json.dumps({'done': len(finished) + i, 'of': len(ids),
                                  'seconds': round(time.perf_counter() - started, 1)}), flush=True)
    manifest.update(state='completed', finished_UTC=now(), rows=len(done_ids(out)), output_sha256=file_sha(out))
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n', encoding='utf-8')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('bm25-index')
    b.add_argument('--corpus', type=Path, required=True)
    b.add_argument('--output-dir', type=Path, required=True)
    g = sub.add_parser('check-gpu')
    g.add_argument('--records', type=Path, required=True, help='sweep records with stored P3-20 traces')
    g.add_argument('--output', type=Path, required=True)
    g.add_argument('--limit', type=int, default=None)
    c = sub.add_parser('candidates')
    c.add_argument('--role', choices=('sweep', 'confirm', 'test'), required=True)
    c.add_argument('--ids', type=Path, default=None, help='test: registered JSON ID list')
    c.add_argument('--queries', type=Path, required=True, help='sample_id/query only export')
    c.add_argument('--bm25-dir', type=Path, required=True)
    c.add_argument('--output-dir', type=Path, required=True)
    c.add_argument('--limit', type=int, default=None)
    args = p.parse_args(argv)
    registration = load_registration()
    if args.command == 'bm25-index':
        print(json.dumps(build_bm25(args.corpus, args.output_dir, registration)))
    elif args.command == 'check-gpu':
        print(json.dumps(check_gpu(args.records, args.output, args.limit)))
    else:
        bm25_spec(registration)
        candidates(args, registration)
    return 0


if __name__ == '__main__': raise SystemExit(main())
