"""FEVER recoverability ladder on a FEVER-train research set (configs/controller/fever_recoverability_v1.json).

sample: SUPPORTS / REFUTES claims of FEVER train in a fixed hash order; a claim is eligible when every annotated
        evidence sentence resolves in the FEVER wiki dump, its evidence pages have text, and its claim is neither
        equal nor near-equal to a claim of the team FEVER Test (fever_dev 13,332) nor to an earlier sampled claim.
        The first 1,000 eligible per label form the set: 800 development and 200 held-out per label.
inputs: the GPU input rows (sample_id, claim, gold sentences and pages); labels stay local.
retrieve: on a CPU host with the accepted FEVER backend, each claim's Top-5 and Top-20 hits (page texts stored once),
        so the GPU host can replay them instead of holding the 3.5 GB corpus; P3 still runs the product code.
run:    on the GPU host, C0 / C1 / L1 / L3 / L4 for every input row with the product's shared Qwen, the accepted
        FEVER backend and P3-20 for FEVER; one JSON line per item, resumable.
score:  accuracy, RFR, flips, evidence coverage and cost per condition, for development and held-out separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

REGISTRATION = 'configs/controller/fever_recoverability_v1.json'
LABELS = ('SUPPORTS', 'REFUTES')
CONDITIONS = ('C0', 'C1', 'L1', 'L3', 'L4')
FEVER_RESOURCES = 'configs/controller/fever_resources_v1.json'
CLASSIFIER = 'models/diagnosis/diagnosis_v2_expanded_hierarchical.json'
# Team closed-book FEVER prompt (feature/baseline-evaluation e8c4213, baselines/_lib/prompts.py).
C0_INSTRUCTION = ('Determine whether the claim is supported or refuted. '
                  'Return only SUPPORTS or REFUTES. '
                  'Do not provide explanations.')


def load_registration(root=ROOT):
    return json.loads((Path(root) / REGISTRATION).read_text(encoding='utf-8'))


def order_key(seed, sample_id):
    return hashlib.sha256(json.dumps([seed, 'fever_train', str(sample_id)]).encode()).hexdigest()


def evidence_sentences(evidence):
    """Union of annotated (page, sentence) pairs over all evidence groups, first-seen order (as prepare_retrieval_data)."""
    pairs, seen = [], set()
    for group in evidence or []:
        for item in group:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            page_id, sentence_id = item[2], item[3]
            if page_id in (None, '') or sentence_id is None:
                continue
            pair = (str(page_id), int(sentence_id))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def parse_lines(lines_field):
    """FEVER wiki 'lines': 'i<TAB>sentence<TAB>links...' per line -> {i: sentence}."""
    out = {}
    for line in (lines_field or '').split('\n'):
        parts = line.split('\t')
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].strip():
            out[int(parts[0])] = parts[1].strip()
    return out


def norm_claim(text):
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', text.lower()).split())


def jaccard(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / max(1, len(a | b))


def wiki_pages(zip_path, wanted):
    """{page_id: (has_text, {sentence_id: sentence})} for the wanted pages, streamed from the FEVER wiki zip."""
    out = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith('.jsonl') and '__MACOSX' not in n):
            with archive.open(name) as handle:
                for raw in handle:
                    head = raw[:300].decode('utf-8', 'ignore')
                    match = re.match(r'\{"id": "((?:[^"\\]|\\.)*)"', head)
                    if not match or json.loads(f'"{match.group(1)}"') not in wanted:
                        continue
                    page = json.loads(raw)
                    out[page['id']] = (bool((page.get('text') or '').strip()), parse_lines(page.get('lines')))
    return out


def ineligible(pairs, claim, pages, test_norm, test_by_page, seen_claims, near):
    """The first registered eligibility rule a claim fails, or None."""
    if not pairs:
        return 'no_evidence'
    if any(p not in pages or s not in pages[p][1] for p, s in pairs):
        return 'unresolved_sentence'
    if any(not pages[p][0] for p, _ in pairs):
        return 'page_without_text'
    if claim in test_norm:
        return 'equals_test_claim'
    if any(jaccard(claim, t) >= near for p in {p for p, _ in pairs} for t in test_by_page.get(p, [])):
        return 'near_test_claim'
    if claim in seen_claims:
        return 'repeats_sampled_claim'
    return None


def sample(train_rows, test_rows, pages_of, registration):
    """Ordered eligibility pass; returns (items, exclusion counts). `pages_of(ids)` resolves wiki pages lazily."""
    spec = registration['sample']
    seed, per_label, near = spec['seed'], spec['per_label'], spec['near_duplicate_jaccard']
    test_norm = {norm_claim(r['claim']) for r in test_rows}
    test_by_page = {}
    for r in test_rows:
        for page in r.get('gold_page_ids') or []:
            test_by_page.setdefault(page, []).append(norm_claim(r['claim']))
    pool = sorted((r for r in train_rows if r.get('label') in LABELS), key=lambda r: order_key(seed, r['id']))
    chosen = {label: [] for label in LABELS}
    excluded = {k: 0 for k in ('no_evidence', 'unresolved_sentence', 'page_without_text', 'equals_test_claim',
                               'near_test_claim', 'repeats_sampled_claim')}
    seen_claims = set()
    for start in range(0, len(pool), spec['batch']):
        if all(len(v) >= per_label for v in chosen.values()):
            break
        batch = [r for r in pool[start:start + spec['batch']] if len(chosen[r['label']]) < per_label]
        pages = pages_of({p for r in batch for p, _ in evidence_sentences(r['evidence'])})
        for r in batch:
            if len(chosen[r['label']]) >= per_label:
                continue
            pairs = evidence_sentences(r['evidence'])
            claim = norm_claim(r['claim'])
            reason = ineligible(pairs, claim, pages, test_norm, test_by_page, seen_claims, near)
            if reason:
                excluded[reason] += 1
                continue
            seen_claims.add(claim)
            gold_pages = list(dict.fromkeys(p for p, _ in pairs))
            chosen[r['label']].append({
                'sample_id': str(r['id']), 'claim': r['claim'], 'label': r['label'],
                'gold_page_ids': gold_pages, 'gold_sentence_ids': [[p, s] for p, s in pairs],
                'gold_evidence_text': list(dict.fromkeys(pages[p][1][s] for p, s in pairs)),
                'evidence_groups': len(r['evidence'])})
    if any(len(v) < per_label for v in chosen.values()):
        raise ValueError('FEVER train pool too small for the registered sample')
    items = []
    for label in LABELS:
        for i, row in enumerate(chosen[label]):
            items.append({**row, 'role': 'development' if i < spec['development_per_label'] else 'held_out'})
    items.sort(key=lambda r: order_key(seed, r['sample_id']))
    return items, excluded


def c0_prompt(claim):
    return f"{C0_INSTRUCTION}\n\nClaim: {claim}"


def oracle_passages(gold_text, c1_hits, gold_pages):
    """L3: the gold sentence block first, then C1's pages without evidence pages (first four); L4: the block alone."""
    block = {'passage_id': 'gold_evidence', 'text': ' '.join(gold_text)}
    others = [h for h in c1_hits if h['passage_id'] not in set(gold_pages)]
    return [block, *others[:4]], [block]


def retrieval_cache(items, retrieve):
    """(hit rows without text, {passage_id: text}) for Top-5 and Top-20 of every claim."""
    rows, pages = [], {}
    for r in items:
        entry = {'sample_id': r['sample_id'], 'claim': r['claim']}
        for k in (5, 20):
            hits = retrieve(r['claim'], k)
            for h in hits:
                if pages.setdefault(h['passage_id'], h['text']) != h['text']:
                    raise ValueError('Conflicting text for ' + h['passage_id'])
            entry[f'top{k}'] = [{'passage_id': h['passage_id'], 'score': h['score']} for h in hits]   # order is the rank
        rows.append(entry)
    return rows, pages


class ReplayRetriever:
    """Serves the Top-5 / Top-20 FEVER hits computed beforehand with the accepted backend."""

    def __init__(self, rows, pages):
        self.by_claim, self.pages = {r['claim']: r for r in rows}, pages
        if len(self.by_claim) != len(rows):
            raise ValueError('Duplicate claims in the retrieval cache')

    def retrieve(self, query, *, dataset, top_k):
        if dataset != 'fever' or top_k not in (5, 20):
            raise ValueError('The replay holds FEVER Top-5 and Top-20 only')
        return [{'passage_id': h['passage_id'], 'text': self.pages[h['passage_id']], 'score': h['score']}
                for h in self.by_claim[query][f'top{top_k}']]


def gpu_rows(items):
    return [{k: r[k] for k in ('sample_id', 'claim', 'gold_evidence_text', 'gold_page_ids')} for r in items]


def run_item(row, generate, retrieve, p3):
    """All conditions for one claim. generate(prompt) -> Generation; retrieve(claim, k) -> hits; p3(row, hits, c1) -> dict."""
    from src.evaluation.fever import parse_fever_prediction
    from src.generation.prompts import build_fever_rag_prompt
    out = {'sample_id': row['sample_id']}

    def record(name, prompt, passages=None):
        g = generate(prompt)
        out[name] = {'raw': g.text, 'label': parse_fever_prediction(g.text), 'input_tokens': g.input_tokens,
                     'seconds': g.latency, 'passage_ids': None if passages is None else [p['passage_id'] for p in passages]}
        return g.text

    record('C0', c0_prompt(row['claim']))
    hits = retrieve(row['claim'], 5)
    c1 = record('C1', build_fever_rag_prompt(row['claim'], hits), hits)
    out['L1'] = p3(row, hits, c1)
    l3, l4 = oracle_passages(row['gold_evidence_text'], hits, row['gold_page_ids'])
    record('L3', build_fever_rag_prompt(row['claim'], l3), l3)
    record('L4', build_fever_rag_prompt(row['claim'], l4), l4)
    return out


def run(args):
    import time
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('FEVER recoverability runs need CUDA')
    from src.controller.execution import CallLog, ExecutionConfig, decision_for
    from src.controller.providers import load_pipeline
    from src.controller.schema import ControllerAction
    from src.diagnosis.schema import DiagnosisInput, EvidenceItem
    rows = read_jsonl(args.inputs)
    if any(set(r) != {'sample_id', 'claim', 'gold_evidence_text', 'gold_page_ids'} for r in rows):
        raise ValueError('GPU inputs must hold sample_id, claim, gold sentences and gold pages only')
    config = ExecutionConfig()
    if args.retrieval_cache:
        from src.controller.fever_resources import DatasetRetriever
        cache = read_jsonl(args.retrieval_cache / 'hits.jsonl')
        pages = json.loads((args.retrieval_cache / 'pages.json').read_text(encoding='utf-8'))
        if {r['sample_id'] for r in cache} != {r['sample_id'] for r in rows}:
            raise ValueError('Retrieval cache does not cover the inputs')
        controller, environment = load_pipeline(config, ROOT / CLASSIFIER)
        controller.retriever = DatasetRetriever({'fever': ReplayRetriever(cache, pages)})
        environment = {**environment, 'fever_retrieval': {
            'mode': 'replay of the accepted backend, computed on a CPU host',
            'manifest': json.loads((args.retrieval_cache / 'manifest.json').read_text(encoding='utf-8')),
            'hits_sha256': file_sha(args.retrieval_cache / 'hits.jsonl'), 'pages_sha256': file_sha(args.retrieval_cache / 'pages.json')}}
    else:
        controller, environment = load_pipeline(config, ROOT / CLASSIFIER, fever_resources=ROOT / FEVER_RESOURCES)
    generator = controller.raw_generator
    generate = lambda prompt: generator.generate(prompt, config.answer_max_new_tokens)
    retrieve = lambda claim, k: controller.retriever.retrieve(claim, dataset='fever', top_k=k)

    def p3(row, hits, c1):
        started = time.perf_counter()
        item = DiagnosisInput(row['sample_id'], 'fever', row['claim'],
                              [EvidenceItem(i, h['passage_id'], h['text'], h['score']) for i, h in enumerate(hits, 1)],
                              c1.strip() or 'UNKNOWN')
        calls = CallLog(config)
        result = controller.execute(item, decision_for(ControllerAction.RETRIEVAL_ENHANCEMENT, row['sample_id']), calls=calls)
        trace = result.get('module_trace') or {}
        from src.evaluation.fever import parse_fever_prediction
        return {'raw': result['final_answer'], 'label': parse_fever_prediction(result['final_answer']), 'status': result['status'],
                'passage_ids': [e['passage_id'] for e in trace.get('evidence') or []], 'error': result.get('error'),
                'cost': calls.summary(), 'seconds': time.perf_counter() - started}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / 'results.jsonl'
    done = {json.loads(l)['sample_id'] for l in out.read_text(encoding='utf-8').splitlines() if l.strip()} if out.exists() else set()
    manifest = {'schema_version': 'fever_recoverability_run_v1', 'state': 'running', 'registration_sha256': file_sha(ROOT / REGISTRATION),
                'inputs_sha256': file_sha(args.inputs), 'n': len(rows), 'resumed_with': len(done), 'environment': environment}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n', encoding='utf-8')
    started = time.perf_counter()
    with out.open('a', encoding='utf-8') as handle:
        for i, row in enumerate([r for r in rows if r['sample_id'] not in done], 1):
            handle.write(json.dumps(run_item(row, generate, retrieve, p3), ensure_ascii=False) + '\n')
            handle.flush()
            if i % 50 == 0:
                print(json.dumps({'done': len(done) + i, 'of': len(rows), 'seconds': round(time.perf_counter() - started, 1)}), flush=True)
    manifest.update(state='completed', results_sha256=file_sha(out))
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n', encoding='utf-8')
    return 0


def score(items, results):
    """Per role: accuracy (overall and per label), RFR, flips against C1, evidence coverage and cost per condition."""
    by_id = {r['sample_id']: r for r in results}
    if set(by_id) != {r['sample_id'] for r in items}:
        raise ValueError('Results do not cover the sample')
    report = {}
    for role in ('development', 'held_out', 'all'):
        rows = [r for r in items if role == 'all' or r['role'] == role]
        correct = {c: {r['sample_id']: by_id[r['sample_id']][c]['label'] == r['label'] for r in rows} for c in CONDITIONS}
        wrong_c1 = [r['sample_id'] for r in rows if not correct['C1'][r['sample_id']]]
        cover = lambda c: sum(set(r['gold_page_ids']) <= set(by_id[r['sample_id']][c]['passage_ids'] or []) for r in rows)
        report[role] = {
            'n': len(rows), 'c1_wrong': len(wrong_c1),
            'accuracy': {c: sum(correct[c].values()) / len(rows) for c in CONDITIONS},
            'accuracy_by_label': {label: {c: sum(correct[c][r['sample_id']] for r in rows if r['label'] == label)
                                          / max(1, sum(r['label'] == label for r in rows)) for c in CONDITIONS} for label in LABELS},
            'rfr': {c: sum(correct[c][s] for s in wrong_c1) / max(1, len(wrong_c1)) for c in CONDITIONS if c != 'C1'},
            'flips_vs_c1': {c: {'wrong_to_correct': sum(correct[c][s] for s in wrong_c1),
                                'correct_to_wrong': sum(correct['C1'][r['sample_id']] and not correct[c][r['sample_id']] for r in rows)}
                            for c in CONDITIONS if c != 'C1'},
            'unparsed': {c: sum(by_id[r['sample_id']][c]['label'] not in LABELS for r in rows) for c in CONDITIONS},
            'gold_pages_covered': {'C1_top5': cover('C1'), 'L1_top5': cover('L1')},
            'cost': {c: {'seconds': sum(by_id[r['sample_id']][c]['seconds'] or 0 for r in rows) / len(rows),
                         'input_tokens': sum(by_id[r['sample_id']][c].get('input_tokens') or 0 for r in rows) / len(rows)}
                     for c in CONDITIONS}}
    return report


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--train', type=Path, required=True, help='FEVER train.jsonl')
    s.add_argument('--wiki', type=Path, required=True, help='FEVER wiki-pages.zip')
    s.add_argument('--test', type=Path, required=True, help='team fever_dev.jsonl (13,332 claims with gold pages)')
    s.add_argument('--output-dir', type=Path, required=True)
    i = sub.add_parser('inputs')
    i.add_argument('--items', type=Path, required=True)
    i.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('retrieve')
    q.add_argument('--items', type=Path, required=True)
    q.add_argument('--resources-root', type=Path, default=ROOT, help='root holding the registered FEVER resource paths')
    q.add_argument('--output-dir', type=Path, required=True)
    r = sub.add_parser('run')
    r.add_argument('--inputs', type=Path, required=True)
    r.add_argument('--output-dir', type=Path, required=True)
    r.add_argument('--retrieval-cache', type=Path, default=None, help='retrieve output dir; replaces the full FEVER backend')
    c = sub.add_parser('score')
    c.add_argument('--items', type=Path, required=True)
    c.add_argument('--results', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == 'inputs':
        with args.output.open('x', encoding='utf-8') as handle:
            for row in gpu_rows(read_jsonl(args.items)):
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        return 0
    if args.command == 'retrieve':
        from src.controller.fever_resources import load_backend
        backend, receipt = load_backend(args.resources_root / FEVER_RESOURCES, root=args.resources_root)
        rows, pages = retrieval_cache(read_jsonl(args.items),
                                      lambda claim, k: backend.retrieve(claim, dataset='fever', top_k=k))
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with (args.output_dir / 'hits.jsonl').open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        (args.output_dir / 'pages.json').write_text(json.dumps(pages, ensure_ascii=False, sort_keys=True), encoding='utf-8')
        manifest = {'schema_version': 'fever_recoverability_retrieval_v1', 'items_sha256': file_sha(args.items),
                    'n': len(rows), 'pages': len(pages), 'backend_receipt': receipt}
        (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n', encoding='utf-8')
        print(json.dumps({k: manifest[k] for k in ('n', 'pages')}))
        return 0
    if args.command == 'run':
        return run(args)
    if args.command == 'score':
        report = score(read_jsonl(args.items), read_jsonl(args.results))
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({role: {k: report[role][k] for k in ('n', 'c1_wrong', 'accuracy', 'rfr')} for role in report}, indent=1))
        return 0
    registration = load_registration()
    test_rows = [{'claim': r['claim'], 'gold_page_ids': r.get('gold_page_ids')} for r in read_jsonl(args.test)]
    if len(test_rows) != registration['team_test']['n']:
        raise ValueError('Team FEVER Test file has an unexpected size')
    items, excluded = sample(read_jsonl(args.train), test_rows, lambda ids: wiki_pages(args.wiki, ids), registration)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'items.jsonl').open('x', encoding='utf-8') as handle:
        for row in items:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {'schema_version': 'fever_recoverability_sample_v1', 'registration_sha256': file_sha(ROOT / REGISTRATION),
                'inputs': {'train_sha256': file_sha(args.train), 'wiki_zip_sha256': file_sha(args.wiki),
                           'team_test_sha256': file_sha(args.test)},
                'counts': {role: {label: sum(r['role'] == role and r['label'] == label for r in items) for label in LABELS}
                           for role in ('development', 'held_out')},
                'excluded_before_fill': excluded, 'items_sha256': file_sha(args.output_dir / 'items.jsonl')}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
