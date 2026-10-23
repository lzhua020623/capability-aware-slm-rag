"""Read-only structural audit. Prints counts only, never held-out question text."""
import collections
import hashlib
import itertools
import json
import math
import pathlib
import re
import sys
import unicodedata

root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
parts = json.loads((root / 'data/processed/answer_repair/nq_train_v1/partition_ids.json').read_text())['partitions']
membership = collections.defaultdict(set)
for role, ids in parts.items():
    for sid in ids:
        membership[str(sid)].add(role)
manifest = json.loads((root / 'data/annotations/nq_train_delivery/manifest.json').read_text())
expected = {r['file']: r['sha256'] for r in manifest['shards']}
counts = collections.Counter()
ids = set()
queries = collections.defaultdict(list)
variant_queries = collections.defaultdict(set)
gold_groups = collections.defaultdict(set)
evidence_versions = collections.defaultdict(set)
shards = []

def norm(s):
    return ' '.join(s.lower().split())

for path in sorted((root / 'data/processed/external/nq_train_delivery_20261017').glob('nq_train_annotation_*.jsonl')):
    digest = hashlib.sha256()
    n = 0
    with path.open('rb') as f:
        for line in f:
            digest.update(line)
            if not line.strip():
                counts['blank_lines'] += 1
                continue
            r = json.loads(line)
            n += 1
            counts['rows'] += 1
            sid = r['sample_id']
            counts['non_string_ids'] += not isinstance(sid, str)
            counts['duplicate_sample_ids'] += sid in ids
            ids.add(sid)
            counts['missing_partition_ids'] += sid not in membership
            counts['source_null_labels'] += r.get('label') is None
            q = norm(r['query'])
            queries[q].append(sid)
            variant = re.sub(r'[^\w]+', ' ', unicodedata.normalize('NFKC', q)).strip()
            variant_queries[variant].update(membership[sid])
            counts['empty_queries'] += not q
            counts['empty_initial_answers'] += not r['initial_answer'].strip()
            ev = r['evidence']
            counts['not_five_evidence'] += len(ev) != 5
            counts['bad_rank_sequence'] += [e['rank'] for e in ev] != list(range(1, 6))
            pids = [e['passage_id'] for e in ev]
            texts = [e['text'] for e in ev]
            counts['rows_duplicate_evidence_ids'] += len(set(pids)) != len(pids)
            counts['rows_exact_duplicate_evidence_text'] += len(set(texts)) != len(texts)
            normalized = [norm(t) for t in texts]
            counts['rows_case_whitespace_duplicate_evidence_text'] += len(set(normalized)) != len(normalized)
            shingles = []
            for e, t in zip(ev, normalized):
                counts['evidence_records'] += 1
                counts['empty_evidence_text'] += not t
                counts['nonfinite_scores'] += not math.isfinite(e['score'])
                counts['evidence_with_replacement_character'] += '\ufffd' in e['text']
                counts['evidence_with_literal_unicode_escape'] += bool(re.search(r'\\u[0-9a-fA-F]{4}', e['text']))
                evidence_versions[e['passage_id']].add(hashlib.sha256(e['text'].encode()).digest())
                tokens = re.findall(r'\w+', t)
                shingles.append(set(zip(tokens, tokens[1:], tokens[2:])) if len(tokens) >= 12 else set())
            near = False
            for a, b in itertools.combinations(shingles, 2):
                if a and b and min(len(a), len(b)) / max(len(a), len(b)) >= .9:
                    intersect = len(a & b)
                    if intersect / (len(a) + len(b) - intersect) >= .9:
                        near = True
                        break
            counts['rows_near_duplicate_evidence_shingles_090'] += near
            ref = r['annotation_reference']
            gold_groups[ref['gold_passage_id']].update(membership[sid])
            counts['gold_top5_flag_disagreement'] += ref['gold_in_top5'] != (ref['gold_passage_id'] in pids)
            counts['gold_in_top5'] += ref['gold_in_top5']
    sha = digest.hexdigest()
    shards.append({'file': path.name, 'rows': n, 'sha256': sha, 'matches_main_manifest': sha == expected.get(path.name)})
    print(path.name, n, 'verified', sha == expected.get(path.name), flush=True)

query_cross = sum(len(set().union(*(membership[s] for s in sids))) > 1 for sids in queries.values())
exposed = set(json.loads((root / 'configs/splits/answer_repair_nq_train_exposure_registry_20261022.json').read_text())['known_exposed_ids'])
label_path = root / 'data/processed/answer_repair/nq_train_preliminary_20261022/labels.jsonl'
label_counts = collections.Counter()
label_ids = set()
label_hash = hashlib.sha256()
label_duplicates = 0
label_human_verified = 0
with label_path.open('rb') as f:
    for line in f:
        label_hash.update(line)
        if not line.strip(): continue
        r = json.loads(line)
        sid = r['sample_id']
        label_duplicates += sid in label_ids
        label_ids.add(sid)
        label_counts[str(r.get('label', r.get('primary_label')))] += 1
        label_human_verified += r.get('human_verified') is True

result = {
    'scope': 'All 15 locally delivered NQ Train C1 shards and the preliminary label layer. Not the full NQ corpus, not semantic certification.',
    'counts': dict(counts), 'source_shards': shards,
    'partition_counts': {k: len(v) for k,v in parts.items()},
    'ids_in_multiple_partitions': sum(len(v) > 1 for v in membership.values()),
    'partition_ids_not_in_sources': len(set(membership)-ids),
    'normalized_query_groups': len(queries),
    'duplicate_query_groups': sum(len(v)>1 for v in queries.values()),
    'exact_normalized_query_groups_cross_partition': query_cross,
    'unicode_punctuation_normalized_query_groups_cross_partition': sum(len(v)>1 for v in variant_queries.values()),
    'shared_gold_passage_groups_cross_partition': sum(len(v)>1 for v in gold_groups.values()),
    'unique_retrieved_passage_ids': len(evidence_versions),
    'passage_ids_with_multiple_text_versions': sum(len(v)>1 for v in evidence_versions.values()),
    'known_E_exposure_counts_by_partition': {k:len(exposed & set(v)) for k,v in parts.items()},
    'preliminary_labels': {'sha256':label_hash.hexdigest(),'unique_ids':len(label_ids),'duplicate_ids':label_duplicates,'missing_ids':len(ids-label_ids),'extra_ids':len(label_ids-ids),'label_counts':dict(label_counts),'human_verified_true':label_human_verified},
    'limitations': ['Near-duplicate is a lexical flag, not adjudicated redundancy or contradiction.', 'Shared evidence across splits is permitted in ordinary RAG and is not alone proof of label leakage.', 'No full semantic, temporal, adversarial-instruction or model-pretraining contamination audit.', 'No held-out question or answer text is printed. No source data or split is modified.']
}
(out / 'nq_delivery_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'source_shards'},ensure_ascii=False,indent=2),flush=True)
