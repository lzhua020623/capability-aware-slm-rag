"""Check frozen C1 output coverage and provenance without scoring answers."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from src.controller.fever_c1 import COUNTS, ROLES, read_jsonl, validate_retrieval
from src.compensation.answer_repair import Generation
from src.generation.prompts import build_fever_rag_prompt
from src.retrieval.offset_corpus import file_sha


def audit(collection, output):
    collection,output=Path(collection),Path(output)
    manifest=json.loads((collection/'manifest.json').read_text())
    state=json.loads((output/'manifest.json').read_text())
    if (file_sha(collection/'manifest.json') != state['collection_manifest_sha256']
            or (output/'collection_manifest.json').read_bytes() != (collection/'manifest.json').read_bytes()
            or state['quality_scored'] is not False or state['counts'] != COUNTS):
        raise ValueError('Output collection provenance mismatch')
    counts,all_ids,expected_ids,costs={},[],[],[]
    digests={}
    for role in ROLES:
        info=manifest['files'][role]
        if file_sha(collection/info['path']) != info['sha256']:
            raise ValueError('Frozen input changed')
        rows=read_jsonl(collection/info['path'])
        if len(rows)!=COUNTS[role] or [r['sample_id'] for r in rows]!=manifest['ordered_ids'][role]:
            raise ValueError('Frozen task coverage changed')
        expected_ids.extend(r['sample_id'] for r in rows)
        paths=[output/role/name for name in ('runtime.jsonl','generations.jsonl')]
        values=[read_jsonl(p) if p.exists() else [] for p in paths]
        runtime,traces=values
        if len(runtime)!=len(traces) or len(runtime)>len(rows):
            raise ValueError('Unpaired/excess generated rows')
        for initial,result,trace in zip(rows,runtime,traces):
            validate_retrieval(initial)
            generation=Generation(**trace['generation'])
            expected={**initial,'initial_answer':generation.text.strip() or 'UNKNOWN'}
            if (result!=expected or trace['sample_id']!=initial['sample_id']
                    or trace['dataset']!='fever'
                    or trace['prompt']!=build_fever_rag_prompt(initial['query'],initial['evidence'])):
                raise ValueError('Generated row differs from frozen task or raw generation')
            all_ids.append(result['sample_id']);costs.append(generation.latency)
        counts[role]=len(runtime)
        for path in paths:
            if path.exists():digests[str(path.relative_to(output))]=file_sha(path)
    if (all_ids!=expected_ids[:len(all_ids)] or len(set(all_ids))!=len(all_ids)
            or state['n_completed']!=len(all_ids)):
        raise ValueError('Generated tasks are not the exact frozen prefix')
    if any(path not in digests or digests[path]!=sha for path,sha in state['files'].items()):
        raise ValueError('Recorded output checksum mismatch')
    complete=len(all_ids)==sum(COUNTS.values()) and state['state']=='completed'
    if state['state']=='completed' and not complete:
        raise ValueError('Incomplete output claimed completed')
    return {'schema_version':'controller_fever_c1_coverage_audit_v1','state':state['state'],
            'n_completed':len(all_ids),'n_expected':sum(COUNTS.values()),'counts':counts,
            'complete':complete,'exact_input_prefix':True,'runtime_gold_isolated':True,
            'output_hashes':digests,'successful_generation_latency_seconds':sum(costs),
            'latency_scope':'Successful C1 calls only; failed call, engineering, loading and retrieval excluded',
            'quality_scored':False,'answers_printed':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    print(json.dumps(audit(args.collection,args.output),indent=2))


if __name__=='__main__':main()
