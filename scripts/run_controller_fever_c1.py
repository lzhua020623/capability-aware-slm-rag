"""Generate frozen FEVER C1 from sealed Top5; no index, gold, or enhancement."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.controller.fever_c1 import ROLES, COUNTS, verify_collection, generate_one
from src.retrieval.offset_corpus import file_sha


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


PREFILL_RUNTIME = {'prefill_chunk_size': 256, 'long_fixture_min_input_tokens': 13000}


def generation_runtime(manifest):
    """A registered computational revision; old sealed collections keep defaults."""
    settings = manifest.get('generation_runtime')
    if settings is None:
        return None
    if (settings != PREFILL_RUNTIME
            or any(type(v) is not int for v in settings.values())):
        raise ValueError('Unexpected C1 prefill runtime; register an explicit revision')
    return dict(settings)


def configure_prefill(generator, settings):
    if settings is None:
        return
    config = generator.model.generation_config
    if not hasattr(config, 'prefill_chunk_size'):
        raise RuntimeError('Pinned Transformers must support native chunked prefill')
    config.prefill_chunk_size = settings['prefill_chunk_size']
    generator.runtime_environment['generation_runtime'] = dict(settings)
    generator.runtime_environment['generation_config'] = config.to_dict()


def engineering_smoke(generator, *, min_input_tokens=0):
    traces=[]
    for i,claim in enumerate(('The capital of Luma is Amber.', 'The capital of Luma is Blue.')):
        evidence=[{'rank':rank,'passage_id':f'fiction-{rank}',
                   'text':'Luma is a fictional country. Its capital is Amber.', 'score':1-rank/10}
                  for rank in range(1,6)]
        _,trace=generate_one({'dataset':'fever','sample_id':f'engineering-{i}',
                              'query':claim,'evidence':evidence},generator)
        traces.append(trace)
    if min_input_tokens:
        # Synthetic text only. Exercise the longest sealed input's shape without
        # generating a real-task answer twice, dropping evidence, or scoring gold.
        from src.generation.prompts import build_fever_rag_prompt
        from src.generation.qwen import apply_qwen_chat_template
        import math
        unit = ' Luma has hills.'
        repeats = math.ceil(min_input_tokens/len(generator.tokenizer.encode(unit, add_special_tokens=False)))
        evidence=[{'rank':rank,'passage_id':f'long-fiction-{rank}',
                   'text':('Luma is fictional. Its capital is Amber.'+unit*repeats if rank==1
                           else 'Luma is a fictional country. Its capital is Amber.'),
                   'score':1-rank/10} for rank in range(1,6)]
        row={'dataset':'fever','sample_id':'engineering-long-context',
             'query':'The capital of Luma is Amber.','evidence':evidence}
        prompt=build_fever_rag_prompt(row['query'],evidence)
        measured=len(generator.tokenizer.encode(apply_qwen_chat_template(generator.tokenizer,prompt)))
        if measured < min_input_tokens:
            raise RuntimeError('Synthetic prefill fixture did not cover registered input length')
        _,trace=generate_one(row,generator)
        trace['fixture_min_input_tokens']=min_input_tokens
        traces.append(trace)
    passed=all(r['parsed_label'] in ('SUPPORTS','REFUTES') for r in traces)
    return {'passed':passed,'scope':'fictional engineering inputs; parseability and registered input shape, not QA quality',
            'llm_calls':len(traces),'traces':traces}


def run_collection(collection, output_dir, generator, *, root=ROOT, environment=None):
    manifest, by_role = verify_collection(collection,root=root)
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    state={'schema_version':'controller_fever_c1_run_v1','state':'running', 'counts':COUNTS,
           'collection_manifest_sha256':file_sha(Path(collection)/'manifest.json'),
           'quality_scored':False,'n_completed':0,'files':{},'cost_scope':'C1 generation only; CPU retrieval archived separately'}
    write_json(output_dir/'manifest.json',state)
    write_json(output_dir/'environment.json',environment or {})
    try:
        for role in ROLES:
            folder=output_dir/role;folder.mkdir()
            with (folder/'runtime.jsonl').open('x',encoding='utf-8') as runtime, \
                 (folder/'generations.jsonl').open('x',encoding='utf-8') as traces:
                for row in by_role[role]:
                    value,trace=generate_one(row,generator)
                    runtime.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');runtime.flush()
                    traces.write(json.dumps(trace,ensure_ascii=False,allow_nan=False)+'\n');traces.flush()
                    state['n_completed']+=1
                    if state['n_completed']%25==0:
                        print(json.dumps({'generated':state['n_completed'],'total':900}),flush=True)
            for name in ('runtime.jsonl','generations.jsonl'):
                relative=f'{role}/{name}';state['files'][relative]=file_sha(output_dir/relative)
        state['state']='completed'
    except BaseException as exc:
        state.update(state='failed',error={'type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        write_json(output_dir/'manifest.json',state)
    return state


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args(argv)
    manifest, rows=verify_collection(args.collection,root=ROOT)
    settings=generation_runtime(manifest)
    if args.output_dir.exists():
        parser.error('Use a new C1 output directory; failed/previous runs are preserved')
    if args.preflight:
        print(json.dumps({'preflight':'passed','counts':{r:len(v) for r,v in rows.items()},
                          'gold_included':False,'generation_runtime':settings}))
        return 0
    from src.compensation.qwen_repair import QwenRepairGenerator
    started=perf_counter();generator=QwenRepairGenerator(json_output=False)
    configure_prefill(generator,settings)
    environment={**generator.runtime_environment,'resource_loading_seconds':perf_counter()-started,
                 'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()}
    engineering=engineering_smoke(generator, min_input_tokens=(settings or {}).get('long_fixture_min_input_tokens',0))
    if not engineering['passed']:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        write_json(args.output_dir/'environment.json',environment)
        write_json(args.output_dir/'engineering_smoke.json',engineering)
        write_json(args.output_dir/'manifest.json',{'state':'engineering_failed','n_completed':0})
        raise RuntimeError('Fictional engineering fixture produced an unparsable label; C1 collection not started')
    try:
        state=run_collection(args.collection,args.output_dir,generator,environment=environment)
    finally:
        if args.output_dir.exists():
            write_json(args.output_dir/'engineering_smoke.json',engineering)
    print(json.dumps({'state':state['state'],'n_completed':state['n_completed']}),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
