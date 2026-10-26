"""One frozen selection run. No gold, tuning, promotion, or automatic retry."""
from copy import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_controller_integration import read_inputs, summarize, file_sha
from src.controller.execution import ExecutionConfig, CallLog, MeteredGenerator, fingerprint
from src.controller.runtime import canonical_input

POLICY='configs/controller/t5_selection_execution_v1.json'
MODES=('c1','always_retrieve','always_rerank','always_repair','adaptive')


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def verify(root,manifest):
    root=Path(root)
    if (manifest['role']!='selection' or manifest['contains_gold'] is not False
            or manifest['contains_development_or_final_inputs'] is not False
            or manifest['modes']!=list(MODES) or set(manifest['datasets'])!={'nq','fever'}):
        raise ValueError('Selection-only scope changed')
    for rel,sha in manifest['source_files'].items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts or file_sha(root/rel)!=sha:
            raise ValueError('Frozen source changed: '+rel)
    policy=json.loads((root/POLICY).read_text())
    if file_sha(root/'inputs/candidate_policy.json')!=manifest['candidate_sha256']:
        raise ValueError('Candidate changed after freeze')
    candidate=json.loads((root/'inputs/candidate_policy.json').read_text())
    if (candidate!={'dataset':'nq','threshold':.2,'eligible_action':'no_action',
                    'new_action':'retrieval_enhancement','existing_supported_probe_veto':True}
            or policy['runtime_config']['nq_v2a_mode']!='unavailable'
            or policy['runtime_config']['fever_v2a_mode']!='unavailable'
            or any(policy['runtime_config'][ds+'_repair_strategy']!='evidence_id_v2a' for ds in ('nq','fever'))):
        raise ValueError('Candidate scope or shared E mode changed')
    inputs={}
    for ds,info in manifest['datasets'].items():
        rel=info['input_path']
        if Path(rel).is_absolute() or '..' in Path(rel).parts or file_sha(root/rel)!=info['input_sha256']:
            raise ValueError('Frozen selection inputs changed')
        items=read_inputs(root/rel)
        if (len(items)!=200 or [i.sample_id for i in items]!=info['sample_ids']
                or any(i.dataset.value!=ds for i in items)
                or {i.sample_id:fingerprint(asdict(canonical_input(i))) for i in items}!=info['effective_input_hashes']):
            raise ValueError('Selection IDs, dataset or canonical input mismatch')
        inputs[ds]=items
    return inputs,policy


def active_engineering(pipeline,output,minimum=14000):
    """Engineering path coverage only; E remains unaccepted and disabled."""
    from src.controller.engineering import (long_fixture,FictionalScorer,FictionalRetriever,
                                            inspect_engineering_calls,fictional_item)
    from src.probe.probe import EvidenceProbe,PROBE_MAX_NEW_TOKENS
    isolated=copy(pipeline);isolated.scorer=FictionalScorer()
    checks=[];inactive=[]
    output=Path(output);output.mkdir()
    for ds in ('nq','fever'):
        item,lengths=long_fixture(pipeline.raw_generator.tokenizer,ds,minimum)
        isolated.retriever=FictionalRetriever(item.evidence[0].text)
        for mode in ('probe','always_retrieve','always_rerank'):
            if mode=='probe':
                calls=CallLog(pipeline.config)
                gen=MeteredGenerator(pipeline.raw_generator,calls,'engineering_probe')
                probe=EvidenceProbe(lambda prompt:gen.generate(prompt,PROBE_MAX_NEW_TOKENS).text).run(item)
                record={'calls':calls.events,'probe_result':asdict(probe)}
            else:record=isolated.run(item,mode=mode)
            check={'dataset':ds,'path':mode,'passed':False,'record':record}
            write(output/(ds+'_'+mode+'.json'),check)
            inspection=inspect_engineering_calls(record['calls'],minimum)
            if any(e['status']!='completed' for e in record['calls'] if e['attempted']):
                raise RuntimeError('Active engineering call failed')
            check.update(passed=True,**inspection);checks.append(check)
            write(output/(ds+'_'+mode+'.json'),check)
            print(json.dumps({'engineering':ds+'/'+mode,'passed':True}),flush=True)
        record=isolated.run(fictional_item(ds),mode='always_repair')
        c=record['module_trace']['support_check']
        if (record['calls'] or record['final_answer']!=record['initial_answer']
                or c['original_verification']!='incomplete' or not c['unresolved']):
            raise RuntimeError('Unavailable E called a model or fabricated a verdict')
        inactive.append(record);write(output/(ds+'_E_disabled.json'),record)
    cpu=[]
    for n in (5,20):
        start=perf_counter();scores=list(pipeline.scorer('What is the capital of Luma?',
            [fictional_item('nq',100).evidence[0].text]*n))
        if len(scores)!=n:raise ValueError('Real CPU scorer shape failed')
        cpu.append({'pairs':n,'seconds':perf_counter()-start,'device':'cpu','dtype':'float32'})
    return {'passed':True,'active_long_checks':checks,'inactive_E_checks':inactive,'cpu_scoring':cpu,
            'E_engineering_accepted':False,'quality_scored':False}


def run_batches(pipeline,inputs,output,state):
    lean=[]
    with (Path(output)/'predictions.jsonl').open('x') as handle:
        for ds in ('nq','fever'):
            for ordinal,item in enumerate(inputs[ds],1):
                for mode in MODES:
                    record=pipeline.run(item,mode=mode)
                    handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
                    state['n_predictions']+=1
                    lean.append({k:record[k] for k in ('mode','dataset','cost','routing','answer_changed',
                        'status','unresolved_failure','total_seconds')})
                    write(Path(output)/'manifest.json',state)
                    print(json.dumps({'dataset':ds,'ordinal':ordinal,'mode':mode,
                        'n_predictions':state['n_predictions'],'status':record['status']}),flush=True)
                    if any('out of memory' in str(e.get('error',{})).lower() for e in record['calls']):
                        raise RuntimeError('Infrastructure OOM; preserve prefix, no repeated failing calls')
    return lean


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--preflight',action='store_true')
    args=p.parse_args(argv)
    manifest=json.loads((ROOT/'execution_manifest.json').read_text())
    inputs,policy=verify(ROOT,manifest)
    if args.preflight:
        print(json.dumps({'preflight':True,'role':'selection','nq':200,'fever':200,'gold_read':False}));return 0
    if args.output_dir.exists():raise FileExistsError('Preserve previous selection outputs')
    args.output_dir.mkdir()
    state={'state':'starting','role':'selection','selection_started':False,'n_predictions':0,'quality_scored':False,
           'E_mode':'unavailable','candidate_threshold':.2,'source_commit':manifest['source_commit']}
    write(args.output_dir/'manifest.json',state)
    try:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('Real CUDA required')
        from src.controller.providers import load_pipeline
        from src.controller.nq_train_resources import load_backend as load_nq
        from src.controller.fever_resources import load_backend as load_fever,DatasetRetriever
        config=ExecutionConfig(**policy['runtime_config'])
        pipeline,env=load_pipeline(config,ROOT/'models/diagnosis/diagnosis_v2_expanded_hierarchical.json')
        write(args.output_dir/'environment.json',env)
        receipt=active_engineering(pipeline,args.output_dir/'engineering_checks')
        write(args.output_dir/'engineering_receipt.json',receipt)
        nq,nr=load_nq(ROOT/'configs/controller/nq_train_resources_v1.json',root=ROOT)
        fever,fr=load_fever(ROOT/'configs/controller/fever_resources_v1.json',root=ROOT)
        pipeline.retriever=DatasetRetriever({'nq':nq,'fever':fever})
        lookup=[]
        for item in read_inputs(ROOT/'engineering/historical_inputs.jsonl'):
            hits=pipeline.retriever.retrieve(item.query,dataset=item.dataset.value,top_k=20)
            same=[h['passage_id'] for h in hits[:5]]==[e.passage_id for e in item.evidence]
            lookup.append({'dataset':item.dataset.value,'sample_id':item.sample_id,'same_ordered_top5':same})
            if not same:raise ValueError('Resource lookup differs from historical C1')
        write(args.output_dir/'historical_lookup.json',lookup)
        env.update(nq_train_resources=nr,fever_resources=fr)
        write(args.output_dir/'environment.json',env)
        verify(ROOT,manifest)
        write(args.output_dir/'execution_seal.json',{**manifest,'active_engineering_passed':True,
            'E_engineering_accepted':False,'engineering_receipt_sha256':file_sha(args.output_dir/'engineering_receipt.json'),
            'environment_sha256':file_sha(args.output_dir/'environment.json')})
        state.update(state='running',selection_started=True)
        lean=run_batches(pipeline,inputs,args.output_dir,state)
        errors=sum(bool(r['cost']['failed_calls'] or r['cost']['budget_blocks']) for r in lean)
        write(args.output_dir/'summary.json',summarize(lean))
        state.update(state='completed_with_errors' if errors else 'completed',failed_records=errors)
        if state['n_predictions']!=2000:raise ValueError('Incomplete registered arms')
        return 2 if errors else 0
    except Exception as exc:
        state.update(state='failed',error={'type':type(exc).__name__,'message':str(exc)});raise
    finally:write(args.output_dir/'manifest.json',state)


if __name__=='__main__':raise SystemExit(main())
