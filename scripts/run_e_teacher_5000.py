"""Prepare/verify the 5000-request extension; paid collection needs --execute."""
import argparse
import getpass
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.data.team_nq import BoundaryError
from src.distillation.expansion_5000_prep import prepare,preflight
from src.distillation.expansion_5000_collect import run


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--prompt-key',action='store_true')
    parser.add_argument('--limit-post-pilot',type=int,default=4500)
    args=parser.parse_args(argv)
    if args.prompt_key and not args.execute:parser.error('--prompt-key requires --execute')
    if args.prepare:
        result=prepare(ROOT)
    elif args.execute:
        preflight(ROOT)  # Validate source/data/old charges before asking for any secret.
        environment=dict(os.environ)
        print('Reuse500+54; skip unknown ordinal55 without refund; continue at56; cumulative cap USD40.',flush=True)
        if args.prompt_key:
            if not sys.stdin.isatty():raise BoundaryError('Hidden key input requires an interactive terminal')
            environment['AZURE_OPENAI_API_KEY']=getpass.getpass('Existing Azure API key (hidden): ')
        result=run(ROOT,limit_post_pilot=args.limit_post_pilot,environment=environment,
                   progress=lambda e:print(json.dumps(e,ensure_ascii=False),flush=True))
    else:
        spec,parent,sample,tasks,state=preflight(ROOT)
        result={'verified_post_pilot_tasks':len(tasks),'reused_complete_candidates':500+state['completed'],
            'prior_attempts':state['attempted'],'next_ordinal':state['attempted']+1,
            'future_request_cap':4500-state['attempted'],'budget_usd':40,'new_API_calls':0,'training_runs':0}
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except BoundaryError as error:
        print('采集停止：'+str(error),file=sys.stderr,flush=True);raise SystemExit(2)
