"""Prepare/check the USD20 2000-candidate wave; live calls require --execute."""
import argparse
import getpass
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.data.team_nq import BoundaryError
from src.distillation.expansion_prep import prepare,preflight
from src.distillation.expansion_collect import run


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--execute',action='store_true')
    parser.add_argument('--prompt-key',action='store_true')
    parser.add_argument('--limit-new',type=int,default=1500,help='Cumulative new-request cap, at most 1500')
    args=parser.parse_args(argv)
    if args.prompt_key and not args.execute:parser.error('--prompt-key requires --execute')
    if args.prepare:
        print('Preparing ID-frozen Train-only tasks; no Azure calls.',flush=True)
        result=prepare(ROOT)
    elif args.execute:
        environment=dict(os.environ)
        print('2000 total candidates = 500 reused + at most 1500 new; cumulative price-estimate cap USD20.',flush=True)
        if args.prompt_key:
            if not sys.stdin.isatty():raise BoundaryError('Hidden key input needs an interactive terminal')
            environment['AZURE_OPENAI_API_KEY']=getpass.getpass('Existing Azure API key (hidden): ')
        result=run(ROOT,limit_new=args.limit_new,environment=environment,
            progress=lambda e:print(json.dumps(e,ensure_ascii=False),flush=True))
    else:
        plan,parent,sample,tasks,receipt=preflight(ROOT)
        result={'verified_new_tasks':len(tasks),'reused_pilot':500,'total_candidates':2000,
            'budget_usd':20,'API_calls':0,'training_runs':0}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
