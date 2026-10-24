"""Seal a uniform chunked-prefill C1 collection after preserving the OOM run."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from src.controller.fever_c1 import ROLES, verify_collection
from src.retrieval.offset_corpus import file_sha
from scripts.run_controller_fever_c1 import PREFILL_RUNTIME

RUNNER='scripts/run_controller_fever_c1.py'


def prepare(parent, output, *, root=ROOT):
    parent,output,root=map(Path,(parent,output,root))
    if output.exists():raise FileExistsError('Preserve existing experiments')
    original=json.loads((parent/'manifest.json').read_text())
    changed={name:{'before':sha,'after':file_sha(root/name)}
             for name,sha in original['source_files'].items() if file_sha(root/name)!=sha}
    if set(changed)!={RUNNER}:raise ValueError('Prefill revision permits only the C1 runner delta')
    revised=copy.deepcopy(original)
    revised['source_files'][RUNNER]=changed[RUNNER]['after']
    revised['generation_runtime']=dict(PREFILL_RUNTIME)
    revised['execution_revision']={
        'name':'uniform_chunked_prefill_r2','parent_manifest_sha256':file_sha(parent/'manifest.json'),
        'source_delta':changed,'reason':'Version 2 OOM on input ordinal 646 (12911 tokens)',
        'uniform_full_collection_run':True,'reuse_previous_answers':False,
        'previous_n_completed':645,'no_outcome_dependent_selection':True,
        'unchanged':'all 900 tasks, Top5 text/order, original prompt, model revision, NF4/FP16/double-quant, greedy and 64 token budget',
        'numerical_and_latency_change_possible':True,
    }
    for role in ROLES:
        info=original['files'][role]
        if info['path']!=f'{role}/retrieval.jsonl' or file_sha(parent/info['path'])!=info['sha256']:
            raise ValueError('Frozen task or evidence changed')
    output.mkdir(parents=True)
    for role in ROLES:
        relative=original['files'][role]['path'];(output/relative).parent.mkdir(parents=True)
        shutil.copyfile(parent/relative,output/relative)
    (output/'manifest.json').write_text(json.dumps(revised,ensure_ascii=False,indent=2)+'\n')
    verify_collection(output,root=root)
    return revised['execution_revision']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(prepare(args.parent,args.output),indent=2))


if __name__=='__main__':main()
