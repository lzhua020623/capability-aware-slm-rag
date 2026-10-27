"""ID-only, Train-only expansion inventory; no paid calls or label fabrication."""
from pathlib import Path
import json

from src.data.team_nq import BoundaryError,read_json,sha
from src.distillation.e_prep import (ROOT,load_protocol,validate_sample_manifest,ordered,digest)

RELEASE='artifacts/controller/e_teacher_expansion_release_20261028/user_batch_release.json'
PILOT='artifacts/controller/e_distillation_v1/sample_manifest.json'
SEED='e_expansion_wave1_20261028'


def batch_release(root=ROOT):
    root=Path(root);release=read_json(root/RELEASE)
    if (release.get('schema_version')!='e_teacher_user_batch_release_v1'
            or release.get('source')!='user_chat'
            or release.get('candidate_collection_authorized') is not True
            or release.get('training_authorized') is not False
            or release.get('browser_item_annotations_received') is not False
            or release.get('itemwise_semantic_accuracy_measured') is not False
            or release.get('original_itemized_gate_rewritten') is not False):
        raise BoundaryError('Batch collection release cannot claim per-item measurement or training approval')
    path=(root/release['review_file']).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path)!=release['review_sha256']:
        raise BoundaryError('Released review cohort missing or changed')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    expected=[{'review_id':r['review_id'],'sample_id':r['sample_id'],
               'accepted_by_user_batch_confirmation':True} for r in rows]
    if len(rows)!=100 or release['accepted_review_items']!=expected:
        raise BoundaryError('User batch release differs from the frozen 100-item cohort')
    return release


def build_inventory(boundary):
    root=boundary.root;protocol=load_protocol(root)
    pilot=validate_sample_manifest(boundary,protocol,read_json(root/PILOT))
    release=batch_release(root)
    reused=pilot['sample_ids']['teacher_pilot'];reused_set=set(reused)
    quarantine={r['sample_id'] for r in release['provisional_training_quarantine']}
    if not quarantine<=reused_set:raise BoundaryError('Provisional quarantine outside the pilot')
    pools={label:[] for label in protocol['sampling']['strata']}
    for sid in boundary.allowed_ids('train'):
        if sid in reused_set:continue
        label=boundary.labels[sid]
        if label not in pools:raise BoundaryError('Unknown AI metadata stratum')
        pools[label].append(sid)
    for label in pools:
        pools[label]=ordered(pools[label],SEED,'new:'+label)[:1250]
        if len(pools[label])!=1250:raise BoundaryError('Not enough admitted training IDs')
    new=[sid for group in zip(*pools.values(),strict=True) for sid in group]
    boundary.require(new,'train')
    if len(set(new))!=5000 or reused_set&set(new):raise BoundaryError('Duplicate/new pilot IDs')
    for role in ('development','selection'):
        if set(new)&set(pilot['sample_ids'][role]):raise BoundaryError('Expansion overlaps validation')
    return {'schema_version':'e_expansion_inventory_v1','seed':SEED,'boundary':boundary.receipt(),
        'pilot_manifest_sha256':sha(root/PILOT),'user_release_sha256':sha(root/RELEASE),
        'reused_pilot_ids':reused,'new_order_max5000':new,
        'prefixes':{str(total):{'reuse_n':500,'new_n':total-500,
            'new_ids_sha256':digest(new[:total-500])} for total in (2000,5000)},
        'selection_basis':'sample IDs and frozen AI metadata only; no runtime/gold/EM selection',
        'runtime_content_decoded_for_sampling':False,'AI_strata_are_semantic_gold':False,
        'pilot_batch_release_is_itemwise_quality_measurement':False,
        'provisional_training_quarantine_ids':sorted(quarantine),
        'API_calls':0,'ready_for_API':False,'approved_for_SFT':0,'training_enabled':False,
        'pending':['user budget/first-wave size choice','task export and execution seal']}
