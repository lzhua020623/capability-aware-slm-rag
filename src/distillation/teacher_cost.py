"""Published-price estimates, separate from the old conservative pilot ledger.

Rates are an explicit 2026-10-28 Global Standard short-context snapshot. This
does not query or certify an Azure invoice and never changes old journal costs.
"""
from decimal import Decimal,InvalidOperation
import hashlib
import json
from pathlib import Path
import tarfile

from src.data.team_nq import BoundaryError,read_json,sha

MODEL='gpt-6.1-sol-2026-09-29'
SOURCE='https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/provisioned-throughput-sizing'
RATES={'fresh_input':Decimal('2'),'cached_input':Decimal('0.10'),
       'cache_write':Decimal('2.50'),'completion_including_reasoning':Decimal('10')}


def integer(value,name):
    if type(value) is not int or value<0:raise BoundaryError('Invalid nonnegative token count: '+name)
    return value


def response_price(response):
    if response.get('model')!=MODEL or response.get('service_tier')!='default':
        raise BoundaryError('Published quote supports only the registered model/default tier')
    usage=response['usage'];prompt=integer(usage['prompt_tokens'],'prompt_tokens')
    completion=integer(usage['completion_tokens'],'completion_tokens')
    if prompt>272000:raise BoundaryError('Short-context price cannot quote long-context requests')
    details=usage.get('prompt_tokens_details') or {};out=usage.get('completion_tokens_details') or {}
    cached=integer(details.get('cached_tokens',0),'cached_tokens')
    written=integer(details.get('cache_write_tokens',0),'cache_write_tokens')
    reasoning=integer(out.get('reasoning_tokens',0),'reasoning_tokens')
    if cached+written>prompt or reasoning>completion:
        raise BoundaryError('Overlapping/invalid token accounting')
    if details.get('audio_tokens',0) or out.get('audio_tokens',0):
        raise BoundaryError('Text-only price snapshot cannot quote audio')
    counts={'fresh_input':prompt-cached-written,'cached_input':cached,'cache_write':written,
            'completion_including_reasoning':completion}
    amount=sum(Decimal(n)*RATES[k] for k,n in counts.items())/Decimal(1000000)
    return amount,counts


def project(pilot_usd,pilot_n,total_n):
    if type(pilot_n) is not int or pilot_n<=0 or type(total_n) is not int or total_n<pilot_n:
        raise BoundaryError('Projection must reuse the completed pilot without rebilling it')
    try:amount=Decimal(str(pilot_usd))
    except InvalidOperation:raise BoundaryError('Invalid pilot currency amount') from None
    if not amount.is_finite() or amount<0:raise BoundaryError('Invalid pilot currency amount')
    mean=amount/pilot_n
    return {'total_candidates':total_n,'reused_pilot':pilot_n,'new_candidates':total_n-pilot_n,
        'estimated_new_usd':str(mean*(total_n-pilot_n)),
        'estimated_total_usd':str(mean*total_n),'is_invoice_or_fixed_fee':False}


def reconcile_archive(folder):
    folder=Path(folder);manifest=read_json(folder/'runtime_bundle_manifest.json')
    archive=folder/manifest['archive']
    if sha(archive)!=manifest['sha256']:raise BoundaryError('Changed raw teacher archive')
    total=Decimal(0);n=0;counts={k:0 for k in RATES}
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            if not member.name.endswith('.response.json'):continue
            if not member.isfile() or member.name not in manifest['members']:
                raise BoundaryError('Unexpected response member')
            raw=tar.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest()!=manifest['members'][member.name]['sha256']:
                raise BoundaryError('Changed raw teacher response')
            amount,tokens=response_price(json.loads(raw));total+=amount;n+=1
            for key,value in tokens.items():counts[key]+=value
    summary=read_json(folder/'audit_summary.json')
    if n!=500 or n!=summary['n']:raise BoundaryError('Incomplete pilot cost basis')
    return {'currency':'USD','pilot_n':n,'published_price_estimate_usd':str(total),
        'old_conservative_accounting_usd':str(summary['accounting_usd_upper_estimate']),
        'rates_usd_per_million':{k:str(v) for k,v in RATES.items()},'token_counts':counts,
        'source':SOURCE,'price_snapshot_date':'2026-10-28','invoice_verified':False,
        'projections':[project(total,n,size) for size in (2000,5000)],
        'old_results_or_ledgers_rewritten':False,'new_expansion_API_calls':0}
