"""Billing-category arithmetic and real frozen receipt reconciliation, no API."""
import copy
import json
from decimal import Decimal
from pathlib import Path
import unittest

from src.data.team_nq import BoundaryError
from src.distillation.e_prep import ROOT
from src.distillation.teacher_cost import MODEL,response_price,project,reconcile_archive


class TeacherCostTests(unittest.TestCase):
    def response(self):
        return {'model':MODEL,'service_tier':'default','usage':{'prompt_tokens':1000,
            'completion_tokens':100,'prompt_tokens_details':{'cached_tokens':100,'cache_write_tokens':600},
            'completion_tokens_details':{'reasoning_tokens':60}}}

    def test_each_input_category_and_reasoning_are_billed_once(self):
        amount,counts=response_price(self.response())
        self.assertEqual(counts,{'fresh_input':300,'cached_input':100,'cache_write':600,
                                'completion_including_reasoning':100})
        self.assertEqual(amount,Decimal('0.00311'))

    def test_absent_cache_details_mean_uncached_not_free_input(self):
        response=self.response();response['usage'].pop('prompt_tokens_details')
        self.assertEqual(response_price(response)[0],Decimal('0.003'))

    def test_invalid_counts_and_overlapping_categories_are_rejected(self):
        for mutate in [lambda r:r['usage'].update(prompt_tokens=-1),
                       lambda r:r['usage'].update(completion_tokens=True),
                       lambda r:r['usage']['prompt_tokens_details'].update(cache_write_tokens=901),
                       lambda r:r['usage']['completion_tokens_details'].update(reasoning_tokens=101),
                       lambda r:r['usage']['prompt_tokens_details'].update(audio_tokens=2)]:
            r=self.response();mutate(r)
            with self.assertRaises(BoundaryError):response_price(r)

    def test_other_model_tier_and_long_context_do_not_use_this_price(self):
        for key,value in [('model','gpt-6.1-sol'),('service_tier','priority')]:
            r=self.response();r[key]=value
            with self.assertRaises(BoundaryError):response_price(r)
        r=self.response();r['usage']['prompt_tokens']=272001
        with self.assertRaises(BoundaryError):response_price(r)

    def test_projection_reuses_500_without_charging_them_again(self):
        quote=project('3.42705',500,5000)
        self.assertEqual(Decimal(quote['estimated_new_usd']),Decimal('30.84345'))
        self.assertEqual(Decimal(quote['estimated_total_usd']),Decimal('34.27050'))
        self.assertEqual(quote['new_candidates'],4500)
        self.assertFalse(quote['is_invoice_or_fixed_fee'])
        smaller=project('3.42705',500,2000)
        self.assertEqual(Decimal(smaller['estimated_total_usd']),Decimal('13.70820'))
        self.assertEqual(Decimal(smaller['estimated_new_usd']),Decimal('10.28115'))

    def test_projection_cannot_erase_previous_cost_or_accept_invalid_population(self):
        for n,total in [(0,500),(500,499),(True,2000),(500,2000.0)]:
            with self.assertRaises(BoundaryError):project('3.42705',n,total)
        for value in ('NaN','Infinity','-1','not-a-number'):
            with self.assertRaises(BoundaryError):project(value,500,2000)

    def test_real_500_receipts_match_published_estimate_and_preserve_legacy_ledger(self):
        quote=reconcile_archive(ROOT/'artifacts/controller/e_teacher_pilot_20261028')
        self.assertEqual(quote['pilot_n'],500)
        self.assertEqual(Decimal(quote['published_price_estimate_usd']),Decimal('3.42705'))
        self.assertEqual(Decimal(quote['old_conservative_accounting_usd']),Decimal('6.58152'))
        self.assertEqual(quote['token_counts']['cache_write'],1150356)
        self.assertFalse(quote['invoice_verified']);self.assertFalse(quote['old_results_or_ledgers_rewritten'])
        self.assertEqual(quote['new_expansion_API_calls'],0)

    def test_budget_draft_cannot_collect_before_selection_or_auto_approve_training(self):
        plan=json.loads((ROOT/'configs/distillation/e_expansion_wave1_v1.json').read_text())
        if plan['budget_selected'] is None:self.assertFalse(plan['candidate_collection_enabled'])
        self.assertFalse(plan['training_enabled']);self.assertFalse(plan['semantic_approval_automatic'])
        for total,option in plan['budget_options'].items():
            expected=project('3.42705',500,int(total))
            self.assertEqual(option['primary_new_requests'],expected['new_candidates'])
            self.assertEqual(Decimal(option['published_price_estimated_total_usd']),Decimal(expected['estimated_total_usd']))
            self.assertGreater(Decimal(option['proposed_cap_usd']),Decimal(expected['estimated_total_usd']))


if __name__=='__main__':unittest.main()
