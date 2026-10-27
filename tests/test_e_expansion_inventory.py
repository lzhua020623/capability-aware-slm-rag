"""User batch provenance and ID-only nested expansion boundaries."""
import copy
import json
import unittest
from unittest.mock import patch
from collections import Counter

from src.data.team_nq import BoundaryError,TeamNQBoundary
from src.distillation.e_prep import ROOT
from src.distillation.expansion_inventory import RELEASE,batch_release,build_inventory


class ExpansionInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boundary=TeamNQBoundary(ROOT);cls.inventory=build_inventory(cls.boundary)

    def test_user_batch_release_never_claims_browser_annotations_or_measured_quality(self):
        release=batch_release(ROOT)
        self.assertEqual(len(release['accepted_review_items']),100)
        self.assertTrue(release['candidate_collection_authorized'])
        self.assertFalse(release['training_authorized'])
        self.assertFalse(release['browser_item_annotations_received'])
        self.assertFalse(release['itemwise_semantic_accuracy_measured'])
        self.assertTrue(all('semantic_correct' not in r for r in release['accepted_review_items']))

    def test_fake_training_or_itemized_quality_approval_is_rejected(self):
        original=json.loads((ROOT/RELEASE).read_text())
        for key in ('training_authorized','browser_item_annotations_received','itemwise_semantic_accuracy_measured'):
            changed=copy.deepcopy(original);changed[key]=True
            with patch('src.distillation.expansion_inventory.read_json',return_value=changed):
                with self.assertRaises(BoundaryError):batch_release(ROOT)
        changed=copy.deepcopy(original);changed['accepted_review_items'][0]['sample_id']='wrong'
        with patch('src.distillation.expansion_inventory.read_json',return_value=changed):
            with self.assertRaises(BoundaryError):batch_release(ROOT)

    def test_all_new_ids_are_train_and_disjoint_from_pilot_validation_test_reserve(self):
        m=self.inventory;new=m['new_order_max5000'];pilot=m['reused_pilot_ids']
        self.assertEqual(len(new),5000);self.assertEqual(len(set(new)),5000)
        self.assertEqual(len(pilot),500);self.assertFalse(set(new)&set(pilot))
        self.boundary.require(new,'train')
        for blocked in (self.boundary.splits['test'],self.boundary.splits['validation'],self.boundary.legacy):
            self.assertFalse(set(new)&set(blocked))
        self.assertFalse(m['runtime_content_decoded_for_sampling'])
        self.assertFalse(m['ready_for_API']);self.assertEqual(m['API_calls'],0)
        self.assertEqual(m['approved_for_SFT'],0)

    def test_small_scale_is_exact_balanced_prefix_of_large_scale(self):
        m=self.inventory;new=m['new_order_max5000']
        for n,per_group in [(1500,375),(4500,1125),(5000,1250)]:
            counts=Counter(self.boundary.labels[sid] for sid in new[:n])
            self.assertEqual(set(counts.values()),{per_group})
            self.assertEqual(len(counts),4)
        self.assertEqual(m['prefixes']['2000']['new_n'],1500)
        self.assertEqual(m['prefixes']['5000']['new_n'],4500)

    def test_candidate_order_does_not_depend_on_source_iteration_order(self):
        changed=copy.copy(self.boundary)
        changed.allowed_ids=lambda role:list(reversed(self.boundary.allowed_ids(role)))
        self.assertEqual(build_inventory(changed),self.inventory)


if __name__=='__main__':unittest.main()
