"""Cohort isolation, outcome-blind selection, gold separation and immutable outputs."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_controller_data import ROOT, POLICY, asset_inventory, extract_exposure, prepare
from scripts.check_controller_data import REGISTRY, verify
from src.controller.data_preparation import (
    Candidate, OLD_ROLES, ROLES, QueryGuard, canonical_query, fever_outputs,
    fever_role, jsonl_bytes, nq_outputs, precision_notes, select_cohorts, sha,
)


def nq_row(identifier="nq1", query="What is the crystal isotope?"):
    evidence = [{"rank": i, "passage_id": f"p{i}", "text": f"Evidence {i}: alpha.", "score": 1.0 / i}
                for i in range(1, 6)]
    return {"dataset": "nq", "sample_id": identifier, "query": query,
            "initial_answer": "beta", "evidence": evidence, "label": "evidence_utilisation",
            "annotation_reference": {"gold_answer": "alpha", "gold_passage_id": "p1",
                "gold_evidence_text": evidence[0]["text"],
                "gold_in_top1": True, "gold_in_top3": True, "gold_in_top5": True}}


class SelectionTests(unittest.TestCase):
    def pool(self):
        return [Candidate("nq", f"{role}-{i}", f"What is mineral{n}{i} isotope{n}{i}?", role)
                for n, role in enumerate(ROLES) for i in range(4)]

    def test_selection_ignores_input_order_and_preserves_roles(self):
        pool = self.pool()
        args = ({"nq": set()}, {"nq": []}, dict.fromkeys(ROLES, 2))
        a, _, _ = select_cohorts(pool, *args)
        b, _, _ = select_cohorts(list(reversed(pool)), *args)
        self.assertEqual(a, b)
        for role, identifiers in a["nq"].items():
            self.assertEqual(len(identifiers), 2)
            self.assertTrue(all(i.startswith(role) for i in identifiers))

    def test_exposed_ids_and_query_variants_cannot_enter(self):
        pool = self.pool()
        exposed = pool[0]
        a, _, _ = select_cohorts(pool, {"nq": {exposed.sample_id}}, {"nq": [pool[4].query]}, dict.fromkeys(ROLES, 2))
        chosen = set(sum(a["nq"].values(), []))
        self.assertNotIn(exposed.sample_id, chosen)
        self.assertNotIn(pool[4].sample_id, chosen)

    def test_cross_partition_variant_group_quarantines_both_sides(self):
        pool = self.pool() + [Candidate("nq", "variant1", "What’s  Ａmber?", "development"),
                              Candidate("nq", "variant2", "whats amber", "final_candidate")]
        a, report, excluded = select_cohorts(pool, {}, {}, dict.fromkeys(ROLES, 2))
        self.assertEqual(report["nq"]["pre_sampling_exclusions"]["cross_partition_query_variant"], 2)
        self.assertTrue({"variant1", "variant2"} <= {r["sample_id"] for r in excluded})
        self.assertFalse({"variant1", "variant2"} & set(sum(a["nq"].values(), [])))

    def test_lexical_guard_is_not_only_exact_text(self):
        guard = QueryGuard(["Where did Nikola Tesla study electrical engineering?"])
        self.assertTrue(guard.conflicts("At which university did Nikola Tesla study electrical engineering?"))
        # No stemming or semantic model: this paraphrase falls below 0.60.
        self.assertFalse(guard.conflicts("Nikola Tesla studied electrical engineering at which university?"))
        self.assertFalse(guard.conflicts("Which mineral contains carbon atoms?"))
        self.assertEqual(canonical_query("A—Ｂ's  Name!"), canonical_query("abs name"))

    def test_existing_variant_audit_is_respected_even_if_normalization_differs(self):
        pool = self.pool()
        blocked = pool[0].sample_id
        selected, report, exclusions = select_cohorts(pool, {}, {}, dict.fromkeys(ROLES, 2),
                                                       quarantined_ids={"nq": {blocked}})
        self.assertNotIn(blocked, sum(selected["nq"].values(), []))
        self.assertEqual(report["nq"]["pre_sampling_exclusions"]["previously_flagged_query_variant"], 1)
        self.assertIn(blocked, {r["sample_id"] for r in exclusions})

    def test_duplicate_ids_and_short_pools_fail_instead_of_silent_shrinkage(self):
        pool = self.pool()
        for data, n in ((pool + pool[:1], 1), (pool, 5)):
            with self.assertRaises(ValueError):
                select_cohorts(data, {}, {}, dict.fromkeys(ROLES, n))

    def test_gold_and_ai_labels_are_removed_from_runtime(self):
        source = nq_row()
        runtime, reference = nq_outputs(source)
        self.assertEqual(set(runtime), {"sample_id", "dataset", "query", "initial_answer", "evidence"})
        changed = deepcopy(source)
        changed["label"] = "correct"
        changed["annotation_reference"]["gold_answer"] = "gamma"
        runtime2, reference2 = nq_outputs(changed)
        self.assertEqual(runtime, runtime2)
        self.assertNotEqual(reference, reference2)
        self.assertEqual(runtime["initial_answer"], "beta")

    def test_fever_tasks_do_not_fake_an_answer_or_evidence(self):
        row = {"id": 20, "claim": "Amber is a mineral.", "label": "REFUTES",
               "gold_page_ids": ["Amber"], "gold_evidence_text": ["Amber is fossil resin."]}
        task, gold = fever_outputs(row)
        self.assertEqual(set(task), {"dataset", "sample_id", "query"})
        self.assertEqual(gold["gold_answer_or_label"], "REFUTES")
        for bad in ({**row, "gold_evidence_text": []}, {**row, "label": "NOT ENOUGH INFO"}):
            with self.assertRaises(ValueError):
                fever_outputs(bad)

    def test_exposure_keeps_namespaces_and_does_not_treat_passages_as_samples(self):
        record = {"splits": {"nq": {"train": ["11"]}, "fever": {"test": ["11"]}},
                  "records": [{"dataset": "fever", "sample_id": "22", "query": "Claim here",
                               "evidence": [{"passage_id": "33", "text": "body"}]}]}
        ids, queries = extract_exposure(record, "nq")
        self.assertEqual(ids["nq"], {"11"})
        self.assertEqual(ids["fever"], {"11", "22"})
        self.assertEqual(queries["fever"], {"Claim here"})

    def test_precision_is_labelled_as_planning_not_observed_gain(self):
        report = precision_notes(500)
        self.assertEqual(report["single_accuracy_worst_case_95pct_half_width_pp"], 4.38)
        self.assertEqual(report["paired_gain_if_discordance_20pct_half_width_pp"], 3.92)
        self.assertIn("not observed", report["interpretation"])


class BuildTests(unittest.TestCase):
    def write(self, root, path, raw):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)

    def fixture(self, root):
        policy = json.loads((ROOT / POLICY).read_text())
        policy["counts_per_dataset"] = dict.fromkeys(ROLES, 1)
        # Three-row fixtures have one row per role; production policy stays 10%-15%.
        policy["evaluation_reserve"]["minimum_fraction"] = 0.30
        policy["evaluation_reserve"]["maximum_fraction"] = 0.35
        self.write(root, POLICY, json.dumps(policy).encode())
        self.write(root, policy["preexisting_variant_quarantine"], b'{"groups": []}')
        for path in [policy["protocol_path"], str(Path(policy["protocol_path"]).with_suffix(".sha256")),
                     "scripts/prepare_controller_data.py", "src/controller/data_preparation.py",
                     "src/compensation/nq_train_adapter.py", "src/diagnosis/features.py", "src/diagnosis/schema.py"]:
            self.write(root, path, (ROOT / path).read_bytes())
        nq = [nq_row(str(i), f"What is crystal{i} isotope{i}?") for i in range(3)]
        raw = jsonl_bytes(nq)
        self.write(root, policy["nq_delivery_directory"] + "/shard.jsonl", raw)
        self.write(root, policy["nq_delivery_manifest"], json.dumps({"total_samples": 3,
            "shards": [{"file": "shard.jsonl", "n_samples": 3, "sha256": sha(raw)}]}).encode())
        self.write(root, policy["nq_generation_stats"], json.dumps({"generation_model": "Qwen/Qwen2.5-7B-Instruct",
            "quantization": "nf4", "decoding": "greedy do_sample=False", "max_new_tokens": 64}).encode())
        self.write(root, policy["nq_partition_source"], json.dumps({"partitions": {
            old: [str(i)] for i, old in enumerate(OLD_ROLES)}}).encode())
        fever = {}
        for i in range(100):
            query = f"Mineral{i} contains isotope{i+1000}."
            role = fever_role(query)
            fever.setdefault(role, {"id": i+100, "claim": query, "label": "SUPPORTS",
                                   "gold_page_ids": ["Mineral"], "gold_evidence_text": ["Some evidence."]})
            if len(fever) == 3:
                break
        self.write(root, policy["fever_source"], jsonl_bytes(fever.values()))
        return policy

    def build(self, root):
        with patch("scripts.prepare_controller_data.exposure_audit",
                   return_value=(defaultdict(set), defaultdict(set),
                                 {"files": [], "known_ids": {"nq": [], "fever": []}})):
            return prepare(root, root / "out")

    def register(self, root):
        manifest = root / "out/manifest.json"
        value = {"schema_version": "controller_data_registration_v1", "manifest_path": "out/manifest.json",
                 "manifest_sha256": sha(manifest.read_bytes()),
                 "active_sample_ids": json.loads((root/"out/selection_ids.json").read_text())}
        self.write(root, REGISTRY, json.dumps(value).encode())

    def relock_test_artifact(self, root, path, raw):
        """Exercise semantic validation even when a faulty producer hashes its output."""
        self.write(root / "out", path, raw)
        manifest = json.loads((root/"out/manifest.json").read_text())
        manifest["artifacts"][path] = {"sha256": sha(raw), "bytes": len(raw)}
        self.write(root, "out/manifest.json", json.dumps(manifest).encode())
        self.register(root)

    def test_complete_small_build_keeps_missing_fever_c1_and_assets_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            manifest = self.build(root)
            self.assertFalse(manifest["full_T2_ready"])
            self.assertTrue(manifest["nq"]["paired_runtime_and_gold_ready"])
            self.assertFalse(manifest["fever"]["paired_runtime_and_gold_ready"])
            self.assertEqual(manifest["fever"]["missing_c1_count"], 3)
            self.assertEqual(manifest["evaluation_reserve"]["nq"]["protected_count"], 1)
            self.assertEqual(manifest["evaluation_reserve"]["fever"]["protected_count"], 1)
            for path, item in manifest["artifacts"].items():
                self.assertEqual(sha((root / "out" / path).read_bytes()), item["sha256"])
            self.assertEqual(json.loads((root/"out/manifest.json").read_text()), manifest)
            self.register(root)
            checked = verify(root, check_sources=True)
            self.assertTrue(checked["data_integrity_passed"])
            self.assertFalse(checked["full_T2_ready"])
            with self.assertRaises(FileExistsError):
                self.build(root)

    def test_corrupted_delivery_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = self.fixture(root)
            (root / policy["nq_delivery_directory"] / "shard.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                self.build(root)
            self.assertFalse((root/"out").exists())

    def test_too_small_protected_reserve_fails_before_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = self.fixture(root)
            policy["evaluation_reserve"]["minimum_fraction"] = 0.40
            self.write(root, POLICY, json.dumps(policy).encode())
            with self.assertRaisesRegex(ValueError, "protected evaluation pool"):
                self.build(root)
            self.assertFalse((root/"out").exists())

    def test_checker_rejects_changed_bytes_and_gold_in_runtime_even_if_relocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            self.build(root)
            self.register(root)
            path = "nq/development/runtime.jsonl"
            row = json.loads((root/"out"/path).read_text())
            row["gold_answer_or_label"] = "alpha"
            raw = jsonl_bytes([row])
            self.write(root/"out", path, raw)
            with self.assertRaisesRegex(ValueError, "Artifact hash mismatch"):
                verify(root)
            self.relock_test_artifact(root, path, raw)
            with self.assertRaisesRegex(ValueError, "Unexpected runtime/task fields"):
                verify(root)

    def test_checker_rejects_reference_id_mismatch_even_if_relocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            self.build(root)
            self.register(root)
            path = "offline/fever/development/gold.jsonl"
            row = json.loads((root/"out"/path).read_text())
            row["sample_id"] = "wrong"
            self.relock_test_artifact(root, path, jsonl_bytes([row]))
            with self.assertRaisesRegex(ValueError, "membership"):
                verify(root)

    def test_checker_rejects_using_reserved_ids_for_optimization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            self.build(root)
            self.register(root)
            path = "evaluation_reserve_ids.json"
            ids = json.loads((root/"out/selection_ids.json").read_text())
            reserve = json.loads((root/"out"/path).read_text())
            reserve["nq"] = ids["nq"]["development"]
            self.relock_test_artifact(root, path, json.dumps(reserve).encode())
            with self.assertRaisesRegex(ValueError, "used for optimization"):
                verify(root)

    def test_missing_assets_never_get_a_fabricated_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = asset_inventory(Path(tmp))
            for records in result["datasets"].values():
                self.assertTrue(all(r["sha256"] is None and not r["exists"] for r in records))
            self.assertFalse(result["index_c1_match_verified"])


if __name__ == "__main__":
    unittest.main()
