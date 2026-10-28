"""Online P60 FeatureProvider: schema, unseen query, historical P26 alignment, select."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from src.controller.p26_semantic import P26_FEATURE_NAMES, P26SemanticRuntime
from src.controller.p60_online_features import (
    P60OnlineFeatureProvider,
    build_p60_online_feature_provider,
    expected_feature_names,
    frozen_feature_names,
)
from src.controller.selection import FrozenOnlineSelectionAdapter
from src.diagnosis.features import FEATURE_NAMES
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import DatasetName, DiagnosisInput, EvidenceItem
from src.pipeline.current import readiness

ROOT = Path(__file__).resolve().parents[1]
SEM_DIR = ROOT / "data/processed/diagnosis/nq_train_74097_rep08_semantic_runtime_v3"
SEM_ARRAYS = SEM_DIR / "semantic_arrays_v3.npz"
SPLIT_PATH = ROOT / "baselines/splits/diagnosis_nq_train_74097_split.json"
DATASET_PATH = ROOT / "data/processed/diagnosis/nq_train_74097_diagnosis.jsonl"

# Float tolerance for BGE / CrossEncoder recomputation vs historical checkpoint.
# Models are deterministic given same weights + inputs; residual comes from
# GPU/CPU matmul and ST encode batching. Historical arrays are float32.
ALIGN_ATOL = 2e-3
ALIGN_RTOL = 1e-4


def _failure_diag() -> HierarchicalDiagnosis:
    p_fail = 0.8
    cond = {
        DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.5,
        DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.3,
        DiagnosisLabel.EVIDENCE_UTILISATION: 0.2,
    }
    joint = {DiagnosisLabel.CORRECT: 1.0 - p_fail}
    joint.update({k: p_fail * v for k, v in cond.items()})
    return HierarchicalDiagnosis(
        stage1_failure_probability=p_fail,
        stage1_label="failure",
        stage2_class_scores=cond,
        final_label=DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
        confidence=joint[DiagnosisLabel.EVIDENCE_INSUFFICIENCY],
        class_scores=joint,
    )


def _synthetic_item(sample_id: str = "unseen-synthetic-001") -> DiagnosisInput:
    passages = [
        EvidenceItem(
            passage_id=f"syn-pass-{i}",
            rank=i,
            score=1.0 - 0.1 * i,
            text=f"Passage {i} about Canberra capital city Australia.",
        )
        for i in range(1, 6)
    ]
    return DiagnosisInput(
        sample_id=sample_id,
        dataset=DatasetName.NQ,
        query="What is the capital of Australia?",
        evidence=passages,
        initial_answer="Canberra",
    )


@unittest.skipUnless(SEM_ARRAYS.is_file(), "semantic_arrays_v3.npz not present")
class P60OnlineProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names = frozen_feature_names()
        cls.provider = P60OnlineFeatureProvider()

    def test_schema_matches_frozen_78(self):
        self.assertEqual(len(self.names), 78)
        self.assertEqual(self.names, expected_feature_names())
        self.assertEqual(self.names[:27], FEATURE_NAMES)
        self.assertEqual(self.names[27:62], P26_FEATURE_NAMES)

    def test_unseen_sample_id_produces_78_without_lookup(self):
        item = _synthetic_item("never-seen-query-xyz-999")
        row = self.provider.features_for(item, _failure_diag())
        self.assertEqual(row.shape, (78,))
        self.assertTrue(np.isfinite(row).all())
        receipt = self.provider.runtime_receipt()
        self.assertIsInstance(receipt, dict)
        self.assertEqual(set(receipt), {"input_sha256", "calls"})
        kinds = {c["kind"] for c in receipt["calls"]}
        self.assertIn("bge_encode", kinds)
        self.assertIn("bge_reranker", kinds)

    def test_fail_fast_on_wrong_evidence_count(self):
        item = _synthetic_item()
        item = DiagnosisInput(
            sample_id=item.sample_id,
            dataset=item.dataset,
            query=item.query,
            evidence=item.evidence[:3],
            initial_answer=item.initial_answer,
        )
        with self.assertRaisesRegex(ValueError, "exactly 5"):
            self.provider.features_for(item, _failure_diag())

    def test_frozen_selection_adapter_runs_on_online_features(self):
        item = _synthetic_item("select-smoke-1")
        row = self.provider.features_for(item, _failure_diag())
        adapter = FrozenOnlineSelectionAdapter.from_frozen_defaults(repo_root=ROOT)
        decision = adapter.select(_failure_diag(), row, sample_id=item.sample_id)
        self.assertIsNotNone(decision.selection_lane)
        self.assertEqual(decision.failure_type, DiagnosisLabel.EVIDENCE_INSUFFICIENCY)

    def test_factory_registered_shape(self):
        provider = build_p60_online_feature_provider(execution_backend=None)
        self.assertEqual(tuple(provider.feature_names), self.names)
        self.assertTrue(callable(provider.runtime_receipt))


@unittest.skipUnless(
    SEM_ARRAYS.is_file() and SPLIT_PATH.is_file() and DATASET_PATH.is_file(),
    "P26 historical artifacts / diagnosis jsonl missing",
)
class P26OnlineHistoricalAlignmentTests(unittest.TestCase):
    """Recompute P26 for a few Train/Val IDs; compare to semantic_arrays_v3."""

    @classmethod
    def setUpClass(cls):
        arrays = np.load(SEM_ARRAYS, allow_pickle=False)
        split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        train_ids = [str(x) for x in split["splits"]["nq"]["train"]]
        val_ids = [str(x) for x in split["splits"]["nq"]["validation"]]
        order = train_ids + val_ids
        want = set(order)
        by_id = {}
        with DATASET_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                sid = str(rec["input"]["sample_id"])
                if sid in want:
                    by_id[sid] = rec
        missing = [sid for sid in order if sid not in by_id]
        if missing:
            raise RuntimeError(f"diagnosis jsonl missing {len(missing)} split IDs")
        cls.records = [by_id[sid] for sid in order]
        assert len(cls.records) == len(arrays["qa_support_max"])
        cls.arrays = {k: arrays[k] for k in arrays.files}
        n_tr = len(train_ids)
        cls.probe_idx = [0, 1, 2, n_tr, n_tr + 1]
        cls.runtime = P26SemanticRuntime()

    def test_online_matches_historical_p26_rows(self):
        max_abs = 0.0
        mismatches = []
        scalar_keys = [
            "qa_support_max",
            "qa_support_mean_top3",
            "qa_support_first_rank",
            "qa_support_margin",
            "qa_support_concentration",
            "support_peak_dispersion",
            "top_support_margin",
            "evidence_agreement_mean",
            "evidence_agreement_min",
            "answer_evidence_max_similarity",
            "answer_evidence_top1_similarity",
            "answer_evidence_mean_topk",
            "qa_type_match",
            "s5_rerank_best_chunk",
            "s5_rerank_top1_passage",
        ]
        for gi in self.probe_idx:
            rec = self.records[gi]
            inp = rec["input"]
            passages = sorted(inp["evidence"], key=lambda p: int(p.get("rank", 999)))
            texts = [str(p.get("text") or "") for p in passages]
            online = self.runtime.compute(
                str(inp["query"]),
                str(inp.get("initial_answer") or ""),
                texts,
            )
            for key in scalar_keys:
                hist = float(self.arrays[key][gi])
                got = float(getattr(online, key))
                diff = abs(got - hist)
                max_abs = max(max_abs, diff)
                if not math.isclose(got, hist, rel_tol=ALIGN_RTOL, abs_tol=ALIGN_ATOL):
                    mismatches.append(
                        {
                            "gi": gi,
                            "sample_id": inp["sample_id"],
                            "feature": key,
                            "hist": hist,
                            "online": got,
                            "abs_diff": diff,
                        }
                    )
            for arr_name, online_arr in (
                ("chunk_support", online.chunk_support),
                ("pass_support", online.pass_support),
                ("qtypes", online.qtypes),
                ("atypes", online.atypes),
            ):
                hist = np.asarray(self.arrays[arr_name][gi], dtype=np.float64)
                got = np.asarray(online_arr, dtype=np.float64)
                diff = float(np.max(np.abs(got - hist)))
                max_abs = max(max_abs, diff)
                if not np.allclose(got, hist, rtol=ALIGN_RTOL, atol=ALIGN_ATOL):
                    mismatches.append(
                        {
                            "gi": gi,
                            "sample_id": inp["sample_id"],
                            "feature": arr_name,
                            "max_abs_diff": diff,
                            "hist": hist.tolist(),
                            "online": got.tolist(),
                        }
                    )
        if mismatches:
            self.fail(
                f"P26 online vs historical mismatch (max_abs={max_abs:.6g}, "
                f"atol={ALIGN_ATOL}): {mismatches[:5]}"
            )
        # Soft assert that we are well inside tolerance on this machine.
        self.assertLessEqual(max_abs, ALIGN_ATOL)


class ReadinessProviderProvenanceTests(unittest.TestCase):
    def test_provider_and_provenance_no_longer_null_missing(self):
        report = readiness(ROOT)
        missing = report["missing"]
        self.assertNotIn("runtime_78_feature_provider_not_delivered", missing)
        self.assertNotIn("P60_training_tuning_calibration_ID_provenance_not_delivered", missing)
        # Source sha must match registered provider file.
        self.assertNotIn("runtime_feature_provider_source_mismatch", missing)
        self.assertNotIn("runtime_feature_provider_cost_contract_unregistered", missing)


if __name__ == "__main__":
    unittest.main()
