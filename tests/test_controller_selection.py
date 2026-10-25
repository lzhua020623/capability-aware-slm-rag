"""Tests for FrozenOnlineSelectionAdapter (P60 formal). No Test dataset, no gold."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from src.controller.controller import decision_from_selection, run_controller
from src.controller.schema import ControllerAction, SelectionLane
from src.controller.selection import (
    DEFAULT_CONFIG_PATH,
    FrozenAdapterModels,
    FrozenOnlineSelectionAdapter,
    d_combo_score,
)
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
from src.diagnosis.labels import DiagnosisLabel
from tests.test_probe_mapping import nq_input

ROOT = Path(__file__).resolve().parents[1]


def failure_diag(
    *,
    p_fail: float = 0.80,
    p_insuff: float = 0.60,
    p_access: float = 0.10,
    p_util: float = 0.10,
    label: DiagnosisLabel = DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
    confidence: float = 0.55,
) -> HierarchicalDiagnosis:
    p_correct = max(0.0, 1.0 - p_fail)
    # Renormalize failure mass into the three subtypes for a valid joint.
    mass = p_insuff + p_access + p_util
    if mass <= 0:
        p_insuff, p_access, p_util = p_fail, 0.0, 0.0
    else:
        scale = p_fail / mass
        p_insuff, p_access, p_util = p_insuff * scale, p_access * scale, p_util * scale
    joint = {
        DiagnosisLabel.CORRECT: p_correct,
        DiagnosisLabel.EVIDENCE_INSUFFICIENCY: p_insuff,
        DiagnosisLabel.EVIDENCE_ACCESSIBILITY: p_access,
        DiagnosisLabel.EVIDENCE_UTILISATION: p_util,
    }
    stage2 = {
        DiagnosisLabel.EVIDENCE_INSUFFICIENCY: p_insuff / p_fail if p_fail else 0.0,
        DiagnosisLabel.EVIDENCE_ACCESSIBILITY: p_access / p_fail if p_fail else 0.0,
        DiagnosisLabel.EVIDENCE_UTILISATION: p_util / p_fail if p_fail else 0.0,
    }
    return HierarchicalDiagnosis(
        stage1_failure_probability=p_fail,
        stage1_label="failure",
        stage2_class_scores=stage2,
        final_label=label,
        confidence=confidence,
        class_scores=joint,
    )


def correct_diag(p_fail: float = 0.20) -> HierarchicalDiagnosis:
    return HierarchicalDiagnosis(
        p_fail, "correct", None, DiagnosisLabel.CORRECT, 1.0 - p_fail, None
    )


class _ConstProbaModel:
    def __init__(self, classes, proba_row):
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = 78
        self._proba = np.asarray(proba_row, dtype=float)

    def predict_proba(self, X):
        X = np.asarray(X)
        return np.repeat(self._proba.reshape(1, -1), len(X), axis=0)


def stub_adapter(*, n_features: int = 78) -> FrozenOnlineSelectionAdapter:
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert len(config["feature_names"]) == n_features
    models = FrozenAdapterModels(
        fail_gate=_ConstProbaModel([0, 1], [0.2, 0.8]),
        rec_given_fail=_ConstProbaModel([0, 1], [0.3, 0.7]),
        multiclass_3c=_ConstProbaModel([0, 1, 2], [0.2, 0.3, 0.5]),
    )
    return FrozenOnlineSelectionAdapter(config, models)


class FrozenAdapterLoadTests(unittest.TestCase):
    def test_loads_frozen_config_and_models(self):
        adapter = FrozenOnlineSelectionAdapter.from_frozen_defaults(repo_root=ROOT)
        self.assertEqual(adapter.version, "p60_online_adapter_v1")
        self.assertFalse(adapter.probe_enabled)
        self.assertEqual(adapter.n_features, 78)
        self.assertAlmostEqual(adapter.t_insuff, 0.5349365680553068)
        self.assertAlmostEqual(adapter.t_combo, 0.49535149245589105)
        self.assertTrue((ROOT / "models/controller/p60_online_adapter/hgb_fail_gate.joblib").is_file())
        # Runtime path must not require _experiments
        self.assertNotIn("_experiments", str(adapter.config_path).replace("\\", "/"))


class SingleQueryLaneTests(unittest.TestCase):
    def setUp(self):
        self.adapter = stub_adapter()
        self.features = np.zeros(78, dtype=float)

    def test_lane1_insuff_preserve(self):
        diag = failure_diag(p_insuff=0.70, p_access=0.05, p_util=0.05, p_fail=0.80)
        decision = self.adapter.select(diag, self.features, sample_id="s-lane1")
        self.assertTrue(decision.needs_enhancement)
        self.assertIs(decision.selection_lane, SelectionLane.INSUFF_PRESERVE)
        self.assertGreaterEqual(decision.p_insuff, self.adapter.t_insuff)
        self.assertIs(decision.failure_type, DiagnosisLabel.EVIDENCE_INSUFFICIENCY)
        self.assertFalse(decision.used_probe)

    def test_lane2_residual_combo(self):
        # Low insuff; force high D_combo via overrides (no batch ranking).
        diag = failure_diag(
            p_insuff=0.05,
            p_access=0.40,
            p_util=0.35,
            p_fail=0.90,
            label=DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
        )
        # Choose rec/p_corr so frozen-z D_combo >= t_combo.
        rec, p_corr = 0.9, 0.01
        combo = d_combo_score(
            0.90,
            rec,
            p_corr,
            alpha=self.adapter.alpha,
            lam=self.adapter.lam,
            z_stats=self.adapter.z_stats,
        )
        self.assertGreaterEqual(combo, self.adapter.t_combo)
        decision = self.adapter.select(
            diag, self.features, p_insuff=0.05, rec=rec, p_corr_3c=p_corr
        )
        self.assertTrue(decision.needs_enhancement)
        self.assertIs(decision.selection_lane, SelectionLane.RESIDUAL_COMBO)
        self.assertIs(decision.failure_type, DiagnosisLabel.EVIDENCE_ACCESSIBILITY)
        self.assertFalse(decision.used_probe)

    def test_no_action(self):
        diag = failure_diag(
            p_insuff=0.05,
            p_access=0.40,
            p_util=0.35,
            p_fail=0.55,
            label=DiagnosisLabel.EVIDENCE_UTILISATION,
        )
        decision = self.adapter.select(
            diag, self.features, p_insuff=0.05, rec=0.0, p_corr_3c=0.9
        )
        self.assertFalse(decision.needs_enhancement)
        self.assertIs(decision.selection_lane, SelectionLane.NO_ACTION)
        self.assertIs(decision.failure_type, DiagnosisLabel.EVIDENCE_UTILISATION)
        self.assertFalse(decision.used_probe)

    def test_correct_diagnosis_forced_no_action(self):
        decision = self.adapter.select(correct_diag(), self.features)
        self.assertFalse(decision.needs_enhancement)
        self.assertIs(decision.selection_lane, SelectionLane.NO_ACTION)
        self.assertIs(decision.failure_type, DiagnosisLabel.CORRECT)

    def test_failure_type_not_overridden_by_lane(self):
        # Lane1 fires on high p_insuff but Diagnosis subtype is utilisation.
        diag = failure_diag(
            p_insuff=0.70,
            p_access=0.05,
            p_util=0.05,
            p_fail=0.80,
            label=DiagnosisLabel.EVIDENCE_UTILISATION,
        )
        decision = self.adapter.select(diag, self.features, p_insuff=0.70)
        self.assertIs(decision.selection_lane, SelectionLane.INSUFF_PRESERVE)
        self.assertIs(decision.failure_type, DiagnosisLabel.EVIDENCE_UTILISATION)


class ProbeAndControllerWiringTests(unittest.TestCase):
    def test_probe_none_without_adapter(self):
        class StubClassifier:
            def diagnose(self, item):
                return failure_diag(label=DiagnosisLabel.EVIDENCE_ACCESSIBILITY)

        decision = run_controller(nq_input(), StubClassifier(), probe=None)
        self.assertFalse(decision.used_probe)
        self.assertFalse(decision.probe_overrode_diagnosis)
        self.assertIs(decision.action, ControllerAction.EVIDENCE_REFINEMENT)

    def test_run_controller_with_selection_adapter_ignores_probe(self):
        adapter = stub_adapter()
        features = np.zeros(78)

        class StubClassifier:
            def diagnose(self, item):
                return failure_diag(p_insuff=0.70, label=DiagnosisLabel.EVIDENCE_INSUFFICIENCY)

        decision = run_controller(
            nq_input(),
            StubClassifier(),
            probe=None,
            selection_adapter=adapter,
            adapter_features=features,
        )
        self.assertFalse(decision.used_probe)
        self.assertIs(decision.action, ControllerAction.RETRIEVAL_ENHANCEMENT)
        self.assertTrue(decision.needs_compensation)

    def test_decision_from_selection_no_action(self):
        adapter = stub_adapter()
        sel = adapter.select(
            failure_diag(p_insuff=0.05, label=DiagnosisLabel.EVIDENCE_UTILISATION),
            np.zeros(78),
            p_insuff=0.05,
            rec=0.0,
            p_corr_3c=0.9,
        )
        decision = decision_from_selection(sel)
        self.assertIs(decision.action, ControllerAction.NO_ACTION)
        self.assertFalse(decision.used_probe)
        self.assertFalse(decision.needs_compensation)


class SafetyTests(unittest.TestCase):
    def test_no_gold_required(self):
        adapter = stub_adapter()
        diag = failure_diag(p_insuff=0.70)
        # Only runtime diagnosis + features; no gold fields on either object.
        decision = adapter.select(diag, np.zeros(78))
        self.assertIn(decision.selection_lane, SelectionLane)

    def test_invalid_feature_length_raises(self):
        adapter = stub_adapter()
        with self.assertRaises(ValueError):
            adapter.select(failure_diag(p_insuff=0.7), np.zeros(10))

    def test_non_finite_features_raise(self):
        adapter = stub_adapter()
        bad = np.zeros(78)
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            adapter.select(failure_diag(p_insuff=0.7), bad)

    def test_no_batch_api(self):
        source = (ROOT / "src/controller/selection.py").read_text(encoding="utf-8")
        for token in ("np.percentile", "np.argsort", "topk_mask", "top_k_mask", "argpartition"):
            self.assertNotIn(token, source)

    def test_probe_enabled_config_rejected(self):
        config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        config = copy.deepcopy(config)
        config["probe_enabled"] = True
        models = FrozenAdapterModels(
            fail_gate=_ConstProbaModel([0, 1], [0.5, 0.5]),
            rec_given_fail=None,
            multiclass_3c=_ConstProbaModel([0, 1, 2], [0.3, 0.3, 0.4]),
        )
        with self.assertRaises(ValueError):
            FrozenOnlineSelectionAdapter(config, models)


class HierarchicalUnchangedTests(unittest.TestCase):
    def test_hierarchical_module_not_edited_for_selection(self):
        # Smoke: constructing HierarchicalDiagnosis still works as before.
        d = failure_diag()
        self.assertEqual(d.stage1_label, "failure")
        self.assertIn(DiagnosisLabel.EVIDENCE_INSUFFICIENCY, d.class_scores)


if __name__ == "__main__":
    unittest.main()
