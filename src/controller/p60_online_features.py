"""Production online 78-dim FeatureProvider for frozen P60.

Assembles CURRENT27 + online P26 + Diagnosis-derived + runtime-safe aliases
in the exact ``configs/controller/p60_online_adapter.json`` feature_names order.
Never looks up historical semantic tables by sample_id.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from src.controller.execution import fingerprint
from src.controller.p26_semantic import P26_FEATURE_NAMES, P26SemanticRuntime
from src.diagnosis.features import FEATURE_NAMES, extract_features, tokenize
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import DiagnosisInput

ROOT = Path(__file__).resolve().parents[2]
SELECTOR_CONFIG = ROOT / "configs/controller/p60_online_adapter.json"

DIAG_NAMES: tuple[str, ...] = (
    "diag_p_correct",
    "diag_p_insuff",
    "diag_p_access",
    "diag_p_util",
    "diag_p_fail",
    "diag_fail_margin",
    "diag_entropy",
    "diag_max_class_prob",
    "diag_access_minus_util",
)
ALIAS_NAMES: tuple[str, ...] = (
    "score_top1",
    "score_gap_1_2",
    "score_std_top5",
    "score_range_top5",
    "answer_len_tokens",
    "question_len_tokens",
    "answer_in_evidence_jaccard",
)


def frozen_feature_names(repo_root: Path | None = None) -> tuple[str, ...]:
    root = Path(repo_root) if repo_root is not None else ROOT
    names = json.loads((root / "configs/controller/p60_online_adapter.json").read_text(encoding="utf-8"))[
        "feature_names"
    ]
    return tuple(names)


def expected_feature_names() -> tuple[str, ...]:
    return tuple(FEATURE_NAMES) + P26_FEATURE_NAMES + DIAG_NAMES + ALIAS_NAMES


def _diag_vector(diagnosis: HierarchicalDiagnosis) -> np.ndarray:
    """Map HierarchicalDiagnosis → 9 diag_* dims (P53/P60 recipe online path)."""
    if not isinstance(diagnosis, HierarchicalDiagnosis):
        raise TypeError(f"diagnosis must be HierarchicalDiagnosis, got {type(diagnosis).__name__}")
    p_fail = float(diagnosis.stage1_failure_probability)
    if not math.isfinite(p_fail) or not (0.0 <= p_fail <= 1.0):
        raise ValueError(f"Invalid stage1_failure_probability: {p_fail}")

    if diagnosis.class_scores is not None:
        p_c = float(diagnosis.class_scores.get(DiagnosisLabel.CORRECT, 0.0))
        p_i = float(diagnosis.class_scores.get(DiagnosisLabel.EVIDENCE_INSUFFICIENCY, 0.0))
        p_a = float(diagnosis.class_scores.get(DiagnosisLabel.EVIDENCE_ACCESSIBILITY, 0.0))
        p_u = float(diagnosis.class_scores.get(DiagnosisLabel.EVIDENCE_UTILISATION, 0.0))
    elif diagnosis.final_label is DiagnosisLabel.CORRECT or diagnosis.stage1_label == "correct":
        p_c = 1.0 - p_fail
        p_i = p_a = p_u = 0.0
    elif diagnosis.stage2_class_scores is not None:
        p_c = 1.0 - p_fail
        p_i = p_fail * float(
            diagnosis.stage2_class_scores.get(DiagnosisLabel.EVIDENCE_INSUFFICIENCY, 0.0)
        )
        p_a = p_fail * float(
            diagnosis.stage2_class_scores.get(DiagnosisLabel.EVIDENCE_ACCESSIBILITY, 0.0)
        )
        p_u = p_fail * float(
            diagnosis.stage2_class_scores.get(DiagnosisLabel.EVIDENCE_UTILISATION, 0.0)
        )
    else:
        raise ValueError(
            "Failure-path HierarchicalDiagnosis requires class_scores or stage2_class_scores"
        )

    for name, val in (
        ("diag_p_correct", p_c),
        ("diag_p_insuff", p_i),
        ("diag_p_access", p_a),
        ("diag_p_util", p_u),
    ):
        if not math.isfinite(val):
            raise ValueError(f"Non-finite {name}")
    fps = sorted([p_i, p_a, p_u], reverse=True)
    proba = [p_c, p_i, p_a, p_u]
    ent = -sum(p * math.log(max(float(p), 1e-12)) for p in proba)
    return np.asarray(
        [
            p_c,
            p_i,
            p_a,
            p_u,
            p_fail,
            float(fps[0] - fps[1]),
            float(ent),
            float(max(proba)),
            float(p_a - p_u),
        ],
        dtype=np.float64,
    )


def _alias_vector(item: DiagnosisInput) -> np.ndarray:
    """P53 runtime-safe aliases (padded Top-5 scores + Jaccard)."""
    evidence = sorted(item.evidence, key=lambda p: int(p.rank))
    scores = [float(p.score) for p in evidence[:5]]
    while len(scores) < 5:
        scores.append(0.0)
    ans = str(item.initial_answer or "")
    q = str(item.query or "")
    at = set(tokenize(ans))
    ev_toks = [set(tokenize(p.text)) for p in evidence[:5]]
    if at:
        aie = max((len(at & pt) / max(len(at | pt), 1) for pt in ev_toks), default=0.0)
    else:
        aie = 0.0
    return np.asarray(
        [
            scores[0],
            scores[0] - scores[1],
            float(np.std(scores)),
            scores[0] - scores[4],
            float(len(tokenize(ans))),
            float(len(tokenize(q))),
            float(aie),
        ],
        dtype=np.float64,
    )


class P60OnlineFeatureProvider:
    """Production FeatureProvider: any DiagnosisInput → frozen 78-dim row."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str] | None = None,
        semantic_runtime: P26SemanticRuntime | None = None,
        device: str | None = None,
        repo_root: Path | None = None,
    ):
        names = tuple(feature_names) if feature_names is not None else frozen_feature_names(repo_root)
        expected = expected_feature_names()
        frozen = frozen_feature_names(repo_root)
        if names != frozen:
            raise ValueError("feature_names must match configs/controller/p60_online_adapter.json")
        if expected != frozen:
            raise ValueError("Recovered 78-dim composition drifted from frozen P60 schema")
        self.feature_names = names
        self._runtime = semantic_runtime or P26SemanticRuntime(device=device)
        self._last_receipt: dict | None = None
        self._last_receipts: list[dict] | None = None
        self._last_item_fp: str | None = None

    def runtime_receipt(self) -> dict | None:
        return self._last_receipt

    def _assemble_row(
        self,
        item: DiagnosisInput,
        diagnosis: HierarchicalDiagnosis,
        p26: np.ndarray,
    ) -> np.ndarray:
        feat_map = extract_features(item)
        missing27 = [n for n in FEATURE_NAMES if n not in feat_map]
        if missing27:
            raise ValueError(f"CURRENT27 missing features: {missing27}")
        cur27 = np.asarray([float(feat_map[n]) for n in FEATURE_NAMES], dtype=np.float64)
        if p26.shape != (35,):
            raise ValueError(f"P26 vector must have length 35, got {p26.shape}")
        diag = _diag_vector(diagnosis)
        alias = _alias_vector(item)
        row = np.concatenate([cur27, p26, diag, alias])
        if row.shape != (78,):
            raise ValueError(f"Assembled features must have length 78, got {row.shape}")
        if not np.isfinite(row).all():
            bad = [self.feature_names[i] for i, v in enumerate(row) if not np.isfinite(v)]
            raise ValueError(f"Non-finite feature values: {bad}")
        return row.astype(np.float64, copy=False)

    def features_for(self, item: DiagnosisInput, diagnosis: HierarchicalDiagnosis) -> np.ndarray:
        rows = self.features_for_batch([item], [diagnosis])
        return rows[0]

    def features_for_batch(
        self,
        items: Sequence[DiagnosisInput],
        diagnoses: Sequence[HierarchicalDiagnosis],
    ) -> list[np.ndarray]:
        """Batch P26 encode/CE; CURRENT27/diag/alias remain per-sample (CPU, cheap)."""
        if len(items) != len(diagnoses):
            raise ValueError("items/diagnoses length mismatch")
        if not items:
            self._last_receipt = None
            self._last_receipts = []
            return []
        for item in items:
            if not isinstance(item, DiagnosisInput):
                raise TypeError(f"item must be DiagnosisInput, got {type(item).__name__}")

        payloads = []
        fingerprints = []
        for item in items:
            passages = [p.text for p in sorted(item.evidence, key=lambda e: int(e.rank))]
            payloads.append((item.query, item.initial_answer, passages))
            fingerprints.append(fingerprint(asdict(item)))

        try:
            sem_rows = self._runtime.compute_batch(payloads)
        except Exception:
            calls = self._runtime.runtime_calls()
            self._last_receipts = [
                {"input_sha256": fp, "calls": list(calls)} for fp in fingerprints
            ]
            self._last_receipt = self._last_receipts[-1] if self._last_receipts else None
            self._last_item_fp = fingerprints[-1] if fingerprints else None
            raise

        batch_calls = self._runtime.runtime_calls()
        rows: list[np.ndarray] = []
        receipts: list[dict] = []
        for item, diagnosis, sem, fp in zip(items, diagnoses, sem_rows, fingerprints):
            row = self._assemble_row(item, diagnosis, sem.p60_vector())
            rows.append(row)
            receipts.append({"input_sha256": fp, "calls": list(batch_calls)})
        self._last_receipts = receipts
        self._last_receipt = receipts[-1]
        self._last_item_fp = fingerprints[-1]
        return rows

    def runtime_receipts(self) -> list[dict] | None:
        return list(self._last_receipts) if getattr(self, "_last_receipts", None) else None


def build_p60_online_feature_provider(*, execution_backend=None, device: str | None = None, **_kwargs):
    """Factory registered in configs/controller/current_baseline.json."""
    del execution_backend  # required by load_current_pipeline; features do not use backend
    return P60OnlineFeatureProvider(device=device)
