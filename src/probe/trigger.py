"""Prototype Probe trigger (selected on validation only; not a final benchmark setting).

v1 (should_probe): probe when 0.40 <= P(failure) <= 0.55, or when
P(failure) > 0.55 and the maximum Stage 2 class probability is below 0.50.
TriggerPolicy generalises the same rule to other thresholds; the v2-expanded
thresholds were selected on the v2 validation split. Triggers read only
Diagnosis outputs, never gold data, target labels or correctness.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

LOW_THRESHOLD = 0.40
HIGH_THRESHOLD = 0.55
STAGE2_THRESHOLD = 0.50


def _probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if math.isnan(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")
    return value


def should_probe(
    stage1_failure_probability: float,
    stage2_class_scores: Mapping[object, float] | None,
) -> bool:
    p_failure = _probability("stage1_failure_probability", stage1_failure_probability)
    if LOW_THRESHOLD <= p_failure <= HIGH_THRESHOLD:
        return True
    if p_failure > HIGH_THRESHOLD:
        if not stage2_class_scores:
            raise ValueError("stage2_class_scores are required when P(failure) > high threshold")
        stage2_max = max(_probability("stage2_class_scores", v) for v in stage2_class_scores.values())
        return stage2_max < STAGE2_THRESHOLD
    return False


@dataclass(frozen=True)
class TriggerPolicy:
    """Probe if low <= P(failure) <= high, or if Stage 1 predicted failure and max Stage 2 score < stage2_threshold.

    stage2_class_scores must be None exactly when Stage 1 predicted correct
    (as produced by HierarchicalDiagnosis).
    """

    low: float
    high: float
    stage2_threshold: float

    def __post_init__(self) -> None:
        for name in ("low", "high", "stage2_threshold"):
            _probability(name, getattr(self, name))
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must not exceed high ({self.high})")

    def should_probe(
        self,
        stage1_failure_probability: float,
        stage2_class_scores: Mapping[object, float] | None,
    ) -> bool:
        p_failure = _probability("stage1_failure_probability", stage1_failure_probability)
        if self.low <= p_failure <= self.high:
            return True
        if stage2_class_scores is None:
            return False
        if not stage2_class_scores:
            raise ValueError("stage2_class_scores must not be empty")
        stage2_max = max(_probability("stage2_class_scores", v) for v in stage2_class_scores.values())
        return stage2_max < self.stage2_threshold


V1_TRIGGER = TriggerPolicy(LOW_THRESHOLD, HIGH_THRESHOLD, STAGE2_THRESHOLD)
# Selected by scripts/search_probe_trigger_v2_expanded.py on the v2-expanded validation split.
V2_EXPANDED_TRIGGER = TriggerPolicy(low=0.50, high=0.65, stage2_threshold=0.50)
