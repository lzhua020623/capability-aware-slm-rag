"""DiagnosisInput -> Hierarchical Diagnosis -> Probe trigger -> optional Probe.

No repair and no Controller logic. Diagnosis probabilities decide only
whether the Probe runs; the Probe sees only the DiagnosisInput.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis, HierarchicalDiagnosisClassifier
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import DiagnosisInput
from src.probe.probe import EvidenceProbe, refine_label
from src.probe.schema import ProbeResult
from src.probe.trigger import TriggerPolicy, should_probe


@dataclass(frozen=True)
class ProbedDiagnosis:
    diagnosis: HierarchicalDiagnosis
    probe_triggered: bool
    probe_result: ProbeResult | None
    final_label: DiagnosisLabel


def apply_probe(
    item: DiagnosisInput,
    diagnosis: HierarchicalDiagnosis,
    probe: EvidenceProbe,
    trigger: TriggerPolicy | None = None,
) -> ProbedDiagnosis:
    """Run the Probe if triggered; trigger=None keeps the v1 should_probe rule."""
    trigger_fn = should_probe if trigger is None else trigger.should_probe
    triggered = trigger_fn(diagnosis.stage1_failure_probability, diagnosis.stage2_class_scores)
    result = probe.run(item) if triggered else None
    return ProbedDiagnosis(diagnosis, triggered, result, refine_label(diagnosis.final_label, result))


def diagnose_with_probe(
    item: DiagnosisInput,
    classifier: HierarchicalDiagnosisClassifier,
    probe: EvidenceProbe,
    trigger: TriggerPolicy | None = None,
) -> ProbedDiagnosis:
    return apply_probe(item, classifier.diagnose(item), probe, trigger)
