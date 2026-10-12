"""Tests for Probe mapping, parsing, prompts and Diagnosis fallback. No model is loaded."""

from __future__ import annotations

import inspect
import itertools
import json
import subprocess
import sys
import unittest
from pathlib import Path

from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
from src.diagnosis.labels import DiagnosisLabel
from src.probe.pipeline import apply_probe
from src.probe.probe import EvidenceProbe, ProbeParseError, parse_probe_output, refine_label
from src.probe.prompts import build_probe_prompt
from src.probe.schema import (
    AnswerSupport,
    EvidenceAccessibility,
    EvidenceSufficiency,
    ProbeLabel,
    ProbeResult,
    map_probe_label,
)
from tests.test_diagnosis_features import fever_input, nq_input

ROOT = Path(__file__).resolve().parents[1]


def probe_json(**overrides) -> str:
    values = {
        "answer_supported": "unsupported",
        "evidence_sufficiency": "sufficient",
        "evidence_accessibility": "hard_to_access",
        "supporting_ranks": [],
        "conflicting_ranks": [1],
        "reason": "Rank 1 gives another name.",
    }
    values.update(overrides)
    return json.dumps(values)


class RecordingGenerator:
    def __init__(self, output: str):
        self.output = output
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.output


def diagnosis(p_failure: float, final: DiagnosisLabel, stage2: dict | None) -> HierarchicalDiagnosis:
    return HierarchicalDiagnosis(
        stage1_failure_probability=p_failure,
        stage1_label="failure" if stage2 else "correct",
        stage2_class_scores=stage2,
        final_label=final,
        confidence=0.4321,
        class_scores=None,
    )


UNSURE_STAGE2 = {
    DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.4567,
    DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.3333,
    DiagnosisLabel.EVIDENCE_UTILISATION: 0.2100,
}


class MappingTests(unittest.TestCase):
    def test_four_deterministic_labels(self):
        S, I, A = AnswerSupport, EvidenceSufficiency, EvidenceAccessibility
        self.assertIs(map_probe_label(S.SUPPORTED, I.SUFFICIENT, A.CLEAR), ProbeLabel.CORRECT)
        self.assertIs(map_probe_label(S.UNSUPPORTED, I.INSUFFICIENT, A.UNCERTAIN), ProbeLabel.EVIDENCE_INSUFFICIENCY)
        self.assertIs(map_probe_label(S.UNSUPPORTED, I.SUFFICIENT, A.HARD_TO_ACCESS), ProbeLabel.EVIDENCE_ACCESSIBILITY)
        self.assertIs(map_probe_label(S.UNSUPPORTED, I.SUFFICIENT, A.CLEAR), ProbeLabel.EVIDENCE_UTILISATION)
        self.assertIs(map_probe_label(S.UNCERTAIN, I.INSUFFICIENT, A.CLEAR), ProbeLabel.EVIDENCE_INSUFFICIENCY)
        self.assertIs(map_probe_label(S.UNCERTAIN, I.SUFFICIENT, A.CLEAR), ProbeLabel.EVIDENCE_UTILISATION)

    def test_full_mapping_table(self):
        for s, i, a in itertools.product(AnswerSupport, EvidenceSufficiency, EvidenceAccessibility):
            label = map_probe_label(s, i, a)
            if s is AnswerSupport.SUPPORTED:
                expected = ProbeLabel.CORRECT
            elif i is EvidenceSufficiency.INSUFFICIENT:
                expected = ProbeLabel.EVIDENCE_INSUFFICIENCY
            elif i is EvidenceSufficiency.SUFFICIENT and a is EvidenceAccessibility.HARD_TO_ACCESS:
                expected = ProbeLabel.EVIDENCE_ACCESSIBILITY
            elif i is EvidenceSufficiency.SUFFICIENT and a is EvidenceAccessibility.CLEAR:
                expected = ProbeLabel.EVIDENCE_UTILISATION
            else:
                expected = ProbeLabel.UNCERTAIN
            self.assertIs(label, expected, (s, i, a))

    def test_uncertain_mapping(self):
        S, I, A = AnswerSupport, EvidenceSufficiency, EvidenceAccessibility
        for s in (S.UNSUPPORTED, S.UNCERTAIN):
            self.assertIs(map_probe_label(s, I.UNCERTAIN, A.CLEAR), ProbeLabel.UNCERTAIN)
            self.assertIs(map_probe_label(s, I.SUFFICIENT, A.UNCERTAIN), ProbeLabel.UNCERTAIN)
            self.assertIs(map_probe_label(s, I.UNCERTAIN, A.UNCERTAIN), ProbeLabel.UNCERTAIN)


class ParserTests(unittest.TestCase):
    def test_valid_and_fenced_json(self):
        parsed = parse_probe_output(probe_json())
        self.assertIs(parsed.probe_label, ProbeLabel.EVIDENCE_ACCESSIBILITY)
        self.assertEqual(parsed.conflicting_ranks, (1,))
        fenced = parse_probe_output("```json\n" + probe_json(answer_supported="supported") + "\n```")
        self.assertIs(fenced.probe_label, ProbeLabel.CORRECT)

    def test_invalid_outputs_rejected(self):
        bad_outputs = (
            "The answer is supported.",
            "{not json}",
            json.dumps({"answer_supported": "supported"}),
            probe_json(answer_supported="yes"),
            probe_json(evidence_sufficiency="partial"),
            probe_json(supporting_ranks=[6]),
            probe_json(conflicting_ranks=[0]),
            probe_json(supporting_ranks="1"),
            probe_json(reason=["long"]),
        )
        for text in bad_outputs:
            with self.subTest(text=text), self.assertRaises(ProbeParseError):
                parse_probe_output(text)
        with self.assertRaises(ProbeParseError):
            parse_probe_output(probe_json(supporting_ranks=[4]), max_rank=3)

    def test_run_falls_back_to_uncertain(self):
        generator = RecordingGenerator("I think it is supported")
        result = EvidenceProbe(generator).run(nq_input())
        self.assertIs(result.probe_label, ProbeLabel.UNCERTAIN)
        self.assertIsNotNone(result.parse_error)
        self.assertEqual(len(generator.prompts), 1)


class PromptTests(unittest.TestCase):
    def test_prompt_contents(self):
        nq_prompt = build_probe_prompt(nq_input(sample_id="sample-777"))
        self.assertIn("Question: who proposed the quantum hypothesis", nq_prompt)
        self.assertIn("Proposed answer: Max Planck", nq_prompt)
        self.assertIn("[1] The quantum theory began in 1900 .", nq_prompt)
        self.assertIn("semantically equivalent", nq_prompt)
        self.assertNotIn("sample-777", nq_prompt)
        self.assertNotIn("nq_p_1", nq_prompt)
        fever_prompt = build_probe_prompt(fever_input())
        self.assertIn("Claim: Boeing 787 Dreamliner is a machine.", fever_prompt)
        self.assertNotIn("Predicted verdict", fever_prompt)
        self.assertNotIn("Boeing_787_Dreamliner", fever_prompt)
        self.assertNotIn("106023", fever_prompt)
        self.assertEqual(build_probe_prompt(nq_input()), build_probe_prompt(nq_input()))

    def test_prompt_takes_only_runtime_input(self):
        self.assertEqual(list(inspect.signature(build_probe_prompt).parameters), ["item"])
        with self.assertRaises(TypeError):
            build_probe_prompt({"input": {}, "target_label": "correct"})

    def test_diagnosis_values_not_in_prompt(self):
        generator = RecordingGenerator(probe_json())
        apply_probe(nq_input(), diagnosis(0.4999, DiagnosisLabel.EVIDENCE_ACCESSIBILITY, UNSURE_STAGE2),
                    EvidenceProbe(generator))
        self.assertEqual(len(generator.prompts), 1)
        prompt = generator.prompts[0]
        for leaked in ("0.4999", "0.4567", "0.3333", "0.21", "0.4321", "stage1", "stage2", "confidence",
                       "probability", "diagnosis", "gold", "target"):
            self.assertNotIn(leaked, prompt.lower())
        self.assertEqual(prompt, build_probe_prompt(nq_input()))

    def test_probe_needs_no_gold(self):
        generator = RecordingGenerator(probe_json(answer_supported="supported"))
        result = EvidenceProbe(generator).run(nq_input())
        self.assertIs(result.probe_label, ProbeLabel.CORRECT)
        self.assertEqual(list(inspect.signature(EvidenceProbe.run).parameters), ["self", "item"])


class FallbackTests(unittest.TestCase):
    def test_refine_label(self):
        original = DiagnosisLabel.EVIDENCE_ACCESSIBILITY
        self.assertIs(refine_label(original, None), original)
        self.assertIs(refine_label(original, ProbeResult.unparsable("x", "bad")), original)
        uncertain = ProbeResult.from_judgments("unsupported", "uncertain", "uncertain")
        self.assertIs(refine_label(original, uncertain), original)
        supported = ProbeResult.from_judgments("supported", "sufficient", "clear", [1])
        self.assertIs(refine_label(original, supported), DiagnosisLabel.CORRECT)

    def test_pipeline_not_triggered_skips_probe(self):
        generator = RecordingGenerator(probe_json(answer_supported="supported"))
        outcome = apply_probe(nq_input(), diagnosis(0.20, DiagnosisLabel.CORRECT, None), EvidenceProbe(generator))
        self.assertFalse(outcome.probe_triggered)
        self.assertIsNone(outcome.probe_result)
        self.assertIs(outcome.final_label, DiagnosisLabel.CORRECT)
        self.assertEqual(generator.prompts, [])

    def test_pipeline_uses_probe_or_keeps_original(self):
        original = DiagnosisLabel.EVIDENCE_ACCESSIBILITY
        refined = apply_probe(nq_input(), diagnosis(0.45, original, UNSURE_STAGE2),
                              EvidenceProbe(RecordingGenerator(probe_json(answer_supported="supported"))))
        self.assertTrue(refined.probe_triggered)
        self.assertIs(refined.final_label, DiagnosisLabel.CORRECT)
        kept = apply_probe(nq_input(), diagnosis(0.45, original, UNSURE_STAGE2),
                           EvidenceProbe(RecordingGenerator("garbage")))
        self.assertIs(kept.probe_result.probe_label, ProbeLabel.UNCERTAIN)
        self.assertIs(kept.final_label, original)

    def test_probe_modules_do_not_load_models(self):
        code = (
            "import sys; import src.probe.probe, src.probe.pipeline, src.probe.trigger, src.probe.prompts; "
            "loaded = [m for m in ('torch', 'transformers', 'src.generation.qwen') if m in sys.modules]; "
            "sys.exit(1 if loaded else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
