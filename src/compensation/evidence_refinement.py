"""The `evidence_refinement` enhancement: rerank the evidence, then re-ask once.

This is the module the frozen Controller routes `evidence_accessibility` to
(`ControllerAction.EVIDENCE_REFINEMENT`). It reranks and prunes the retrieved
evidence with a cross-encoder (`src/compensation/rerank.py`), re-asks the
frozen Reference RAG generator with the refined context using the frozen C1
prompt, and accepts the new answer only if it passes a grounding check.

Harm safety is the first requirement, because the Controller is known to
misroute: on the frozen PRIMARY test only 1 of the 29 samples routed here is a
true accessibility failure and 15 already had a correct answer
(`docs/DIAGNOSIS_PROBE_CONTROLLER_HANDOFF.md`, section F). So the refiner

* does nothing at all unless the action really is `evidence_refinement`,
* keeps the original answer when reranking changed nothing,
* keeps the original answer when the new answer is empty, a refusal, or not
  grounded in the kept evidence,
* and reports what it did in `RefinementResult`, so gains and harm on
  already-correct samples can both be measured offline.

It never re-runs Diagnosis, the Probe or Safe Binary, never edits the frozen
prompt text, and never reads gold answers, gold evidence, correctness, EM/F1,
C0/C3 or larger-model outcomes.

Integration (the frozen contract in section G of the handoff):

    refiner = EvidenceRefiner.from_frozen_defaults()
    decision = run_controller(item, classifier, probe)
    if decision.action is ControllerAction.EVIDENCE_REFINEMENT:
        final_answer = refiner(item, decision)        # or evidence_refinement(item, decision, refiner)

Use `refiner.run(item, decision, probe_result)` when the caller wants the full
`RefinementResult` instead of just the answer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from src.compensation.rerank import CandidateProvider, CrossEncoderScorer, RelevanceScorer, build_candidate_pool, rerank_evidence
from src.compensation.schema import (
    DEFAULT_REFINEMENT_CONFIG,
    REFUSAL_ANSWERS,
    RefinementConfig,
    RefinementResult,
    RefinementStatus,
)
from src.controller.schema import ControllerAction, ControllerDecision
from src.diagnosis.schema import DatasetName, DiagnosisInput, EvidenceItem
from src.evaluation.fever import parse_fever_prediction
from src.evaluation.qa import normalize_answer
from src.probe.schema import ProbeResult

PromptBuilder = Callable[[DiagnosisInput, Sequence[EvidenceItem]], str]


def build_refinement_prompt(item: DiagnosisInput, evidence: Sequence[EvidenceItem]) -> str:
    """The frozen Reference RAG C1 prompt, rebuilt over the refined evidence."""
    from src.generation.qwen import build_fever_rag_prompt, build_rag_prompt

    passages = [{"text": passage.text} for passage in evidence]
    if item.dataset is DatasetName.FEVER:
        return build_fever_rag_prompt(item.query, passages)
    return build_rag_prompt(item.query, passages)


def is_refusal(answer: str) -> bool:
    """True for an empty answer or one of the known "declined to answer" strings."""
    return normalize_answer(answer) in REFUSAL_ANSWERS


def is_grounded(answer: str, evidence: Sequence[EvidenceItem]) -> bool:
    """True if the normalised answer appears as a contiguous token run in some kept passage."""
    tokens = normalize_answer(answer).split()
    if not tokens:
        return False
    span = len(tokens)
    for passage in evidence:
        passage_tokens = normalize_answer(passage.text).split()
        for start in range(len(passage_tokens) - span + 1):
            if passage_tokens[start : start + span] == tokens:
                return True
    return False


def rejection_reason(
    item: DiagnosisInput,
    answer: str,
    evidence: Sequence[EvidenceItem],
    config: RefinementConfig = DEFAULT_REFINEMENT_CONFIG,
) -> str | None:
    """Why a regenerated answer must not replace the original, or None if it may."""
    if is_refusal(answer):
        return "refusal_or_empty"
    if item.dataset is DatasetName.FEVER:
        if parse_fever_prediction(answer) == "UNKNOWN":
            return "unparsable_fever_label"
        return None
    if config.require_grounded and not is_grounded(answer, evidence):
        return "not_grounded_in_kept_evidence"
    return None


class EvidenceRefiner:
    """Rerank-based `evidence_refinement`. The scorer and generator are injected so tests load no model."""

    def __init__(
        self,
        scorer: RelevanceScorer,
        generate_fn: Callable[[str], str],
        config: RefinementConfig = DEFAULT_REFINEMENT_CONFIG,
        candidate_provider: CandidateProvider | None = None,
        prompt_builder: PromptBuilder = build_refinement_prompt,
    ):
        if not callable(scorer):
            raise TypeError("scorer must be callable")
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable")
        if not isinstance(config, RefinementConfig):
            raise TypeError(f"config must be a RefinementConfig, got {type(config).__name__}")
        if candidate_provider is not None and not callable(candidate_provider):
            raise TypeError("candidate_provider must be callable or None")
        if not callable(prompt_builder):
            raise TypeError("prompt_builder must be callable")
        self._scorer = scorer
        self._generate = generate_fn
        self.config = config
        self.candidate_provider = candidate_provider
        self._build_prompt = prompt_builder

    @classmethod
    def from_frozen_defaults(
        cls,
        config: RefinementConfig = DEFAULT_REFINEMENT_CONFIG,
        candidate_provider: CandidateProvider | None = None,
        max_new_tokens: int | None = None,
    ) -> EvidenceRefiner:
        """Load the cross-encoder and the frozen Reference RAG generator (Qwen2.5-7B, 4-bit NF4, greedy).

        max_new_tokens defaults to configs/base.yaml; nothing about the frozen
        generator or its prompt is changed here.
        """
        from src.generation.qwen import generate_greedy, load_qwen_nf4

        scorer = CrossEncoderScorer.from_pretrained(config.reranker_model)
        tokenizer, model = load_qwen_nf4()

        def generate(prompt: str) -> str:
            text, _latency = generate_greedy(tokenizer, model, prompt, max_new_tokens=max_new_tokens)
            return text

        return cls(scorer, generate, config=config, candidate_provider=candidate_provider)

    def generate(self, prompt: str) -> str:
        """One call to the injected generator.

        Offline evaluation uses it to re-ask the same backend on the evidence in
        its original order, so a "before" answer and an "after" answer come from
        the same numeric setup. Refinement itself never calls this directly.
        """
        answer = self._generate(prompt)
        if not isinstance(answer, str):
            raise TypeError(f"generate_fn must return a string, got {type(answer).__name__}")
        return answer.strip()

    def _untouched(self, item: DiagnosisInput, status: RefinementStatus, reject_reason: str | None = None) -> RefinementResult:
        evidence = tuple(item.evidence)
        return RefinementResult(
            status=status,
            initial_answer=item.initial_answer,
            final_answer=item.initial_answer,
            evidence=evidence,
            rerank_scores=(),
            original_passage_ids=tuple(passage.passage_id for passage in evidence),
            dropped_passage_ids=(),
            candidate_pool_size=len(evidence),
            regenerated=False,
            used_probe_hint=False,
            reject_reason=reject_reason,
            sample_id=item.sample_id,
        )

    def run(
        self,
        item: DiagnosisInput,
        decision: ControllerDecision,
        probe_result: ProbeResult | None = None,
    ) -> RefinementResult:
        """Refine one sample. Returns an untouched result for any action other than evidence_refinement."""
        if not isinstance(item, DiagnosisInput):
            raise TypeError(f"item must be a DiagnosisInput, got {type(item).__name__}")
        if not isinstance(decision, ControllerDecision):
            raise TypeError(f"decision must be a ControllerDecision, got {type(decision).__name__}")
        if probe_result is not None and not isinstance(probe_result, ProbeResult):
            raise TypeError(f"probe_result must be a ProbeResult or None, got {type(probe_result).__name__}")
        if decision.action is not ControllerAction.EVIDENCE_REFINEMENT:
            return self._untouched(item, RefinementStatus.NOT_ROUTED)

        candidates = build_candidate_pool(item, self.candidate_provider)
        if not candidates:
            return self._untouched(item, RefinementStatus.NO_CHANGE)

        outcome = rerank_evidence(item.query, candidates, self._scorer, self.config, probe_result)
        original_ids = tuple(passage.passage_id for passage in item.evidence)
        unchanged = tuple(passage.passage_id for passage in outcome.evidence) == original_ids
        if unchanged and not self.config.regenerate_when_unchanged:
            return RefinementResult(
                status=RefinementStatus.NO_CHANGE,
                initial_answer=item.initial_answer,
                final_answer=item.initial_answer,
                evidence=outcome.evidence,
                rerank_scores=outcome.scores,
                original_passage_ids=original_ids,
                dropped_passage_ids=outcome.dropped_passage_ids,
                candidate_pool_size=outcome.candidate_pool_size,
                regenerated=False,
                used_probe_hint=outcome.used_probe_hint,
                sample_id=item.sample_id,
            )

        answer = self._generate(self._build_prompt(item, outcome.evidence))
        if not isinstance(answer, str):
            raise TypeError(f"generate_fn must return a string, got {type(answer).__name__}")
        answer = answer.strip()
        reason = rejection_reason(item, answer, outcome.evidence, self.config)
        accepted = reason is None
        return RefinementResult(
            status=RefinementStatus.ANSWER_REPLACED if accepted else RefinementStatus.KEPT_ORIGINAL,
            initial_answer=item.initial_answer,
            final_answer=answer if accepted else item.initial_answer,
            evidence=outcome.evidence,
            rerank_scores=outcome.scores,
            original_passage_ids=original_ids,
            dropped_passage_ids=outcome.dropped_passage_ids,
            candidate_pool_size=outcome.candidate_pool_size,
            regenerated=True,
            used_probe_hint=outcome.used_probe_hint,
            reject_reason=reason,
            candidate_answer=answer,
            sample_id=item.sample_id,
        )

    def __call__(self, item: DiagnosisInput, decision: ControllerDecision) -> str:
        """The frozen integration signature: returns the final answer for the host RAG."""
        return self.run(item, decision).final_answer


def evidence_refinement(item: DiagnosisInput, decision: ControllerDecision, refiner: EvidenceRefiner) -> str:
    """Explicit form of `refiner(item, decision)` for callers that pass the refiner around."""
    if not isinstance(refiner, EvidenceRefiner):
        raise TypeError(f"refiner must be an EvidenceRefiner, got {type(refiner).__name__}")
    return refiner(item, decision)
