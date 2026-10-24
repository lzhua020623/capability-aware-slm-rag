"""Execute the frozen router and compensation modules with observable receipts.

This is an integration layer, not a new diagnosis or repair policy. Providers
are injected and shared by the runner. Gold and evaluation labels never enter
this module. A valid citation/label is not a claim of semantic correctness.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from time import perf_counter

from src.compensation.answer_repair import Generation, NQ_ONLY_STRATEGIES
from src.compensation.controller_adapter import generation_repair
from src.compensation.evidence_refinement import EvidenceRefiner
from src.compensation.retrieval_enhancement.p3 import P320RetrievalEnhancer
from src.controller.controller import ACTION_BY_LABEL, decide_action
from src.controller.schema import ControllerAction, ControllerDecision
from src.diagnosis.schema import DiagnosisInput
from src.probe.probe import EvidenceProbe, PROBE_MAX_NEW_TOKENS
from src.probe.schema import ProbeResult
from src.probe.trigger import V2_EXPANDED_TRIGGER

SCHEMA_VERSION = "controller_integration_v1"
MODES = ("adaptive", "c1", "always_retrieve", "always_rerank", "always_repair")
MODE_ACTION = dict(zip(MODES[1:], ControllerAction))
# No implicit loading of an NLI verifier. These are existing Qwen strategies.
QWEN_STRATEGIES = frozenset({
    "verify_repair", "grounded_reanswer", "query_grounded", "guarded_reanswer",
    "extractive_reanswer", "short_answer_quote", "conservative_repair",
    "conservative_no_original_gate", "conservative_no_candidate_gate",
})


def fingerprint(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ExecutionConfig:
    # These defaults name an integration comparator, not a promoted E policy.
    nq_repair_strategy: str = "short_answer_quote"
    fever_repair_strategy: str = "verify_repair"
    max_llm_calls: int = 4  # at most one Probe plus up to three E calls
    max_retrieval_calls: int = 1
    max_scoring_calls: int = 1
    answer_max_new_tokens: int = 64
    repair_max_new_tokens: int = 384

    def __post_init__(self):
        for name in ("nq_repair_strategy", "fever_repair_strategy"):
            if getattr(self, name) not in QWEN_STRATEGIES:
                raise ValueError(f"Unsupported Qwen strategy: {getattr(self, name)}")
        if self.fever_repair_strategy in NQ_ONLY_STRATEGIES:
            raise ValueError("FEVER cannot use an NQ-only repair strategy")
        for name in ("max_llm_calls", "max_retrieval_calls", "max_scoring_calls",
                     "answer_max_new_tokens", "repair_max_new_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name.endswith("tokens") else 0):
                raise ValueError(f"Invalid {name}")


class BudgetExceeded(RuntimeError):
    pass


class CallLog:
    """Per-query counters count attempted provider calls, including exceptions."""

    def __init__(self, config: ExecutionConfig):
        self.limits = {"llm": config.max_llm_calls, "retrieval": config.max_retrieval_calls,
                       "scoring": config.max_scoring_calls}
        self.events = []

    def invoke(self, kind, stage, arguments, fn):
        count = sum(e["kind"] == kind and e["attempted"] for e in self.events)
        event = {"kind": kind, "stage": stage, "arguments": arguments,
                 "attempted": False, "status": "budget_exceeded", "seconds": 0.0}
        self.events.append(event)
        if count >= self.limits[kind]:
            raise BudgetExceeded(f"{kind} call budget exhausted")
        event.update(attempted=True, status="started")
        started = perf_counter()
        try:
            value = fn()
            if kind == "llm" and not isinstance(value, Generation):
                raise TypeError("The generator must return Generation, including token accounting")
            event["output"] = asdict(value) if isinstance(value, Generation) else value
            event["status"] = "completed"
            return value
        except Exception as exc:
            event.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            event["seconds"] = perf_counter() - started

    def summary(self):
        llm = [e for e in self.events if e["kind"] == "llm" and e["attempted"]]
        def token_total(field):
            values = [e.get("output", {}).get(field) for e in llm]
            return sum(values) if all(v is not None for v in values) else None
        return {
            **{f"{kind}_calls": sum(e["kind"] == kind and e["attempted"] for e in self.events)
               for kind in self.limits},
            "input_tokens": token_total("input_tokens"),
            "output_tokens_estimate": token_total("output_tokens_estimate"),
            "provider_seconds": sum(e["seconds"] for e in self.events),
            "failed_calls": sum(e["status"] == "failed" for e in self.events),
            "budget_blocks": sum(e["status"] == "budget_exceeded" for e in self.events),
            "scope": "incremental_after_C1; excludes cached C1 and resource loading",
        }


class MeteredGenerator:
    def __init__(self, provider, calls, stage):
        self.provider, self.calls, self.stage = provider, calls, stage

    def _call(self, prompt, max_new_tokens, labels=None, insufficient=False):
        arguments = {"prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                     "max_new_tokens": max_new_tokens}
        if labels is not None:
            arguments.update(labels=list(labels), insufficient=insufficient)
        def generate():
            if labels is not None and hasattr(self.provider, "generate_labeled"):
                return self.provider.generate_labeled(prompt, max_new_tokens,
                                                      labels=labels, insufficient=insufficient)
            return self.provider.generate(prompt, max_new_tokens)
        return self.calls.invoke("llm", self.stage, arguments, generate)

    def generate(self, prompt, max_new_tokens):
        return self._call(prompt, max_new_tokens)

    def generate_labeled(self, prompt, max_new_tokens, *, labels, insufficient=False):
        return self._call(prompt, max_new_tokens, labels, insufficient)


class MeteredScorer:
    def __init__(self, provider, calls):
        self.provider, self.calls = provider, calls
        self.model_name = getattr(provider, "model_name", None)
        self.revision = getattr(provider, "revision", None)

    def __call__(self, query, texts):
        texts = list(texts)
        def score():
            values = [float(x) for x in self.provider(query, texts)]
            if len(values) != len(texts) or not all(math.isfinite(x) for x in values):
                raise ValueError("Scorer returned non-finite or unaligned scores")
            return values
        return self.calls.invoke("scoring", "reranker", {"query": query, "texts": texts},
                                 score)


class MeteredRetriever:
    def __init__(self, provider, calls):
        self.provider, self.calls = provider, calls

    def retrieve(self, query, *, dataset, top_k):
        if self.provider is None:
            raise RuntimeError("No matching index provider configured")
        def retrieve():
            value = self.provider.retrieve(query, dataset=dataset, top_k=top_k)
            # Fail inside the attempted call rather than leave an unserialisable
            # trace that prevents the runner from saving the fallback record.
            json.dumps(value, allow_nan=False)
            return value
        return self.calls.invoke("retrieval", "additional_retrieval",
                                 {"query": query, "dataset": dataset, "top_k": top_k},
                                 retrieve)


def validate_input(item):
    if not isinstance(item, DiagnosisInput):
        raise TypeError("Expected runtime-only DiagnosisInput")
    if not item.query.strip() or not item.initial_answer.strip():
        raise ValueError("Empty query or C1 prediction")
    if [e.rank for e in item.evidence] != [1, 2, 3, 4, 5]:
        raise ValueError("Integration requires the original ordered Top-5 evidence")
    if len({e.passage_id for e in item.evidence}) != 5:
        raise ValueError("Duplicate Top-5 passage IDs")
    if any(not e.text.strip() or not math.isfinite(e.score) for e in item.evidence):
        raise ValueError("Empty evidence or non-finite retrieval scores")


class IntegratedController:
    """Frozen Diagnosis/Safe Binary, one chosen action, no retry or optimisation.

    raw_generator supplies the Probe and C1-style regeneration. repair_generators
    maps nq/fever to adapters over the SAME loaded Qwen with strategy-specific
    stopping. scorer and retriever are externally loaded and shared as well.
    """

    def __init__(self, classifier, raw_generator, repair_generators, scorer, retriever=None,
                 *, config=ExecutionConfig(), trigger=V2_EXPANDED_TRIGGER):
        self.classifier, self.raw_generator = classifier, raw_generator
        self.repair_generators, self.scorer, self.retriever = repair_generators, scorer, retriever
        self.config, self.trigger = config, trigger

    def run(self, item: DiagnosisInput, *, mode="adaptive") -> dict:
        validate_input(item)
        if mode not in MODES:
            raise ValueError(f"Unknown execution mode {mode}")
        started = perf_counter()
        calls = CallLog(self.config)
        diagnosis = probe_result = decision = None
        probe_error = None
        diagnosis_seconds = 0.0
        if mode == "adaptive":
            before = perf_counter()
            # A broken classifier is an infrastructure error, not a 'correct' label.
            diagnosis = self.classifier.diagnose(item)
            diagnosis_seconds = perf_counter() - before
            triggered = self.trigger.should_probe(diagnosis.stage1_failure_probability,
                                                  diagnosis.stage2_class_scores)
            if triggered:
                generator = MeteredGenerator(self.raw_generator, calls, "probe")
                try:
                    probe_result = EvidenceProbe(
                        lambda prompt: generator.generate(prompt, PROBE_MAX_NEW_TOKENS).text
                    ).run(item)
                except Exception as exc:
                    probe_error = {"type": type(exc).__name__, "message": str(exc)}
                    probe_result = ProbeResult.unparsable("", str(exc))
            decision = decide_action(diagnosis, triggered, probe_result, item.sample_id)
        else:
            # These are explicit action baselines, never predicted/oracle labels.
            action = MODE_ACTION[mode]
            label = next(label for label, value in ACTION_BY_LABEL.items() if value == action)
            decision = ControllerDecision(action, label, 1.0, False, False, item.sample_id)
        action_result = self.execute(item, decision, calls=calls, probe_result=probe_result)
        return {
            "schema_version": SCHEMA_VERSION, "sample_id": item.sample_id,
            "dataset": item.dataset.value, "mode": mode,
            "input": asdict(item), "input_sha256": fingerprint(asdict(item)),
            "config": asdict(self.config),
            "routing": {"source": "diagnosis" if mode == "adaptive" else "fixed_action_baseline",
                        "diagnosis": asdict(diagnosis) if diagnosis is not None else None,
                        "diagnosis_seconds": diagnosis_seconds,
                        "probe": asdict(probe_result) if probe_result is not None else None,
                        "probe_error": probe_error, "decision": decision.to_dict()},
            **action_result, "cost": calls.summary(), "calls": calls.events,
            "total_seconds": perf_counter() - started,
        }

    def execute(self, item, decision, *, calls=None, probe_result=None):
        """Execute one already-resolved decision; useful for isolated module tests."""
        validate_input(item)
        if not isinstance(decision, ControllerDecision):
            raise TypeError("Expected ControllerDecision")
        if decision.sample_id is not None and decision.sample_id != item.sample_id:
            raise ValueError("Decision/input sample IDs differ")
        if ACTION_BY_LABEL[decision.diagnosis_label] != decision.action:
            raise ValueError("Decision action and resolved label disagree")
        calls = calls if calls is not None else CallLog(self.config)
        initial = item.initial_answer
        result = {"initial_answer": initial, "final_answer": initial, "answer_changed": False,
                  "status": "no_action", "executed": False, "module_trace": None,
                  "error": None, "final_evidence": [asdict(e) for e in item.evidence]}
        action = decision.action
        if action == ControllerAction.NO_ACTION:
            return result
        fever_enabled = 'fever' in getattr(self.retriever, 'supported_datasets', ())
        if (action == ControllerAction.RETRIEVAL_ENHANCEMENT and item.dataset.value == 'fever'
                and not fever_enabled):
            result.update(status="unsupported_dataset", error={"type": "UnsupportedDataset",
                          "message": "No verified FEVER retrieval provider; original answer retained"})
            return result
        result["executed"] = True
        try:
            scorer = MeteredScorer(self.scorer, calls)
            generator = MeteredGenerator(self.raw_generator, calls, action.value)
            regenerate = lambda prompt: generator.generate(prompt, self.config.answer_max_new_tokens).text
            if action == ControllerAction.RETRIEVAL_ENHANCEMENT:
                native = P320RetrievalEnhancer(MeteredRetriever(self.retriever, calls), scorer,
                                               regenerate, enable_fever=fever_enabled).run(item, decision)
                selected = native["evidence"]
                final, status = native["final_answer"], native["status"]
                result["error"] = native.get("error")
            elif action == ControllerAction.EVIDENCE_REFINEMENT:
                native_result = EvidenceRefiner(scorer, regenerate).run(item, decision, probe_result)
                native = native_result.to_dict()
                selected = [asdict(e) for e in native_result.evidence]
                native["evidence"] = selected
                final, status = native_result.final_answer, native_result.status.value
            else:
                strategy = (self.config.nq_repair_strategy if item.dataset.value == "nq"
                            else self.config.fever_repair_strategy)
                repair_generator = MeteredGenerator(self.repair_generators[item.dataset.value], calls,
                                                     "generation_repair")
                native_result = generation_repair(item, decision, repair_generator, strategy=strategy,
                                                 max_new_tokens=self.config.repair_max_new_tokens)
                native = {**asdict(native_result), "strategy": strategy}
                selected = [asdict(e) for e in item.evidence]
                final, status = native_result.final_answer, native_result.status
            result.update(final_answer=final, answer_changed=final != initial,
                          status=status, module_trace=native)
            if final != initial:
                # A retained C1 answer keeps its original context. The attempted
                # enhanced context is still retained in module_trace/call receipts.
                result["final_evidence"] = selected
        except Exception as exc:
            result.update(status="execution_failed", error={"type": type(exc).__name__, "message": str(exc)})
        if any(e["status"] == "budget_exceeded" for e in calls.events):
            # P3 catches provider errors internally; expose the same budget status.
            result["status"] = "budget_exceeded"
        return result
