"""Optional FEVER-only independent MNLI classifier experiment.

This changes the model budget and is not an equal-Qwen-budget comparison.
Threshold is fixed before this experiment; no FEVER labels enter inference.
"""
from __future__ import annotations

import math
import time

from src.compensation.answer_repair import Citation, RepairResult
from src.diagnosis import DatasetName, DiagnosisInput

NLI_MODEL = "microsoft/deberta-large-mnli"
NLI_REVISION = "7296194b9009373def4f7c5dad292651e4b5cf4e"
NLI_VERSION = "v6_mnli_windows_threshold_090"
NLI_THRESHOLD = 0.90
NLI_OVERLAP = 64
NLI_BATCH_SIZE = 4
LABELS = {"CONTRADICTION", "NEUTRAL", "ENTAILMENT"}


def evidence_windows(request, tokenizer, max_length=512):
    """Cover every premise token without truncating the claim or hiding evidence."""
    claim_length = len(tokenizer.encode(request.query, add_special_tokens=False))
    capacity = max_length - claim_length - tokenizer.num_special_tokens_to_add(pair=True)
    if capacity <= NLI_OVERLAP:
        raise ValueError("Claim too long for safe overlapping NLI evidence windows")
    windows = []
    for item in request.evidence:
        encoded = tokenizer(item.text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        start = 0
        while start < len(offsets):
            end = min(len(offsets), start + capacity)
            # Retokenization at a character boundary can alter BPE length.
            # Shrink explicitly until this exact string pair fits, never truncate.
            while end > start:
                a, b = offsets[start][0], offsets[end - 1][1]
                quote = item.text[a:b]
                length = len(tokenizer(quote, request.query, truncation=False)["input_ids"])
                if length <= max_length:
                    break
                end -= 1
            if end <= start:
                raise ValueError("NLI evidence window cannot fit without truncation")
            windows.append({"rank": item.rank, "passage_id": item.passage_id,
                            "start": a, "end": b, "quote": quote, "input_tokens": length})
            if end == len(offsets):
                break
            start = max(start + 1, end - NLI_OVERLAP)
    return windows


class MNLIVerifier:
    """Lazy GPU backend, FP32 evaluation; classifier costs are logged separately."""
    def __init__(self):
        import importlib.metadata
        import platform
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("The frozen NLI experiment requires CUDA")
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL, revision=NLI_REVISION, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError("NLI windows require a tokenizer with character offsets")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL, revision=NLI_REVISION, torch_dtype=torch.float32).to("cuda").eval()
        self.labels = {int(k): str(v).upper() for k, v in self.model.config.id2label.items()}
        if set(self.labels.values()) != LABELS or len(self.labels) != 3:
            raise RuntimeError("Unexpected NLI label mapping")
        self.max_length = int(self.model.config.max_position_embeddings)
        self.runtime_environment = {
            "python": platform.python_version(), "model_name": NLI_MODEL,
            "model_revision": NLI_REVISION, "dtype": "float32", "batch_size": NLI_BATCH_SIZE,
            "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers")},
            "id2label": self.labels, "max_length": self.max_length,
        }

    def score(self, request: DiagnosisInput) -> dict:
        import torch
        windows = evidence_windows(request, self.tokenizer, self.max_length)
        calls = 0
        torch.cuda.synchronize()
        started = time.perf_counter()
        for start in range(0, len(windows), NLI_BATCH_SIZE):
            group = windows[start:start + NLI_BATCH_SIZE]
            inputs = self.tokenizer([w["quote"] for w in group], [request.query] * len(group),
                                    padding=True, truncation=False, return_tensors="pt").to("cuda")
            if inputs["input_ids"].shape[1] > self.max_length:
                raise RuntimeError("NLI batch exceeds context length")
            with torch.inference_mode():
                probs = torch.softmax(self.model(**inputs).logits.float(), dim=-1).cpu().tolist()
            calls += 1
            for w, scores in zip(group, probs):
                w["probabilities"] = {self.labels[i]: p for i, p in enumerate(scores)}
        torch.cuda.synchronize()
        return {"model_name": NLI_MODEL, "model_revision": NLI_REVISION,
                "threshold": NLI_THRESHOLD, "windows": windows, "calls": calls,
                "latency": time.perf_counter() - started,
                "input_tokens": sum(w["input_tokens"] for w in windows)}


def repair_from_nli_trace(request: DiagnosisInput, trace: dict) -> RepairResult:
    """Deterministic decision/replay; never equate a classifier score with truth."""
    if request.dataset != DatasetName.FEVER:
        raise ValueError("fever_nli supports FEVER only")
    if (trace["model_name"] != NLI_MODEL or trace["model_revision"] != NLI_REVISION
            or trace["threshold"] != NLI_THRESHOLD):
        raise ValueError("NLI model/threshold differs from the frozen protocol")
    if (type(trace["calls"]) is not int or trace["calls"] < 0
            or not math.isfinite(trace["latency"]) or trace["latency"] < 0
            or type(trace["input_tokens"]) is not int or trace["input_tokens"] < 0):
        raise ValueError("Invalid NLI costs")
    if (trace["calls"] != math.ceil(len(trace["windows"]) / NLI_BATCH_SIZE)
            or trace["input_tokens"] != sum(w["input_tokens"] for w in trace["windows"])):
        raise ValueError("NLI window and cost counts disagree")
    evidence = {e.rank: e for e in request.evidence}
    for w in trace["windows"]:
        item = evidence[w["rank"]]
        if (w["passage_id"] != item.passage_id or not w["quote"]
                or item.text[w["start"]:w["end"]] != w["quote"]):
            raise ValueError("NLI window does not match source evidence")
        probs = w["probabilities"]
        if (set(probs) != LABELS or any(type(p) not in (int, float) or not math.isfinite(p)
                                       or not 0 <= p <= 1 for p in probs.values())
                or not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-5)):
            raise ValueError("Invalid NLI probabilities")
    supported = [w for w in trace["windows"] if w["probabilities"]["ENTAILMENT"] >= NLI_THRESHOLD]
    refuted = [w for w in trace["windows"] if w["probabilities"]["CONTRADICTION"] >= NLI_THRESHOLD]
    common = dict(verifier_calls=trace["calls"], verifier_latency=trace["latency"],
                  verifier_input_tokens=trace["input_tokens"], verifier_trace=trace)
    if bool(supported) == bool(refuted):
        state = "conflict" if supported else "uncertain"
        return RepairResult(request.initial_answer, request.initial_answer, "nli_" + state,
                            independent_support_status="nli_" + state, **common)
    label, dimension, candidates = (("SUPPORTS", "ENTAILMENT", supported) if supported else
                                     ("REFUTES", "CONTRADICTION", refuted))
    best = max(candidates, key=lambda w: w["probabilities"][dimension])
    kept = label == request.initial_answer.strip()
    return RepairResult(request.initial_answer, request.initial_answer if kept else label,
                        "kept" if kept else "repaired", verdict="keep" if kept else "repair",
                        citations=[Citation(best["rank"], best["passage_id"], best["quote"])],
                        independent_support_status="nli_" + dimension.lower(), **common)


def repair_fever(request: DiagnosisInput, verifier) -> RepairResult:
    if request.dataset != DatasetName.FEVER:
        raise ValueError("fever_nli supports FEVER only")
    return repair_from_nli_trace(request, verifier.score(request))
