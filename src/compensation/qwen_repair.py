"""Lazy adapter to the existing frozen 7B NF4 generator. No hosted-model path."""

from __future__ import annotations

import importlib.metadata
import platform
import json

from src.compensation.answer_repair import Generation, MODEL_NAME, MODEL_REVISION


def complete_json_object(text: str) -> bool:
    """Generation stop signal only; strict schema/quote validation follows."""
    text = text.lstrip()
    if not text.startswith("{"):
        return False
    try:
        value, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict)


class QwenRepairGenerator:
    def __init__(self, *, json_output: bool = True) -> None:
        # Import only for a real run; preflight and tests need neither CUDA nor torch.
        import torch
        self.json_output = json_output
        if not torch.cuda.is_available():
            raise RuntimeError("E requires CUDA for the frozen 7B NF4 experiment; run the environment checker first")
        from src.config import load_base_config, load_generator_config
        from src.generation.qwen import load_qwen_nf4

        config = load_generator_config()
        if config["model_name"] != MODEL_NAME or config.get("do_sample") is not False:
            raise RuntimeError("E requires the shared Qwen2.5-7B-Instruct greedy generator")
        seed = load_base_config()["seed"]
        if seed != 42:
            raise RuntimeError("E requires the shared seed 42")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.tokenizer, self.model = load_qwen_nf4(MODEL_NAME, revision=MODEL_REVISION)
        import bitsandbytes as bnb
        quant = getattr(self.model.config, "quantization_config", {})
        if hasattr(quant, "to_dict"):
            quant = quant.to_dict()
        fourbit = [module for module in self.model.modules() if isinstance(module, bnb.nn.Linear4bit)]
        quant_types = sorted({str(getattr(module.weight, "quant_type", None)) for module in fourbit})
        if (getattr(self.model.config, "model_type", None) != "qwen2"
                or not getattr(self.model, "is_loaded_in_4bit", False)
                or not fourbit or quant_types != ["nf4"]
                or quant.get("bnb_4bit_compute_dtype") != "float16"
                or quant.get("bnb_4bit_use_double_quant") is not True):
            raise RuntimeError("Loaded model does not satisfy the Qwen NF4/FP16/double-quant protocol")
        packages = {}
        for name in ("torch", "transformers", "accelerate", "bitsandbytes", "PyYAML"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        self.runtime_environment = {
            "python": platform.python_version(), "packages": packages,
            "cuda_version": torch.version.cuda,
            "gpus": [{"name": torch.cuda.get_device_name(i),
                      "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                     for i in range(torch.cuda.device_count())],
            "model_revision": MODEL_REVISION,
            "json_output": self.json_output,
            "tokenizer_revision": MODEL_REVISION,
            "model_class": type(self.model).__module__ + "." + type(self.model).__name__,
            "quantization_config": quant,
            "fourbit_module_count": len(fourbit), "fourbit_quant_types": quant_types,
            "parameter_devices": sorted({str(parameter.device) for parameter in self.model.parameters()}),
            "model_device_map": {str(k): str(v) for k, v in getattr(self.model, "hf_device_map", {}).items()},
        }

    def generate(self, prompt: str, max_new_tokens: int) -> Generation:
        from src.generation.qwen import apply_qwen_chat_template, generate_greedy

        chat = apply_qwen_chat_template(self.tokenizer, prompt)
        input_tokens = len(self.tokenizer.encode(chat))
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        if context_limit is not None and input_tokens + max_new_tokens > context_limit:
            raise RuntimeError("E prompt exceeds the model context window; refusing to silently truncate evidence")
        stopping = None
        if self.json_output:
            from transformers import StoppingCriteria, StoppingCriteriaList
            tokenizer = self.tokenizer
            boundary = {"observed": None, "framed": None}

            class CompleteObject(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    continuation = tokenizer.decode(input_ids[0, input_tokens:], skip_special_tokens=True)
                    if not complete_json_object(continuation):
                        return False
                    observed = continuation.strip()
                    _, end = json.JSONDecoder().raw_decode(observed)
                    last_token = tokenizer.decode(input_ids[0, -1:], skip_special_tokens=True)
                    # A tokenizer may merge the closing brace and a formatting
                    # mark. Frame only the complete object observed at this stop,
                    # with any excess confined to the same final token.
                    if len(observed) - end <= len(last_token):
                        boundary.update(observed=observed, framed=observed[:end])
                    return True

            stopping = StoppingCriteriaList([CompleteObject()])
        text, latency = generate_greedy(self.tokenizer, self.model, prompt, max_new_tokens,
                                       stopping_criteria=stopping)
        # Retokenized decoded text excludes EOS; keep its estimate label explicit.
        output_estimate = len(self.tokenizer.encode(text, add_special_tokens=False))
        raw = text
        if self.json_output and boundary["observed"] == text.strip():
            text = boundary["framed"]
        return Generation(text, latency, input_tokens, output_estimate, raw_text=raw)
