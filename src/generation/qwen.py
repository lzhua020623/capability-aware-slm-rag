"""Qwen2.5 generator with required 4-bit NF4 loading."""

from __future__ import annotations

import gc
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import load_generator_config

PROMPT_INSTRUCTION = (
    "Answer the question using only the provided context. "
    "Return only the shortest answer span. "
    "Do not provide explanations or full sentences."
)
FEVER_PROMPT_INSTRUCTION = (
    "Based only on the provided evidence, determine whether the claim is "
    "supported or refuted. Return only SUPPORTS or REFUTES."
)


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def free_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def cpu_offload_enabled() -> bool:
    """Return the machine-local offload choice without changing shared config."""
    value = os.environ.get("SLM_RAG_CPU_OFFLOAD", "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"", "0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        "SLM_RAG_CPU_OFFLOAD must be one of: 1/0, true/false, yes/no, on/off"
    )


def require_nf4_support() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit NF4 loading requires CUDA. This environment has no GPU. "
            "Refusing to fall back to another model or unquantized loading."
        )
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "4-bit NF4 loading requires bitsandbytes. Install bitsandbytes and retry. "
            "Refusing to fall back to another model."
        ) from exc


def _raise_cuda_oom(exc: BaseException, model_name: str, stage: str) -> None:
    print(
        f"CUDA OOM during {stage} for {model_name} "
        "(4-bit NF4, bnb_4bit_compute_dtype=float16, device_map=auto). "
        "Refusing to fall back to any other model."
    )
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(
            f"CUDA memory allocated={allocated:.1f} MiB, reserved={reserved:.1f} MiB"
        )
    raise RuntimeError(
        f"CUDA OOM during {stage} for {model_name}. "
        "Not switching model or changing quantization."
    ) from exc


def _configured_model_name(requested: str | None = None) -> str:
    generator = load_generator_config()
    configured = str(generator["model_name"])
    if requested is not None and requested != configured:
        raise RuntimeError(
            f"Requested generator {requested!r} does not match "
            f"configs/base.yaml ({configured!r}). "
            "Refusing to load a different model."
        )
    return configured


def load_qwen_nf4(model_name: str | None = None):
    generator = load_generator_config()
    model_name = _configured_model_name(model_name)
    quant_type = str(generator.get("quantization") or "nf4")
    load_in_4bit = bool(generator.get("load_in_4bit", True))
    cpu_offload = cpu_offload_enabled()
    if not load_in_4bit or quant_type.lower() != "nf4":
        raise RuntimeError(
            "This project requires 4-bit NF4 loading from configs/base.yaml. "
            "Refusing to fall back to another quantization."
        )
    require_nf4_support()
    free_cuda_memory()
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        # When enabled by the shared config, Accelerate may place modules that
        # do not fit in VRAM on the CPU in fp32. The flag name is inherited
        # from bitsandbytes but is also checked by the 4-bit integration.
        llm_int8_enable_fp32_cpu_offload=cpu_offload,
    )
    if cpu_offload:
        # Keep every quantized transformer block on CUDA. Offloading arbitrary
        # Linear4bit modules through an automatic map can leave their bnb
        # quantization state on the meta device. lm_head is kept in fp32 on CPU.
        device_map = {"model": 0, "lm_head": "cpu"}
    else:
        device_map = {"": 0}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=device_map,
            trust_remote_code=True,
        )
    except Exception as exc:
        if is_cuda_oom(exc):
            _raise_cuda_oom(exc, model_name, "model load")
        raise RuntimeError(
            f"Failed to load {model_name} with 4-bit NF4 quantization. "
            f"CPU offload configured: {cpu_offload}. "
            "Refusing to fall back to another model or a different quantization. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def build_rag_prompt(question: str, passages: list[dict]) -> str:
    context_blocks = []
    for i, passage in enumerate(passages, start=1):
        context_blocks.append(f"[{i}] {passage['text']}")
    context = "\n\n".join(context_blocks)
    return (
        f"{PROMPT_INSTRUCTION}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


def build_fever_rag_prompt(claim: str, passages: list[dict]) -> str:
    context_blocks = []
    for i, passage in enumerate(passages, start=1):
        context_blocks.append(f"[{i}] {passage['text']}")
    context = "\n\n".join(context_blocks)
    return (
        f"{FEVER_PROMPT_INSTRUCTION}\n\n"
        f"Evidence:\n{context}\n\n"
        f"Claim: {claim}"
    )


def apply_qwen_chat_template(tokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _input_device(model) -> torch.device:
    if hasattr(model, "device") and model.device is not None:
        device = model.device
        if isinstance(device, torch.device) and device.type != "meta":
            return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def generate_greedy(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int | None = None,
) -> tuple[str, float]:
    generator = load_generator_config()
    if max_new_tokens is None:
        max_new_tokens = int(generator.get("max_new_tokens", 64))
    chat_prompt = apply_qwen_chat_template(tokenizer, prompt)
    inputs = tokenizer(chat_prompt, return_tensors="pt")
    device = _input_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    start = time.perf_counter()
    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    except Exception as exc:
        if is_cuda_oom(exc):
            _raise_cuda_oom(exc, str(generator["model_name"]), "generation")
        raise
    latency = time.perf_counter() - start
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer, latency
