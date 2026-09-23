"""Backend-agnostic text generation for the larger-model control runs.

The frozen 7B path in `src.generation.qwen` loads weights locally with
bitsandbytes NF4 and is deliberately rigid. The larger-model control (32B / 70B /
72B) cannot work that way on a 6GB laptop, so this module keeps the *call*
abstract and lets the compute source be chosen at run time.

`OpenAICompatibleBackend` covers nearly every option with one implementation,
because Together, Fireworks, DeepInfra, OpenRouter, Groq, a self-hosted vLLM
server, llama.cpp's server, and Ollama all speak the same
`/chat/completions` protocol. Only the base URL, model id, and key change.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable


@runtime_checkable
class GenerationBackend(Protocol):
    """Anything that turns a prompt into (text, latency_seconds)."""

    name: str

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, float]:
        ...


class EchoBackend:
    """Dry-run backend. Runs the whole pipeline with no model and no compute.

    Use it to validate prompt assembly, subset selection, resume logic, and the
    output schema before spending anything on a real 70B run.
    """

    def __init__(self, reply: str = "ECHO") -> None:
        self.name = "echo"
        self.reply = reply

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, float]:
        start = time.perf_counter()
        return self.reply, time.perf_counter() - start


class OpenAICompatibleBackend:
    """Any OpenAI-compatible /chat/completions endpoint.

    Verified shape for: Together, Fireworks, DeepInfra, OpenRouter, Groq, vLLM,
    llama.cpp server, Ollama. `base_url` is the root (no trailing /chat/completions).

    Determinism caveat: the frozen protocol is greedy (`do_sample: false`). This
    sends temperature=0 / top_p=1, which is greedy in intent, but hosted
    inference is not bit-reproducible across calls -- batching and kernel
    nondeterminism can change a token. Record the provider and date alongside
    results; do not claim bit-exact reproduction of a hosted run.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 180.0,
        max_retries: int = 5,
        extra_body: dict | None = None,
    ) -> None:
        self.name = f"{model} @ {base_url}"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or ""
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = extra_body or {}

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, float]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            **self.extra_body,
        }
        start = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                data = self._post(payload)
                text = data["choices"][0]["message"]["content"]
                return (text or "").strip(), time.perf_counter() - start
            except urllib.error.HTTPError as exc:
                # 4xx other than rate-limiting will not fix themselves.
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                    raise RuntimeError(
                        f"{self.base_url} rejected the request "
                        f"(HTTP {exc.code}): {detail}"
                    ) from exc
                last_error = exc
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
            backoff = min(2.0**attempt, 30.0)
            print(
                f"  retry {attempt + 1}/{self.max_retries} after "
                f"{type(last_error).__name__}; sleeping {backoff:.0f}s"
            )
            time.sleep(backoff)
        raise RuntimeError(
            f"{self.base_url} failed after {self.max_retries} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        )


class LocalTransformersBackend:
    """Local weights on a GPU large enough to hold them (rented A100/A6000).

    Unlike the frozen 7B loader this does not force NF4 -- a 40GB+ card can run
    the 70B in 4-bit comfortably, and an 80GB card can go higher. Imports are
    deferred so the module stays importable on a machine with no GPU.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
        load_in_4bit: bool = True,
        max_memory: dict | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.name = model_name
        self.torch = torch
        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            max_memory=max_memory,
            dtype=torch.float16 if quant_config is None else None,
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, float]:
        torch = self.torch
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(chat, return_tensors="pt").to(self.model.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        latency = time.perf_counter() - start
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip(), latency


# Base URLs for providers that serve Llama 3.3 70B behind the OpenAI protocol.
# Model ids differ per provider -- check the provider's model list before use.
KNOWN_ENDPOINTS = {
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "local-vllm": "http://localhost:8000/v1",
    "local-llamacpp": "http://localhost:8080/v1",
    "ollama": "http://localhost:11434/v1",
}


def build_backend(
    kind: str,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> GenerationBackend:
    """Resolve a backend from CLI-friendly arguments."""
    if kind == "echo":
        return EchoBackend()
    if kind == "local":
        return LocalTransformersBackend(model_name=model)
    url = base_url or KNOWN_ENDPOINTS.get(kind)
    if not url:
        raise RuntimeError(
            f"Unknown backend {kind!r}. Use one of "
            f"{sorted(KNOWN_ENDPOINTS)}, 'local', 'echo', or pass --base-url."
        )
    return OpenAICompatibleBackend(base_url=url, model=model, api_key=api_key)
