"""Tests for the pluggable generation backends. No network call is made."""

from __future__ import annotations

import json
import unittest
import urllib.error

from src.generation.backend import (
    KNOWN_ENDPOINTS,
    EchoBackend,
    GenerationBackend,
    OpenAICompatibleBackend,
    build_backend,
)


class RecordingBackend(OpenAICompatibleBackend):
    """Captures the payload instead of posting it."""

    def __init__(self, *args, reply: str = "Paris", **kwargs):
        super().__init__(*args, **kwargs)
        self.payloads: list[dict] = []
        self.reply = reply

    def _post(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {"choices": [{"message": {"content": self.reply}}]}


class EchoBackendTest(unittest.TestCase):
    def test_returns_its_reply_and_a_latency(self):
        backend = EchoBackend("ECHO")
        text, latency = backend.generate("any prompt", 64)
        self.assertEqual(text, "ECHO")
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(backend.name, "echo")

    def test_satisfies_the_backend_protocol(self):
        self.assertIsInstance(EchoBackend(), GenerationBackend)


class OpenAICompatibleBackendTest(unittest.TestCase):
    def test_sends_a_greedy_single_turn_request(self):
        backend = RecordingBackend("https://example.test/v1", "qwen/qwen-2.5-7b-instruct", api_key="k")
        text, _latency = backend.generate("Answer the question.", 64)
        self.assertEqual(text, "Paris")
        payload = backend.payloads[0]
        self.assertEqual(payload["model"], "qwen/qwen-2.5-7b-instruct")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Answer the question."}])
        self.assertEqual(payload["max_tokens"], 64)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_p"], 1.0)

    def test_strips_whitespace_and_tolerates_a_null_reply(self):
        backend = RecordingBackend("https://example.test/v1", "m", reply="  Paris \n")
        self.assertEqual(backend.generate("p", 8)[0], "Paris")
        backend.reply = None
        self.assertEqual(backend.generate("p", 8)[0], "")

    def test_trailing_slash_in_the_base_url_is_normalised(self):
        backend = OpenAICompatibleBackend("https://example.test/v1/", "m")
        self.assertEqual(backend.base_url, "https://example.test/v1")

    def test_api_key_falls_back_to_the_environment(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"LLM_API_KEY": "from-env"}, clear=False):
            self.assertEqual(OpenAICompatibleBackend("https://example.test/v1", "m").api_key, "from-env")
        self.assertEqual(OpenAICompatibleBackend("https://example.test/v1", "m", api_key="explicit").api_key, "explicit")

    def test_client_errors_are_not_retried(self):
        class Rejecting(OpenAICompatibleBackend):
            def _post(self, payload: dict) -> dict:
                raise urllib.error.HTTPError(
                    "https://example.test/v1/chat/completions", 401, "Unauthorized", {}, None
                )

        backend = Rejecting("https://example.test/v1", "m", max_retries=5)
        with self.assertRaises(RuntimeError) as caught:
            backend.generate("p", 8)
        self.assertIn("401", str(caught.exception))

    def test_extra_body_is_merged(self):
        backend = RecordingBackend("https://example.test/v1", "m", extra_body={"provider": {"order": ["x"]}})
        backend.generate("p", 8)
        self.assertEqual(backend.payloads[0]["provider"], {"order": ["x"]})

    def test_a_malformed_reply_is_reported(self):
        class Garbled(OpenAICompatibleBackend):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.calls = 0

            def _post(self, payload: dict) -> dict:
                self.calls += 1
                raise json.JSONDecodeError("bad", "", 0)

        backend = Garbled("https://example.test/v1", "m", max_retries=1)
        with self.assertRaises(RuntimeError):
            backend.generate("p", 8)
        self.assertEqual(backend.calls, 1)


class BuildBackendTest(unittest.TestCase):
    def test_echo(self):
        self.assertIsInstance(build_backend("echo", "ignored"), EchoBackend)

    def test_known_providers_resolve_to_their_base_url(self):
        for kind, url in KNOWN_ENDPOINTS.items():
            backend = build_backend(kind, "some-model", api_key="k")
            self.assertIsInstance(backend, OpenAICompatibleBackend)
            self.assertEqual(backend.base_url, url.rstrip("/"))

    def test_an_explicit_base_url_wins(self):
        backend = build_backend("openrouter", "m", base_url="http://localhost:9999/v1", api_key="k")
        self.assertEqual(backend.base_url, "http://localhost:9999/v1")

    def test_an_unknown_provider_without_a_url_is_rejected(self):
        with self.assertRaises(RuntimeError):
            build_backend("not-a-provider", "m")


if __name__ == "__main__":
    unittest.main()
