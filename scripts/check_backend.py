"""Verify an inference endpoint before spending a real run on it.

Makes one tiny call and reports exactly which part is wrong -- base URL, key, or
model id -- because those three fail in ways that look alike when a long job
dies halfway through.

    python scripts/check_backend.py --backend deepinfra --model Qwen/Qwen2.5-32B-Instruct

Reads the key from --api-key, else $LLM_API_KEY.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.backend import KNOWN_ENDPOINTS, build_backend

# The judge prompt shape, so a pass here means the real judge call will parse.
PROBE = (
    "Question: what is the capital of France\n"
    "Reference answer: Paris\n"
    "Predicted answer: paris, france\n\n"
    "Reply with exactly one word: CORRECT or INCORRECT."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True, help=f"one of {sorted(KNOWN_ENDPOINTS)}, or 'local'/'echo'")
    parser.add_argument("--model", required=True, help="exact model id as the provider lists it")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    args = parser.parse_args()

    url = args.base_url or KNOWN_ENDPOINTS.get(args.backend, "(n/a)")
    print(f"backend  : {args.backend}")
    print(f"base URL : {url}")
    print(f"model    : {args.model}")

    try:
        backend = build_backend(args.backend, args.model, args.base_url, args.api_key)
    except RuntimeError as exc:
        print(f"\nFAIL: {exc}")
        return 1

    key = getattr(backend, "api_key", None)
    if key is not None:
        print(f"api key  : {'set, ' + str(len(key)) + ' chars' if key else 'MISSING'}")
        if not key:
            print("\nFAIL: no API key. Pass --api-key or set $env:LLM_API_KEY")
            return 1

    print("\nsending one probe request...")
    try:
        text, latency = backend.generate(PROBE, 8)
    except RuntimeError as exc:
        message = str(exc)
        print(f"\nFAIL: {message}\n")
        if "401" in message or "403" in message:
            print("  -> the key was rejected. Check it is for this provider and still active.")
        elif "404" in message:
            print("  -> endpoint or model id not found. Check the model id against the")
            print("     provider's model list -- ids differ between providers for the same")
            print("     weights (e.g. a '-Turbo' or vendor prefix).")
        elif "402" in message or "billing" in message.lower() or "credit" in message.lower():
            print("  -> the account has no credit. Add funds or claim signup credit.")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL: {type(exc).__name__}: {exc}")
        return 1

    print(f"reply    : {text!r}")
    print(f"latency  : {latency:.2f}s")
    verdict = "CORRECT" in text.upper() or "INCORRECT" in text.upper()
    print(f"\nPASS: endpoint, key, and model id all work.")
    if not verdict:
        print("NOTE: the reply did not contain CORRECT/INCORRECT. The model answered but")
        print("      not in the expected form -- check the model id is an *Instruct* variant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
