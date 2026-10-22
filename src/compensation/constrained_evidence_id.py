"""Isolated v9 contract experiment; do not alter the retired smoke or defaults.

The existing semantic pipeline and strict postchecks are reused. A backend may
add grammar masks, but legal JSON/IDs never establish semantic correctness.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from src.compensation import evidence_id_repair as original
from src.compensation.answer_repair import Generation
from src.diagnosis import DatasetName

VERSION = "v9_contract_v1_xgrammar"
STRATEGIES = ("evidence_id", "factored_evidence_id")


def schema_hash(schema):
    return hashlib.sha256(json.dumps(schema, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def text():
    return {"type": "string"}


def ids(units, minimum=0):
    return {"type": "array", "items": {"type": "string", "enum": [u.id for u in units]},
            "minItems": minimum, "maxItems": 5}


def plan_schema():
    return obj({"questions": {"type": "array", "items": text(), "minItems": 1, "maxItems": original.MAX_QUESTIONS}})


def fact_schema(units):
    return {"anyOf": [obj({"status": {"const": status}, "fact": text(),
                           "evidence_ids": ids(units, 0 if status == "insufficient" else 1)})
                      for status in ("answerable", "insufficient", "conflict")]}


def decision_schema(request, units):
    variants = []
    for decision in ("keep", "repair", "insufficient", "conflict"):
        answer = ({"const": request.initial_answer} if decision == "keep" else
                  {"const": ""} if decision in ("insufficient", "conflict") else
                  {"enum": [v for v in ("SUPPORTS", "REFUTES") if v != request.initial_answer]}
                  if request.dataset == DatasetName.FEVER else text())
        variants.append(obj({"decision": {"const": decision}, "answer": answer,
                             "evidence_ids": ids(units, 1 if decision in ("keep", "repair") else 0),
                             "reason": text()}))
    return {"anyOf": variants}


def schema_prompt(prompt, schema):
    # Both ablation arms get this identical schema text. Only decoder masking
    # differs, so an instruction change is not confounded with grammar masks.
    return prompt + "\nOutput JSON Schema (structural constraints only): " + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))


def repair(request, backend, *, strategy="evidence_id", decision=None, all_samples=False):
    if strategy not in STRATEGIES:
        raise ValueError("Contract experiment supports the two structured arms only")
    units, constraints = original.evidence_units(request), []

    class Dispatch:
        position = 0
        question_count = 0

        def generate(self, prompt, max_new_tokens):
            if strategy == "factored_evidence_id" and self.position == 0:
                stage, schema = "plan", plan_schema()
            elif strategy == "factored_evidence_id" and self.position <= self.question_count:
                stage, schema = f"fact_{self.position}", fact_schema(units)
            else:
                stage, schema = "decision", decision_schema(request, units)
            final_prompt = schema_prompt(prompt, schema)
            generated, meta = backend.generate_schema(final_prompt, max_new_tokens, schema)
            constraints.append({"stage": stage, "schema_sha256": schema_hash(schema),
                                "schema_prompt_sha256": hashlib.sha256(final_prompt.encode()).hexdigest(), "backend": meta})
            self.position += 1
            if stage == "plan":
                # Let the original pipeline record the actual call/cost before
                # rejecting an invalid plan. Dispatch must not lose its trace.
                try:
                    value = original.strict_object(generated.text, ("questions",))
                    if isinstance(value["questions"], list):
                        self.question_count = len(value["questions"])
                except (ValueError, TypeError, KeyError):
                    pass
            return generated

    result = original.repair(request, Dispatch(), strategy=strategy, decision=decision, all_samples=all_samples)
    if result.repair_trace is not None:
        result.repair_trace.update(version=VERSION, semantic_pipeline_version=original.VERSION,
                                   constraints=constraints,
                                   schema_limitations=["Duplicate IDs/questions still checked after decoding",
                                                       "Empty facts/reasons, unchanged repair and NQ spans still checked",
                                                       "Valid structure does not establish semantic support"])
    return result


def replay(request, saved, *, strategy, decision=None, all_samples=False):
    trace = saved.get("repair_trace") or {}
    calls, constraints = trace.get("calls", []), trace.get("constraints", [])

    class Backend:
        position = 0

        def generate_schema(self, prompt, max_new_tokens, schema):
            if self.position >= len(calls) or self.position >= len(constraints):
                raise ValueError("Contract replay exhausted")
            call, constraint = calls[self.position], constraints[self.position]
            if (constraint["schema_prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest()
                    or call["max_new_tokens"] != max_new_tokens
                    or constraint["schema_sha256"] != schema_hash(schema)):
                raise ValueError("Contract replay prompt/schema/budget mismatch")
            self.position += 1
            return Generation(**call["generation"]), constraint["backend"]

    backend = Backend()
    rebuilt = repair(request, backend, strategy=strategy, decision=decision, all_samples=all_samples)
    if (backend.position != len(calls) or backend.position != len(constraints)
            or any(saved.get(k) != v for k, v in asdict(rebuilt).items())):
        raise ValueError("Contract replay differs")
    return rebuilt
