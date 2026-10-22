from dataclasses import asdict, replace
import json
import unittest

from src.compensation.answer_repair import Generation
from src.compensation.constrained_evidence_id import repair, replay, decision_schema, plan_schema, fact_schema
from src.compensation.evidence_id_repair import evidence_units
from src.diagnosis import DiagnosisInput
from scripts.prepare_answer_repair_contract import fixtures, request_from_case


def request(n=0):
    return request_from_case(fixtures()[n])


class Backend:
    def __init__(self, outputs):
        self.outputs, self.calls = iter(outputs), []

    def generate_schema(self, prompt, limit, schema):
        self.calls.append((prompt, limit, schema))
        value = next(self.outputs)
        return Generation(value if isinstance(value, str) else json.dumps(value), .25, 100, 25), {
            "decoder": "test_stub_not_real_inference", "schema_compile_seconds": 0.0}


class ContractTests(unittest.TestCase):
    def test_conditional_keep_and_legal_id_schema_preserve_all_original_postchecks(self):
        r = request()
        schema = decision_schema(r, evidence_units(r))
        self.assertEqual(schema["anyOf"][0]["properties"]["answer"], {"const": r.initial_answer})
        self.assertEqual(schema["anyOf"][0]["properties"]["evidence_ids"]["items"]["enum"],
                         [u.id for u in evidence_units(r)])
        for output in [
            {"decision": "keep", "answer": "rewritten", "evidence_ids": ["R1S1"], "reason": "x"},
            {"decision": "repair", "answer": "not in source", "evidence_ids": ["R1S1"], "reason": "x"},
            {"decision": "repair", "answer": "Dr. Mira Vale", "evidence_ids": ["R1S1", "R1S1"], "reason": "x"}]:
            result = repair(r, Backend([output]), all_samples=True)
            self.assertEqual(result.status, "invalid_decision")
            self.assertEqual(result.final_answer, r.initial_answer)

    def test_replay_and_blind_stage_isolation(self):
        r = request(1)
        backend = Backend([{"questions": ["Who plays Captain Rowan in Glass Harbor?"]},
            {"status": "answerable", "fact": "Ilan Reed", "evidence_ids": ["R1S1"]},
            {"decision": "repair", "answer": "Ilan Reed", "evidence_ids": ["R1S1"], "reason": "The actor is Ilan Reed."}])
        result = repair(r, backend, strategy="factored_evidence_id", all_samples=True)
        self.assertEqual(result.final_answer, "Ilan Reed")
        self.assertEqual(len(result.repair_trace["constraints"]), 3)
        # C1 text may overlap the question; use an explicit sentinel to check isolation.
        r2 = replace(request(1), initial_answer="C1_SENTINEL_NOT_IN_BLIND_CALLS")
        b2 = Backend([{"questions": ["Who plays Captain Rowan?"]},
            {"status": "answerable", "fact": "Ilan Reed", "evidence_ids": ["R1S1"]},
            {"decision": "repair", "answer": "Ilan Reed", "evidence_ids": ["R1S1"], "reason": "Actor."}])
        repair(r2, b2, strategy="factored_evidence_id", all_samples=True)
        self.assertTrue(all(r2.initial_answer not in call[0] for call in b2.calls[:2]))
        self.assertEqual(asdict(replay(r, asdict(result), strategy="factored_evidence_id", all_samples=True)), asdict(result))
        saved = asdict(result); saved["repair_trace"]["constraints"][0]["schema_sha256"] = "changed"
        with self.assertRaises(ValueError):
            replay(r, saved, strategy="factored_evidence_id", all_samples=True)

    def test_invalid_plan_retains_actual_call_and_cost(self):
        result = repair(request(), Backend(["not JSON"]), strategy="factored_evidence_id", all_samples=True)
        self.assertEqual(result.status, "invalid_plan")
        self.assertEqual(result.extra_calls, 1)
        self.assertEqual(len(result.repair_trace["constraints"]), 1)
        self.assertEqual(result.latency, .25)

    def test_fever_schema_and_synthetic_scope(self):
        rows = fixtures()
        self.assertEqual(len(rows), 12)
        self.assertEqual({r["request"]["dataset"] for r in rows}, {"fever", "nq"})
        self.assertTrue(all(r["synthetic"] and len(r["request"]["evidence"]) == 5 for r in rows))
        r = request(7)
        self.assertEqual(decision_schema(r, evidence_units(r))["anyOf"][1]["properties"]["answer"], {"enum": ["REFUTES"]})


if __name__ == "__main__":
    unittest.main()
