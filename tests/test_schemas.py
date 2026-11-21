"""Machine-readable schemas of findings.json and of an action line (docs/tasks/B_qa_fixes.md item 5, QA L9).

The project has no dependencies, so a small validator for the subset of JSON Schema 2020-12 the two files
use lives here, with the semantics of 2020-12 (1.0 is an integer there). The schema must agree with
auditenv.submission.check_submission on the 15 broken structures and the 4 edge cases of V's rehearsal
(qa/rehearsal/probe_policy.py), except the two rules no schema can say: episode_id must be the episode's own
id, and an integer must be written without a fraction (the env takes only a JSON int, E-28).
"""
import json
import re
import unittest
from pathlib import Path

from auditenv.submission import InvalidSubmission, check_submission

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = {name: json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
           for name in ("findings.schema.json", "action.schema.json")}
TYPES = {"object": dict, "array": list, "string": str, "null": type(None)}


def _type_ok(value, name):
    if name == "integer":                             # 2020-12: 1.0 is an integer, true is a boolean
        return type(value) is int or (type(value) is float and value.is_integer())
    return isinstance(value, TYPES[name]) and not (name != "null" and value is None)


def valid(value, schema, root):
    """True when value satisfies schema (the subset used in schemas/)."""
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref.startswith("#/$defs/"):
            return valid(value, root["$defs"][ref.split("/")[-1]], root)
        return valid(value, SCHEMAS[ref], SCHEMAS[ref])
    if "oneOf" in schema and sum(valid(value, s, root) for s in schema["oneOf"]) != 1:
        return False
    types = schema.get("type")
    if types is not None and not any(_type_ok(value, t) for t in ([types] if isinstance(types, str) else types)):
        return False
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", len(value)):
            return False
        if "pattern" in schema and not re.search(schema["pattern"], value):
            return False
    number = type(value) in (int, float)
    if number and "minimum" in schema and value < schema["minimum"]:
        return False
    if number and "maximum" in schema and value > schema["maximum"]:
        return False
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            return False
        if "items" in schema and not all(valid(v, schema["items"], root) for v in value):
            return False
    if isinstance(value, dict):
        if any(key not in value for key in schema.get("required", ())):
            return False
        for key, sub in schema.get("properties", {}).items():
            if key in value and not valid(value[key], sub, root):
                return False
    return True


def findings_valid(value):
    return valid(value, SCHEMAS["findings.schema.json"], SCHEMAS["findings.schema.json"])


def action_valid(value):
    return valid(value, SCHEMAS["action.schema.json"], SCHEMAS["action.schema.json"])


EID = "KARPSR"


def finding(n=1, evidence=1, explanation=1, amount=1):
    return {"type": "duplicate_invoice", "affected_ids": [f"invoice/PROBE{n % 10}"], "claimed_amount_minor": amount,
            "currency": "KZT", "evidence_ids": [f"invoice/E{i:05d}" for i in range(evidence)],
            "explanation": "x" * explanation}


# The broken structures and edge cases of V's rehearsal, qa/rehearsal/probe_policy.py BAD and EDGE.
BAD = {
    "findings_not_list": {"episode_id": EID, "findings": "all of them"},
    "no_episode_id": {"findings": []},
    "wrong_episode_id": {"episode_id": "ZZZZZZ", "findings": []},
    "amount_float": {"episode_id": EID, "findings": [finding(amount=1.0)]},
    "amount_bool": {"episode_id": EID, "findings": [finding(amount=True)]},
    "amount_negative": {"episode_id": EID, "findings": [finding(amount=-5)]},
    "amount_string": {"episode_id": EID, "findings": [finding(amount="100")]},
    "currency_lower": {"episode_id": EID, "findings": [dict(finding(), currency="kzt")]},
    "type_unknown": {"episode_id": EID, "findings": [dict(finding(), type="fraud")]},
    "affected_empty": {"episode_id": EID, "findings": [dict(finding(), affected_ids=[])]},
    "evidence_not_list": {"episode_id": EID, "findings": [dict(finding(), evidence_ids="invoice/X")]},
    "findings_31": {"episode_id": EID, "findings": [finding(n) for n in range(31)]},
    "evidence_11": {"episode_id": EID, "findings": [finding(evidence=11)]},
    "explanation_1001": {"episode_id": EID, "findings": [finding(explanation=1001)]},
    "findings_object": "not an object",
}
EDGE = {
    "findings_30": {"episode_id": EID, "findings": [finding(n) for n in range(30)]},
    "evidence_10": {"episode_id": EID, "findings": [finding(evidence=10)]},
    "explanation_1000": {"episode_id": EID, "findings": [finding(explanation=1000)]},
    "empty": {"episode_id": EID, "findings": []},
}
ONLY_AT_RUNTIME = {"wrong_episode_id", "amount_float"}


def accepted(payload):
    try:
        check_submission(payload, EID)
    except InvalidSubmission:
        return False
    return True


class Schemas(unittest.TestCase):
    def test_both_files_are_json_schema_2020_12(self):
        for name, schema in SCHEMAS.items():
            with self.subTest(schema=name):
                self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
                self.assertEqual(schema["$id"], name)

    def test_the_15_broken_structures_are_rejected_and_the_edges_accepted(self):
        self.assertEqual(len(BAD), 15)
        for name, payload in BAD.items():
            with self.subTest(case=name):
                self.assertFalse(accepted(payload))
                self.assertEqual(findings_valid(payload), name in ONLY_AT_RUNTIME)
        for name, payload in EDGE.items():
            with self.subTest(case=name):
                self.assertTrue(accepted(payload))
                self.assertTrue(findings_valid(payload))

    def test_extra_keys_are_ignored_as_the_env_does(self):
        payload = {"episode_id": EID, "note": 1, "findings": [dict(finding(), confidence=0.9)]}
        self.assertTrue(accepted(payload))
        self.assertTrue(findings_valid(payload))

    def test_actions(self):
        good = [{"action": "list_records", "table": "ledger", "limit": 2}, {"action": "list_records", "table": "fx",
                "cursor": None, "limit": None}, {"action": "get_record", "table": "invoices", "id": "invoice/WRJD8T"},
                {"action": "get_document", "doc_id": "WRJD8T"}, {"action": "calculator", "expression": "1 + 1"},
                {"action": "submit", "findings": {"episode_id": EID, "findings": []}}]
        bad = [{"action": "list_records", "table": "ledger", "limit": 0}, {"action": "list_records", "table": "ledger",
               "limit": 51}, {"action": "list_records", "table": "truth"}, {"action": "get_record", "table": "fx"},
               {"action": "get_document", "doc_id": ""}, {"action": "calculator", "expression": 1},
               {"action": "submit", "findings": BAD["findings_31"]}, {"action": "noop"}, ["list_records"]]
        for action in good:
            with self.subTest(good=action):
                self.assertTrue(action_valid(action))
        for action in bad:
            with self.subTest(bad=action):
                self.assertFalse(action_valid(action))


if __name__ == "__main__":
    unittest.main()
