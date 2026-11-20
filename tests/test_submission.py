"""findings.json structure (case §05) and document-id aliases in the grader (case §03)."""
import copy
import unittest

from auditenv.grader import score
from auditenv.submission import MAX_EXPLANATION, MAX_FINDINGS, InvalidSubmission, check_submission
from tests.golden import oracle_findings, truth_v2

EPISODE = "KARPSR"


def submission(findings=None):
    truth = truth_v2()
    return {"episode_id": EPISODE, "findings": oracle_findings(truth) if findings is None else findings}


class SubmissionStructure(unittest.TestCase):
    def test_oracle_submission_passes_and_scores_one(self):
        payload = submission()
        findings = check_submission(payload, EPISODE)
        self.assertEqual(score(findings, truth_v2(), 0)["reward"], 1.0)

    def test_limits_are_inclusive(self):
        f = oracle_findings(truth_v2())[0]
        thirty = [dict(f, explanation="x" * MAX_EXPLANATION, evidence_ids=(f["evidence_ids"] * 10)[:10])] * MAX_FINDINGS
        self.assertEqual(len(check_submission(submission(thirty), EPISODE)), MAX_FINDINGS)
        self.assertEqual(check_submission(submission([]), EPISODE), [])

    def test_extra_keys_are_ignored(self):
        payload = submission()
        payload["note"] = "x"
        payload["findings"][0]["confidence"] = 0.9
        self.assertEqual(len(check_submission(payload, EPISODE)), 8)

    def test_broken_structure_is_rejected_with_a_code(self):
        good = submission()
        f = good["findings"][0]

        def with_finding(**changes):
            return {**good, "findings": [{**f, **changes}] + good["findings"][1:]}

        def without(key):
            return {**good, "findings": [{k: v for k, v in f.items() if k != key}]}

        bad = {
            "not an object": ([], "submit_not_object"),
            "no episode_id": ({"findings": good["findings"]}, "episode_id_mismatch"),
            "foreign episode_id": ({**good, "episode_id": "ZZZZZZ"}, "episode_id_mismatch"),
            "episode_id not a string": ({**good, "episode_id": 7}, "episode_id_mismatch"),
            "findings missing": ({"episode_id": EPISODE}, "findings_not_list"),
            "findings an object": ({**good, "findings": {"0": f}}, "findings_not_list"),
            "31 findings": ({**good, "findings": [f] * (MAX_FINDINGS + 1)}, "too_many_findings"),
            "finding not an object": ({**good, "findings": ["x"]}, "finding_not_object"),
            "old field names": ({**good, "findings": [{"type": f["type"], "affected_ids": f["affected_ids"],
                                                        "amount_minor": 1, "currency": "KZT", "evidence": [],
                                                        "explanation": ""}]}, "missing_field"),
            "no explanation": (without("explanation"), "missing_field"),
            "unknown type": (with_finding(type="fraud"), "bad_type"),
            "empty affected_ids": (with_finding(affected_ids=[]), "bad_affected_ids"),
            "affected_ids a string": (with_finding(affected_ids="ledger/NP7DTC"), "bad_affected_ids"),
            "amount as string": (with_finding(claimed_amount_minor="187500000"), "bad_claimed_amount"),
            "amount as float": (with_finding(claimed_amount_minor=1.0), "bad_claimed_amount"),
            "amount as bool": (with_finding(claimed_amount_minor=True), "bad_claimed_amount"),
            "negative amount": (with_finding(claimed_amount_minor=-1), "bad_claimed_amount"),
            "lowercase currency": (with_finding(currency="kzt"), "bad_currency"),
            "evidence_ids not a list": (with_finding(evidence_ids="ledger/NP7DTC"), "bad_evidence_ids"),
            "11 evidence ids": (with_finding(evidence_ids=["ledger/NP7DTC"] * 11), "too_many_evidence_ids"),
            "explanation not a string": (with_finding(explanation=None), "bad_explanation"),
            "explanation too long": (with_finding(explanation="x" * (MAX_EXPLANATION + 1)), "explanation_too_long"),
        }
        for name, (payload, code) in bad.items():
            with self.subTest(case=name):
                with self.assertRaises(InvalidSubmission) as ctx:
                    check_submission(copy.deepcopy(payload), EPISODE)
                self.assertEqual(str(ctx.exception), code)

    def test_hostile_objects_become_invalid_submission(self):
        class LyingDict(dict):
            def get(self, key, default=None):
                raise RuntimeError("boom")

        for payload in (LyingDict(submission()), {"episode_id": EPISODE, "findings": [LyingDict(submission()["findings"][0])]}):
            with self.assertRaises(InvalidSubmission):
                check_submission(payload, EPISODE)


class DocumentAliases(unittest.TestCase):
    """A document may be named by doc_id or by its invoice_id / payment_id (decision v2, case §03)."""

    def test_doc_ids_map_to_canonical_ids(self):
        truth = truth_v2()
        aliases = {"DOC001": "1TNM24", "DOC002": "F7SF66"}
        oracle = oracle_findings(truth)
        dup = next(f for f in oracle if f["affected_ids"] == ["invoice/1TNM24", "invoice/F7SF66"])
        by_doc = dict(dup, affected_ids=["DOC001", "doc/DOC002"],
                      evidence_ids=["DOC001", "DOC002", "ledger/CKBBZ4", "ledger/KBYZZ8"])
        self.assertEqual(score([by_doc], truth, 0, aliases)["tp"], 1)
        self.assertEqual(score([by_doc], truth, 0)["tp"], 0)
        self.assertEqual(score(oracle, truth, 0, aliases)["reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
