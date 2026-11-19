"""Matcher and reward (SPEC.md §5). QA cases E-24, order E-26, prefixes E-27, formats E-28."""
import copy
import json
import random
import unittest
from fractions import Fraction
from pathlib import Path

from auditenv.grader import f1_score, normalize_id, parse_truth, score, terminal_reward, timeout_score
from tests.golden import clean_truth, oracle_findings, public_record_ids, truth_v2
from tests.support import run_python

ISSUE_TYPES = ("duplicate_invoice", "fx_rate", "payment_allocation", "period_cutoff")


def counts(result):
    return result["tp"], result["fp"], result["fn"]


def pad_evidence(finding, size):
    pool = [r for r in public_record_ids() if r not in finding["evidence"]]
    return {**finding, "evidence": finding["evidence"] + pool[: size - len(finding["evidence"])]}


def swap_c1_types(findings):
    swapped = copy.deepcopy(findings)
    c1 = [f for f in swapped if f["affected_ids"] == ["ledger/Z3H7BB"]]
    assert sorted(f["type"] for f in c1) == ["fx_rate", "period_cutoff"]
    for f in c1:
        f["type"] = "fx_rate" if f["type"] == "period_cutoff" else "period_cutoff"
    return swapped


def qa_cases():
    """The STRICT column of the QA v0.1 reward table: (name, findings, truth, steps, (tp, fp, fn), reward)."""
    truth, clean = truth_v2(), clean_truth()
    oracle = oracle_findings(truth)
    records = public_record_ids()
    flag_all = [
        {"type": t, "affected_ids": [r], "amount_minor": 0, "currency": "KZT", "evidence": [r]}
        for r in records for t in ISSUE_TYPES
    ]
    spray = [
        {"type": t, "affected_ids": records, "amount_minor": 0, "currency": "KZT", "evidence": records[:10]}
        for t in ISSUE_TYPES
    ]
    false_claim = {**oracle[0]}
    third = Fraction(1, 3)
    return [
        ("oracle, 0 steps", oracle, truth, 0, (8, 0, 0), 1.0),
        ("oracle, 40 steps", oracle, truth, 40, (8, 0, 0), 0.92),
        ("silence, book with errors", [], truth, 0, (0, 0, 8), -1.0),
        ("silence, clean company", [], clean, 0, (0, 0, 0), 1.0),
        ("one false accusation, clean company", [false_claim], clean, 0, (0, 1, 0), -1.0),
        ("flag everything, book with errors", flag_all, truth, 0, (0, 300, 8), -1.0),
        ("flag everything, clean company", flag_all, clean, 0, (0, 300, 0), -1.0),
        ("spray all 75 ids in 4 findings", spray, truth, 0, (0, 4, 8), -1.0),
        ("oracle, every finding twice", oracle + oracle, truth, 0, (8, 8, 0), float(third)),
        ("oracle, amount off by 2", [{**f, "amount_minor": f["amount_minor"] + 2} for f in oracle], truth, 0, (0, 8, 8), -1.0),
        ("oracle, wrong currency", [{**f, "currency": "EUR" if f["currency"] != "EUR" else "USD"} for f in oracle], truth, 0, (0, 8, 8), -1.0),
        ("oracle without evidence", [{**f, "evidence": []} for f in oracle], truth, 0, (0, 8, 8), -1.0),
        ("oracle, evidence padded to 10", [pad_evidence(f, 10) for f in oracle], truth, 0, (8, 0, 0), 1.0),
        ("oracle, 11 evidence ids", [pad_evidence(f, 11) for f in oracle], truth, 0, (0, 8, 8), -1.0),
        ("non-existent ids", [{**f, "affected_ids": [f"invoice/00000{k}"]} for k, f in enumerate(oracle)], truth, 0, (0, 8, 8), -1.0),
        ("oracle plus one false finding", oracle + [flag_all[0]], truth, 0, (8, 1, 0), float(Fraction(15, 17))),
        ("half of oracle", oracle[:4], truth, 0, (4, 0, 4), float(third)),
        ("C1, fx and period swapped", swap_c1_types(oracle), truth, 0, (6, 2, 2), 0.5),
        ("timeout, clean company", [], clean, 100, (0, 0, 0), 0.8),
    ]


class QaRewardTable(unittest.TestCase):
    """E-24: the grader reproduces every STRICT number of the QA v0.1 table."""

    def test_strict_column(self):
        for name, findings, truth, steps, expected_counts, expected_reward in qa_cases():
            with self.subTest(case=name):
                result = score(findings, truth, steps)
                self.assertEqual(counts(result), expected_counts)
                self.assertEqual(result["reward"], expected_reward)

    def test_honest_answer_beats_every_attack(self):
        truth = truth_v2()
        honest = score(oracle_findings(truth), truth, 0)["reward"]
        honest_variants = {"oracle, 0 steps", "oracle, 40 steps", "oracle, evidence padded to 10"}
        attacks = [c for c in qa_cases() if not c[2]["clean"] and c[0] not in honest_variants]
        self.assertEqual(len(attacks), 12)
        for name, findings, _, steps, _, _ in attacks:
            with self.subTest(case=name):
                self.assertLess(score(findings, truth, steps)["reward"], honest)

    def test_flag_everything_never_beats_silence(self):
        for truth in (truth_v2(), clean_truth()):
            records = public_record_ids()
            flag_all = [
                {"type": t, "affected_ids": [r], "amount_minor": 0, "currency": "KZT", "evidence": [r]}
                for r in records for t in ISSUE_TYPES
            ]
            self.assertLessEqual(score(flag_all, truth, 0)["reward"], score([], truth, 0)["reward"])


class IdPrefixes(unittest.TestCase):
    """E-27 and P-04: ids with and without a table prefix are the same id."""

    def test_normalize_id(self):
        self.assertEqual(normalize_id("invoice/1TNM24"), "1TNM24")
        self.assertEqual(normalize_id("1TNM24"), "1TNM24")
        self.assertEqual(normalize_id("fx/2026-09-14/USD/KZT"), "2026-09-14/USD/KZT")
        self.assertEqual(normalize_id("2026-09-14/USD/KZT"), "2026-09-14/USD/KZT")
        self.assertEqual(normalize_id("je/A12BT6"), "je/A12BT6")
        self.assertEqual(normalize_id("ledger/"), "ledger/")

    def test_bare_and_mixed_prefixes_give_the_same_reward(self):
        truth = truth_v2()
        oracle = oracle_findings(truth)
        bare = [{**f, "affected_ids": [normalize_id(x) for x in f["affected_ids"]],
                 "evidence": [normalize_id(x) for x in f["evidence"]]} for f in oracle]
        mixed = [{**f, "evidence": [normalize_id(x) if k % 2 else x for k, x in enumerate(f["evidence"])]} for f in oracle]
        for findings in (oracle, bare, mixed):
            self.assertEqual(counts(score(findings, truth, 0)), (8, 0, 0))

    def test_journal_entry_ids_are_not_records(self):
        truth = truth_v2()
        old_style = [{**f, "affected_ids": ["je/A12BT6"]} if f["affected_ids"] == ["ledger/NP7DTC"] else f
                     for f in oracle_findings(truth)]
        self.assertEqual(counts(score(old_style, truth, 0)), (7, 1, 1))

    def test_unknown_prefix_is_not_stripped(self):
        truth = truth_v2()
        oracle = oracle_findings(truth)
        broken = [{**f, "affected_ids": [f"doc_ref/{normalize_id(x)}" for x in f["affected_ids"]]} for f in oracle]
        self.assertEqual(counts(score(broken, truth, 0)), (0, 8, 8))


class OrderAndDeterminism(unittest.TestCase):
    """E-26 and E-15 for the grader."""

    def test_finding_and_issue_order_do_not_matter(self):
        truth = truth_v2()
        findings = oracle_findings(truth) + oracle_findings(truth)[:3] + [{"type": "fx_rate"}]
        base = score(findings, truth, 7)
        for seed in range(5):
            shuffled_findings = findings[:]
            random.Random(seed).shuffle(shuffled_findings)
            shuffled_truth = {**truth, "issues": truth["issues"][:]}
            random.Random(seed + 100).shuffle(shuffled_truth["issues"])
            got = score(shuffled_findings, shuffled_truth, 7)
            with self.subTest(seed=seed):
                self.assertEqual((counts(got), got["reward"]), (counts(base), base["reward"]))

    def test_same_result_under_different_hash_seeds(self):
        root = Path(__file__).resolve().parent.parent
        code = (
            f"import json, sys\nsys.path.insert(0, {str(root)!r})\n"
            "from auditenv.grader import score\n"
            "from tests.golden import truth_v2, oracle_findings, public_record_ids\n"
            "t = truth_v2()\n"
            "f = oracle_findings(t) + [{'type': 'fx_rate', 'affected_ids': public_record_ids(),"
            " 'amount_minor': 1, 'currency': 'KZT', 'evidence': []}]\n"
            "print(json.dumps(score(f, t, 13), sort_keys=True))\n"
        )
        outputs ={seed: run_python(["-c", code], seed, cwd=root) for seed in ("0", "1", "12345")}
        for seed, res in outputs.items():
            self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(len({res.stdout for res in outputs.values()}), 1)
        self.assertEqual(json.loads(outputs["0"].stdout)["tp"], 8)


class MaximumMatching(unittest.TestCase):
    def two_twin_issues(self):
        base = {"type": "duplicate_invoice", "affected_ids": ["invoice/AAAAAA", "invoice/BBBBBB"],
                "expected_amount_minor": 500, "currency": "KZT"}
        return {"clean": False, "issues": [
            {**base, "issue_id": "I1", "evidence_sets": [["ledger/EV0001"]]},
            {**base, "issue_id": "I2", "evidence_sets": [["ledger/EV0002"]]},
        ]}

    def test_beats_greedy_assignment(self):
        finding = {"type": "duplicate_invoice", "affected_ids": ["AAAAAA", "BBBBBB"], "amount_minor": 500, "currency": "KZT"}
        both = {**finding, "evidence": ["EV0001", "EV0002"]}
        only_first = {**finding, "evidence": ["EV0001"]}
        result = score([both, only_first], self.two_twin_issues(), 0)
        self.assertEqual(counts(result), (2, 0, 0))

    def test_one_finding_closes_one_issue(self):
        finding = {"type": "duplicate_invoice", "affected_ids": ["AAAAAA", "BBBBBB"], "amount_minor": 500,
                   "currency": "KZT", "evidence": ["EV0001", "EV0002"]}
        self.assertEqual(counts(score([finding], self.two_twin_issues(), 0)), (1, 0, 1))

    def test_alternative_evidence_sets(self):
        truth = {"clean": False, "issues": [{
            "issue_id": "I1", "type": "fx_rate", "affected_ids": ["ledger/QQ5GHK"], "expected_amount_minor": 1270000,
            "currency": "KZT", "evidence_sets": [["invoice/FRVMVD", "ledger/QQ5GHK", "fx/F1"], ["ledger/QQ5GHK", "fx/F2"]],
        }]}
        finding = {"type": "fx_rate", "affected_ids": ["QQ5GHK"], "amount_minor": 1270000, "currency": "KZT",
                   "evidence": ["QQ5GHK", "F2", "invoice/ZZZZZZ"]}
        self.assertEqual(counts(score([finding], truth, 0)), (1, 0, 0))
        self.assertEqual(counts(score([{**finding, "evidence": ["QQ5GHK"]}], truth, 0)), (0, 1, 1))

    def test_amount_tolerance_is_inclusive(self):
        truth = truth_v2()
        oracle = oracle_findings(truth)
        for delta, expected in ((-1, 8), (1, 8), (2, 0), (-2, 0)):
            shifted = [{**f, "amount_minor": f["amount_minor"] + delta} for f in oracle]
            with self.subTest(delta=delta):
                self.assertEqual(score(shifted, truth, 0)["tp"], expected)


class FindingFormat(unittest.TestCase):
    """E-28: a malformed finding is an FP, never an error and never a TP."""

    def setUp(self):
        self.truth = truth_v2()
        self.good = oracle_findings(self.truth)[0]
        self.assertEqual(score([self.good], self.truth, 0)["tp"], 1)

    def test_malformed_findings_count_as_false_positives(self):
        g = self.good
        bad = {
            "amount as string": {**g, "amount_minor": str(g["amount_minor"])},
            "amount as float": {**g, "amount_minor": float(g["amount_minor"])},
            "amount as bool": {**g, "amount_minor": True},
            "negative amount": {**g, "amount_minor": -g["amount_minor"]},
            "lowercase currency": {**g, "currency": g["currency"].lower()},
            "four-letter currency": {**g, "currency": g["currency"] + "X"},
            "unknown type": {**g, "type": "fraud"},
            "affected ids as string": {**g, "affected_ids": g["affected_ids"][0]},
            "empty affected ids": {**g, "affected_ids": []},
            "evidence as string": {**g, "evidence": " ".join(g["evidence"])},
            "non-string id": {**g, "affected_ids": [1]},
            "missing currency": {k: v for k, v in g.items() if k != "currency"},
            "duplicated evidence over 10": {**g, "evidence": (g["evidence"] * 11)[:11]},
            "not an object": [g],
            "null": None,
        }
        for name, finding in bad.items():
            with self.subTest(case=name):
                result = score([finding], self.truth, 0)
                self.assertEqual(counts(result), (0, 1, 8))
                self.assertEqual(result["invalid_findings"], [0])

    def test_extra_keys_are_ignored(self):
        self.assertEqual(score([{**self.good, "comment": "found it"}], self.truth, 0)["tp"], 1)

    def test_bad_call_arguments(self):
        with self.assertRaises(TypeError):
            score({"findings": []}, self.truth, 0)
        for steps in (-1, 1.5, True, "3"):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                score([], self.truth, steps)


class TruthValidation(unittest.TestCase):
    def test_golden_v2_truth_parses(self):
        self.assertEqual(len(parse_truth(truth_v2())), 8)

    def test_malformed_truth_is_rejected(self):
        good = truth_v2()
        issue = good["issues"][0]

        def with_issue(**changes):
            return {**good, "issues": [{**issue, **changes}] + good["issues"][1:]}

        bad = {
            "clean flag on a book with errors": {**good, "clean": True},
            "issues missing on a clean company": {"clean": True},
            "clean flag missing": {"issues": []},
            "duplicate issue_id": {**good, "issues": good["issues"] + [issue]},
            "empty evidence_sets": with_issue(evidence_sets=[]),
            "empty evidence set": with_issue(evidence_sets=[[]]),
            "zero amount": with_issue(expected_amount_minor=0),
            "float amount": with_issue(expected_amount_minor=1.0),
            "unknown type": with_issue(type="fraud"),
            "non-string ids": with_issue(affected_ids=[1]),
            "missing field": {**good, "issues": [{k: v for k, v in issue.items() if k != "currency"}]},
        }
        for name, truth in bad.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                parse_truth(truth)


class RewardFormula(unittest.TestCase):
    def test_f1_edges(self):
        self.assertEqual(f1_score(0, 0, 0), 1)
        self.assertEqual(f1_score(0, 1, 0), 0)
        self.assertEqual(f1_score(0, 0, 1), 0)
        self.assertEqual(f1_score(3, 1, 1), Fraction(6, 8))

    def test_clip_and_exact_step_cost(self):
        self.assertEqual(terminal_reward(1, 0), 1.0)
        self.assertEqual(terminal_reward(1, 40), 0.92)
        self.assertEqual(terminal_reward(1, 100), 0.8)
        self.assertEqual(terminal_reward(1, 5000), -1.0)
        self.assertEqual(terminal_reward(0, 0), -1.0)

    def test_timeout_charges_all_steps(self):
        self.assertEqual(timeout_score(clean_truth())["reward"], 0.8)
        self.assertEqual(counts(timeout_score(truth_v2())), (0, 0, 8))
        self.assertEqual(timeout_score(truth_v2())["reward"], -1.0)


if __name__ == "__main__":
    unittest.main()
