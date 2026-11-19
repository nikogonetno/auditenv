"""Independent black-box tests for the grader (N-version check).

Written only from SPEC.md sections 3.3 (IDs), 3.6 (truth.json) and 5 (grading)
plus the golden helpers in tests/golden.py. The grader source and the other
grader tests were not read while writing this file. Every expected value below
is derived from the spec text, and rewards are recomputed here in Fraction.

Where the spec leaves an interface detail open (the shape of `matched`, the
base of the numbers in `invalid_findings`, the signatures of f1_score and
terminal_reward), the assumption is stated next to the helper that makes it.
"""
import copy
import inspect
import random
import unittest
from collections.abc import Mapping
from fractions import Fraction

from auditenv.grader import (
    f1_score,
    normalize_id,
    parse_finding,
    parse_truth,
    score,
    terminal_reward,
    timeout_score,
)
from tests.golden import clean_truth, oracle_findings, public_record_ids, truth_v2

TYPES = ("duplicate_invoice", "payment_allocation", "fx_rate", "period_cutoff")
# SPEC 3.3 lists seven prefixes, SPEC 5.2 adds `doc`.
KNOWN_PREFIXES = ("allocation", "counterparty", "doc", "fx", "invoice", "ledger", "line", "payment")
RESULT_FIELDS = ("tp", "fp", "fn", "precision", "recall", "f1", "reward", "matched", "invalid_findings")
MISSING = object()


# --------------------------------------------------------------------------- helpers

def field(result, name):
    """SPEC 5.4 lists the returned names but not the container type."""
    if isinstance(result, Mapping):
        return result[name]
    return getattr(result, name)


def counts(result):
    return field(result, "tp"), field(result, "fp"), field(result, "fn")


def invalid(result):
    """SPEC 5.1: the finding's number goes to invalid_findings. Assumed 0-based."""
    return sorted(field(result, "invalid_findings"))


def _pair(a, b):
    if isinstance(a, str) and isinstance(b, int) and not isinstance(b, bool):
        return (a, b)
    if isinstance(b, str) and isinstance(a, int) and not isinstance(a, bool):
        return (b, a)
    raise AssertionError(f"unrecognised matched entry: {a!r}, {b!r}")


def matched_pairs(result):
    """Normalise `matched` into {(issue_id, finding_index)}.

    SPEC does not fix the shape, so accept a mapping in either direction, a
    sequence of 2-tuples in either order, or a sequence of dicts holding one
    str (issue_id) and one int (finding index).
    """
    m = field(result, "matched")
    pairs = set()
    if isinstance(m, Mapping):
        for k, v in m.items():
            pairs.add(_pair(k, v))
        return pairs
    for item in m:
        if isinstance(item, Mapping):
            strs = [v for v in item.values() if isinstance(v, str)]
            ints = [v for v in item.values() if isinstance(v, int) and not isinstance(v, bool)]
            if len(strs) != 1 or len(ints) != 1:
                raise AssertionError(f"unrecognised matched entry: {item!r}")
            pairs.add((strs[0], ints[0]))
        else:
            a, b = item
            pairs.add(_pair(a, b))
    return pairs


def golden():
    return truth_v2()


def golden_issue(issue_id):
    (i,) = [i for i in golden()["issues"] if i["issue_id"] == issue_id]
    return i


def single(issue_id):
    """Golden truth reduced to one issue."""
    return {"clean": False, "issues": [golden_issue(issue_id)]}


def finding_for(issue):
    return oracle_findings({"issues": [issue]})[0]


def make_issue(issue_id="AAAAAA", type_="duplicate_invoice",
               affected=("invoice/1TNM24", "invoice/F7SF66"), amount=325000, currency="EUR",
               evidence_sets=(("invoice/1TNM24", "invoice/F7SF66", "ledger/CKBBZ4", "ledger/KBYZZ8"),),
               tolerance=1, **extra):
    issue = {
        "issue_id": issue_id,
        "type": type_,
        "affected_ids": sorted(affected),
        "expected_amount_minor": amount,
        "currency": currency,
        "evidence_sets": [sorted(s) for s in evidence_sets],
    }
    if tolerance is not MISSING:
        issue["tolerance_minor"] = tolerance
    issue.update(extra)
    return issue


def truth_of(*issues):
    return {"clean": not issues, "issues": list(issues)}


def junk(n):
    """n valid-format findings that match nothing in the golden truth."""
    ids = [r for r in public_record_ids() if r.startswith("ledger/")]
    return [{
        "type": TYPES[k % 4],
        "affected_ids": [ids[k % len(ids)]],
        "amount_minor": 7 + k,
        "currency": "KZT",
        "evidence": [ids[k % len(ids)]],
    } for k in range(n)]


def exact_f1(tp, fp, fn):
    if tp == 0:
        return Fraction(1) if fp == 0 and fn == 0 else Fraction(0)
    p = Fraction(tp, tp + fp)
    r = Fraction(tp, tp + fn)
    return 2 * p * r / (p + r)


def exact_reward(tp, fp, fn, steps):
    raw = 2 * exact_f1(tp, fp, fn) - 1 - Fraction(2, 1000) * steps
    return max(Fraction(-1), min(Fraction(1), raw))


def noise_ids(exclude, n):
    ex = {normalize_ref(x) for x in exclude}
    out = [r for r in public_record_ids() if normalize_ref(r) not in ex]
    return out[:n]


def normalize_ref(s):
    """Reference normalize_id written from SPEC 5.2, used only by this file."""
    head, sep, tail = s.partition("/")
    if sep and head in KNOWN_PREFIXES:
        return tail
    return s


class GraderCase(unittest.TestCase):
    def score(self, findings, truth, steps=0):
        return score(copy.deepcopy(findings), copy.deepcopy(truth), steps)

    def assertCounts(self, result, tp, fp, fn, msg=None):
        self.assertEqual(counts(result), (tp, fp, fn), msg)

    def assertReward(self, result, expected, msg=None):
        r = field(result, "reward")
        self.assertIsInstance(r, float, msg)
        self.assertEqual(r, float(expected), msg)


# --------------------------------------------------------------------------- API

class PublicApiTest(GraderCase):
    def test_public_names_are_callable(self):
        for fn in (score, timeout_score, normalize_id, parse_finding, parse_truth, f1_score, terminal_reward):
            self.assertTrue(callable(fn), fn)

    def test_score_returns_every_documented_field(self):
        for truth in (clean_truth(), golden()):
            res = self.score([], truth, 0)
            for name in RESULT_FIELDS:
                field(res, name)
            res = timeout_score(copy.deepcopy(truth))
            for name in RESULT_FIELDS:
                field(res, name)


# --------------------------------------------------------------------------- 3.3 / 5.2 normalize_id

class NormalizeIdTest(unittest.TestCase):
    def test_each_known_prefix_is_stripped(self):
        for p in KNOWN_PREFIXES:
            with self.subTest(prefix=p):
                self.assertEqual(normalize_id(f"{p}/1TNM24"), "1TNM24")

    def test_bare_id_is_unchanged(self):
        for s in ("1TNM24", "F7SF66", "Z3H7BB", "0N1PQJ"):
            self.assertEqual(normalize_id(s), s)

    def test_prefix_is_stripped_only_once(self):
        self.assertEqual(normalize_id("invoice/invoice/1TNM24"), "invoice/1TNM24")
        self.assertEqual(normalize_id("ledger/invoice/1TNM24"), "invoice/1TNM24")
        self.assertEqual(normalize_id("fx/fx/2026-09-14/USD/KZT"), "fx/2026-09-14/USD/KZT")

    def test_fx_path_keeps_its_tail(self):
        self.assertEqual(normalize_id("fx/2026-09-14/USD/KZT"), "2026-09-14/USD/KZT")
        self.assertEqual(normalize_id("2026-09-14/USD/KZT"), "2026-09-14/USD/KZT")

    def test_je_and_unknown_prefixes_are_kept(self):
        for s in ("je/A12BT6", "invoices/1TNM24", "account/1210", "entry/1TNM24",
                  "counterparties/5N6CKK", "ledgers/CKBBZ4", "doc_id/1TNM24", "/1TNM24"):
            with self.subTest(s=s):
                self.assertEqual(normalize_id(s), s)

    def test_case_is_significant(self):
        self.assertEqual(normalize_id("Invoice/1TNM24"), "Invoice/1TNM24")
        self.assertEqual(normalize_id("INVOICE/1TNM24"), "INVOICE/1TNM24")
        self.assertEqual(normalize_id("invoice/1tnm24"), "1tnm24")

    def test_other_separators_and_whitespace_are_not_prefixes(self):
        for s in ("invoice:1TNM24", "invoice\\1TNM24", "invoice1TNM24", "invoice", " invoice/1TNM24",
                  "invoice /1TNM24"):
            with self.subTest(s=s):
                self.assertEqual(normalize_id(s), s)


# --------------------------------------------------------------------------- 5.1 finding format

def _set(f, key, value):
    g = copy.deepcopy(f)
    if value is MISSING:
        g.pop(key, None)
    else:
        g[key] = value
    return g


def _bad_variants():
    good = finding_for(golden_issue("G21S27"))
    ev = list(good["evidence"])
    noise = noise_ids(ev, 7)
    out = []
    for v in ("duplicate", "Duplicate_Invoice", "DUPLICATE_INVOICE", "duplicate_invoice ", "", None, 1,
              ["duplicate_invoice"], {"duplicate_invoice": 1}, MISSING):
        out.append((f"type={v!r}", _set(good, "type", v)))
    for v in ([], "invoice/1TNM24", None, 7, {"invoice/1TNM24": 1}, [1, 2],
              ["invoice/1TNM24", None], ["invoice/1TNM24", ["invoice/F7SF66"]], MISSING):
        out.append((f"affected_ids={v!r}", _set(good, "affected_ids", v)))
    for v in (325000.0, 325000.5, "325000", True, False, -1, -325000, None, [325000], MISSING):
        out.append((f"amount_minor={v!r}", _set(good, "amount_minor", v)))
    for v in ("eur", "Eur", "EURO", "EU", "", "ЕUR", "EUR\n", " EUR", "EUR ", "E1R", "ÉUR",
              "ＥＵＲ", None, 978, ["EUR"], MISSING):
        out.append((f"currency={v!r}", _set(good, "currency", v)))
    for v in (ev + noise,                      # 11 distinct ids
              ev + ev + ev[:3],                # 11 with repeats
              "invoice/1TNM24", None, 3, [1], ev + [None], [ev], {k: 1 for k in ev}, MISSING):
        out.append((f"evidence={v!r}", _set(good, "evidence", v)))
    return good, out


class FindingFormatTest(GraderCase):
    truth = staticmethod(lambda: single("G21S27"))

    def test_badly_formed_finding_is_fp_listed_and_never_tp(self):
        good, variants = _bad_variants()
        for name, bad in variants:
            with self.subTest(variant=name):
                res = self.score([bad], self.truth())
                self.assertCounts(res, 0, 1, 1)
                self.assertEqual(invalid(res), [0])
                self.assertReward(res, -1)
                # The well-formed copy after it still wins the issue.
                res = self.score([bad, good], self.truth())
                self.assertCounts(res, 1, 1, 0)
                self.assertEqual(invalid(res), [0])
                self.assertEqual(matched_pairs(res), {("G21S27", 1)})

    def test_non_dict_findings_do_not_crash(self):
        good = finding_for(golden_issue("G21S27"))
        garbage = [None, "finding", 42, 4.2, True, [], ["duplicate_invoice"], {}, set()]
        findings = garbage[:4] + [good] + garbage[4:]
        res = self.score(findings, self.truth())
        self.assertCounts(res, 1, len(garbage), 0)
        self.assertEqual(invalid(res), [0, 1, 2, 3, 5, 6, 7, 8, 9])

    def test_extra_keys_are_ignored(self):
        good = finding_for(golden_issue("G21S27"))
        extra = dict(good, issue_id="ZZZZZZ", comment="x", confidence=0.3, scenario_id="S1",
                     tolerance_minor=10 ** 9, evidence_sets=[[]], expected_amount_minor=1)
        res = self.score([extra], self.truth())
        self.assertCounts(res, 1, 0, 0)
        self.assertEqual(invalid(res), [])

    def test_well_formed_but_wrong_finding_is_fp_but_not_invalid(self):
        good = finding_for(golden_issue("G21S27"))
        for name, f in (("amount 0", _set(good, "amount_minor", 0)),
                        ("empty evidence", _set(good, "evidence", [])),
                        ("other currency", _set(good, "currency", "USD")),
                        ("other type", _set(good, "type", "fx_rate")),
                        ("unknown id", _set(good, "affected_ids", ["invoice/XXXXXX"]))):
            with self.subTest(name):
                res = self.score([f], self.truth())
                self.assertCounts(res, 0, 1, 1)
                self.assertEqual(invalid(res), [])

    def test_evidence_of_exactly_ten_is_allowed_including_repeats(self):
        good = finding_for(golden_issue("G21S27"))
        ev = list(good["evidence"])
        for name, evidence in (("10 distinct", ev + noise_ids(ev, 6)),
                               ("10 with repeats", ev + ev + ev[:2])):
            with self.subTest(name):
                self.assertEqual(len(evidence), 10)
                res = self.score([_set(good, "evidence", evidence)], self.truth())
                self.assertCounts(res, 1, 0, 0)
                self.assertEqual(invalid(res), [])

    def test_amount_zero_is_valid_and_can_match_within_tolerance(self):
        issue = make_issue(amount=1, tolerance=1)
        f = _set(finding_for(issue), "amount_minor", 0)
        res = self.score([f], truth_of(issue))
        self.assertCounts(res, 1, 0, 0)
        self.assertEqual(invalid(res), [])

    def test_bool_amount_is_rejected_even_when_value_would_match(self):
        issue = make_issue(amount=1, tolerance=0)
        f = _set(finding_for(issue), "amount_minor", True)
        res = self.score([f], truth_of(issue))
        self.assertCounts(res, 0, 1, 1)
        self.assertEqual(invalid(res), [0])

    def test_invalid_numbers_point_at_the_right_positions(self):
        t = golden()
        oracle = oracle_findings(t)
        findings = []
        for k, f in enumerate(oracle):
            findings.append(f)
            if k % 3 == 0:
                findings.append(_set(f, "currency", f["currency"].lower()))
        expected_invalid = [i for i, f in enumerate(findings) if f["currency"].islower()]
        res = self.score(findings, t)
        self.assertCounts(res, 8, len(expected_invalid), 0)
        self.assertEqual(invalid(res), expected_invalid)


# --------------------------------------------------------------------------- 5.2 five conditions

class MatchConditionsTest(GraderCase):
    def test_oracle_is_perfect(self):
        t = golden()
        res = self.score(oracle_findings(t), t)
        self.assertCounts(res, 8, 0, 0)
        self.assertReward(res, 1)
        self.assertEqual(invalid(res), [])

    def test_condition_1_type_must_be_equal(self):
        for issue in golden()["issues"]:
            for other in TYPES:
                if other == issue["type"]:
                    continue
                with self.subTest(issue=issue["issue_id"], type=other):
                    f = _set(finding_for(issue), "type", other)
                    self.assertCounts(self.score([f], truth_of(issue)), 0, 1, 1)

    def test_condition_2_affected_set_must_be_equal(self):
        issue = golden_issue("G21S27")
        good = finding_for(issue)
        all_ids = public_record_ids()
        cases = {
            "subset": ["invoice/1TNM24"],
            "other half": ["invoice/F7SF66"],
            "superset +1": ["invoice/1TNM24", "invoice/F7SF66", "ledger/CKBBZ4"],
            "spray all 75": all_ids,
            "disjoint": ["invoice/RRH8VY", "invoice/ZZRFBK"],
        }
        for name, ids in cases.items():
            with self.subTest(name):
                res = self.score([_set(good, "affected_ids", ids)], truth_of(issue))
                self.assertCounts(res, 0, 1, 1)

    def test_condition_2_order_and_repeats_do_not_matter(self):
        issue = golden_issue("G21S27")
        good = finding_for(issue)
        for ids in (["invoice/F7SF66", "invoice/1TNM24"],
                    ["invoice/1TNM24", "invoice/1TNM24", "invoice/F7SF66"],
                    ["invoice/1TNM24", "1TNM24", "doc/F7SF66", "F7SF66"]):
            with self.subTest(ids=ids):
                self.assertCounts(self.score([_set(good, "affected_ids", ids)], truth_of(issue)), 1, 0, 0)

    def test_condition_3_default_tolerance_is_one_and_inclusive(self):
        for tol in (1, MISSING):
            issue = make_issue(amount=325000, tolerance=tol)
            good = finding_for(issue)
            for delta, tp in ((0, 1), (1, 1), (-1, 1), (2, 0), (-2, 0), (1000, 0)):
                with self.subTest(tolerance="default" if tol is MISSING else tol, delta=delta):
                    f = _set(good, "amount_minor", 325000 + delta)
                    self.assertCounts(self.score([f], truth_of(issue)), tp, 1 - tp, 1 - tp)

    def test_condition_3_tolerance_comes_from_issue(self):
        for tol, deltas in ((0, ((0, 1), (1, 0), (-1, 0))),
                            (5, ((5, 1), (-5, 1), (6, 0), (-6, 0))),
                            (100, ((100, 1), (-100, 1), (101, 0)))):
            issue = make_issue(amount=325000, tolerance=tol)
            for delta, tp in deltas:
                with self.subTest(tolerance=tol, delta=delta):
                    f = _set(finding_for(issue), "amount_minor", 325000 + delta)
                    self.assertCounts(self.score([f], truth_of(issue)), tp, 1 - tp, 1 - tp)

    def test_condition_3_on_large_golden_amounts(self):
        issue = golden_issue("XG1494")  # 510120000 KZT, tolerance 1
        for delta, tp in ((1, 1), (-1, 1), (2, 0), (-2, 0)):
            with self.subTest(delta=delta):
                f = _set(finding_for(issue), "amount_minor", issue["expected_amount_minor"] + delta)
                self.assertCounts(self.score([f], truth_of(issue)), tp, 1 - tp, 1 - tp)

    def test_condition_4_currency_must_be_equal(self):
        for issue in golden()["issues"]:
            for cur in ("KZT", "USD", "EUR", "RUB"):
                if cur == issue["currency"]:
                    continue
                with self.subTest(issue=issue["issue_id"], currency=cur):
                    f = _set(finding_for(issue), "currency", cur)
                    res = self.score([f], truth_of(issue))
                    self.assertCounts(res, 0, 1, 1)
                    self.assertEqual(invalid(res), [])

    def test_condition_5_every_evidence_id_is_needed(self):
        for issue in golden()["issues"]:
            needed = issue["evidence_sets"][0]
            for drop in needed:
                with self.subTest(issue=issue["issue_id"], drop=drop):
                    ev = [x for x in needed if x != drop]
                    f = _set(finding_for(issue), "evidence", ev)
                    self.assertCounts(self.score([f], truth_of(issue)), 0, 1, 1)

    def test_condition_5_superset_up_to_ten_is_enough(self):
        for issue in golden()["issues"]:
            needed = issue["evidence_sets"][0]
            ev = list(needed) + noise_ids(needed, 10 - len(needed))
            random.Random(len(ev)).shuffle(ev)
            with self.subTest(issue=issue["issue_id"]):
                f = _set(finding_for(issue), "evidence", ev)
                self.assertCounts(self.score([f], truth_of(issue)), 1, 0, 0)

    def test_condition_5_any_one_of_several_sets(self):
        a, b, c, d = "invoice/1TNM24", "invoice/F7SF66", "ledger/CKBBZ4", "ledger/KBYZZ8"
        issue = make_issue(evidence_sets=((a, c), (b, d)))
        good = finding_for(issue)
        for ev, tp in (([a, c], 1), ([b, d], 1), ([d, b], 1), ([a, b, c, d], 1),
                       ([a, d], 0), ([b, c], 0), ([a], 0), ([], 0)):
            with self.subTest(evidence=ev):
                self.assertCounts(self.score([_set(good, "evidence", ev)], truth_of(issue)), tp, 1 - tp, 1 - tp)

    def test_condition_5_eleven_ids_lose_even_with_the_set_inside(self):
        issue = golden_issue("FDZ2T6")
        needed = issue["evidence_sets"][0]
        ev = list(needed) + noise_ids(needed, 11 - len(needed))
        res = self.score([_set(finding_for(issue), "evidence", ev)], truth_of(issue))
        self.assertCounts(res, 0, 1, 1)
        self.assertEqual(invalid(res), [0])


# --------------------------------------------------------------------------- 3.3 ID forms inside matching

class IdFormsTest(GraderCase):
    def _bare(self, f):
        return dict(f, affected_ids=[normalize_ref(x) for x in f["affected_ids"]],
                    evidence=[normalize_ref(x) for x in f["evidence"]])

    def test_bare_ids_match_prefixed_truth(self):
        t = golden()
        findings = [self._bare(f) for f in oracle_findings(t)]
        res = self.score(findings, t)
        self.assertCounts(res, 8, 0, 0)
        self.assertReward(res, 1)

    def test_prefixed_findings_match_bare_truth(self):
        t = golden()
        findings = oracle_findings(t)
        for i in t["issues"]:
            i["affected_ids"] = sorted(normalize_ref(x) for x in i["affected_ids"])
            i["evidence_sets"] = [sorted(normalize_ref(x) for x in s) for s in i["evidence_sets"]]
        self.assertCounts(self.score(findings, t), 8, 0, 0)

    def test_doc_prefix_and_foreign_table_prefix_are_accepted(self):
        issue = golden_issue("FDZ2T6")  # invoice/REB8R3 + payment/WFV9R1
        good = finding_for(issue)
        for ids in (["doc/REB8R3", "doc/WFV9R1"],
                    ["payment/REB8R3", "invoice/WFV9R1"],
                    ["ledger/REB8R3", "line/WFV9R1"],
                    ["counterparty/REB8R3", "allocation/WFV9R1"],
                    ["fx/REB8R3", "WFV9R1"]):
            with self.subTest(ids=ids):
                self.assertCounts(self.score([_set(good, "affected_ids", ids)], truth_of(issue)), 1, 0, 0)

    def test_fx_evidence_in_every_accepted_form(self):
        issue = golden_issue("XT92KR")  # evidence carries fx/2026-09-14/USD/KZT
        good = finding_for(issue)
        rest = [x for x in good["evidence"] if not x.startswith("fx/")]
        for fx_ref, tp in (("fx/2026-09-14/USD/KZT", 1), ("2026-09-14/USD/KZT", 1),
                           ("ledger/2026-09-14/USD/KZT", 1), ("fx/fx/2026-09-14/USD/KZT", 0),
                           ("fx/2026-09-15/USD/KZT", 0), ("FX/2026-09-14/USD/KZT", 0),
                           ("fx/2026-09-14/usd/kzt", 0)):
            with self.subTest(fx_ref=fx_ref):
                f = _set(good, "evidence", rest + [fx_ref])
                self.assertCounts(self.score([f], truth_of(issue)), tp, 1 - tp, 1 - tp)

    def test_case_double_prefix_unknown_prefix_and_spaces_do_not_match(self):
        issue = golden_issue("G21S27")
        good = finding_for(issue)
        for first in ("INVOICE/1TNM24", "Invoice/1TNM24", "invoice/1tnm24", "1tnm24",
                      "invoice/invoice/1TNM24", "invoices/1TNM24", "je/1TNM24",
                      " 1TNM24", "1TNM24 ", "invoice/ 1TNM24", "invoice:1TNM24"):
            with self.subTest(first=first):
                f = _set(good, "affected_ids", [first, "invoice/F7SF66"])
                self.assertCounts(self.score([f], truth_of(issue)), 0, 1, 1)

    def test_prefix_noise_in_evidence_is_normalised_too(self):
        issue = golden_issue("G21S27")
        good = finding_for(issue)
        ev = ["doc/1TNM24", "F7SF66", "invoice/CKBBZ4", "payment/KBYZZ8"]
        self.assertCounts(self.score([_set(good, "evidence", ev)], truth_of(issue)), 1, 0, 0)
        ev = ["INVOICE/1TNM24", "F7SF66", "ledger/CKBBZ4", "ledger/KBYZZ8"]
        self.assertCounts(self.score([_set(good, "evidence", ev)], truth_of(issue)), 0, 1, 1)

    def test_je_ids_are_compared_as_is(self):
        issue = make_issue(type_="period_cutoff", affected=("je/A12BT6",), amount=187500000, currency="KZT",
                           evidence_sets=(("invoice/VX37TR", "je/A12BT6"),))
        good = finding_for(issue)
        for ids, tp in ((["je/A12BT6"], 1), (["ledger/je/A12BT6"], 1), (["A12BT6"], 0),
                        (["ledger/A12BT6"], 0), (["JE/A12BT6"], 0)):
            with self.subTest(ids=ids):
                self.assertCounts(self.score([_set(good, "affected_ids", ids)], truth_of(issue)), tp, 1 - tp, 1 - tp)


# --------------------------------------------------------------------------- 3.6 / 5.2 truth validation

def _bad_truths():
    ok = make_issue()
    out = [
        ("clean true with issues", {"clean": True, "issues": [ok]}),
        ("clean false without issues", {"clean": False, "issues": []}),
        ("duplicate issue_id", truth_of(make_issue("AAAAAA"), make_issue("AAAAAA", amount=1))),
        ("duplicate issue_id, different types",
         truth_of(make_issue("AAAAAA"), make_issue("AAAAAA", type_="fx_rate", affected=("ledger/QQ5GHK",)))),
    ]
    for amount in (0, -1, -325000, 1.5, 325000.0, "325000", True, None, [325000]):
        out.append((f"expected_amount_minor={amount!r}", truth_of(make_issue(amount=amount))))
    out.append(("expected_amount_minor missing", truth_of({k: v for k, v in ok.items()
                                                            if k != "expected_amount_minor"})))
    eleven = public_record_ids()[:11]
    out.append(("empty evidence set", truth_of(make_issue(evidence_sets=((),)))))
    out.append(("evidence set of 11", truth_of(make_issue(evidence_sets=(eleven,)))))
    out.append(("second evidence set empty",
                truth_of(make_issue(evidence_sets=(("invoice/1TNM24",), ())))))
    out.append(("second evidence set of 11",
                truth_of(make_issue(evidence_sets=(("invoice/1TNM24",), eleven)))))
    g = golden()
    g["issues"][3]["issue_id"] = g["issues"][0]["issue_id"]
    out.append(("golden with a repeated issue_id", g))
    g = golden()
    g["clean"] = True
    out.append(("golden marked clean", g))
    return out


class TruthValidationTest(GraderCase):
    def test_bad_truth_raises_value_error_in_parse_truth(self):
        for name, t in _bad_truths():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    parse_truth(copy.deepcopy(t))

    def test_bad_truth_raises_value_error_in_score(self):
        f = finding_for(make_issue())
        for name, t in _bad_truths():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    self.score([], t, 0)
                with self.assertRaises(ValueError):
                    self.score([f], t, 0)

    def test_bad_truth_raises_value_error_in_timeout_score(self):
        for name, t in _bad_truths():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    timeout_score(copy.deepcopy(t))

    def test_valid_truths_are_accepted(self):
        ten = public_record_ids()[:10]
        cases = {
            "golden": golden(),
            "clean": clean_truth(),
            "one-id evidence set": truth_of(make_issue(evidence_sets=(("invoice/1TNM24",),))),
            "ten-id evidence set": truth_of(make_issue(evidence_sets=(ten,))),
            "amount 1": truth_of(make_issue(amount=1)),
            "no tolerance_minor": truth_of(make_issue(tolerance=MISSING)),
            "extra A fields": truth_of(make_issue(scenario_id="S1", tolerance=1, note="x")),
            "extra top-level fields": dict(golden(), episode_id="KARPSR"),
        }
        for name, t in cases.items():
            with self.subTest(name):
                parse_truth(copy.deepcopy(t))
                self.score([], t, 0)


# --------------------------------------------------------------------------- 5.3 assignment

class AssignmentTest(GraderCase):
    def test_repeat_of_a_correct_finding_is_fp(self):
        issue = golden_issue("G21S27")
        f = finding_for(issue)
        for n in (2, 3, 5):
            with self.subTest(copies=n):
                res = self.score([f] * n, truth_of(issue))
                self.assertCounts(res, 1, n - 1, 0)
                self.assertEqual(matched_pairs(res), {("G21S27", 0)})

    def test_repeat_with_different_spelling_is_still_fp(self):
        issue = golden_issue("G21S27")
        f = finding_for(issue)
        g = dict(f, affected_ids=["F7SF66", "1TNM24"], amount_minor=f["amount_minor"] + 1,
                 evidence=list(reversed(f["evidence"])) + ["ledger/0N1PQJ"])
        res = self.score([g, f], truth_of(issue))
        self.assertCounts(res, 1, 1, 0)
        self.assertEqual(matched_pairs(res), {("G21S27", 0)})

    def test_c1_period_and_fx_share_ids_and_type_decides(self):
        t = truth_of(golden_issue("XG1494"), golden_issue("ZHB4FR"))
        period = finding_for(golden_issue("XG1494"))
        fx = finding_for(golden_issue("ZHB4FR"))
        self.assertEqual(period["affected_ids"], fx["affected_ids"])
        res = self.score([period, fx], t)
        self.assertCounts(res, 2, 0, 0)
        self.assertEqual(matched_pairs(res), {("XG1494", 0), ("ZHB4FR", 1)})
        res = self.score([fx], t)
        self.assertCounts(res, 1, 0, 1)
        self.assertEqual(matched_pairs(res), {("ZHB4FR", 0)})
        swapped = [dict(period, type="fx_rate"), dict(fx, type="period_cutoff")]
        self.assertCounts(self.score(swapped, t), 0, 2, 2)

    def test_c1_period_finding_with_fx_evidence_superset(self):
        t = truth_of(golden_issue("XG1494"), golden_issue("ZHB4FR"))
        period = finding_for(golden_issue("XG1494"))
        fx = finding_for(golden_issue("ZHB4FR"))
        f = dict(period, evidence=fx["evidence"])  # superset of the period set
        res = self.score([f], t)
        self.assertCounts(res, 1, 0, 1)
        self.assertEqual(matched_pairs(res), {("XG1494", 0)})

    def _two_issue_world(self):
        a, b, c = "ledger/CKBBZ4", "ledger/KBYZZ8", "ledger/M5DC58"
        base = dict(type_="duplicate_invoice", affected=("invoice/1TNM24", "invoice/F7SF66"),
                    amount=325000, currency="EUR")
        ia = make_issue("AAAAAA", evidence_sets=((a,),), **base)
        ib = make_issue("BBBBBB", evidence_sets=((b,),), **base)
        f_both = dict(finding_for(ia), evidence=[a, b, c])
        f_only_a = dict(finding_for(ia), evidence=[a])
        return ia, ib, f_both, f_only_a

    def test_maximum_matching_beats_greedy(self):
        ia, ib, f_both, f_only_a = self._two_issue_world()
        for order, expected in (((f_both, f_only_a), {("AAAAAA", 1), ("BBBBBB", 0)}),
                                ((f_only_a, f_both), {("AAAAAA", 0), ("BBBBBB", 1)})):
            for issues in ((ia, ib), (ib, ia)):
                with self.subTest(order=[len(f["evidence"]) for f in order],
                                  issues=[i["issue_id"] for i in issues]):
                    res = self.score(list(order), truth_of(*issues))
                    self.assertCounts(res, 2, 0, 0)
                    self.assertEqual(matched_pairs(res), expected)

    def test_tie_goes_to_lower_issue_id(self):
        base = make_issue("ZZZZZZ")
        twin = dict(base, issue_id="AAAAAA")
        f = finding_for(base)
        for issues in ((base, twin), (twin, base)):
            with self.subTest(first=issues[0]["issue_id"]):
                res = self.score([f], truth_of(*issues))
                self.assertCounts(res, 1, 0, 1)
                self.assertEqual(matched_pairs(res), {("AAAAAA", 0)})

    def test_tie_with_two_findings_is_deterministic(self):
        # SPEC 5.3 says ties go "by issue_id order, then by finding number" but does
        # not say whether that is the outcome (AAAAAA<-0, ZZZZZZ<-1) or only the
        # processing order. The grader returns AAAAAA<-1, ZZZZZZ<-0 (augmenting
        # path re-routes the first issue). Reward is unaffected, so assert only what
        # the spec fixes: a full matching that does not depend on truth order.
        base = make_issue("ZZZZZZ")
        twin = dict(base, issue_id="AAAAAA")
        f = finding_for(base)
        seen = []
        for issues in ((base, twin), (twin, base)):
            with self.subTest(first=issues[0]["issue_id"]):
                res = self.score([f, copy.deepcopy(f)], truth_of(*issues))
                self.assertCounts(res, 2, 0, 0)
                pairs = matched_pairs(res)
                self.assertEqual({p[0] for p in pairs}, {"AAAAAA", "ZZZZZZ"})
                self.assertEqual({p[1] for p in pairs}, {0, 1})
                seen.append(pairs)
        self.assertEqual(seen[0], seen[1])

    def test_issue_order_in_truth_does_not_matter(self):
        t = golden()
        findings = oracle_findings(t)[:5] + junk(2) + oracle_findings(t)[:2]
        ref = self.score(findings, t)
        rng = random.Random(7)
        for k in range(5):
            shuffled = copy.deepcopy(t)
            rng.shuffle(shuffled["issues"])
            with self.subTest(k=k):
                res = self.score(findings, shuffled)
                self.assertEqual(counts(res), counts(ref))
                self.assertEqual(matched_pairs(res), matched_pairs(ref))
                self.assertEqual(field(res, "reward"), field(ref, "reward"))

    def test_finding_order_changes_nothing_but_numbers(self):
        t = golden()
        findings = oracle_findings(t) + junk(3)
        rng = random.Random(11)
        for k in range(5):
            perm = list(range(len(findings)))
            rng.shuffle(perm)
            with self.subTest(k=k):
                res = self.score([findings[p] for p in perm], t)
                self.assertCounts(res, 8, 3, 0)
                expected = {(t["issues"][j]["issue_id"], perm.index(j)) for j in range(8)}
                self.assertEqual(matched_pairs(res), expected)

    def test_same_input_same_output(self):
        t = golden()
        findings = oracle_findings(t)[:6] + junk(4)
        a = self.score(findings, t, 17)
        b = self.score(findings, t, 17)
        for name in RESULT_FIELDS:
            if name == "matched":
                self.assertEqual(matched_pairs(a), matched_pairs(b))
            else:
                self.assertEqual(field(a, name), field(b, name), name)


# --------------------------------------------------------------------------- 5.4 reward

class RewardTest(GraderCase):
    def test_forty_steps_give_exactly_092(self):
        t = golden()
        self.assertReward(self.score(oracle_findings(t), t, 40), 0.92)
        self.assertReward(self.score([], clean_truth(), 40), 0.92)

    def test_clean_company_silence(self):
        res = self.score([], clean_truth(), 0)
        self.assertCounts(res, 0, 0, 0)
        self.assertReward(res, 1)
        self.assertAlmostEqual(float(field(res, "f1")), 1.0)
        self.assertAlmostEqual(float(field(res, "precision")), 1.0)
        self.assertAlmostEqual(float(field(res, "recall")), 1.0)

    def test_clean_company_false_positive(self):
        for n, steps in ((1, 0), (1, 30), (5, 0)):
            with self.subTest(n=n, steps=steps):
                res = self.score(junk(n), clean_truth(), steps)
                self.assertCounts(res, 0, n, 0)
                self.assertReward(res, -1)
                self.assertAlmostEqual(float(field(res, "f1")), 0.0)
                self.assertAlmostEqual(float(field(res, "precision")), 0.0)
                self.assertAlmostEqual(float(field(res, "recall")), 1.0)

    def test_silence_on_dirty_company(self):
        res = self.score([], golden(), 0)
        self.assertCounts(res, 0, 0, 8)
        self.assertReward(res, -1)
        self.assertAlmostEqual(float(field(res, "f1")), 0.0)
        self.assertAlmostEqual(float(field(res, "precision")), 1.0)
        self.assertAlmostEqual(float(field(res, "recall")), 0.0)

    def test_step_cost_and_clipping(self):
        t = golden()
        oracle = oracle_findings(t)
        for steps, expected in ((0, 1), (1, Fraction(499, 500)), (13, Fraction(487, 500)),
                                (100, Fraction(4, 5)), (999, Fraction(-499, 500)),
                                (1000, -1), (1500, -1), (10 ** 6, -1)):
            with self.subTest(steps=steps):
                self.assertReward(self.score(oracle, t, steps), expected)

    def test_exact_fraction_arithmetic(self):
        # Cases where naive float arithmetic lands one ulp off the exact value.
        t = golden()
        oracle = oracle_findings(t)
        for tp, fp, steps in ((1, 1, 40), (1, 1, 33), (1, 3, 0), (1, 3, 7), (1, 0, 25), (1, 2, 100)):
            with self.subTest(tp=tp, fp=fp, steps=steps):
                res = self.score(oracle[:tp] + junk(fp), t, steps)
                self.assertCounts(res, tp, fp, 8 - tp)
                self.assertReward(res, exact_reward(tp, fp, 8 - tp, steps))
        self.assertReward(self.score(oracle[:1] + junk(1), t, 40), -0.68)
        self.assertReward(self.score(oracle[:1] + junk(3), t, 0), Fraction(-2, 3))

    def test_reward_grid(self):
        t = golden()
        oracle = oracle_findings(t)
        for tp in range(9):
            for fp in range(4):
                for steps in (0, 13, 25, 40, 100):
                    with self.subTest(tp=tp, fp=fp, steps=steps):
                        res = self.score(oracle[:tp] + junk(fp), t, steps)
                        fn = 8 - tp
                        self.assertCounts(res, tp, fp, fn)
                        self.assertReward(res, exact_reward(tp, fp, fn, steps))
                        p = Fraction(1) if tp + fp == 0 else Fraction(tp, tp + fp)
                        r = Fraction(tp, tp + fn)
                        self.assertAlmostEqual(float(field(res, "precision")), float(p), places=12)
                        self.assertAlmostEqual(float(field(res, "recall")), float(r), places=12)
                        self.assertAlmostEqual(float(field(res, "f1")), float(exact_f1(tp, fp, fn)), places=12)


class TimeoutTest(GraderCase):
    def test_clean_timeout_is_08_and_below_silence(self):
        res = timeout_score(clean_truth())
        self.assertCounts(res, 0, 0, 0)
        self.assertReward(res, 0.8)
        self.assertLess(field(res, "reward"), field(self.score([], clean_truth(), 0), "reward"))

    def test_dirty_timeout(self):
        res = timeout_score(golden())
        self.assertCounts(res, 0, 0, 8)
        self.assertReward(res, -1)

    def test_timeout_equals_empty_answer_at_100_steps(self):
        for t in (clean_truth(), golden(), single("G21S27")):
            with self.subTest(issues=len(t["issues"])):
                a = timeout_score(copy.deepcopy(t))
                b = self.score([], t, 100)
                for name in ("tp", "fp", "fn", "reward"):
                    self.assertEqual(field(a, name), field(b, name), name)
                self.assertEqual(invalid(a), [])
                self.assertEqual(matched_pairs(a), set())


# --------------------------------------------------------------------------- QA reward table (reconstructed)

def _qa_rows():
    """Attack table reconstructed from SPEC 5.2 / 5.4 text; STRICT column computed here.

    The 23 cases of qa_reward.py are not in the repo, so this is our own table
    of the attacks the spec names: type+ID only, ID spraying, intersection
    credit, duplicates, tolerance edges, format tricks, timeout, step cost.
    """
    t = golden()
    o = oracle_findings(t)
    ids = public_record_ids()
    rows = []

    def row(name, findings, truth, steps, tp, fp, fn):
        rows.append((name, findings, truth, steps, tp, fp, fn, exact_reward(tp, fp, fn, steps)))

    row("oracle", o, t, 0, 8, 0, 0)
    row("oracle at 40 steps", o, t, 40, 8, 0, 0)
    row("empty answer", [], t, 0, 0, 0, 8)
    row("timeout on dirty", [], t, 100, 0, 0, 8)
    row("clean, empty", [], clean_truth(), 0, 0, 0, 0)
    row("clean, timeout", [], clean_truth(), 100, 0, 0, 0)
    row("clean, one FP", junk(1), clean_truth(), 0, 0, 1, 0)
    row("oracle twice", o + o, t, 0, 8, 8, 0)
    row("half oracle", o[:4], t, 0, 4, 0, 4)
    row("type and ids only, amount 0", [dict(f, amount_minor=0) for f in o], t, 0, 0, 8, 8)
    row("type and ids only, no evidence", [dict(f, evidence=[]) for f in o], t, 0, 0, 8, 8)
    row("wrong currency", [dict(f, currency="RUB") for f in o], t, 0, 0, 8, 8)
    row("amount off by one", [dict(f, amount_minor=f["amount_minor"] + 1) for f in o], t, 0, 8, 0, 0)
    row("amount off by two", [dict(f, amount_minor=f["amount_minor"] - 2) for f in o], t, 0, 0, 8, 8)
    row("bare ids", [dict(f, affected_ids=[normalize_ref(x) for x in f["affected_ids"]],
                          evidence=[normalize_ref(x) for x in f["evidence"]]) for f in o], t, 0, 8, 0, 0)
    swap = {"period_cutoff": "fx_rate", "fx_rate": "period_cutoff"}
    c1 = [dict(f, type=swap[f["type"]]) if f["affected_ids"] == ["ledger/Z3H7BB"] else f for f in o]
    row("C1 period/fx types swapped", c1, t, 0, 6, 2, 2)
    row("id spray, one finding per type, all ids", [
        {"type": ty, "affected_ids": ids, "amount_minor": 1, "currency": "KZT", "evidence": ids[:10]}
        for ty in TYPES], t, 0, 0, 4, 8)
    row("id spray, oracle with every id in affected_ids",
        [dict(f, affected_ids=ids) for f in o], t, 0, 0, 8, 8)
    row("intersection credit: one of two ids",
        [dict(f, affected_ids=f["affected_ids"][:1]) for f in o if len(f["affected_ids"]) == 2], t, 0, 0, 4, 8)
    row("evidence padded to 10", [dict(f, evidence=list(f["evidence"]) + noise_ids(f["evidence"], 10 - len(f["evidence"])))
                                  for f in o], t, 0, 8, 0, 0)
    row("evidence padded to 11", [dict(f, evidence=list(f["evidence"]) + noise_ids(f["evidence"], 11 - len(f["evidence"])))
                                  for f in o], t, 0, 0, 8, 8)
    row("amount as string", [dict(f, amount_minor=str(f["amount_minor"])) for f in o], t, 0, 0, 8, 8)
    row("amount as float", [dict(f, amount_minor=float(f["amount_minor"])) for f in o], t, 0, 0, 8, 8)
    row("lowercase currency", [dict(f, currency=f["currency"].lower()) for f in o], t, 0, 0, 8, 8)
    row("oracle plus 100 junk", o + junk(100), t, 0, 8, 100, 0)
    row("per-record spray of all types", [
        {"type": ty, "affected_ids": [r], "amount_minor": 1, "currency": "KZT", "evidence": [r]}
        for r in ids for ty in TYPES], t, 0, 0, 300, 8)
    return rows


class QaRewardTableTest(GraderCase):
    def test_strict_column(self):
        for name, findings, truth, steps, tp, fp, fn, reward in _qa_rows():
            with self.subTest(name):
                res = self.score(findings, truth, steps)
                self.assertCounts(res, tp, fp, fn)
                self.assertReward(res, reward)

    def test_no_cheap_attack_beats_silence_on_dirty_company(self):
        t = golden()
        silence = field(self.score([], t, 0), "reward")
        for name, findings, truth, steps, tp, fp, fn, reward in _qa_rows():
            if truth is not t and truth != t:
                continue
            if tp == 0:
                with self.subTest(name):
                    self.assertLessEqual(field(self.score(findings, truth, steps), "reward"), silence)


# --------------------------------------------------------------------------- reference matcher fuzz (N-version)

def ref_valid(f):
    if not isinstance(f, dict):
        return False
    t = f.get("type")
    if not isinstance(t, str) or t not in TYPES:
        return False
    a = f.get("affected_ids")
    if not isinstance(a, list) or not a or not all(isinstance(x, str) for x in a):
        return False
    m = f.get("amount_minor", MISSING)
    if type(m) is not int or m < 0:
        return False
    c = f.get("currency")
    if not isinstance(c, str) or len(c) != 3 or not all("A" <= ch <= "Z" for ch in c):
        return False
    e = f.get("evidence")
    if not isinstance(e, list) or len(e) > 10 or not all(isinstance(x, str) for x in e):
        return False
    return True


def ref_admissible(f, issue):
    if f["type"] != issue["type"] or f["currency"] != issue["currency"]:
        return False
    if {normalize_ref(x) for x in f["affected_ids"]} != {normalize_ref(x) for x in issue["affected_ids"]}:
        return False
    if abs(f["amount_minor"] - issue["expected_amount_minor"]) > issue.get("tolerance_minor", 1):
        return False
    ev = {normalize_ref(x) for x in f["evidence"]}
    return any({normalize_ref(x) for x in s} <= ev for s in issue["evidence_sets"])


def ref_score(findings, truth, steps):
    issues = truth["issues"]
    valid = [ref_valid(f) for f in findings]
    adj = [[j for j, i in enumerate(issues) if valid[k] and ref_admissible(f, i)] for k, f in enumerate(findings)]
    owner = {}

    def augment(k, seen):
        for j in adj[k]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = k
                return True
        return False

    tp = sum(1 for k in range(len(findings)) if augment(k, set()))
    fp = len(findings) - tp
    fn = len(issues) - tp
    return tp, fp, fn, [k for k, v in enumerate(valid) if not v], exact_reward(tp, fp, fn, steps)


def _mutate(rng, truth):
    ids = public_record_ids()
    out = []
    for f in oracle_findings(truth):
        if rng.random() < 0.15:
            continue
        g = copy.deepcopy(f)
        op = rng.randrange(16)
        if op == 0:
            g["amount_minor"] += rng.choice((-3, -2, -1, 1, 2, 3))
        elif op == 1:
            g["currency"] = rng.choice(("KZT", "USD", "EUR", "usd", "US", "USDT"))
        elif op == 2:
            g["type"] = rng.choice(TYPES)
        elif op == 3 and g["evidence"]:
            g["evidence"].pop(rng.randrange(len(g["evidence"])))
        elif op == 4:
            g["evidence"] += rng.sample(ids, rng.randint(0, 9))
        elif op == 5:
            g["affected_ids"] = [normalize_ref(x) for x in g["affected_ids"]]
            g["evidence"] = [normalize_ref(x) for x in g["evidence"]]
        elif op == 6:
            g["affected_ids"] = [rng.choice(KNOWN_PREFIXES) + "/" + normalize_ref(x) for x in g["affected_ids"]]
        elif op == 7:
            g["affected_ids"].append(rng.choice(ids))
        elif op == 8:
            g["amount_minor"] = rng.choice((float(g["amount_minor"]), str(g["amount_minor"]), True, -1))
        elif op == 9:
            out.append(copy.deepcopy(g))
        elif op == 10 and len(g["affected_ids"]) > 1:
            g["affected_ids"].pop()
        elif op == 11:
            g["affected_ids"] = [x.upper() if rng.random() < 0.5 else x for x in g["affected_ids"]]
        elif op == 12:
            g["evidence"] = g["evidence"] * 3
        elif op == 13:
            g["affected_ids"] = g["affected_ids"] * 2
        out.append(g)
    garbage = [None, 1, "x", [], {}, {"type": "fx_rate"}, junk(1)[0]]
    for _ in range(rng.randint(0, 3)):
        out.append(copy.deepcopy(rng.choice(garbage)))
    rng.shuffle(out)
    return out


class ReferenceFuzzTest(GraderCase):
    def test_counts_invalid_and_reward_agree_with_reference(self):
        base = golden()
        for seed in range(400):
            rng = random.Random(seed)
            t = copy.deepcopy(base)
            for i in t["issues"]:
                tol = rng.choice((0, 1, 2, MISSING))
                if tol is MISSING:
                    i.pop("tolerance_minor", None)
                else:
                    i["tolerance_minor"] = tol
            findings = _mutate(rng, t)
            steps = rng.choice((0, 1, 17, 40, 99, 100))
            tp, fp, fn, bad, reward = ref_score(findings, t, steps)
            with self.subTest(seed=seed):
                res = self.score(findings, t, steps)
                self.assertCounts(res, tp, fp, fn)
                self.assertEqual(invalid(res), bad)
                self.assertReward(res, reward)
                pairs = matched_pairs(res)
                self.assertEqual(len(pairs), tp)
                # every reported pair must be admissible and one-to-one
                self.assertEqual(len({p[0] for p in pairs}), tp)
                self.assertEqual(len({p[1] for p in pairs}), tp)
                by_id = {i["issue_id"]: i for i in t["issues"]}
                for issue_id, k in pairs:
                    self.assertTrue(ref_valid(findings[k]) and ref_admissible(findings[k], by_id[issue_id]),
                                    (issue_id, k))


# --------------------------------------------------------------------------- helper functions (signature not in SPEC)

class HelperFunctionsTest(unittest.TestCase):
    def test_f1_score_from_counts(self):
        params = list(inspect.signature(f1_score).parameters)
        if params[:3] != ["tp", "fp", "fn"]:
            self.skipTest(f"SPEC does not fix f1_score signature; got {params}")
        for tp, fp, fn in ((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (4, 0, 4), (8, 0, 0), (1, 3, 7)):
            with self.subTest(tp=tp, fp=fp, fn=fn):
                self.assertEqual(Fraction(f1_score(tp, fp, fn)), exact_f1(tp, fp, fn))

    def test_terminal_reward(self):
        params = list(inspect.signature(terminal_reward).parameters)
        if len(params) != 2 or params[0] != "f1":
            self.skipTest(f"SPEC does not fix terminal_reward signature; got {params}")
        for f1, steps, expected in ((Fraction(1), 40, 0.92), (Fraction(1), 0, 1.0), (Fraction(0), 0, -1.0),
                                    (Fraction(2, 3), 0, float(Fraction(1, 3))), (Fraction(1), 1500, -1.0)):
            with self.subTest(f1=f1, steps=steps):
                self.assertEqual(terminal_reward(f1, steps), expected)


if __name__ == "__main__":
    unittest.main()
