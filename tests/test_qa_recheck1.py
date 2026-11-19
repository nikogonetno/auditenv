"""QA re-check 1 of the grader, commit 1f98152. Drop into tests/ as is.
D-4 (two truth issues with equal type and affected_ids) belongs to the verifier INV-18,
not to parse_truth: the matching tests of B need such truths on purpose.

On 1f98152 two tests fail on purpose (D-1, D-2) and T-1 passes, it only adds
the missing coverage for the negative tolerance guard. After B's fix all three pass.
"""
import copy
import unittest

from auditenv.grader import parse_truth, score
from tests.golden import oracle_findings, truth_v2


class QaRecheck1(unittest.TestCase):
    def setUp(self):
        self.truth = truth_v2()
        self.oracle = oracle_findings(self.truth)

    def test_d1_type_must_be_exact_str(self):
        class AnyType(str):
            def __eq__(self, other):
                return True

            __hash__ = str.__hash__

        issue = self.truth["issues"][0]
        finding = copy.deepcopy(self.oracle[0])
        wrong = "fx_rate" if issue["type"] != "fx_rate" else "period_cutoff"
        finding["type"] = AnyType(wrong)
        result = score([finding], {"clean": False, "issues": [issue]}, 0)
        self.assertEqual((result["tp"], result["fp"], result["invalid_findings"]), (0, 1, [0]))

    def test_d2_je_ids_rejected_after_prefix_is_stripped(self):
        truth = copy.deepcopy(self.truth)
        truth["issues"][0]["affected_ids"] = ["ledger/je/A12BT6"]
        with self.assertRaises(ValueError):
            parse_truth(truth)

    def test_t1_negative_or_non_int_tolerance_rejected(self):
        for bad in (-1, 1.0, True, "1"):
            with self.subTest(tolerance=bad):
                truth = copy.deepcopy(self.truth)
                truth["issues"][0]["tolerance_minor"] = bad
                with self.assertRaises(ValueError):
                    parse_truth(truth)


if __name__ == "__main__":
    unittest.main()
