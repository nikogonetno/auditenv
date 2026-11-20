"""Baseline «Простой алгоритм сверки», policies/reconcile.py (case §06; docs/tasks/B_next.md item 6).

The policy is naive on purpose. On the case §03 test book (tests/fixtures/mini_v03, 8 errors and
correct look-alike records) it must land strictly between silence and the oracle: some findings
right, some wrong, and at least one of the wrong ones on a correct look-alike record. It also has
to stay inside the 100-step budget, submit a well-formed answer, be byte-for-byte repeatable, and
score nothing but false positives on a clean company.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from auditenv.env import AuditEnv
from auditenv.grader import MAX_STEPS
from auditenv.run import SUBPROCESS_POLICIES
from auditenv.runner import REPO_ROOT, run_inprocess_episode, run_subprocess_episode
from policies.oracle import make_env_oracle
from policies.reconcile import MAX_EVIDENCE, MAX_FINDINGS, reconcile

from tests.tiny_episode import EPISODE_ID, submit, write_episode

MINI = REPO_ROOT / "tests" / "fixtures" / "mini_v03"
CLEAN_TRUTH = {"clean": True, "episode_id": EPISODE_ID, "issues": []}

# What the naive rules get wrong on mini_v03 (see docs/DOMAIN_SPEC_A.md, «Корректные двойники» and C2).
FALSE_ON_MINI = {
    # monthly subscription: same vendor and total, different vendor_ref and service period
    ("duplicate_invoice", ("invoice/G0QJMF", "invoice/P18WZP")),
    # C2: the payment names both copies of a duplicated bill and is cleared to the copy
    ("payment_allocation", ("invoice/RRH8VY", "payment/AM952T")),
    # midnight in UTC: 21:10 +02:00 on 30 September is already 1 October for the company
    ("period_cutoff", ("ledger/TA1NB3",)),
    # weekend document: Saturday 26 September takes Friday's rate, the rule takes Monday's quote
    ("fx_rate", ("ledger/A35WK1",)),
}
# C1: the payment's UTC date is inside September, its company-local date is not.
MISSED_ON_MINI = {("period_cutoff", ("ledger/Z3H7BB",))}


def reconcile_cmd(seed=0):
    return [sys.executable, "-m", "policies.reconcile", str(seed)]


def silence(_message):
    return submit([])


def key(finding):
    return finding["type"], tuple(sorted(finding["affected_ids"]))


def run(config, mode="eval", seed=0):
    """(grader entry, runner result, trajectory rows) of one reconcile episode."""
    env, trajectory = AuditEnv(mode), io.StringIO()
    result = run_subprocess_episode(reconcile_cmd(seed), env, 1, config, trajectory=trajectory)
    rows = [json.loads(line) for line in trajectory.getvalue().splitlines()]
    return env.grader_log[-1], result, rows


def submitted(rows):
    return rows[-1]["action"]["findings"]["findings"]


def grade_of(make_policy, config, mode="eval"):
    """Grader entry of an in-process policy; make_policy(env) returns the policy function."""
    env = AuditEnv(mode)
    run_inprocess_episode(make_policy(env), env, 1, config)
    return env.grader_log[-1]


def make_silence(_env):
    return silence


class ReconcileCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.mini = {"episode_dir": str(MINI)}
        self.tiny = {"episode_dir": str(write_episode(self.root / "tiny"))}

    def tearDown(self):
        self.tmp.cleanup()


class Protocol(ReconcileCase):
    def test_finishes_with_a_well_formed_submit(self):
        for name, config in (("mini_v03", self.mini), ("tiny", self.tiny)):
            for mode in ("train", "eval"):
                with self.subTest(book=name, mode=mode):
                    entry, result, rows = run(config, mode)
                    self.assertEqual((result["terminated"], result["truncated"]), (True, False))
                    self.assertLessEqual(result["steps"], MAX_STEPS)
                    self.assertEqual(rows[-1]["action"]["action"], "submit")
                    self.assertEqual(entry["end"], "submit")
                    self.assertNotIn("error", entry)
                    self.assertEqual(entry["invalid_findings"], [])
                    findings = submitted(rows)
                    self.assertLessEqual(len(findings), MAX_FINDINGS)
                    for finding in findings:
                        self.assertIs(type(finding["claimed_amount_minor"]), int)
                        self.assertGreater(finding["claimed_amount_minor"], 0)
                        self.assertLessEqual(len(finding["evidence_ids"]), MAX_EVIDENCE)

    def test_pages_every_table_to_the_end_then_submits(self):
        _, result, rows = run(self.mini)
        actions = [row["action"] for row in rows[1:]]
        self.assertEqual([a["table"] for a in actions[:-1]], ["ledger", "invoices", "payments", "fx", "fx"])
        self.assertTrue(all(a["action"] == "list_records" and a["limit"] == 50 for a in actions[:-1]))
        self.assertEqual(["cursor" in a for a in actions[:-1]], [False, False, False, False, True])
        self.assertEqual(result["steps"], 6)

    def test_stops_reading_in_time_to_submit_on_a_long_book(self):
        """5000 ledger rows need 100 pages: it reads 99 of them and submits on the last step."""
        config = {"episode_dir": str(write_episode(self.root / "long", ledger_rows=5000))}
        entry, result, rows = run(config)
        self.assertEqual((result["terminated"], result["truncated"], result["steps"]), (True, False, MAX_STEPS))
        self.assertEqual([row["action"]["action"] for row in rows[1:]], ["list_records"] * 99 + ["submit"])
        self.assertEqual((entry["end"], entry["inspection_steps"]), ("submit", 99))
        self.assertNotIn("error", entry)

    def test_two_runs_give_byte_identical_trajectories(self):
        """Deterministic, and the seed argument is accepted but unused."""
        files = []
        for seed in (0, 0, 7):
            path = self.root / f"traj{len(files)}.jsonl"
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                run_subprocess_episode(reconcile_cmd(seed), AuditEnv("train"), 1, self.mini, trajectory=f)
            files.append(path.read_bytes())
        self.assertEqual(files[0], files[1])
        self.assertEqual(files[0], files[2])
        self.assertGreater(len(files[0].splitlines()), 2)


class Grade(ReconcileCase):
    def test_on_the_test_book_it_sits_between_silence_and_the_oracle(self):
        entry, _, _ = run(self.mini)
        self.assertGreater(entry["tp"], 0)
        self.assertGreater(entry["fp"], 0)
        quiet = grade_of(make_silence, self.mini)["reward"]
        oracle = grade_of(make_env_oracle, self.mini)["reward"]
        self.assertEqual(oracle, 1.0)
        self.assertLess(quiet, entry["reward"])
        self.assertLess(entry["reward"], oracle)

    def test_grade_on_the_test_book(self):
        """Observed on mini_v03: 7 TP, 3 FP, 1 FN after 5 reads, reward 2·7/9 − 1 − 0.002·5."""
        entry, _, _ = run(self.mini)
        self.assertEqual((entry["tp"], entry["fp"], entry["fn"], entry["inspection_steps"]), (7, 4, 1, 5))
        self.assertEqual(entry["reward"], float(2 * Fraction(14, 19) - 1 - Fraction(2, 1000) * 5))

    def test_falls_for_correct_look_alike_records(self):
        _, _, rows = run(self.mini)
        truth = json.loads((MINI / "private" / "truth.json").read_text(encoding="utf-8"))
        expected = {(i["type"], tuple(sorted(i["affected_ids"]))) for i in truth["issues"]}
        made = {key(f) for f in submitted(rows)}
        self.assertEqual(made - expected, FALSE_ON_MINI)
        self.assertEqual(expected - made, MISSED_ON_MINI)

    def test_on_a_clean_company_every_finding_is_false(self):
        config = {"episode_dir": str(write_episode(self.root / "clean", truth=CLEAN_TRUTH))}
        entry, _, rows = run(config)
        findings = submitted(rows)
        self.assertGreater(len(findings), 0)  # the tiny book has a payment that leaves its invoice open
        self.assertEqual((entry["tp"], entry["fp"], entry["fn"]), (0, len(findings), 0))
        self.assertLess(entry["reward"], grade_of(make_silence, config)["reward"])


def manifest(**extra):
    return {"episode_id": "EPRULE", "base_currency": "KZT", "timezone": "+05:00",
            "period_start": "2026-09-01", "period_end": "2026-09-30", **extra}


def invoice(invoice_id, issued_at="2026-09-10T10:00:00+05:00", currency="KZT", total=100000, **extra):
    return {"doc_id": f"D{invoice_id}", "invoice_id": invoice_id, "entity_id": "E1", "issued_at": issued_at,
            "currency": currency, "total_minor": total, "direction": "AR", **extra}


def row(row_id, account, base, invoice_id=None, payment_id=None, timestamp="2026-09-10T12:00:00+05:00",
        doc_currency="KZT", doc_amount=None, rate="1"):
    return {"row_id": row_id, "timestamp": timestamp, "account": account, "debit_minor": base, "credit_minor": 0,
            "currency": "KZT", "entity_id": "E1", "invoice_id": invoice_id, "payment_id": payment_id,
            "doc_currency": doc_currency, "doc_amount_minor": base if doc_amount is None else doc_amount,
            "exchange_rate": rate}


def tables(invoices=(), payments=(), ledger=(), fx=()):
    return {"invoices": list(invoices), "payments": list(payments), "ledger": list(ledger), "fx": list(fx)}


class Rules(unittest.TestCase):
    """The four rules on hand-made records, through the pure function reconcile(manifest, tables)."""

    def test_rate_rounds_half_up_in_decimal(self):
        """1 cent at 500.50 is 500.5 tiyn: ROUND_HALF_UP gives 501, so the gap to 500 is 1 (banker's would give 0)."""
        found = reconcile(manifest(), tables(
            invoices=[invoice("INV1", currency="USD", total=1)],
            ledger=[row("R1", "1210", 500, invoice_id="INV1", doc_currency="USD", doc_amount=1, rate="500.00")],
            fx=[{"date": "2026-09-10", "source_currency": "USD", "target_currency": "KZT", "rate": "500.50",
                 "fx_id": "FX1"}]))
        self.assertEqual(found, [{
            "type": "fx_rate", "affected_ids": ["ledger/R1"], "claimed_amount_minor": 1, "currency": "KZT",
            "evidence_ids": ["invoice/INV1", "ledger/R1", "fx/FX1"], "explanation": found[0]["explanation"]}])

    def test_rate_takes_no_fallback_for_a_day_without_a_quote(self):
        """A Saturday invoice has no fx row on its date: the naive rule skips it rather than look back."""
        found = reconcile(manifest(), tables(
            invoices=[invoice("INV1", issued_at="2026-09-26T09:00:00+05:00", currency="USD", total=100)],
            ledger=[row("R1", "3310", 51264, invoice_id="INV1", timestamp="2026-09-26T12:00:00+05:00",
                        doc_currency="USD", doc_amount=100, rate="512.64")],
            fx=[{"date": "2026-09-25", "source_currency": "USD", "target_currency": "KZT", "rate": "512.64",
                 "fx_id": "FX1"}]))
        self.assertEqual(found, [])

    def test_period_uses_the_utc_date_of_the_document(self):
        """00:20 +05:00 on 1 October is 30 September in UTC: posted on 1 October, it is flagged."""
        found = reconcile(manifest(), tables(
            invoices=[invoice("INV1", issued_at="2026-10-01T00:20:00+05:00")],
            ledger=[row("R1", "1210", 7000, invoice_id="INV1", timestamp="2026-10-01T12:00:00+05:00")]))
        self.assertEqual([(f["type"], f["affected_ids"], f["claimed_amount_minor"]) for f in found],
                         [("period_cutoff", ["ledger/R1"], 7000)])

    def test_fully_cleared_payment_toward_an_open_invoice_is_not_flagged(self):
        """An instalment cleared in full leaves nothing unallocated: amount 0, no finding."""
        found = reconcile(manifest(), tables(
            invoices=[invoice("INV1", total=100000)],
            payments=[{"payment_id": "PAY1", "currency": "KZT", "amount_minor": 40000, "invoice_refs": ["INV1"]}],
            ledger=[row("R1", "1210", 40000, invoice_id="INV1", payment_id="PAY1")]))
        self.assertEqual(found, [])

    def test_at_most_30_findings(self):
        bills = [invoice(f"AP{k:03d}", direction="AP") for k in range(9)]  # one group, 36 pairs
        found = reconcile(manifest(), tables(invoices=bills))
        self.assertEqual(len(found), MAX_FINDINGS)
        self.assertEqual(found[0]["affected_ids"], ["invoice/AP000", "invoice/AP001"])
        self.assertEqual(found[-1]["affected_ids"], ["invoice/AP004", "invoice/AP008"])  # 8+7+6+5+4 pairs

    def test_at_most_10_evidence_ids(self):
        """Twelve clearing rows of one payment: the evidence list is cut at 10, payment and invoice first."""
        clearing = [row(f"C{k:02d}", "1210", 1000, invoice_id="INV1", payment_id="PAY1") for k in range(12)]
        found = reconcile(manifest(), tables(
            invoices=[invoice("INV1", total=100000)], ledger=clearing,
            payments=[{"payment_id": "PAY1", "currency": "KZT", "amount_minor": 50000, "invoice_refs": ["INV1"]}]))
        self.assertEqual([(f["type"], f["claimed_amount_minor"]) for f in found], [("payment_allocation", 38000)])
        self.assertEqual(found[0]["evidence_ids"][:3], ["payment/PAY1", "invoice/INV1", "ledger/C00"])
        self.assertEqual(len(found[0]["evidence_ids"]), MAX_EVIDENCE)


class Registration(unittest.TestCase):
    def test_the_cli_and_the_demo_run_it(self):
        self.assertEqual(SUBPROCESS_POLICIES["reconcile"], "policies.reconcile")
        demo = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertRegex(demo, r"for policy in [^;]*\breconcile\b[^;]*; do")
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        proc = subprocess.run([sys.executable, "-m", "auditenv.run", "--policy", "reconcile", "--episode-dir", str(MINI),
                               "--mode", "eval"], capture_output=True, text=True, cwd=REPO_ROOT, env=env, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual((report["policy"], report["tp"], report["fp"], report["fn"]), ("reconcile", 7, 4, 1))


if __name__ == "__main__":
    unittest.main()
