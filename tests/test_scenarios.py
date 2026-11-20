"""Coupled scenarios C1 and C2 (docs/DOMAIN_SPEC_A.md, docs/tasks/B_night_stage3.md items 3 and 4)."""
import csv
import datetime as dt
import json
import tempfile
import unittest
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from auditenv.env import AuditEnv
from auditenv.generate import generate_episode
from auditenv.runner import run_inprocess_episode
from auditenv.verify import verify_episode
from policies.oracle import make_env_oracle
from policies.reconcile import reconcile


def read(out):
    pub = Path(out) / "public"
    invoices = {r["invoice_id"]: r for r in map(json.loads, (pub / "invoices.jsonl").read_text().splitlines())}
    payments = {r["payment_id"]: r for r in map(json.loads, (pub / "payments.jsonl").read_text().splitlines())}
    with open(pub / "ledger.csv", encoding="utf-8") as f:
        ledger = {r["row_id"]: r for r in csv.DictReader(f)}
    with open(pub / "fx.csv", encoding="utf-8") as f:
        fx = {r["fx_id"]: r for r in csv.DictReader(f)}
    policy = json.loads((pub / "policy.json").read_text())
    truth = json.loads((Path(out) / "private" / "truth.json").read_text())
    return invoices, payments, ledger, fx, policy, truth


def oracle(out):
    env = AuditEnv("train")
    run_inprocess_episode(make_env_oracle(env), env, 0, {"episode_dir": str(out)})
    return env.grader_log[-1]["reward"]


def base(row):
    return int(row["debit_minor"]) or int(row["credit_minor"])


class C1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.outs = []
        for seed, offsets in ((1, ["+05:00"]), (2, ["+03:00", "+06:00"]), (3, ["-05:00"])):
            out = Path(cls.tmp.name) / str(seed)
            generate_episode(seed, {"errors": {}, "scenarios": {"C1": 1}, "company_offsets": offsets}, str(out))
            cls.outs.append(out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_three_issues_one_payment_and_amounts_by_type(self):
        for out in self.outs:
            invoices, payments, ledger, fx, policy, truth = read(out)
            with self.subTest(out=out.name):
                self.assertEqual(sorted(i["type"] for i in truth["issues"]),
                                 ["fx_rate", "payment_allocation", "period_cutoff"])
                self.assertEqual({i["scenario_id"] for i in truth["issues"]}, {"C1"})
                by_type = {i["type"]: i for i in truth["issues"]}
                period, rate, alloc = by_type["period_cutoff"], by_type["fx_rate"], by_type["payment_allocation"]
                self.assertEqual(period["affected_ids"], rate["affected_ids"])      # pairs differ by type only
                (row,) = [ledger[x.split("/")[1]] for x in period["affected_ids"]]
                pay = payments[row["payment_id"]]
                self.assertEqual((row["account"], pay["direction"]), ("1030", "in"))
                self.assertNotEqual(pay["currency"], "KZT")
                tz = dt.datetime.fromisoformat("2026-01-01T00:00:00" + policy["company_utc_offset"]).tzinfo
                when = dt.datetime.fromisoformat(pay["timestamp"])
                end = dt.date.fromisoformat(policy["period"]["end_local"])
                local, utc = when.astimezone(tz).date(), when.astimezone(dt.timezone.utc).date()
                self.assertNotEqual(local <= end, utc <= end)                     # two sides of the period end
                self.assertEqual(row["timestamp"][:10], utc.isoformat())          # posted on the UTC date
                self.assertEqual(period["expected_amount_minor"], base(row))
                (quote,) = [fx[x.split("/")[1]] for x in rate["evidence_sets"][0] if x.startswith("fx/")]
                self.assertEqual(quote["date"], local.isoformat())                # the company-date rate
                right = int((Decimal(row["doc_amount_minor"]) * Decimal(quote["rate"])).quantize(Decimal(1), ROUND_HALF_UP))
                self.assertEqual(rate["expected_amount_minor"], abs(base(row) - right))
                held = [r for r in ledger.values() if r["payment_id"] == pay["payment_id"] and r["account"] == "3510"]
                cleared = [r for r in ledger.values() if r["payment_id"] == pay["payment_id"] and r["invoice_id"]]
                self.assertEqual(alloc["expected_amount_minor"], int(held[0]["doc_amount_minor"]))
                self.assertEqual(alloc["expected_amount_minor"],
                                 pay["amount_minor"] - sum(int(r["doc_amount_minor"]) for r in cleared))
                for issue in truth["issues"]:
                    self.assertGreater(issue["expected_amount_minor"], 2 * issue["tolerance_minor"])

    def test_strict_verifier_and_oracle(self):
        for out in self.outs:
            with self.subTest(out=out.name):
                self.assertEqual(verify_episode(str(out)), [])
                self.assertEqual(oracle(out), 1.0)

    def test_impossible_boundaries_are_value_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            for config in ({"period": "2026-10", "scenarios": {"C1": 1}},         # 31 Oct Sat, 1 Nov Sun
                           {"company_offsets": ["+00:00"], "scenarios": {"C1": 1}}):
                with self.subTest(config=config), self.assertRaises(ValueError):
                    generate_episode(1, config, tmp)


class C2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "c2"
        generate_episode(4, {"errors": {}, "scenarios": {"C2": 1}}, str(cls.out))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_one_duplicate_issue_and_the_payment_on_the_copy(self):
        invoices, payments, ledger, _, _, truth = read(self.out)
        (issue,) = truth["issues"]
        self.assertEqual((issue["type"], issue["scenario_id"]), ("duplicate_invoice", "C2"))
        bills = [x.split("/")[1] for x in issue["affected_ids"]]
        paying = [p for p in payments.values() if p["invoice_refs"] and set(bills) <= set(p["invoice_refs"])]
        self.assertEqual(len(paying), 1)
        cleared = {r["invoice_id"] for r in ledger.values() if r["payment_id"] == paying[0]["payment_id"] and r["invoice_id"]}
        self.assertEqual(len(cleared), 1)
        (copy,) = cleared
        (original,) = set(bills) - cleared
        self.assertEqual(invoices[copy]["vendor_ref"] != invoices[original]["vendor_ref"], True)
        self.assertLess(dt.datetime.fromisoformat(invoices[copy]["entered_at"]),
                        dt.datetime.fromisoformat(paying[0]["entered_at"]))       # keyed before it was paid
        self.assertEqual(verify_episode(str(self.out)), [])
        self.assertEqual(oracle(self.out), 1.0)

    def test_naive_reconciliation_flags_a_false_allocation_on_the_original(self):
        from auditenv.episode import load_episode
        episode = load_episode(self.out)
        truth = episode.load_truth()
        bills = {x for x in truth["issues"][0]["affected_ids"]}
        found = reconcile(episode.manifest, episode.tables)
        allocation = [f for f in found if f["type"] == "payment_allocation" and bills & set(f["affected_ids"])]
        self.assertTrue(allocation)


if __name__ == "__main__":
    unittest.main()
