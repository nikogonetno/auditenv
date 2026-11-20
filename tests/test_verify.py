"""Independent episode verifier (role V, docs/tasks/V_verifier.md).

Every test works on a copy of tests/fixtures/mini_v03 in a temporary folder. The verifier has its own
detector of the policy rules INV-13…INV-16, so a change in the public files alone must show up as a
violation that truth does not list, and a change in truth alone must show up as an issue the detector
does not see.
"""
import ast
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from auditenv.verify import verify_episode
from auditenv.verify.ids import normalize_id
from tests.support import run_python

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "mini_v03"


class VerifyMini(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "episode"
        shutil.copytree(FIXTURE, self.dir)

    def tearDown(self):
        self._tmp.cleanup()

    # --- helpers -------------------------------------------------------------------------------

    def problems(self, strict_volume=False):
        return verify_episode(str(self.dir), strict_volume=strict_volume)

    def codes(self, strict_volume=False):
        return {p["code"] for p in self.problems(strict_volume)}

    def details(self, code):
        return [p["detail"] for p in self.problems() if p["code"] == code]

    def edit_truth(self, change):
        path = self.dir / "private" / "truth.json"
        truth = json.loads(path.read_text(encoding="utf-8"))
        change(truth)
        path.write_text(json.dumps(truth, sort_keys=True, ensure_ascii=False, indent=2), encoding="utf-8")

    def issue(self, truth, issue_id):
        return next(i for i in truth["issues"] if i["issue_id"] == issue_id)

    def edit_jsonl(self, name, key, change):
        path = self.dir / "public" / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        change(next(r for r in rows if r[key[0]] == key[1]))
        path.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows),
                        encoding="utf-8")

    def edit_csv(self, name, key, change):
        path = self.dir / "public" / name
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames, list(reader)
        change(next(r for r in rows if r[key[0]] == key[1]))
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def append(self, name, records):
        """Add records to a public table and keep manifest.counts and manifest.documents in step."""
        path = self.dir / "public" / name
        if name.endswith(".jsonl"):
            with open(path, "a", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records)
        else:
            with open(path, encoding="utf-8", newline="") as f:
                fields = next(csv.reader(f))
            with open(path, "a", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=fields, lineterminator="\n").writerows(records)
        table = name.split(".")[0]
        manifest_path = self.dir / "public" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["counts"][table] += len(records)
        if table in ("invoices", "payments"):
            manifest["documents"] += len(records)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- the fixture itself --------------------------------------------------------------------

    def test_mini_passes_without_volume_rule(self):
        self.assertEqual(self.problems(strict_volume=False), [])

    def test_mini_fails_volume_rule_when_strict(self):
        # Only 2 rows of each neighbouring period in mini_v03, INV-03 asks for at least 10.
        self.assertEqual(self.codes(strict_volume=True), {"INV-03"})

    def test_problem_shape(self):
        self.edit_truth(lambda t: t["issues"].pop(0))
        for p in self.problems():
            self.assertEqual(set(p), {"code", "detail"})
            self.assertIsInstance(p["code"], str)
            self.assertIsInstance(p["detail"], str)

    # --- INV-18: detector against truth --------------------------------------------------------

    def test_issue_removed_from_truth_is_inv18(self):
        self.edit_truth(lambda t: t["issues"].remove(self.issue(t, "XT92KR")))
        self.assertEqual(self.codes(), {"INV-18"})
        [detail] = self.details("INV-18")
        self.assertIn("fx_rate", detail)
        self.assertIn("QQ5GHK", detail)

    def test_invented_issue_in_truth_is_inv18(self):
        def invent(truth):
            truth["issues"].append({
                "issue_id": "Z00000", "type": "duplicate_invoice",
                "affected_ids": ["invoice/G0QJMF", "invoice/P18WZP"],
                "expected_amount_minor": 120000, "currency": "USD", "tolerance_minor": 1,
                "evidence_sets": [["invoice/G0QJMF", "invoice/P18WZP", "ledger/A35WK1", "ledger/XNPCY7"]],
            })
        self.edit_truth(invent)
        self.assertIn("INV-18", self.codes())
        self.assertTrue(any("Z00000" in d for d in self.details("INV-18")))

    def test_type_swapped_in_truth_is_inv18_both_ways(self):
        # C1: period and fx on the same 1030 row; the type must take part in the match.
        self.edit_truth(lambda t: self.issue(t, "ZHB4FR").update(type="period_cutoff"))
        details = self.details("INV-18")
        self.assertIn("duplicate truth issue: XG1494 and ZHB4FR are both period_cutoff on ['Z3H7BB']", details)
        self.assertTrue(any("ZHB4FR" in d for d in details))
        self.assertTrue(any("fx_rate" in d and "Z3H7BB" in d and "ZHB4FR" not in d for d in details))

    def test_affected_ids_with_bare_and_doc_ids_still_match(self):
        def rename(truth):
            self.issue(truth, "G21S27")["affected_ids"] = ["1TNM24", "doc/6V88KD"]
        self.edit_truth(rename)
        self.assertEqual(self.problems(), [])

    def test_new_fx_violation_in_public_data_is_inv18(self):
        # The Saturday Tailspin bill takes Friday's rate; a different Friday rate breaks the twin.
        self.edit_csv("fx.csv", ("fx_id", "0TZMZ3"), lambda r: r.update(rate="512.65"))
        self.assertEqual(self.codes(), {"INV-18"})
        [detail] = self.details("INV-18")
        self.assertIn("fx_rate", detail)
        self.assertIn("A35WK1", detail)

    def test_new_period_violation_in_public_data_is_inv18(self):
        # Late entry twin S-2026-0446: issued 23.40 local on 30.09. Twenty-five minutes later it is October.
        self.edit_jsonl("invoices.jsonl", ("invoice_id", "39487B"),
                        lambda r: r.update(issued_at="2026-10-01T00:05:00+05:00"))
        self.assertEqual(self.codes(), {"INV-18"})
        [detail] = self.details("INV-18")
        self.assertIn("period_cutoff", detail)
        self.assertIn("RD1JK8", detail)

    def test_new_duplicate_in_public_data_is_inv18(self):
        # Subscription twin: give the September bill the August vendor_ref in another spelling.
        self.edit_jsonl("invoices.jsonl", ("invoice_id", "P18WZP"), lambda r: r.update(vendor_ref="tsh 0826"))
        self.assertEqual(self.codes(), {"INV-18"})
        [detail] = self.details("INV-18")
        self.assertIn("duplicate_invoice", detail)
        self.assertIn("G0QJMF", detail)
        self.assertIn("P18WZP", detail)

    def test_new_allocation_violation_in_public_data_is_inv18(self):
        # Fabrikam partial payment twin: the bank now shows more than was cleared, and the invoice is
        # still open by more than the payment, so the rest may not sit on 3510.
        self.edit_jsonl("payments.jsonl", ("payment_id", "7VZYZY"), lambda r: r.update(amount_minor=150000000))
        details = self.details("INV-18")
        self.assertTrue(any("payment_allocation" in d and "7VZYZY" in d and "EY31ZP" in d for d in details))

    def test_violation_not_detected_is_inv18_for_unposted_duplicate(self):
        # A duplicate counts only when both bills are posted. Drop the copy's 3310 row from the book.
        path = self.dir / "public" / "ledger.csv"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(l for l in lines if not l.startswith(("KBYZZ8,", "TZSB5M,"))), encoding="utf-8")
        self.assertTrue(any("G21S27" in d for d in self.details("INV-18")))

    # --- INV-17: amount from each evidence set --------------------------------------------------

    def test_amount_shifted_by_five_is_inv17(self):
        self.edit_truth(lambda t: self.issue(t, "FDZ2T6").update(expected_amount_minor=300005))
        self.assertEqual(self.codes(), {"INV-17"})
        self.assertTrue(all("FDZ2T6" in d for d in self.details("INV-17")))

    def test_amount_within_tolerance_passes(self):
        self.edit_truth(lambda t: self.issue(t, "XT92KR").update(expected_amount_minor=1270001))
        self.assertEqual(self.problems(), [])

    def test_every_evidence_set_is_checked(self):
        def add_short_set(truth):
            issue = self.issue(truth, "XT92KR")
            issue["evidence_sets"].append(["invoice/FRVMVD", "ledger/QQ5GHK"])  # no fx row
        self.edit_truth(add_short_set)
        self.assertEqual(self.codes(), {"INV-17"})
        [detail] = self.details("INV-17")
        self.assertIn("XT92KR", detail)

    def test_fx_row_for_wrong_date_is_inv17(self):
        # The rate of the entry date (18.09) instead of the document date (14.09).
        self.edit_truth(lambda t: self.issue(t, "XT92KR").update(
            evidence_sets=[["fx/5PDMWJ", "invoice/FRVMVD", "ledger/QQ5GHK"]]))
        self.assertEqual(self.codes(), {"INV-17"})

    def test_allocation_set_missing_clearing_row_is_inv17(self):
        self.edit_truth(lambda t: self.issue(t, "FDZ2T6").update(
            evidence_sets=[["invoice/REB8R3", "ledger/CTT6J3", "payment/WFV9R1"]]))
        self.assertEqual(self.codes(), {"INV-17"})

    def test_unknown_evidence_id_is_inv17(self):
        self.edit_truth(lambda t: self.issue(t, "7F7V0Z")["evidence_sets"][0].append("ledger/ZZZZZZ"))
        self.assertIn("INV-17", self.codes())

    def test_wrong_currency_is_inv17(self):
        self.edit_truth(lambda t: self.issue(t, "G21S27").update(currency="KZT"))
        self.assertEqual(self.codes(), {"INV-17"})

    # --- INV-01, INV-02: the book balances ----------------------------------------------------

    def test_corrupted_ledger_amount_is_inv01(self):
        self.edit_csv("ledger.csv", ("row_id", "MEFQWF"), lambda r: r.update(credit_minor="102614001"))
        codes = self.codes()
        self.assertIn("INV-01", codes)
        self.assertIn("INV-02", codes)
        self.assertTrue(any("08D9E8" in d for d in self.details("INV-01")))

    def test_row_with_both_sides_is_inv01(self):
        self.edit_csv("ledger.csv", ("row_id", "V2CMCR"), lambda r: r.update(debit_minor="1"))
        self.assertIn("INV-01", self.codes())

    # --- bookkeeping consistency ---------------------------------------------------------------

    def test_row_base_not_matching_its_rate_is_inv04(self):
        # A revenue row is not a carrying row, so only the rate arithmetic can catch it.
        self.edit_csv("ledger.csv", ("row_id", "XEVXN8"), lambda r: r.update(exchange_rate="513.08"))
        self.assertEqual(self.codes(), {"INV-04"})

    def test_fx_row_on_the_wrong_side_is_inv05(self):
        self.edit_csv("ledger.csv", ("row_id", "V2CMCR"), lambda r: r.update(account="7430"))
        self.assertEqual(self.codes(), {"INV-05"})

    def test_line_items_not_summing_to_total_is_inv06(self):
        def change(invoice):
            invoice["line_items"][0].update(qty=2, unit_price_minor=60000)
            invoice["line_items"][0]["amount_minor"] = 110000
        self.edit_jsonl("invoices.jsonl", ("invoice_id", "G0QJMF"), change)
        self.assertEqual(self.codes(), {"INV-06"})

    def test_carrying_row_not_copying_its_document_is_inv08(self):
        self.edit_jsonl("invoices.jsonl", ("invoice_id", "P18WZP"), lambda r: r.update(entity_id="5N6CKK"))
        self.assertIn("INV-08", self.codes())

    def test_refs_not_following_remittance_is_inv08(self):
        # The detector reads invoice_refs, so they must say what the remittance text says.
        self.edit_jsonl("payments.jsonl", ("payment_id", "GDSY3H"), lambda r: r.update(remittance_info="S-2026-0446"))
        self.assertEqual(self.codes(), {"INV-08"})
        [detail] = self.details("INV-08")
        self.assertIn("39487B", detail)

    def test_reference_needs_a_boundary(self):
        # S-2026-04150 is not S-2026-0415.
        self.edit_jsonl("payments.jsonl", ("payment_id", "WHD1JM"), lambda r: r.update(remittance_info="S-2026-04150"))
        self.assertTrue(any("WHD1JM" in d and "None" in d for d in self.details("INV-08")))

    def test_receipt_not_split_into_cleared_and_unapplied_is_inv10(self):
        self.edit_jsonl("payments.jsonl", ("payment_id", "GDSY3H"), lambda r: r.update(amount_minor=31550001))
        self.assertIn("INV-10", self.codes())

    def test_payment_before_invoice_is_inv12(self):
        self.edit_jsonl("invoices.jsonl", ("invoice_id", "EY31ZP"),
                        lambda r: r.update(issued_at="2026-09-22T10:00:00+05:00"))
        self.assertIn("INV-12", self.codes())

    def test_entry_on_two_dates_is_reported(self):
        self.edit_csv("ledger.csv", ("row_id", "2NMKY7"), lambda r: r.update(timestamp="2026-10-01T12:00:00+05:00"))
        self.assertEqual(self.codes(), {"INV-13"})
        [detail] = self.details("INV-13")
        self.assertIn("A12BT6", detail)

    # --- INV-10: refund to a customer (PM decision on case §02) ---------------------------------

    def overpay(self):
        """Fabrikam (BW2C75) pays 2 100 000.00 KZT on 21.09 against its 2 000 000.00 bill S-2026-0420.
        The bill is cleared in full, 100 000.00 KZT stays on 3510."""
        self.edit_jsonl("payments.jsonl", ("payment_id", "7VZYZY"), lambda r: r.update(amount_minor=210000000))
        self.edit_csv("ledger.csv", ("row_id", "K92XEE"),
                      lambda r: r.update(debit_minor="210000000", doc_amount_minor="210000000"))
        self.edit_csv("ledger.csv", ("row_id", "JS0ZFQ"),
                      lambda r: r.update(credit_minor="200000000", doc_amount_minor="200000000"))
        self.append("ledger.csv", [self.row("RFOVER", "2026-09-21", "3510", 0, 10000000, "BW2C75", "7VZYZY",
                                            "XFKZJC", "KZT", 10000000, "1", "unapplied receipt")])

    def row(self, row_id, day, account, debit, credit, entity, payment, je, currency, amount, rate, memo):
        return {"row_id": row_id, "timestamp": f"{day}T12:00:00+05:00", "account": account, "debit_minor": debit,
                "credit_minor": credit, "currency": "KZT", "entity_id": entity, "invoice_id": "",
                "payment_id": payment, "je_id": je, "entered_at": f"{day}T10:30:00+05:00", "doc_currency": currency,
                "doc_amount_minor": amount, "exchange_rate": rate, "memo": memo}

    def refund(self, entity, amount, day="2026-09-25", tag="A", currency="KZT", rate="1", base=None,
               account="3510"):
        """A payment out to a customer that names no invoice, posted Dr 3510 Cr 1030 for the whole amount."""
        base = amount if base is None else base
        pid, je = f"RF{tag}PAY", f"RF{tag}JE"
        self.append("payments.jsonl", [{
            "doc_id": f"RF{tag}DOC", "payment_id": pid, "timestamp": f"{day}T05:00:00+00:00", "currency": currency,
            "amount_minor": amount, "invoice_refs": None, "entity_id": entity, "direction": "out",
            "counterparty_name_raw": "Fabrikam Inc", "entered_at": f"{day}T10:30:00+05:00",
            "remittance_info": "Refund of overpayment"}])
        self.append("ledger.csv", [
            self.row(f"RF{tag}DR", day, account, base, 0, entity, pid, je, currency, amount, rate, "refund"),
            self.row(f"RF{tag}CR", day, "1030", 0, base, entity, pid, je, currency, amount, rate, "refund")])
        return pid

    def test_overpayment_alone_passes(self):
        self.overpay()
        self.assertEqual(self.problems(), [])

    def test_refund_after_overpayment_passes(self):
        self.overpay()
        self.refund("BW2C75", 10000000)
        self.assertEqual(self.problems(), [])

    def test_refund_without_unapplied_balance_is_inv10(self):
        # The unidentified KZT receipt GDSY3H on 3510 belongs to nobody, so it does not fund Fabrikam.
        pid = self.refund("BW2C75", 10000000)
        self.assertEqual(self.codes(), {"INV-10"})
        [detail] = self.details("INV-10")
        self.assertIn(pid, detail)
        self.assertIn("customer BW2C75 holds 0 KZT on 3510 before it", detail)

    def test_refund_above_balance_is_inv10(self):
        self.overpay()
        pid = self.refund("BW2C75", 10000001)
        self.assertEqual(self.codes(), {"INV-10"})
        [detail] = self.details("INV-10")
        self.assertIn(pid, detail)
        self.assertIn("customer BW2C75 holds 10000000 KZT on 3510 before it", detail)

    def test_refund_in_foreign_currency_is_inv10(self):
        # Contoso (YTB3KB) holds 3 000.00 USD on 3510 from WFV9R1, but only base-currency refunds are modelled.
        # Rate and base are right for 24.09, so nothing but INV-10 can fire.
        pid = self.refund("YTB3KB", 100000, day="2026-09-24", currency="USD", rate="511.14", base=51114000)
        self.assertEqual(self.codes(), {"INV-10"})
        [detail] = self.details("INV-10")
        self.assertIn(pid, detail)
        self.assertIn("refund in USD, refunds are in KZT only", detail)

    def test_refund_before_the_overpayment_is_inv10(self):
        self.overpay()
        self.refund("BW2C75", 10000000, day="2026-09-15")
        self.assertEqual(self.codes(), {"INV-10"})
        [detail] = self.details("INV-10")
        self.assertIn("holds 0 KZT", detail)

    def test_refunds_together_above_balance_is_inv10(self):
        self.overpay()
        self.refund("BW2C75", 6000000, day="2026-09-24", tag="A")
        second = self.refund("BW2C75", 6000000, day="2026-09-25", tag="B")
        self.assertEqual(self.codes(), {"INV-10"})
        [detail] = self.details("INV-10")
        self.assertIn(second, detail)
        self.assertIn("holds 4000000 KZT", detail)

    def test_refund_not_through_3510_is_inv10(self):
        self.overpay()
        self.refund("BW2C75", 10000000, account="6010")
        self.assertEqual(self.codes(), {"INV-10"})
        [detail] = self.details("INV-10")
        self.assertIn("not Dr 3510 Cr 1030", detail)

    def test_supplier_payment_short_of_its_bills_is_still_inv10(self):
        self.edit_jsonl("payments.jsonl", ("payment_id", "AM952T"), lambda r: r.update(amount_minor=148000001))
        [detail] = self.details("INV-10")
        self.assertIn("AM952T", detail)
        self.assertIn("it names invoices", detail)

    # --- INV-20 and QA D-4: truth itself -------------------------------------------------------

    def test_amount_not_above_two_tolerances_is_inv20(self):
        self.edit_truth(lambda t: self.issue(t, "XT92KR").update(tolerance_minor=635000))
        self.assertEqual(self.codes(), {"INV-20"})

    def test_repeated_issue_in_truth_is_inv18(self):
        # QA D-4: SPEC §6 puts a repeated (type, affected_ids) under INV-18, oracle would get 0.882.
        def repeat(truth):
            copy = dict(self.issue(truth, "G21S27"), issue_id="Z11111")
            truth["issues"].append(copy)
        self.edit_truth(repeat)
        self.assertEqual(self.codes(), {"INV-18"})
        [detail] = self.details("INV-18")
        self.assertTrue(detail.startswith("duplicate truth issue: "), detail)
        self.assertIn("G21S27", detail)
        self.assertIn("Z11111", detail)

    def test_repeated_issue_with_other_id_spelling_is_inv18(self):
        def repeat(truth):
            copy = dict(self.issue(truth, "G21S27"), issue_id="Z11111", affected_ids=["F7SF66", "doc/WRJD8T"])
            truth["issues"].append(copy)
        self.edit_truth(repeat)
        self.assertEqual(self.codes(), {"INV-18"})
        self.assertTrue(self.details("INV-18")[0].startswith("duplicate truth issue: "))

    def test_clean_flag_must_match_issues(self):
        self.edit_truth(lambda t: t.update(clean=True))
        self.assertIn("TRUTH", self.codes())

    # --- INV-23: no private words in public files ----------------------------------------------

    def test_type_name_in_memo_is_inv23(self):
        self.edit_csv("ledger.csv", ("row_id", "NQKGW2"), lambda r: r.update(memo="payment_allocation pending"))
        self.assertEqual(self.codes(), {"INV-23"})

    def test_private_word_in_remittance_is_inv23(self):
        self.edit_jsonl("payments.jsonl", ("payment_id", "GDSY3H"), lambda r: r.update(remittance_info="Twin OPLATA"))
        self.assertEqual(self.codes(), {"INV-23"})

    def test_issue_id_in_public_file_is_inv23(self):
        self.edit_csv("ledger.csv", ("row_id", "NQKGW2"), lambda r: r.update(memo="ref C1V8ZA"))
        self.assertEqual(self.codes(), {"INV-23"})

    # --- never raises ----------------------------------------------------------------------------

    def test_missing_directory_is_a_problem_not_an_exception(self):
        problems = verify_episode(str(self.dir / "nope"))
        self.assertTrue(problems)
        self.assertEqual({p["code"] for p in problems}, {"LOAD"})

    def test_broken_truth_is_a_problem_not_an_exception(self):
        (self.dir / "private" / "truth.json").write_text("{not json", encoding="utf-8")
        self.assertIn("LOAD", self.codes())

    def test_truth_of_wrong_shape_is_a_problem_not_an_exception(self):
        (self.dir / "private" / "truth.json").write_text('{"clean": false, "issues": [7, {"type": 1}]}',
                                                          encoding="utf-8")
        self.assertTrue(self.problems())

    def test_ledger_with_garbage_is_a_problem_not_an_exception(self):
        self.edit_csv("ledger.csv", ("row_id", "MEFQWF"),
                      lambda r: r.update(timestamp="yesterday", exchange_rate="abc", account=""))
        self.assertTrue(self.problems())

    # --- CLI -------------------------------------------------------------------------------------

    def cli(self, *args, hash_seed="0"):
        # -P keeps the working directory off sys.path, so the child adds the repository root itself.
        code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); from auditenv.verify.__main__ import main; "
                f"sys.exit(main({[*args, str(self.dir)]!r}))")
        return run_python(["-c", code], hash_seed)

    def test_cli_exit_codes_and_json_lines(self):
        ok = self.cli("--no-strict-volume")
        self.assertEqual((ok.returncode, ok.stdout), (0, ""), ok.stderr)
        bad = self.cli()
        self.assertEqual(bad.returncode, 1, bad.stderr)
        lines = [json.loads(line) for line in bad.stdout.splitlines()]
        self.assertEqual({p["code"] for p in lines}, {"INV-03"})

    def test_module_entry_point(self):
        res = subprocess.run([sys.executable, "-m", "auditenv.verify", "--no-strict-volume", str(self.dir)],
                             cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual((res.returncode, res.stdout), (0, ""), res.stderr)

    def test_output_does_not_depend_on_hash_seed(self):
        self.edit_truth(lambda t: t["issues"].pop(3))
        self.edit_csv("ledger.csv", ("row_id", "NQKGW2"), lambda r: r.update(memo="twin", credit_minor="1"))
        zero, one = self.cli(hash_seed="0"), self.cli(hash_seed="1")
        self.assertEqual(zero.returncode, 1, zero.stderr)
        self.assertGreater(len(zero.stdout.splitlines()), 3)
        self.assertEqual(zero.stdout, one.stdout)


class OwnIdNormalization(unittest.TestCase):
    """N-5: the verifier normalizes ids itself (SPEC §3.3, §5.2) and must agree with the grader."""

    CASES = (
        "invoice/1TNM24", "1TNM24", "payment/WFV9R1", "ledger/NP7DTC", "allocation/A1", "counterparty/C1",
        "line/L1", "fx/X2MWG8", "fx/2026-09-14/USD/KZT", "doc/WRJD8T", "je/A12BT6", "je/", "je",
        "invoice/invoice/1TNM24", "doc/invoice/1TNM24", "ledger/fx/X2MWG8", "unknown/ABC", "vendor/ABC",
        "Invoice/ABC", "INVOICE/ABC", "invoice/abc", "abc123", "invoice/", "fx/", "/ABC", "//", "/", "",
        "invoice", "invoice//X", " invoice/X", "invoice/ X", "invoice\\X", "a/b/c", "доc/X",
    )

    def test_same_as_grader(self):
        from auditenv.grader import normalize_id as grader_normalize
        for ref in self.CASES:
            with self.subTest(ref=ref):
                self.assertEqual(normalize_id(ref), grader_normalize(ref))

    def test_spec_examples(self):
        self.assertEqual(normalize_id("fx/2026-09-14/USD/KZT"), "2026-09-14/USD/KZT")
        self.assertEqual(normalize_id("invoice/invoice/X"), "invoice/X")
        self.assertEqual(normalize_id("je/X"), "je/X")
        self.assertEqual(normalize_id("Invoice/X"), "Invoice/X")
        self.assertEqual(normalize_id("invoice/"), "invoice/")

    def test_verifier_imports_neither_grader_nor_generator(self):
        allowed_outside = {"auditenv.episode"}
        for path in sorted((ROOT / "auditenv" / "verify").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0:
                        names = [node.module]
                    else:
                        base = ["auditenv", "verify"][:3 - node.level]
                        names = [".".join(base + ([node.module] if node.module else []))]
                else:
                    continue
                for name in names:
                    with self.subTest(file=path.name, module=name):
                        if name == "auditenv" or name.startswith("auditenv."):
                            self.assertTrue(name in allowed_outside or name == "auditenv.verify"
                                            or name.startswith("auditenv.verify."), name)


if __name__ == "__main__":
    unittest.main()
