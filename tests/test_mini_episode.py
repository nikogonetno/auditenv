"""Mini episode in the case §03 layout, B's own test book until golden v0.2 (PM decisions).

tests/fixtures/mini_v03 is golden v0.1 rewritten by tests/fixtures/build_mini_v03.py into
the schema PM closed in "Схема §03, закрываем поля". Checked here: the loader accepts it,
mandatory columns and keys come first in case §03 order, every truth id resolves to a
public record, oracle findings in case §05 field names score 1.0 (also by doc_id through
aliases), the fixture rebuilds byte for byte under different hash seeds, nothing private
leaks into public files, and invoice_refs follows the PM rule (C2 lists both bills).
"""
import csv
import datetime as dt
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from auditenv.episode import DOCUMENT_TABLES, TABLES, load_episode
from auditenv.grader import score
from tests.support import run_python

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BUILDER = FIXTURES / "build_mini_v03.py"
EPISODE = FIXTURES / "mini_v03"
PUBLIC = EPISODE / "public"
HASH_SEEDS = ("0", "1")

# Case §03 mandatory fields, in the order the case lists them.
MANDATORY = {
    "ledger.csv": ("row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
                   "entity_id", "invoice_id", "payment_id"),
    "invoices.jsonl": ("doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency",
                       "total_minor", "line_items"),
    "payments.jsonl": ("doc_id", "payment_id", "timestamp", "currency", "amount_minor", "invoice_refs"),
    "fx.csv": ("date", "source_currency", "target_currency", "rate"),
}
# Domain fields that follow the mandatory ones (PM default rule: extras stay, after the case fields).
EXTRAS = {
    "ledger.csv": ("je_id", "entered_at", "doc_currency", "doc_amount_minor", "exchange_rate", "memo"),
    "invoices.jsonl": ("direction", "counterparty_name", "invoice_number", "vendor_ref", "po_ref",
                       "entered_at", "service_from", "service_to"),
    "payments.jsonl": ("entity_id", "direction", "counterparty_name_raw", "entered_at", "remittance_info"),
    "fx.csv": ("fx_id",),
}
LINE_ITEM_KEYS = ("line_id", "description", "qty", "unit_price_minor", "amount_minor")
MANIFEST_KEYS = ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone",
                 "counts", "documents", "rounding")
PUBLIC_FILES = ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv")
PREFIX_TABLE = {"invoice": "invoices", "payment": "payments", "ledger": "ledger", "fx": "fx"}
ISSUE_TYPES = ("duplicate_invoice", "payment_allocation", "fx_rate", "period_cutoff")
PRIVATE_WORDS = ("truth", "provenance", "inject", "twin", "scenario")
CROCKFORD_ID = re.compile(r"[0-9ABCDEFGHJKMNPQRSTVWXYZ]{6}")
OFFSET = re.compile(r"[+-]\d\d:\d\d")


def sha256_tree(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def load_json(rel):
    return json.loads((EPISODE / rel).read_text(encoding="utf-8"))


def csv_header(name):
    with open(PUBLIC / name, encoding="utf-8", newline="") as f:
        return tuple(next(csv.reader(f)))


def jsonl_key_orders(name):
    with open(PUBLIC / name, encoding="utf-8") as f:
        return [tuple(json.loads(line)) for line in f if line.strip()]


def split_ref(ref):
    prefix, _, rest = ref.partition("/")
    return prefix, rest


def oracle_findings(truth):
    """Case §05 findings.json entries that restate truth: the maximum-reward answer."""
    return [
        {
            "type": i["type"],
            "affected_ids": list(i["affected_ids"]),
            "claimed_amount_minor": i["expected_amount_minor"],
            "currency": i["currency"],
            "evidence_ids": list(i["evidence_sets"][0]),
            "explanation": f"restated {i['type']} for the listed records",
        }
        for i in truth["issues"]
    ]


def has_offset(timestamp):
    return dt.datetime.fromisoformat(timestamp).utcoffset() is not None


class MiniEpisodeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episode = load_episode(EPISODE)
        cls.truth = cls.episode.load_truth()


class Loading(MiniEpisodeTestCase):
    def test_loader_accepts_the_fixture(self):
        ep = self.episode
        self.assertEqual(ep.episode_id, "KARPSR")
        self.assertEqual(self.truth["episode_id"], ep.episode_id)
        self.assertIs(self.truth["clean"], False)
        self.assertEqual(len(self.truth["issues"]), 8)
        for name in TABLES:
            with self.subTest(table=name):
                self.assertTrue(ep.tables[name])
        self.assertEqual(len(ep.documents), len(ep.tables["invoices"]) + len(ep.tables["payments"]))

    def test_manifest_counts_match_tables(self):
        manifest = self.episode.manifest
        self.assertEqual(manifest["counts"], {name: len(self.episode.tables[name]) for name in TABLES})
        self.assertEqual(manifest["documents"], sum(len(self.episode.tables[t]) for t in DOCUMENT_TABLES))

    def test_every_document_has_an_alias_to_its_record(self):
        ep = self.episode
        for doc_id, (table, record) in sorted(ep.documents.items()):
            with self.subTest(doc_id=doc_id):
                self.assertEqual(ep.aliases[doc_id], record[TABLES[table][1]])


class Layout(MiniEpisodeTestCase):
    def test_csv_mandatory_columns_come_first_in_case_order(self):
        for name in ("ledger.csv", "fx.csv"):
            header = csv_header(name)
            with self.subTest(file=name):
                self.assertEqual(header[:len(MANDATORY[name])], MANDATORY[name])
                self.assertEqual(header, MANDATORY[name] + EXTRAS[name])

    def test_jsonl_mandatory_keys_come_first_in_case_order(self):
        for name in ("invoices.jsonl", "payments.jsonl"):
            orders = jsonl_key_orders(name)
            self.assertTrue(orders)
            for n, keys in enumerate(orders):
                with self.subTest(file=name, line=n + 1):
                    self.assertEqual(keys[:len(MANDATORY[name])], MANDATORY[name])
                    self.assertEqual(keys, MANDATORY[name] + EXTRAS[name])

    def test_line_items_keep_their_field_order_and_add_up(self):
        for inv in self.episode.tables["invoices"]:
            with self.subTest(invoice=inv["invoice_id"]):
                self.assertTrue(inv["line_items"])
                for item in inv["line_items"]:
                    self.assertEqual(tuple(item), LINE_ITEM_KEYS)
                    self.assertIs(type(item["qty"]), int)
                    self.assertEqual(item["qty"] * item["unit_price_minor"], item["amount_minor"])
                self.assertEqual(sum(i["amount_minor"] for i in inv["line_items"]), inv["total_minor"])

    def test_manifest_holds_every_case_key(self):
        # manifest.json is written with sort_keys=True, so the case keys are checked as a set.
        manifest = load_json("public/manifest.json")
        self.assertEqual(tuple(sorted(manifest)), tuple(sorted(MANIFEST_KEYS)))
        self.assertEqual(manifest["schema_version"], "case03-mini-1")
        self.assertEqual(manifest["base_currency"], "KZT")
        self.assertEqual((manifest["period_start"], manifest["period_end"]), ("2026-09-01", "2026-09-30"))
        self.assertEqual(manifest["timezone"], "+05:00")
        self.assertRegex(manifest["timezone"], OFFSET)
        self.assertEqual(tuple(manifest["counts"]), tuple(sorted(TABLES)))
        self.assertEqual(manifest["rounding"],
                         {"minor_units": {"EUR": 2, "KZT": 2, "USD": 2}, "mode": "ROUND_HALF_UP"})

    def test_policy_states_rate_date_period_and_rounding(self):
        policy = self.episode.policy
        fx = policy["fx"]
        self.assertIn("date_rule", fx)
        self.assertIn("rounding", fx)
        self.assertEqual(fx["weekend_days"], [6, 7])
        self.assertEqual(fx["holidays"], ["2026-08-31"])
        self.assertNotIn("non_business_days", fx)
        self.assertEqual(policy["period"]["start_local"], self.episode.manifest["period_start"])
        self.assertEqual(policy["period"]["end_local"], self.episode.manifest["period_end"])
        self.assertEqual(policy["minor_units"], self.episode.manifest["rounding"]["minor_units"])

    def test_policy_chart_covers_every_ledger_account(self):
        accounts = self.episode.policy["accounts"]
        codes = [a["code"] for a in accounts]
        self.assertEqual(codes, sorted(set(codes)))
        self.assertEqual(len(codes), 9)
        for a in accounts:
            self.assertEqual(tuple(a), ("code", "name"))
            self.assertTrue(a["name"])
        used = sorted({r["account"] for r in self.episode.tables["ledger"]})
        self.assertEqual(used, codes)

    def test_ledger_is_in_base_currency_with_offset_timestamps(self):
        for r in self.episode.tables["ledger"]:
            with self.subTest(row=r["row_id"]):
                self.assertEqual(r["currency"], "KZT")
                self.assertTrue(r["timestamp"].endswith("T12:00:00+05:00"))
                self.assertTrue(has_offset(r["timestamp"]))
                self.assertTrue((r["debit_minor"] == 0) != (r["credit_minor"] == 0))

    def test_document_times_carry_an_offset(self):
        for inv in self.episode.tables["invoices"]:
            for key in ("issued_at", "due_at", "entered_at"):
                with self.subTest(invoice=inv["invoice_id"], field=key):
                    self.assertTrue(has_offset(inv[key]))
        for p in self.episode.tables["payments"]:
            for key in ("timestamp", "entered_at"):
                with self.subTest(payment=p["payment_id"], field=key):
                    self.assertTrue(has_offset(p[key]))

    def test_journal_entries_balance(self):
        totals = {}
        for r in self.episode.tables["ledger"]:
            totals[r["je_id"]] = totals.get(r["je_id"], 0) + r["debit_minor"] - r["credit_minor"]
        self.assertEqual({je: t for je, t in totals.items() if t}, {})

    def test_allocations_show_as_clearing_rows(self):
        # allocations.csv is gone; each 1210/3310 line of a payment names both sides.
        clearing = [r for r in self.episode.tables["ledger"]
                    if r["payment_id"] is not None and r["account"] in ("1210", "3310")]
        self.assertEqual(len(clearing), 6)
        for r in clearing:
            with self.subTest(row=r["row_id"]):
                self.assertIn(r["invoice_id"], self.episode.index["invoices"])
                self.assertIn(r["payment_id"], self.episode.index["payments"])

    def test_fx_rates_are_decimal_strings_into_base(self):
        for r in self.episode.tables["fx"]:
            with self.subTest(fx=r["fx_id"]):
                self.assertEqual(r["target_currency"], "KZT")
                self.assertRegex(r["rate"], r"^\d+\.\d+$")
                dt.date.fromisoformat(r["date"])


class Ids(MiniEpisodeTestCase):
    def all_ids(self):
        ep = self.episode
        ids = []
        for name, (_, column) in TABLES.items():
            ids += [(r[column], f"{name}.{column}") for r in ep.tables[name]]
        ids += [(r["doc_id"], f"{t}.doc_id") for t in DOCUMENT_TABLES for r in ep.tables[t]]
        ids += [(item["line_id"], "line_id") for r in ep.tables["invoices"] for item in r["line_items"]]
        ids += [(je, "je_id") for je in sorted({r["je_id"] for r in ep.tables["ledger"]})]
        ids += [(cp, "entity_id") for cp in sorted({r["entity_id"] for r in ep.tables["invoices"]})]
        return ids

    def test_ids_are_unique_across_the_episode(self):
        owner = {}
        for rid, where in self.all_ids():
            with self.subTest(id=rid):
                self.assertNotIn(rid, owner, f"{rid} in {where} and {owner.get(rid)}")
            owner[rid] = where
        for issue in self.truth["issues"]:
            self.assertNotIn(issue["issue_id"], owner)

    def test_new_ids_are_crockford_base32(self):
        for rid, where in self.all_ids():
            if where.endswith("doc_id") or where == "fx.fx_id":
                with self.subTest(id=rid, where=where):
                    self.assertRegex(rid, CROCKFORD_ID)
                    self.assertEqual(len(rid), 6)

    def test_every_truth_id_resolves_to_a_public_record(self):
        for issue in self.truth["issues"]:
            refs = list(issue["affected_ids"]) + [x for s in issue["evidence_sets"] for x in s]
            for ref in refs:
                with self.subTest(issue=issue["issue_id"], ref=ref):
                    prefix, rid = split_ref(ref)
                    self.assertIn(prefix, PREFIX_TABLE, "truth ids carry a table prefix")
                    self.assertIn(rid, self.episode.index[PREFIX_TABLE[prefix]])

    def test_truth_ids_are_sorted_and_free_of_je(self):
        text = (EPISODE / "private" / "truth.json").read_text(encoding="utf-8")
        self.assertNotIn("je/", text)
        self.assertNotIn("allocation/", text)
        for issue in self.truth["issues"]:
            with self.subTest(issue=issue["issue_id"]):
                self.assertEqual(issue["affected_ids"], sorted(issue["affected_ids"]))
                for s in issue["evidence_sets"]:
                    self.assertEqual(s, sorted(s))
                for key in ("issue_id", "type", "affected_ids", "expected_amount_minor", "currency",
                            "evidence_sets", "tolerance_minor", "scenario_id"):
                    self.assertIn(key, issue)

    def test_rate_evidence_names_fx_rows_by_id(self):
        fx_refs = [x for i in self.truth["issues"] for s in i["evidence_sets"] for x in s if x.startswith("fx/")]
        self.assertEqual(len(fx_refs), 2)
        for ref in fx_refs:
            self.assertEqual(ref.count("/"), 1)


class OracleScore(MiniEpisodeTestCase):
    def test_oracle_findings_score_one(self):
        result = score(oracle_findings(self.truth), self.truth, 0, self.episode.aliases)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (8, 0, 0))
        self.assertEqual(result["reward"], 1.0)

    def by_doc_id(self, findings, table, bare):
        """Rename every reference to a record of `table` into its doc_id."""
        doc_of = {rec[TABLES[t][1]]: doc for doc, (t, rec) in self.episode.documents.items() if t == table}
        prefix = {"invoices": "invoice", "payments": "payment"}[table]

        def rename(ref):
            p, rid = split_ref(ref)
            if p != prefix:
                return ref
            return doc_of[rid] if bare else f"doc/{doc_of[rid]}"

        renamed = 0
        for f in findings:
            for key in ("affected_ids", "evidence_ids"):
                new = [rename(x) for x in f[key]]
                renamed += sum(a != b for a, b in zip(new, f[key]))
                f[key] = new
        self.assertGreater(renamed, 0)
        return findings

    def test_oracle_naming_invoices_by_doc_id_scores_one_with_aliases(self):
        for bare in (False, True):
            with self.subTest(bare=bare):
                findings = self.by_doc_id(oracle_findings(self.truth), "invoices", bare)
                result = score(findings, self.truth, 0, self.episode.aliases)
                self.assertEqual(result["tp"], 8)
                self.assertEqual(result["reward"], 1.0)
                # Control: without aliases the same answer must not match, or the check proves nothing.
                self.assertLess(score(findings, self.truth, 0)["reward"], 1.0)

    def test_oracle_naming_payments_by_doc_id_scores_one_with_aliases(self):
        findings = self.by_doc_id(oracle_findings(self.truth), "payments", bare=False)
        self.assertEqual(score(findings, self.truth, 0, self.episode.aliases)["reward"], 1.0)


class Rebuild(unittest.TestCase):
    def test_rebuild_is_byte_identical_under_hash_seeds(self):
        expected = sha256_tree(EPISODE)
        self.assertEqual(sorted(expected), sorted(["private/truth.json"] + [f"public/{n}" for n in PUBLIC_FILES]))
        with tempfile.TemporaryDirectory() as tmp:
            for seed in HASH_SEEDS:
                out = Path(tmp) / f"hashseed_{seed}"
                res = run_python([BUILDER, out], seed, cwd=tmp)
                with self.subTest(hash_seed=seed):
                    self.assertEqual(res.returncode, 0, res.stderr)
                    self.assertEqual(sha256_tree(out), expected)


class Leaks(MiniEpisodeTestCase):
    def test_public_files_hold_no_private_markers(self):
        markers = set(ISSUE_TYPES) | set(PRIVATE_WORDS)
        markers |= {i["issue_id"] for i in self.truth["issues"]}
        self.assertEqual(sorted(p.name for p in PUBLIC.iterdir()), sorted(PUBLIC_FILES))
        for name in PUBLIC_FILES:
            text = (PUBLIC / name).read_text(encoding="utf-8").lower()
            for marker in sorted(markers):
                with self.subTest(file=name, marker=marker):
                    self.assertNotIn(marker.lower(), text)

    def test_no_underscore_columns_or_keys(self):
        for name in ("ledger.csv", "fx.csv"):
            self.assertFalse([c for c in csv_header(name) if c.startswith("_")])
        for name in ("invoices.jsonl", "payments.jsonl"):
            for keys in jsonl_key_orders(name):
                self.assertFalse([k for k in keys if k.startswith("_")])


class InvoiceRefs(MiniEpisodeTestCase):
    def payments(self):
        return self.episode.tables["payments"]

    def test_c2_payment_lists_both_copies_of_the_bill(self):
        (dup,) = [i for i in self.truth["issues"]
                  if i["type"] == "duplicate_invoice" and i["scenario_id"] == "C2"]
        copies = sorted(split_ref(x)[1] for x in dup["affected_ids"])
        paid = {r["payment_id"] for r in self.episode.tables["ledger"]
                if r["payment_id"] is not None and r["invoice_id"] in copies}
        self.assertEqual(len(paid), 1)
        payment = self.episode.index["payments"][paid.pop()]
        self.assertEqual(payment["invoice_refs"], copies)
        self.assertEqual(len(copies), 2)

    def test_unidentified_receipt_has_null_refs(self):
        unknown = [p for p in self.payments() if p["entity_id"] is None]
        self.assertEqual(len(unknown), 1)
        self.assertIsNone(unknown[0]["invoice_refs"])

    def test_refs_are_sorted_known_invoices_or_null(self):
        for p in self.payments():
            with self.subTest(payment=p["payment_id"]):
                refs = p["invoice_refs"]
                if refs is None:
                    continue
                self.assertTrue(refs)
                self.assertEqual(refs, sorted(set(refs)))
                for rid in refs:
                    self.assertIn(rid, self.episode.index["invoices"])


if __name__ == "__main__":
    unittest.main()
