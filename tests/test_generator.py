"""Generator stages 1 and 2 (docs/tasks/A_generator.md, docs/tasks/B_stage2.md): clean world and all four error types."""
import csv
import hashlib
import json
import re
import tempfile
import time
import unittest
from pathlib import Path

from auditenv.config import normalize_config
from auditenv.env import AuditEnv
from auditenv.episode import load_episode
from auditenv.generate import generate_episode
from auditenv.grader import parse_truth
from auditenv.runner import run_inprocess_episode
from auditenv.verify import verify_episode
from auditenv.world import WorldError, build_world, check_book, post
from policies.oracle import make_env_oracle
from tests.support import run_python

ROOT = Path(__file__).resolve().parent.parent
STAGE1 = {"errors": {"duplicate_invoice": 1, "payment_allocation": 1}}
CHILD = (
    "import sys, json\n"
    f"sys.path.insert(0, {str(ROOT)!r})\n"
    "from auditenv.generate import generate_episode\n"
    "generate_episode(int(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3])\n"
)


def digest(root):
    h = hashlib.sha256()
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        h.update(path.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def oracle_reward(directory):
    env = AuditEnv("train")
    run_inprocess_episode(make_env_oracle(env), env, 0, {"episode_dir": str(directory)})
    return env.grader_log[-1]


class Generated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dirs = {}
        for seed, config in ((3, STAGE1), (11, {"errors": {"duplicate_invoice": 2, "payment_allocation": 3}}),
                             (5, {"errors": {}}), (8, {"documents": [20, 20], **STAGE1})):
            out = Path(cls.tmp.name) / f"{seed}"
            started = time.monotonic()
            generate_episode(seed, config, str(out))
            cls.dirs[seed] = (out, normalize_config(config), time.monotonic() - started)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_episode_loads_truth_parses_and_oracle_gets_one(self):
        for seed, (out, config, _) in self.dirs.items():
            with self.subTest(seed=seed):
                episode = load_episode(out)
                truth = episode.load_truth()
                parse_truth(truth, episode.aliases)
                entry = oracle_reward(out)
                self.assertEqual((entry["reward"], entry["fp"], entry["fn"]), (1.0, 0, 0))
                expected = sum(config["errors"].values())
                self.assertEqual((len(truth["issues"]), truth["clean"]), (expected, expected == 0))

    def test_independent_verifier_finds_no_problem(self):
        for seed, (out, _, _) in self.dirs.items():
            with self.subTest(seed=seed):
                self.assertEqual(verify_episode(str(out)), [])

    def test_volume_follows_the_config(self):
        for seed, (out, config, seconds) in self.dirs.items():
            with self.subTest(seed=seed):
                manifest = json.loads((out / "public" / "manifest.json").read_text(encoding="utf-8"))
                low, high = config["documents"]
                self.assertTrue(low <= manifest["documents"] <= high, manifest["documents"])
                self.assertLessEqual(manifest["counts"]["ledger"], 2000)
                self.assertLess(seconds, 5)

    def test_ids_are_unique_across_tables(self):
        out = self.dirs[11][0]
        episode = load_episode(out)
        ids = [episode.episode_id]
        for table, rows in episode.tables.items():
            id_col = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": "fx_id"}[table]
            ids += [r[id_col] for r in rows]
        ids += [r["doc_id"] for r in episode.tables["invoices"] + episode.tables["payments"]]
        ids += [line["line_id"] for r in episode.tables["invoices"] for line in r["line_items"]]
        ids += sorted({r["je_id"] for r in episode.tables["ledger"]})
        ids += sorted({r["entity_id"] for r in episode.tables["invoices"]})
        ids += [i["issue_id"] for i in episode.load_truth()["issues"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{6}", i) for i in ids))

    def test_provenance_names_seed_config_and_every_issue(self):
        out, config, _ = self.dirs[11]
        provenance = json.loads((out / "private" / "provenance.json").read_text(encoding="utf-8"))
        truth = json.loads((out / "private" / "truth.json").read_text(encoding="utf-8"))
        self.assertEqual((provenance["seed"], provenance["config"]), (11, config))
        named = sorted(i for inj in provenance["injections"] for i in inj["issue_ids"])
        self.assertEqual(named, sorted(i["issue_id"] for i in truth["issues"]))
        self.assertTrue(all(inj["changes"] for inj in provenance["injections"]))
        self.assertEqual(json.loads((out / "public" / "policy.json").read_text())["fx"]["weekend_days"], [6, 7])


class Realism(unittest.TestCase):
    """Findings of the stage-1 review (21.11): the data must not give errors away by one field, and the
    declared rounding must actually round."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.episodes = []
        for seed in range(12):
            out = Path(cls.tmp.name) / str(seed)
            generate_episode(100 + seed, {"errors": {"duplicate_invoice": 2, "payment_allocation": 2}}, str(out))
            cls.episodes.append(out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def read(self, out, name):
        return [json.loads(line) for line in (out / "public" / name).read_text(encoding="utf-8").splitlines()]

    def created(self, out):
        provenance = json.loads((out / "private" / "provenance.json").read_text(encoding="utf-8"))
        return {c["record"].split("/")[1] for inj in provenance["injections"] for c in inj["changes"]
                if c["after"] == "created"}

    def test_copy_reference_format_also_occurs_in_clean_bills(self):
        from auditenv.world import ref_shape
        hits = total = 0
        for out in self.episodes:
            created = self.created(out)
            bills = [i for i in self.read(out, "invoices.jsonl") if i["direction"] == "AP"]
            clean = {ref_shape(i["vendor_ref"]) for i in bills if i["invoice_id"] not in created}
            for bill in bills:
                if bill["invoice_id"] in created:
                    total += 1
                    hits += ref_shape(bill["vendor_ref"]) in clean
        self.assertGreaterEqual(hits, total - 1)

    def test_clean_payments_also_cite_two_documents(self):
        for out in self.episodes[:4]:
            created = self.created(out)
            multi = [p for p in self.read(out, "payments.jsonl")
                     if p["invoice_refs"] and len(p["invoice_refs"]) > 1 and not created & set(p["invoice_refs"])]
            with self.subTest(out=out.name):
                self.assertTrue(multi)

    def test_amount_equal_to_the_cited_total_with_3510_is_not_only_an_error(self):
        """A clean full-total payment after a partial one also leaves part on 3510 (review finding 18)."""
        clean_hits = 0
        for out in self.episodes:
            truth = json.loads((out / "private" / "truth.json").read_text(encoding="utf-8"))
            errors = {x.split("/")[1] for i in truth["issues"] for x in i["affected_ids"] if x.startswith("payment/")}
            totals = {i["invoice_id"]: i["total_minor"] for i in self.read(out, "invoices.jsonl")}
            with open(out / "public" / "ledger.csv", encoding="utf-8") as f:
                held = {r["payment_id"] for r in csv.DictReader(f) if r["account"] == "3510" and r["entity_id"]}
            clean_hits += sum(1 for p in self.read(out, "payments.jsonl")
                              if p["payment_id"] in held and p["payment_id"] not in errors and p["invoice_refs"]
                              and len(p["invoice_refs"]) == 1 and p["amount_minor"] == totals[p["invoice_refs"][0]])
        self.assertGreaterEqual(clean_hits, len(self.episodes))

    def test_conversions_round(self):
        from decimal import Decimal
        inexact = 0
        for out in self.episodes:
            with open(out / "public" / "ledger.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                product = Decimal(row["doc_amount_minor"]) * Decimal(row["exchange_rate"])
                inexact += product != product.to_integral_value()
        self.assertGreater(inexact, 0)


class Capacity(unittest.TestCase):
    def test_capacity_depends_on_the_config_not_on_the_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed in range(25):
                generate_episode(seed, {"documents": [20, 200], "errors": {"payment_allocation": 12}},
                                 str(Path(tmp) / str(seed)))
                with self.assertRaises(ValueError):
                    generate_episode(seed, {"documents": [20, 30], "errors": {"payment_allocation": 12}},
                                     str(Path(tmp) / f"bad{seed}"))

    def test_an_id_never_spells_a_private_word(self):
        from auditenv.generate import IdSource

        class Scripted:
            def __init__(self, text):
                self.chars = iter(text)

            def choice(self, _):
                return next(self.chars)

        self.assertEqual(IdSource(Scripted("FXRATE" + "7Q4M2K")).new(), "7Q4M2K")

    def test_minus_zero_offset_is_written_as_utc(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate_episode(3, {"company_offsets": ["-00:00"], "errors": {}}, tmp)
            texts = [p.read_text(encoding="utf-8") for p in (Path(tmp) / "public").iterdir()]
            self.assertFalse(any("-00:00" in t for t in texts))


def read_episode(out):
    """(invoices, payments, ledger rows by id, truth, provenance) of a generated episode."""
    invoices = [json.loads(x) for x in (out / "public" / "invoices.jsonl").read_text(encoding="utf-8").splitlines()]
    payments = [json.loads(x) for x in (out / "public" / "payments.jsonl").read_text(encoding="utf-8").splitlines()]
    with open(out / "public" / "ledger.csv", encoding="utf-8") as f:
        ledger = {r["row_id"]: r for r in csv.DictReader(f)}
    truth = json.loads((out / "private" / "truth.json").read_text(encoding="utf-8"))
    provenance = json.loads((out / "private" / "provenance.json").read_text(encoding="utf-8"))
    return invoices, payments, ledger, truth, provenance


class Stage2(unittest.TestCase):
    """fx_rate rate_of_entry_date, period_cutoff pulled_into_period, clean companies (B_stage2.md items 1-6, 9)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.runs = {}
        configs = {"default": {}, "rate": {"errors": {"fx_rate": 3}}, "period": {"errors": {"period_cutoff": 3}},
                   "clean": {"errors": {}}, "october": {"period": "2026-10", "errors": {"fx_rate": 4}}}
        for name, config in configs.items():
            for seed in range(3):
                out = Path(cls.tmp.name) / f"{name}{seed}"
                generate_episode(200 + seed, config, str(out))
                cls.runs[(name, seed)] = out

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def issues(self, name, kind):
        for seed in range(3):
            out = self.runs[(name, seed)]
            invoices, payments, ledger, truth, provenance = read_episode(out)
            policy = json.loads((out / "public" / "policy.json").read_text(encoding="utf-8"))
            for issue in truth["issues"]:
                if issue["type"] == kind:
                    yield out, issue, invoices, payments, ledger, policy

    def test_default_config_has_one_error_of_each_type_and_verifies(self):
        for seed in range(3):
            out = self.runs[("default", seed)]
            truth = read_episode(out)[3]
            with self.subTest(seed=seed):
                self.assertEqual(sorted(i["type"] for i in truth["issues"]),
                                 ["duplicate_invoice", "fx_rate", "payment_allocation", "period_cutoff"])
                self.assertEqual(verify_episode(str(out)), [])
                self.assertEqual(oracle_reward(out)["reward"], 1.0)

    def test_rate_amount_is_the_carrier_base_against_the_rule_rate(self):
        from decimal import ROUND_HALF_UP, Decimal
        seen = 0
        for out, issue, invoices, _, ledger, policy in self.issues("rate", "fx_rate"):
            (row,) = [ledger[x.split("/")[1]] for x in issue["affected_ids"]]
            evidence = issue["evidence_sets"][0]
            (doc,) = [i for i in invoices if f"invoice/{i['invoice_id']}" in evidence]
            fx_ids = [x.split("/")[1] for x in evidence if x.startswith("fx/")]
            with open(out / "public" / "fx.csv", encoding="utf-8") as f:
                (quote,) = [r for r in csv.DictReader(f) if r["fx_id"] in fx_ids]
            right = int((Decimal(row["doc_amount_minor"]) * Decimal(quote["rate"])).quantize(Decimal(1), ROUND_HALF_UP))
            base = int(row["debit_minor"]) or int(row["credit_minor"])
            self.assertEqual(issue["expected_amount_minor"], abs(base - right))
            self.assertGreater(issue["expected_amount_minor"], 2 * issue["tolerance_minor"])
            self.assertEqual((issue["currency"], row["account"] in ("1210", "3310")), ("KZT", True))
            self.assertNotEqual(row["exchange_rate"], quote["rate"])
            self.assertEqual(quote["source_currency"], doc["currency"])
            # Payments of the invoice clear at the booked rate: no second error in the payment entry.
            for clearing in ledger.values():
                if clearing["invoice_id"] == doc["invoice_id"] and clearing["payment_id"]:
                    self.assertEqual(clearing["exchange_rate"], row["exchange_rate"])
            seen += 1
        self.assertEqual(seen, 9)
        for seed in range(3):
            self.assertEqual(verify_episode(str(self.runs[("rate", seed)])), [])

    def test_rate_error_needs_another_rate_on_the_entry_date(self):
        """31 October and 1 November 2026 are Saturday and Sunday: both take the rate of Friday 30 October."""
        import datetime as dt
        import random
        from auditenv.world import entry_rate_differs
        world = build_world(random.Random(5), normalize_config({"period": "2026-10", "foreign_pct": [100, 100],
                                                                "errors": {}}))
        inv = next(i for i in world.invoices.values() if i.direction == "AR")
        inv.issued_at = dt.datetime(2026, 10, 31, 11, 0, tzinfo=world.tz)
        inv.entered_at = dt.datetime(2026, 11, 1, 12, 0, tzinfo=world.tz)
        self.assertFalse(entry_rate_differs(world, inv))
        inv.entered_at = dt.datetime(2026, 11, 3, 12, 0, tzinfo=world.tz)
        quotes = world.rates
        if quotes.rate(inv.currency, dt.date(2026, 10, 30)) != quotes.rate(inv.currency, dt.date(2026, 11, 3)):
            self.assertTrue(entry_rate_differs(world, inv))
        for seed in range(3):
            out = self.runs[("october", seed)]
            invoices, _, ledger, truth, _ = read_episode(out)
            self.assertEqual(verify_episode(str(out)), [])
            for issue in truth["issues"]:
                self.assertGreater(issue["expected_amount_minor"], 2)

    def test_period_error_moves_a_base_currency_document_of_the_next_month(self):
        import datetime as dt
        seen = 0
        for out, issue, invoices, _, ledger, policy in self.issues("period", "period_cutoff"):
            (row,) = [ledger[x.split("/")[1]] for x in issue["affected_ids"]]
            (doc,) = [i for i in invoices if f"invoice/{i['invoice_id']}" in issue["evidence_sets"][0]]
            tz = dt.datetime.fromisoformat("2026-01-01T00:00:00" + policy["company_utc_offset"]).tzinfo
            end = dt.date.fromisoformat(policy["period"]["end_local"])
            dated = dt.datetime.fromisoformat(doc["issued_at"]).astimezone(tz).date()
            self.assertEqual(doc["currency"], "KZT")
            self.assertTrue(end < dated <= end + dt.timedelta(days=5))
            self.assertEqual(row["timestamp"][:10], end.isoformat())
            self.assertEqual(issue["expected_amount_minor"], int(row["debit_minor"]) or int(row["credit_minor"]))
            self.assertEqual(sorted(issue["evidence_sets"][0]), sorted([f"invoice/{doc['invoice_id']}",
                                                                        f"ledger/{row['row_id']}"]))
            seen += 1
        self.assertEqual(seen, 9)
        for seed in range(3):
            self.assertEqual(verify_episode(str(self.runs[("period", seed)])), [])

    def test_clean_company(self):
        for seed in range(3):
            out = self.runs[("clean", seed)]
            _, _, _, truth, provenance = read_episode(out)
            kinds = {t["kind"] for t in provenance["twins"]}
            with self.subTest(seed=seed):
                self.assertEqual((truth["clean"], truth["issues"], provenance["injections"]), (True, [], []))
                self.assertTrue(3 <= len(provenance["twins"]) <= 8)
                self.assertIn("late_entry", kinds)
                self.assertEqual(verify_episode(str(out)), [])
        policies = [json.loads((self.runs[(n, 0)] / "public" / "policy.json").read_text()) for n in ("clean", "default")]
        self.assertEqual(*(sorted(self.flat_keys(p)) for p in policies))

    def flat_keys(self, obj, prefix=""):
        if isinstance(obj, dict):
            return [k for key, value in obj.items() for k in self.flat_keys(value, f"{prefix}/{key}")] or [prefix]
        return [prefix]

    def test_rate_and_period_look_alikes_are_listed(self):
        out = self.runs[("default", 0)]
        provenance = read_episode(out)[4]
        kinds = {t["kind"]: t["mimics"] for t in provenance["twins"]}
        self.assertEqual((kinds.get("weekend_document"), kinds.get("late_entry")), ("fx_rate", "period_cutoff"))


class Dates(unittest.TestCase):
    """A trading company works on business days: 2-6% of documents on weekends and holidays, and every book
    with foreign documents keeps a foreign invoice on an off day as the rate look-alike (PM 21.11 03.15)."""

    def test_off_day_share_and_the_foreign_weekend_document(self):
        import datetime as dt
        with tempfile.TemporaryDirectory() as tmp:
            for seed in range(6):
                out = Path(tmp) / str(seed)
                generate_episode(900 + seed, {} if seed % 2 else {"errors": {}}, str(out))
                invoices, payments, _, _, _ = read_episode(out)
                policy = json.loads((out / "public" / "policy.json").read_text(encoding="utf-8"))
                tz = dt.datetime.fromisoformat("2026-01-01T00:00:00" + policy["company_utc_offset"]).tzinfo
                holidays = set(policy["fx"]["holidays"])

                def off(ts):
                    day = dt.datetime.fromisoformat(ts).astimezone(tz).date()
                    return day.isoweekday() in policy["fx"]["weekend_days"] or day.isoformat() in holidays

                days = [off(i["issued_at"]) for i in invoices] + [off(p["timestamp"]) for p in payments]
                with self.subTest(seed=seed):
                    self.assertTrue(0.02 <= sum(days) / len(days) <= 0.065, sum(days) / len(days))
                    foreign = [i for i in invoices if i["currency"] != "KZT"]
                    if foreign:
                        self.assertTrue(any(off(i["issued_at"]) for i in foreign))


class DomainReview(unittest.TestCase):
    """Fixes of the domain review of three episodes (docs/tasks/B_domain_fixes.md, PM 21.11 11.40)."""

    OFFICIAL_2026 = {"2026-01-01", "2026-01-02", "2026-01-07", "2026-03-09", "2026-03-23", "2026-03-24",
                     "2026-03-25", "2026-05-01", "2026-05-07", "2026-05-11", "2026-05-27", "2026-07-06",
                     "2026-08-31", "2026-10-26", "2026-12-16"}

    def test_holidays_of_2026_are_the_official_calendar_with_transfers(self):
        from auditenv.world import HOLIDAYS
        self.assertEqual({d.isoformat() for d in HOLIDAYS if d.year == 2026}, self.OFFICIAL_2026)
        # Rate tables of January and December books reach into the neighbour years: 16 December 2025 only,
        # and in 2027 Saturday 2 January moves to Monday 4 January.
        self.assertEqual({d.isoformat() for d in HOLIDAYS if d.year != 2026},
                         {"2025-12-16", "2027-01-01", "2027-01-04", "2027-01-07"})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "2026-12"
            generate_episode(1, {"period": "2026-12", "errors": {}}, str(out))
            policy = json.loads((out / "public" / "policy.json").read_text(encoding="utf-8"))
            self.assertIn("2027-01-04", policy["fx"]["holidays"])
            with open(out / "public" / "fx.csv", encoding="utf-8") as f:
                self.assertFalse([r for r in csv.DictReader(f) if r["date"] == "2027-01-04"])
            for period in ("2026-01", "2026-03", "2026-05", "2026-12"):
                out = Path(tmp) / period
                generate_episode(1, {"period": period, "errors": {}}, str(out))
                policy = json.loads((out / "public" / "policy.json").read_text(encoding="utf-8"))
                published = set(policy["fx"]["holidays"])
                with self.subTest(period=period):
                    self.assertTrue(published & self.OFFICIAL_2026)
                    self.assertTrue({d for d in published if d.startswith("2026")} <= self.OFFICIAL_2026)
                    self.assertNotIn("2026-12-17", published)
                    self.assertNotIn("2025-12-17", published)

    def test_off_day_ceiling_holds_when_no_earlier_business_day_fits(self):
        """25 March is a holiday and the first day of an April window, 1 May the first day after the period:
        free invoices there move later to a business day (default config, seed 5032 was 6.19%)."""
        import datetime as dt
        with tempfile.TemporaryDirectory() as tmp:
            for seed, config in ((5032, {}), *((300 + n, {"period": "2026-04", "errors": {}}) for n in range(6))):
                out = Path(tmp) / str(seed)
                generate_episode(seed, config, str(out))
                invoices, payments, _, _, _ = read_episode(out)
                policy = json.loads((out / "public" / "policy.json").read_text(encoding="utf-8"))
                tz = dt.datetime.fromisoformat("2026-01-01T00:00:00" + policy["company_utc_offset"]).tzinfo
                holidays = set(policy["fx"]["holidays"])

                def off(ts):
                    day = dt.datetime.fromisoformat(ts).astimezone(tz).date()
                    return day.isoweekday() in policy["fx"]["weekend_days"] or day.isoformat() in holidays

                days = [off(i["issued_at"]) for i in invoices] + [off(p["timestamp"]) for p in payments]
                with self.subTest(seed=seed):
                    self.assertTrue(0.02 <= sum(days) / len(days) <= 0.06, sum(days) / len(days))
                    moment = dt.datetime.fromisoformat
                    self.assertTrue(all(moment(i["issued_at"]) < moment(i["entered_at"]) for i in invoices))

    def test_account_7110_is_selling_expenses(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate_episode(1, {"errors": {}}, tmp)
            policy = json.loads((Path(tmp) / "public" / "policy.json").read_text(encoding="utf-8"))
        self.assertIn({"code": "7110", "name": "Selling expenses"}, policy["accounts"])

    def test_a_route_never_ends_in_the_city_it_starts(self):
        from auditenv.world import CITIES
        route = re.compile(r"\b(%s)-(%s)\b" % ("|".join(CITIES), "|".join(CITIES)))
        routes = 0
        with tempfile.TemporaryDirectory() as tmp:
            for seed in range(8):
                out = Path(tmp) / str(seed)
                generate_episode(700 + seed, {"errors": {}}, str(out))
                invoices, _, _, _, _ = read_episode(out)
                for line in (x for i in invoices for x in i["line_items"]):
                    for a, b in route.findall(line["description"]):
                        routes += 1
                        self.assertNotEqual(a, b, line["description"])
        self.assertGreater(routes, 20)

    def test_root_causes_name_no_copy_and_no_typo(self):
        configs = ({"errors": {"payment_allocation": 2, "duplicate_invoice": 1}},
                   {"errors": {}, "scenarios": {"C1": 1}}, {"errors": {}, "scenarios": {"C2": 1}})
        with tempfile.TemporaryDirectory() as tmp:
            for n, config in enumerate(configs):
                out = Path(tmp) / str(n)
                generate_episode(4, config, str(out))
                _, _, _, _, provenance = read_episode(out)
                texts = [i["root_cause"] for i in provenance["injections"]]
                texts += [c["cause"] for i in provenance["injections"] for c in i.get("root_causes", ())]
                texts += [t["distinguishing_feature"] for t in provenance["twins"] if t["kind"] == "c2_trap"]
                with self.subTest(config=config):
                    self.assertTrue(texts)
                    for text in texts:
                        if not config["errors"]:              # C1 and C2 alone: no ordinary duplicate
                            self.assertNotIn("copy", text)
                        self.assertNotIn("typo", text)
                        self.assertNotIn("keyed as", text)
                    allocation = [t for t in texts if "3510" in t]
                    if "C2" not in config.get("scenarios", {}):
                        self.assertTrue(allocation)
                        self.assertTrue(all("only part of the payment allocated to the invoice" in t for t in allocation))


class Determinism(unittest.TestCase):
    def test_same_seed_same_bytes_under_different_hash_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            digests = []
            for hash_seed in ("0", "1", "random"):
                out = Path(tmp) / hash_seed
                result = run_python(["-c", CHILD, "42", json.dumps(STAGE1), out], hash_seed=hash_seed)
                self.assertEqual(result.returncode, 0, result.stderr)
                digests.append(digest(out))
            generate_episode(43, STAGE1, str(Path(tmp) / "other"))
            self.assertEqual(len(set(digests)), 1)
            self.assertNotEqual(digests[0], digest(Path(tmp) / "other"))


class Refusals(unittest.TestCase):
    def test_bad_seed_and_config_are_value_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed, config in ((-1, STAGE1), (True, STAGE1), (1, {"colour": "red"}),
                                 (1, {"documents": [20, 20], "errors": {"payment_allocation": 12}})):
                with self.subTest(seed=seed, config=config), self.assertRaises(ValueError):
                    generate_episode(seed, config, tmp)

    def test_own_invariant_check_catches_a_broken_book(self):
        import random
        world = build_world(random.Random(1), normalize_config({"errors": {}}))
        book = post(world)
        check_book(world, book, set())
        book.rows[0].debit_minor += book.rows[0].debit_minor and 1
        book.rows[0].credit_minor += book.rows[0].credit_minor and 1
        with self.assertRaises(WorldError):
            check_book(world, book, set())


if __name__ == "__main__":
    unittest.main()
