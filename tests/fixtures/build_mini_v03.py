"""Mini test episode in the case §03 layout, converted from golden v0.1 (role B).

A stand-in until A delivers golden v0.2: the same company, rows and issues as
golden_v0.1/episode, rewritten into the file schema closed by PM ("Схема §03,
закрываем поля", docs/PM_DECISIONS.md):

    public/manifest.json, policy.json, ledger.csv, invoices.jsonl, payments.jsonl, fx.csv
    private/truth.json                      schema v2, no je/ ids

counterparties.csv and allocations.csv are dropped (PM default rule): the name goes
into invoices.jsonl, allocations stay visible as 1210/3310 clearing rows that carry
both payment_id and invoice_id. New doc_id and fx_id values come from
random.Random(SEED) and never collide with any golden id.

Deterministic: no set or dict iteration reaches the output unsorted, no hash(),
no clock. Output is byte-identical across runs and PYTHONHASHSEED values.

Usage: python3 tests/fixtures/build_mini_v03.py <out_dir>
"""
import copy
import csv
import datetime as dt
import io
import json
import random
import re
import sys
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent / "golden_v0.1" / "episode"
SEED = 20261120
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
SCHEMA_VERSION = "case03-mini-1"
BASE = "KZT"
TIMEZONE = "+05:00"
ROUNDING_MODE = "ROUND_HALF_UP"
LEDGER_TIME = "T12:00:00"
DUE_TIME = "T00:00:00"
WEEKEND_DAYS = [6, 7]  # ISO weekday numbers: Saturday, Sunday

# Case §03 mandatory columns first, in order, then the domain fields.
LEDGER_COLUMNS = (
    "row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
    "entity_id", "invoice_id", "payment_id",
    "je_id", "entered_at", "doc_currency", "doc_amount_minor", "exchange_rate", "memo",
)
FX_COLUMNS = ("date", "source_currency", "target_currency", "rate", "fx_id")

ACCOUNTS = (
    ("1030", "Bank"),
    ("1210", "Accounts receivable"),
    ("3310", "Accounts payable"),
    ("3510", "Advances received"),
    ("6010", "Revenue"),
    ("6250", "FX gain"),
    ("7110", "Cost of services"),
    ("7210", "Administrative expenses"),
    ("7430", "FX loss"),
)
# Carrier line of a journal entry (SPEC.md §3.6): 1030 for a payment, 1210 or 3310 for an invoice.
CARRIER_ACCOUNTS = {"payment": frozenset({"1030"}), "invoice": frozenset({"1210", "3310"})}
CLEARING_ACCOUNTS = frozenset({"1210", "3310"})


# ---------------------------------------------------------------- input
def read_csv(name):
    with open(GOLDEN / "public" / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(rel):
    with open(GOLDEN / rel, encoding="utf-8") as f:
        return json.load(f)


def null(value):
    """Empty golden strings become explicit nulls (case §03)."""
    return value if value != "" else None


def norm_ref(text):
    """Policy vendor_ref_normalization: uppercase, keep A-Z and 0-9 only."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


class IdSource:
    """6-char Crockford Base32 ids from one seeded generator, never reusing a taken id."""

    def __init__(self, seed, taken):
        self.rng = random.Random(seed)
        self.taken = set(taken)

    def new(self):
        while True:
            candidate = "".join(self.rng.choice(CROCKFORD) for _ in range(6))
            if candidate not in self.taken:
                self.taken.add(candidate)
                return candidate


def golden_ids(g, truth):
    """Every id the golden episode already uses, so new ids cannot alias any of them."""
    ids = []
    ids += [r["invoice_id"] for r in g["invoices"]]
    ids += [r["payment_id"] for r in g["payments"]]
    ids += [r["row_id"] for r in g["ledger"]]
    ids += [r["je_id"] for r in g["ledger"]]
    ids += [r["line_id"] for r in g["lines"]]
    ids += [r["counterparty_id"] for r in g["counterparties"]]
    ids += [r["allocation_id"] for r in g["allocations"]]
    ids += [i["issue_id"] for i in truth["issues"]]
    ids.append(truth["episode_id"])
    return ids


# ---------------------------------------------------------------- public tables
def convert_ledger(rows):
    out = []
    for r in rows:
        source_type, source_id = r["source_type"], r["source_id"]
        if source_type not in CARRIER_ACCOUNTS:
            raise SystemExit(f"ledger {r['row_id']}: unknown source_type {source_type!r}")
        invoice_id = source_id if source_type == "invoice" else r["settles_invoice_id"]
        payment_id = source_id if source_type == "payment" else ""
        out.append({
            "row_id": r["row_id"],
            "timestamp": r["posting_date"] + LEDGER_TIME + TIMEZONE,
            "account": r["account"],
            "debit_minor": r["debit_minor"],
            "credit_minor": r["credit_minor"],
            "currency": BASE,
            "entity_id": null(r["counterparty_id"]),
            "invoice_id": null(invoice_id),
            "payment_id": null(payment_id),
            "je_id": r["je_id"],
            "entered_at": r["entered_at"],
            "doc_currency": r["doc_currency"],
            "doc_amount_minor": r["doc_amount_minor"],
            "exchange_rate": r["exchange_rate"],
            "memo": null(r["memo"]),
        })
    return out


def convert_invoices(g, ids):
    names = {c["counterparty_id"]: c["name"] for c in g["counterparties"]}
    lines = {}
    for line in g["lines"]:
        lines.setdefault(line["invoice_id"], []).append({
            "line_id": line["line_id"],
            "description": line["description"],
            "qty": int(line["qty"]),
            "unit_price_minor": int(line["unit_price_minor"]),
            "amount_minor": int(line["amount_minor"]),
        })
    out = []
    for inv in g["invoices"]:
        items = lines.get(inv["invoice_id"], [])
        if not items or sum(i["amount_minor"] for i in items) != int(inv["total_minor"]):
            raise SystemExit(f"invoice {inv['invoice_id']}: line items do not add up to the total")
        out.append({
            "doc_id": ids.new(),
            "invoice_id": inv["invoice_id"],
            "entity_id": inv["counterparty_id"],
            "issued_at": inv["issued_at"],
            "due_at": inv["due_date"] + DUE_TIME + TIMEZONE,
            "currency": inv["currency"],
            "total_minor": int(inv["total_minor"]),
            "line_items": items,
            "direction": inv["direction"],
            "counterparty_name": names[inv["counterparty_id"]],
            "invoice_number": null(inv["invoice_number"]),
            "vendor_ref": null(inv["vendor_ref"]),
            "po_ref": null(inv["po_ref"]),
            "entered_at": inv["entered_at"],
            "service_from": null(inv["service_from"]),
            "service_to": null(inv["service_to"]),
        })
    return out


def invoice_refs(remittance, invoices):
    """invoice_ids the remittance text points to: AR invoice_number as written, AP vendor_ref normalized.

    A reference that fits several invoices lists all of them (C2). No reference gives null.
    """
    text = remittance or ""
    normalized = norm_ref(text)
    refs = set()
    for inv in invoices:
        number, vendor_ref = inv["invoice_number"], inv["vendor_ref"]
        if inv["direction"] == "AR" and number and number in text:
            refs.add(inv["invoice_id"])
        if inv["direction"] == "AP" and vendor_ref and norm_ref(vendor_ref) in normalized:
            refs.add(inv["invoice_id"])
    return sorted(refs) or None


def convert_payments(g, ids):
    entity = {}
    for r in g["ledger"]:
        if r["source_type"] == "payment":
            entity.setdefault(r["source_id"], set()).add(r["counterparty_id"])
    out = []
    for p in g["payments"]:
        found = entity.get(p["payment_id"])
        if not found or len(found) != 1:
            raise SystemExit(f"payment {p['payment_id']}: expected one counterparty on its ledger rows")
        (entity_id,) = sorted(found)
        out.append({
            "doc_id": ids.new(),
            "payment_id": p["payment_id"],
            # "Z" spelled as an explicit numeric offset, like every other time in the episode.
            "timestamp": dt.datetime.fromisoformat(p["value_at"]).isoformat(),
            "currency": p["currency"],
            "amount_minor": int(p["amount_minor"]),
            "invoice_refs": invoice_refs(p["remittance_info"], g["invoices"]),
            "entity_id": null(entity_id),
            "direction": p["direction"],
            "counterparty_name_raw": p["counterparty_name_raw"],
            "entered_at": p["entered_at"],
            "remittance_info": null(p["remittance_info"]),
        })
    return out


def convert_fx(rows, ids):
    out = []
    for r in rows:
        if r["base"] != BASE:
            raise SystemExit(f"fx {r['date']} {r['currency']}: target is not {BASE}")
        out.append({
            "date": r["date"],
            "source_currency": r["currency"],
            "target_currency": r["base"],
            "rate": r["rate"],
            "fx_id": ids.new(),
        })
    return out


def convert_policy(policy, ledger):
    policy = copy.deepcopy(policy)
    fx = policy["fx"]
    days = fx.pop("non_business_days")
    holidays = sorted(d for d in days if d != "weekends")
    if "weekends" not in days or len(holidays) + 1 != len(days):
        raise SystemExit(f"policy.fx.non_business_days not understood: {days!r}")
    fx["weekend_days"] = list(WEEKEND_DAYS)
    fx["holidays"] = holidays
    # payments.value_at is called timestamp in case §03.
    if "payment.value_at" not in policy["posting_date_rule"]:
        raise SystemExit("policy.posting_date_rule no longer names payment.value_at")
    policy["posting_date_rule"] = policy["posting_date_rule"].replace("payment.value_at", "payment.timestamp")
    used = sorted({r["account"] for r in ledger})
    if used != [code for code, _ in ACCOUNTS]:
        raise SystemExit(f"ledger accounts {used} differ from the chart of accounts")
    policy["accounts"] = [{"code": code, "name": name} for code, name in ACCOUNTS]
    return policy


def build_manifest(episode_id, policy, tables):
    period = policy["period"]
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "base_currency": policy["base_currency"],
        "period_start": period["start_local"],
        "period_end": period["end_local"],
        "timezone": policy["company_utc_offset"],
        "counts": {name: len(tables[name]) for name in ("ledger", "invoices", "payments", "fx")},
        "documents": len(tables["invoices"]) + len(tables["payments"]),
        "rounding": {"mode": ROUNDING_MODE, "minor_units": dict(sorted(policy["minor_units"].items()))},
    }


# ---------------------------------------------------------------- private truth
def carrier_row(je_id, golden_ledger):
    """Same rule as tests/golden.py: the 1030 line of a payment, the 1210/3310 line of an invoice."""
    rows = [
        r for r in golden_ledger
        if r["je_id"] == je_id
        and r["account"] in CARRIER_ACCOUNTS[r["source_type"]]
        and not r["settles_invoice_id"]
    ]
    if len(rows) != 1:
        raise SystemExit(f"je {je_id}: {len(rows)} carrier rows")
    return f"ledger/{rows[0]['row_id']}"


def clearing_row(allocation, ledger):
    """The 1210/3310 line in the allocation's journal entry that settles the allocated invoice."""
    rows = [
        r for r in ledger
        if r["je_id"] == allocation["je_id"]
        and r["account"] in CLEARING_ACCOUNTS
        and r["invoice_id"] == allocation["invoice_id"]
        and r["payment_id"] == allocation["payment_id"]
    ]
    if len(rows) != 1:
        raise SystemExit(f"allocation {allocation['allocation_id']}: {len(rows)} clearing rows")
    row = rows[0]
    if row["doc_amount_minor"] != allocation["amount_minor"] or row["doc_currency"] != allocation["currency"]:
        raise SystemExit(f"allocation {allocation['allocation_id']}: amount differs from its clearing row")
    return f"ledger/{row['row_id']}"


def convert_ref(ref, g, ledger, fx_rows):
    prefix, _, rest = ref.partition("/")
    if prefix == "je":
        return carrier_row(rest, g["ledger"])
    if prefix == "fx":
        date, currency, target = rest.split("/")
        found = [r["fx_id"] for r in fx_rows
                 if (r["date"], r["source_currency"], r["target_currency"]) == (date, currency, target)]
        if len(found) != 1:
            raise SystemExit(f"{ref}: {len(found)} fx rows")
        return f"fx/{found[0]}"
    if prefix == "allocation":
        (allocation,) = [a for a in g["allocations"] if a["allocation_id"] == rest]
        return clearing_row(allocation, ledger)
    if prefix in ("invoice", "payment", "ledger"):
        return ref
    raise SystemExit(f"unknown id form in golden truth: {ref!r}")


def convert_truth(truth, g, ledger, fx_rows):
    issues = []
    for i in truth["issues"]:
        affected = sorted(convert_ref(x, g, ledger, fx_rows) for x in i["affected_ids"])
        evidence = sorted(convert_ref(x, g, ledger, fx_rows) for x in i["evidence"])
        if len(set(affected)) != len(affected) or len(set(evidence)) != len(evidence):
            raise SystemExit(f"issue {i['issue_id']}: repeated id after conversion")
        issues.append({
            "issue_id": i["issue_id"],
            "type": i["type"],
            "affected_ids": affected,
            "expected_amount_minor": i["amount_minor"],
            "currency": i["currency"],
            "evidence_sets": [evidence],
            "tolerance_minor": i["tolerance_minor"],
            "scenario_id": i["scenario_id"],
        })
    issues.sort(key=lambda x: x["issue_id"])
    return {"clean": not issues, "episode_id": truth["episode_id"], "issues": issues}


# ---------------------------------------------------------------- output
def csv_text(rows, columns):
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for r in rows:
        if tuple(r) != columns:
            raise SystemExit("csv row keys out of column order")
        writer.writerow(["" if r[c] is None else r[c] for c in columns])
    return buf.getvalue()


def jsonl_text(rows):
    return "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)


def json_text(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def build():
    """{relative path: file text} of the whole episode."""
    g = {
        "invoices": read_csv("invoices.csv"),
        "lines": read_csv("invoice_lines.csv"),
        "payments": read_csv("payments.csv"),
        "ledger": read_csv("ledger.csv"),
        "fx": read_csv("fx.csv"),
        "counterparties": read_csv("counterparties.csv"),
        "allocations": read_csv("allocations.csv"),
    }
    golden_policy = read_json("public/policy.json")
    golden_truth = read_json("private/truth.json")
    if golden_policy["base_currency"] != BASE or golden_policy["company_utc_offset"] != TIMEZONE:
        raise SystemExit("golden base currency or offset changed")

    ids = IdSource(SEED, golden_ids(g, golden_truth))
    tables = {"ledger": convert_ledger(g["ledger"])}
    tables["invoices"] = convert_invoices(g, ids)
    tables["payments"] = convert_payments(g, ids)
    tables["fx"] = convert_fx(g["fx"], ids)
    policy = convert_policy(golden_policy, tables["ledger"])
    manifest = build_manifest(golden_truth["episode_id"], golden_policy, tables)
    truth = convert_truth(golden_truth, g, tables["ledger"], tables["fx"])
    return {
        "public/manifest.json": json_text(manifest),
        "public/policy.json": json_text(policy),
        "public/ledger.csv": csv_text(tables["ledger"], LEDGER_COLUMNS),
        "public/invoices.jsonl": jsonl_text(tables["invoices"]),
        "public/payments.jsonl": jsonl_text(tables["payments"]),
        "public/fx.csv": csv_text(tables["fx"], FX_COLUMNS),
        "private/truth.json": json_text(truth),
    }


def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: python3 build_mini_v03.py <out_dir>")
    out = Path(argv[1])
    files = build()
    for rel in sorted(files):
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(files[rel])
    print(f"wrote {len(files)} files to {out}")


if __name__ == "__main__":
    main(sys.argv)
