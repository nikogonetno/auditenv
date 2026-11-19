"""Recompute every truth issue from public files + the evidence ids in truth.json only.

Mirrors what an independent verifier (role B, or the jury) must be able to do.
Usage: python -I verify_golden.py <episode_dir>
"""
import csv
import datetime as dt
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP

ep = sys.argv[1]
pub = lambda n: list(csv.DictReader(open(f"{ep}/public/{n}", encoding="utf-8")))
policy = json.load(open(f"{ep}/public/policy.json"))
truth = json.load(open(f"{ep}/private/truth.json"))
OFF = dt.timedelta(hours=int(policy["company_utc_offset"][1:3]))
BASE = policy["base_currency"]

tables = {
    "invoice": {r["invoice_id"]: r for r in pub("invoices.csv")},
    "payment": {r["payment_id"]: r for r in pub("payments.csv")},
    "ledger": {r["row_id"]: r for r in pub("ledger.csv")},
    "allocation": {r["allocation_id"]: r for r in pub("allocations.csv")},
}
fx = {(r["date"], r["currency"]): r["rate"] for r in pub("fx.csv")}


def get(eid):
    kind, rest = eid.split("/", 1)
    if kind == "fx":
        d, ccy, base = rest.split("/")
        return {"date": d, "currency": ccy, "rate": fx[(d, ccy)]}
    return tables[kind][rest]


def local_date(ts):
    return (dt.datetime.fromisoformat(ts).astimezone(dt.timezone.utc) + OFF).date()


def conv(m, r):
    return int((Decimal(m) * Decimal(r)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


p_start = dt.date.fromisoformat(policy["period"]["start_local"])
p_end = dt.date.fromisoformat(policy["period"]["end_local"])
in_period = lambda d: p_start <= d <= p_end

ok = 0
for iss in truth["issues"]:
    ev = {e: get(e) for e in iss["evidence"]}
    by = lambda k: [v for e, v in ev.items() if e.startswith(k + "/")]
    t = iss["type"]
    if t == "duplicate_invoice":
        a, b = by("invoice")
        assert a["direction"] == b["direction"] == "AP"
        assert a["counterparty_id"] == b["counterparty_id"] and a["currency"] == b["currency"]
        assert a["total_minor"] == b["total_minor"] and norm(a["vendor_ref"]) == norm(b["vendor_ref"])
        carriers = {r["source_id"] for r in by("ledger") if r["account"] == "3310"}
        assert carriers == {a["invoice_id"], b["invoice_id"]}, "both copies must be posted"
        got = int(b["total_minor"])
    elif t == "payment_allocation":
        (p,), (inv,) = by("payment"), by("invoice")
        assert inv["invoice_number"] and inv["invoice_number"] in p["remittance_info"]
        assert int(p["amount_minor"]) <= int(inv["total_minor"])
        allocated = sum(int(x["amount_minor"]) for x in by("allocation") if x["payment_id"] == p["payment_id"])
        (u,) = [r for r in by("ledger") if r["account"] == "3510"]
        got = int(p["amount_minor"]) - allocated
        assert int(u["doc_amount_minor"]) == got, "3510 row must hold exactly the unallocated amount"
    elif t == "fx_rate":
        (car,) = by("ledger")
        (doc,) = by("invoice") or by("payment")
        (f,) = by("fx")
        ts = doc.get("issued_at") or doc.get("value_at")
        d = local_date(ts)
        while d.weekday() >= 5 or d.isoformat() in policy["fx"]["non_business_days"]:
            d -= dt.timedelta(days=1)
        assert f["date"] == d.isoformat(), "fx evidence must be the policy-date row"
        posted = int(car["debit_minor"]) or int(car["credit_minor"])
        got = abs(posted - conv(int(car["doc_amount_minor"]), f["rate"]))
    elif t == "period_cutoff":
        (car,) = by("ledger")
        (doc,) = by("invoice") or by("payment")
        ts = doc.get("issued_at") or doc.get("value_at")
        right = local_date(ts)
        posted = dt.date.fromisoformat(car["posting_date"])
        assert in_period(right) != in_period(posted), "posting must cross the period boundary"
        got = int(car["debit_minor"]) or int(car["credit_minor"])
    else:
        raise AssertionError(t)
    assert abs(got - iss["amount_minor"]) <= iss["tolerance_minor"], (t, got, iss["amount_minor"])
    ok += 1
    print(f"OK {t:20s} {iss['scenario_id']:3s} amount={iss['amount_minor']} {iss['currency']}")
print(f"{ok}/{len(truth['issues'])} issues recomputed from evidence only")
