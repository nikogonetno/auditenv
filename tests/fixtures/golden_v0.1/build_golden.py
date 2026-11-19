"""Golden example episode for the AI auditor RL environment (role A, data).

Builds one small, hand-checkable episode for company "Demo Logistics" (base KZT,
UTC+05:00, period 2026-09) with all four error types, both coupled scenarios,
correct twins, and private truth/provenance. Then re-verifies every truth amount
using ONLY public files plus the evidence ids listed in truth.json.

Deterministic: integer random walk for FX, Decimal with ROUND_HALF_UP,
random.Random(seed), sorted keys, no datetime.now(), no hash().
Usage: python -I build_golden.py <out_dir>
"""
import csv
import datetime as dt
import json
import os
import random
import re
import sys
from decimal import Decimal, ROUND_HALF_UP

SEED = 20260919
BASE = "KZT"
CO_OFFSET = dt.timedelta(hours=5)
PERIOD = (dt.date(2026, 9, 1), dt.date(2026, 9, 30))
WINDOW_DAYS = 7
NON_BUSINESS = {dt.date(2026, 8, 31)}
ROUND = ROUND_HALF_UP
rng = random.Random(SEED)

# ---------------------------------------------------------------- helpers
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_used = set()


def new_id():
    while True:
        s = "".join(rng.choice(CROCKFORD) for _ in range(6))
        if s not in _used:
            _used.add(s)
            return s


def is_business(d):
    return d.weekday() < 5 and d not in NON_BUSINESS


def parse_ts(s):
    return dt.datetime.fromisoformat(s)


def local_date(ts):
    """Company-local calendar date of an ISO timestamp with explicit offset."""
    return (parse_ts(ts).astimezone(dt.timezone.utc) + CO_OFFSET).date()


def conv(doc_minor, rate):
    return int((Decimal(doc_minor) * Decimal(rate)).quantize(Decimal(1), rounding=ROUND))


def norm_ref(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


# ---------------------------------------------------------------- FX table
# integer random walk in hundredths of KZT, only on business days
fx = {}  # (date, ccy) -> "512.40"
level = {"USD": 51240, "EUR": 59810}
d = dt.date(2026, 8, 17)
while d <= dt.date(2026, 10, 16):
    if is_business(d):
        for ccy in ("EUR", "USD"):
            level[ccy] += rng.randint(-220, 220)
            fx[(d, ccy)] = f"{level[ccy] // 100}.{level[ccy] % 100:02d}"
    d += dt.timedelta(days=1)


def rate_date(d):
    while not is_business(d):
        d -= dt.timedelta(days=1)
    return d


def rate(ccy, d):
    if ccy == BASE:
        return "1", None
    rd = rate_date(d)
    return fx[(rd, ccy)], rd


# ---------------------------------------------------------------- world
cp = {}
for code, name, role, off in [
    ("contoso", "Contoso Ltd", "customer", "+00:00"),
    ("fabrikam", "Fabrikam Inc", "customer", "+05:00"),
    ("adventure", "Adventure Works LLC", "customer", "+03:00"),
    ("wingtip", "Wingtip Toys GmbH", "customer", "+02:00"),
    ("northwind", "Northwind Freight GmbH", "vendor", "+02:00"),
    ("tailspin", "Tailspin Hosting LLP", "vendor", "+05:00"),
    ("wideworld", "Wide World Importers", "vendor", "+05:00"),
]:
    cp[code] = {"counterparty_id": new_id(), "name": name, "role": role, "utc_offset": off}

invoices, lines, payments, allocs, ledger = [], [], [], [], []
truth_issues, provenance = [], []
twins = []


def add_invoice(key, direction, cpk, number, vendor_ref, issued_at, entered_at,
                ccy, desc, qty, unit_minor, svc_from, svc_to, terms=14, po_ref=""):
    total = qty * unit_minor
    inv = {
        "invoice_id": new_id(), "direction": direction,
        "counterparty_id": cp[cpk]["counterparty_id"],
        "invoice_number": number, "vendor_ref": vendor_ref, "po_ref": po_ref,
        "issued_at": issued_at, "entered_at": entered_at,
        "due_date": (local_date(issued_at) + dt.timedelta(days=terms)).isoformat(),
        "currency": ccy, "total_minor": total,
        "service_from": svc_from, "service_to": svc_to,
        "_key": key,
    }
    invoices.append(inv)
    lines.append({"line_id": new_id(), "invoice_id": inv["invoice_id"], "description": desc,
                  "qty": qty, "unit_price_minor": unit_minor, "amount_minor": total})
    return inv


def row(je, posting_date, entered_at, account, dr, cr, ccy, doc_minor, r,
        src_type, src_id, cp_id, settles="", memo=""):
    ledger.append({
        "row_id": new_id(), "je_id": je, "posting_date": posting_date.isoformat(),
        "entered_at": entered_at, "account": account,
        "debit_minor": dr, "credit_minor": cr,
        "doc_currency": ccy, "doc_amount_minor": doc_minor, "exchange_rate": r,
        "source_type": src_type, "source_id": src_id, "settles_invoice_id": settles,
        "counterparty_id": cp_id, "memo": memo,
    })
    return ledger[-1]


def post_invoice(inv, posting_date=None, rate_override=None, expense_acct="7110"):
    """AR: Dr 1210 / Cr 6010. AP: Dr 7110|7210 / Cr 3310. Carrier is the 1210/3310 row."""
    pd = posting_date or local_date(inv["issued_at"])
    r = rate_override or rate(inv["currency"], local_date(inv["issued_at"]))[0]
    base = conv(inv["total_minor"], r)
    je = new_id()
    ccy, amt = inv["currency"], inv["total_minor"]
    if inv["direction"] == "AR":
        car = row(je, pd, inv["entered_at"], "1210", base, 0, ccy, amt, r, "invoice",
                  inv["invoice_id"], inv["counterparty_id"], memo=inv["invoice_number"])
        row(je, pd, inv["entered_at"], "6010", 0, base, ccy, amt, r, "invoice",
            inv["invoice_id"], inv["counterparty_id"], memo=inv["invoice_number"])
    else:
        row(je, pd, inv["entered_at"], expense_acct, base, 0, ccy, amt, r, "invoice",
            inv["invoice_id"], inv["counterparty_id"], memo=inv["vendor_ref"])
        car = row(je, pd, inv["entered_at"], "3310", 0, base, ccy, amt, r, "invoice",
                  inv["invoice_id"], inv["counterparty_id"], memo=inv["vendor_ref"])
    inv["_je"], inv["_carrier"], inv["_base"], inv["_rate"] = je, car, base, r
    return je, car


open_base = {}  # invoice_id -> remaining base (for residual clearing)
open_doc = {}


def track(inv):
    open_base[inv["invoice_id"]] = inv["_base"]
    open_doc[inv["invoice_id"]] = inv["total_minor"]


def add_payment(direction, cpk, value_at, entered_at, ccy, amount, remittance, raw_name=None):
    p = {"payment_id": new_id(), "direction": direction,
         "counterparty_name_raw": raw_name or cp[cpk]["name"] if cpk else raw_name,
         "value_at": value_at, "entered_at": entered_at, "currency": ccy,
         "amount_minor": amount, "remittance_info": remittance}
    payments.append(p)
    return p


def post_payment(p, cp_id, allocations, posting_date=None, rate_override=None):
    """allocations: list of (invoice, doc_minor). Unallocated rest goes to 3510 (AR only)."""
    pd = posting_date or local_date(p["value_at"])
    r = rate_override or rate(p["currency"], local_date(p["value_at"]))[0]
    je = new_id()
    bank = conv(p["amount_minor"], r)
    ccy = p["currency"]
    rows = []
    clear_total = 0
    if p["direction"] == "in":
        car = row(je, pd, p["entered_at"], "1030", bank, 0, ccy, p["amount_minor"], r,
                  "payment", p["payment_id"], cp_id, memo=p["remittance_info"])
    for inv, a in allocations:
        if a == open_doc[inv["invoice_id"]]:
            c = open_base[inv["invoice_id"]]  # last allocation clears the residual exactly
        else:
            c = conv(a, inv["_rate"])
        open_doc[inv["invoice_id"]] -= a
        open_base[inv["invoice_id"]] -= c
        clear_total += c
        allocs.append({"allocation_id": new_id(), "payment_id": p["payment_id"],
                       "invoice_id": inv["invoice_id"], "currency": ccy, "amount_minor": a,
                       "allocated_at": p["entered_at"], "je_id": je})
        if p["direction"] == "in":
            rows.append(row(je, pd, p["entered_at"], "1210", 0, c, ccy, a, inv["_rate"], "payment",
                            p["payment_id"], cp_id, settles=inv["invoice_id"]))
        else:
            rows.append(row(je, pd, p["entered_at"], "3310", c, 0, ccy, a, inv["_rate"], "payment",
                            p["payment_id"], cp_id, settles=inv["invoice_id"]))
    allocated_doc = sum(a for _, a in allocations)
    unalloc = p["amount_minor"] - allocated_doc
    u_row = None
    if p["direction"] == "in":
        u_base = 0
        if unalloc:
            u_base = conv(unalloc, r)
            u_row = row(je, pd, p["entered_at"], "3510", 0, u_base, ccy, unalloc, r, "payment",
                        p["payment_id"], cp_id, memo="unapplied receipt")
        diff = bank - clear_total - u_base  # >0 gain
    else:
        assert unalloc == 0
        car = row(je, pd, p["entered_at"], "1030", 0, bank, ccy, p["amount_minor"], r,
                  "payment", p["payment_id"], cp_id, memo=p["remittance_info"])
        diff = clear_total - bank  # paid less base than liability -> gain
    if diff > 0:
        row(je, pd, p["entered_at"], "6250", 0, diff, BASE, diff, "1", "payment", p["payment_id"], cp_id,
            memo="realized FX gain")
    elif diff < 0:
        row(je, pd, p["entered_at"], "7430", -diff, 0, BASE, -diff, "1", "payment", p["payment_id"], cp_id,
            memo="realized FX loss")
    return je, car, u_row


def issue(itype, affected, amount, ccy, evidence, scenario, variant, cause):
    iid = new_id()
    truth_issues.append({"issue_id": iid, "type": itype, "affected_ids": sorted(affected),
                         "amount_minor": amount, "currency": ccy, "tolerance_minor": 1,
                         "evidence": sorted(evidence), "scenario_id": scenario})
    provenance.append({"issue_id": iid, "type": itype, "variant": variant, "scenario_id": scenario,
                       "root_cause": cause})
    return iid


def ev_inv(i): return f"invoice/{i['invoice_id']}"
def ev_pay(p): return f"payment/{p['payment_id']}"
def ev_row(r): return f"ledger/{r['row_id']}"
def ev_je(je): return f"je/{je}"
def ev_alloc(a): return f"allocation/{a['allocation_id']}"
def ev_fx(d, ccy): return f"fx/{d.isoformat()}/{ccy}/{BASE}"


C = {k: v["counterparty_id"] for k, v in cp.items()}

# ================================================================ 1. duplicate_invoice (AP, EUR)
b1 = add_invoice("S1_B1", "AP", "northwind", "", "NWF-2026-0418", "2026-09-07T10:12:00+02:00",
                 "2026-09-07T15:40:00+05:00", "EUR", "Container freight Almaty-Hamburg 40ft",
                 1, 325000, "2026-09-01", "2026-09-30", terms=30)
post_invoice(b1); track(b1)
b2 = add_invoice("S1_B2", "AP", "northwind", "", "NWF20260418", "2026-09-07T10:12:00+02:00",
                 "2026-09-15T11:05:00+05:00", "EUR", "Container freight Almaty-Hamburg 40ft",
                 1, 325000, "2026-09-01", "2026-09-30", terms=30)
post_invoice(b2); track(b2)
issue("duplicate_invoice", [ev_inv(b1), ev_inv(b2)], b2["total_minor"], "EUR",
      [ev_inv(b1), ev_inv(b2), ev_row(b1["_carrier"]), ev_row(b2["_carrier"])],
      "S1", "same_vendor_ref_reformatted", "vendor bill keyed twice, second time without dashes")

# twin D: monthly subscription, same amount, different service period and vendor_ref
t_aug = add_invoice("TW_SUB_AUG", "AP", "tailspin", "", "TSH-0826", "2026-08-26T09:00:00+05:00",
                    "2026-08-26T12:00:00+05:00", "USD", "Hosting subscription", 1, 120000,
                    "2026-08-01", "2026-08-31")
post_invoice(t_aug, expense_acct="7210"); track(t_aug)
t_sep = add_invoice("TW_SUB_SEP", "AP", "tailspin", "", "TSH-0926", "2026-09-26T09:00:00+05:00",
                    "2026-09-28T10:30:00+05:00", "USD", "Hosting subscription", 1, 120000,
                    "2026-09-01", "2026-09-30")
post_invoice(t_sep, expense_acct="7210"); track(t_sep)  # Saturday -> Friday rate (also FX twin)
twins += [{"twin_id": "TW-DUP-SUB", "mimics": "duplicate_invoice",
           "records": sorted([ev_inv(t_aug), ev_inv(t_sep)]),
           "distinguishing_feature": "different vendor_ref and service period"},
          {"twin_id": "TW-FX-WEEKEND", "mimics": "fx_rate", "records": [ev_inv(t_sep)],
           "distinguishing_feature": "issued Saturday, rate of previous business day per policy"}]

# ================================================================ 2. payment_allocation (AR, USD)
s2 = add_invoice("S2_INV", "AR", "contoso", "S-2026-0412", "", "2026-09-03T11:00:00+05:00",
                 "2026-09-03T11:20:00+05:00", "USD", "Freight forwarding Shanghai-Almaty",
                 1, 840000, "2026-09-03", "2026-09-03")
post_invoice(s2); track(s2)
p2 = add_payment("in", "contoso", "2026-09-16T06:45:00Z", "2026-09-16T14:10:00+05:00", "USD",
                 840000, "INV S-2026-0412 CONTOSO LTD")
je2, car2, u2 = post_payment(p2, C["contoso"], [(s2, 540000)])
a2 = [a for a in allocs if a["payment_id"] == p2["payment_id"]]
issue("payment_allocation", [ev_pay(p2), ev_inv(s2)], 300000, "USD",
      [ev_pay(p2), ev_inv(s2), ev_row(u2)] + [ev_alloc(a) for a in a2],
      "S2", "partial_allocation", "allocation keyed as 5,400.00 instead of 8,400.00")

# twin A1: real partial payment KZT, fully allocated, invoice legitimately open
f1 = add_invoice("TW_PART_INV", "AR", "fabrikam", "S-2026-0420", "", "2026-09-09T10:00:00+05:00",
                 "2026-09-09T10:15:00+05:00", "KZT", "Warehouse handling", 1, 200000000,
                 "2026-09-09", "2026-09-09")
post_invoice(f1); track(f1)
pf1 = add_payment("in", "fabrikam", "2026-09-21T04:00:00Z", "2026-09-21T09:30:00+05:00", "KZT",
                  120000000, "Partial payment S-2026-0420")
post_payment(pf1, C["fabrikam"], [(f1, 120000000)])
# twin A2: split payment USD with realized FX on each part
w1 = add_invoice("TW_SPLIT_INV", "AR", "wingtip", "S-2026-0415", "", "2026-09-04T15:00:00+05:00",
                 "2026-09-04T15:05:00+05:00", "USD", "Customs brokerage", 1, 600000,
                 "2026-09-04", "2026-09-04")
post_invoice(w1); track(w1)
pw1 = add_payment("in", "wingtip", "2026-09-10T08:00:00Z", "2026-09-10T13:30:00+05:00", "USD",
                  200000, "S-2026-0415 part 1")
post_payment(pw1, C["wingtip"], [(w1, 200000)])
pw2 = add_payment("in", "wingtip", "2026-09-24T08:00:00Z", "2026-09-24T13:30:00+05:00", "USD",
                  400000, "S-2026-0415 part 2")
post_payment(pw2, C["wingtip"], [(w1, 400000)])
# twin A3: unidentified receipt, no reference, amount matches no open item
pu = add_payment("in", None, "2026-09-18T05:00:00Z", "2026-09-18T10:00:00+05:00", "KZT",
                 31550000, "OPLATA", raw_name="IP SAPAROV")
post_payment(pu, "", [])
twins += [{"twin_id": "TW-ALLOC-PARTIAL", "mimics": "payment_allocation",
           "records": sorted([ev_inv(f1), ev_pay(pf1)]),
           "distinguishing_feature": "bank amount is less than invoice, whole payment allocated"},
          {"twin_id": "TW-ALLOC-SPLIT", "mimics": "payment_allocation",
           "records": sorted([ev_inv(w1), ev_pay(pw1), ev_pay(pw2)]),
           "distinguishing_feature": "two payments, each fully allocated, FX diff on each"},
          {"twin_id": "TW-ALLOC-UNIDENTIFIED", "mimics": "payment_allocation", "records": [ev_pay(pu)],
           "distinguishing_feature": "no remittance reference, amount matches no open item"}]

# ================================================================ 3. fx_rate (AR, USD, wrong date = entry date)
s3 = add_invoice("S3_INV", "AR", "adventure", "S-2026-0427", "", "2026-09-14T16:20:00+05:00",
                 "2026-09-18T09:05:00+05:00", "USD", "Air freight Frankfurt-Almaty",
                 1, 500000, "2026-09-14", "2026-09-14")
wrong_r, wrong_d = rate("USD", local_date(s3["entered_at"]))
je3, car3 = post_invoice(s3, rate_override=wrong_r); track(s3)
right_r, right_d = rate("USD", local_date(s3["issued_at"]))
fx3 = abs(car3["debit_minor"] - conv(s3["total_minor"], right_r))
assert fx3 > 1, "FX injection produced no difference, pick another wrong date"
issue("fx_rate", [ev_je(je3)], fx3, BASE,
      [ev_inv(s3), ev_row(car3), ev_fx(right_d, "USD")], "S3", "rate_of_entry_date",
      f"rate of entry date {wrong_d} used instead of issue date {right_d}")

# ================================================================ 4. period_cutoff (AR, KZT, pulled forward)
s4 = add_invoice("S4_INV", "AR", "fabrikam", "S-2026-0447", "", "2026-10-01T10:05:00+05:00",
                 "2026-10-01T10:30:00+05:00", "KZT", "Cross-dock services October",
                 1, 187500000, "2026-10-01", "2026-10-31")
je4, car4 = post_invoice(s4, posting_date=dt.date(2026, 9, 30)); track(s4)
issue("period_cutoff", [ev_je(je4)], car4["debit_minor"], BASE,
      [ev_inv(s4), ev_row(car4)], "S4", "pulled_into_period",
      "revenue of 2026-10-01 posted on 2026-09-30")

# twin P1: vendor timestamp is Sep 30 in vendor and UTC time, but Oct 1 local -> posted Oct 1 correctly
n2 = add_invoice("TW_CUT_TZ", "AP", "northwind", "", "NWF-2026-0502", "2026-09-30T21:10:00+02:00",
                 "2026-10-01T09:00:00+05:00", "EUR", "Rail freight Almaty-Duisburg", 1, 98000,
                 "2026-10-01", "2026-10-31", terms=30)
post_invoice(n2); track(n2)
# twin P2: issued 23:40 local Sep 30, entered Oct 2, posted Sep 30 correctly
q2 = add_invoice("TW_CUT_LAG", "AR", "adventure", "S-2026-0446", "", "2026-09-30T23:40:00+05:00",
                 "2026-10-02T09:15:00+05:00", "KZT", "Storage September", 1, 41000000,
                 "2026-09-01", "2026-09-30")
post_invoice(q2); track(q2)
twins += [{"twin_id": "TW-CUT-TZ", "mimics": "period_cutoff", "records": [ev_inv(n2)],
           "distinguishing_feature": "UTC and vendor date is Sep 30, company local date is Oct 1"},
          {"twin_id": "TW-CUT-LAG", "mimics": "period_cutoff", "records": [ev_inv(q2)],
           "distinguishing_feature": "entered two days late, posting date follows the document"}]

# ================================================================ C1. coupled: period + fx + allocation
c1 = add_invoice("C1_INV", "AR", "contoso", "S-2026-0438", "", "2026-09-21T10:00:00+05:00",
                 "2026-09-21T10:10:00+05:00", "USD", "Project cargo Busan-Almaty",
                 1, 1000000, "2026-09-21", "2026-09-21")
post_invoice(c1); track(c1)
pc1 = add_payment("in", "contoso", "2026-09-30T19:20:00Z", "2026-10-01T09:40:00+05:00", "USD",
                  1000000, "INV S-2026-0438 CONTOSO LTD")
utc_day = parse_ts(pc1["value_at"]).date()                      # clerk's mistake: UTC date
bad_r, bad_d = rate("USD", utc_day)
good_r, good_d = rate("USD", local_date(pc1["value_at"]))
assert bad_r != good_r, "boundary dates share one rate, FX issue would be empty"
jec, carc, uc = post_payment(pc1, C["contoso"], [(c1, 650000)], posting_date=utc_day, rate_override=bad_r)
ac = [a for a in allocs if a["payment_id"] == pc1["payment_id"]]
issue("period_cutoff", [ev_je(jec)], carc["debit_minor"], BASE,
      [ev_pay(pc1), ev_row(carc)], "C1", "utc_date_as_posting_date",
      "bank UTC date used as posting date")
issue("fx_rate", [ev_je(jec)], abs(carc["debit_minor"] - conv(pc1["amount_minor"], good_r)), BASE,
      [ev_pay(pc1), ev_row(carc), ev_fx(good_d, "USD")], "C1", "rate_follows_wrong_posting_date",
      "rate of the wrong posting date")
issue("payment_allocation", [ev_pay(pc1), ev_inv(c1)], 350000, "USD",
      [ev_pay(pc1), ev_inv(c1), ev_row(uc)] + [ev_alloc(a) for a in ac], "C1", "partial_allocation",
      "allocation keyed as 6,500.00")

# ================================================================ C2. coupled: duplicate + payment on the duplicate
d1 = add_invoice("C2_B1", "AP", "wideworld", "", "WWI-7781", "2026-09-08T09:00:00+05:00",
                 "2026-09-08T11:00:00+05:00", "KZT", "Warehouse services September", 1, 148000000,
                 "2026-09-01", "2026-09-30")
post_invoice(d1); track(d1)
d2 = add_invoice("C2_B2", "AP", "wideworld", "", "WWI 7781", "2026-09-08T09:00:00+05:00",
                 "2026-09-15T16:20:00+05:00", "KZT", "Warehouse services September", 1, 148000000,
                 "2026-09-01", "2026-09-30")
post_invoice(d2); track(d2)
pd2 = add_payment("out", "wideworld", "2026-09-22T05:30:00Z", "2026-09-22T10:30:00+05:00", "KZT",
                  148000000, "Payment for invoice WWI-7781")
post_payment(pd2, C["wideworld"], [(d2, 148000000)])
issue("duplicate_invoice", [ev_inv(d1), ev_inv(d2)], d2["total_minor"], "KZT",
      [ev_inv(d1), ev_inv(d2), ev_row(d1["_carrier"]), ev_row(d2["_carrier"])], "C2",
      "duplicate_then_paid", "bill keyed twice, payment applied to the second copy")
twins.append({"twin_id": "TW-C2-TRAP", "mimics": "payment_allocation",
              "records": sorted([ev_pay(pd2), ev_inv(d1)]),
              "distinguishing_feature": "payment is fully allocated to the copy, the only issue is the duplicate"})

# ================================================================ invariants on final books
by_je = {}
for r_ in ledger:
    by_je.setdefault(r_["je_id"], []).append(r_)
for je, rs in by_je.items():
    assert sum(x["debit_minor"] for x in rs) == sum(x["credit_minor"] for x in rs), je
assert sum(x["debit_minor"] - x["credit_minor"] for x in ledger) == 0
win = (PERIOD[0] - dt.timedelta(days=WINDOW_DAYS), PERIOD[1] + dt.timedelta(days=WINDOW_DAYS))
assert all(win[0].isoformat() <= x["posting_date"] <= win[1].isoformat() for x in ledger)
for a in allocs:
    assert a["amount_minor"] > 0
for iid in open_doc:
    assert open_doc[iid] >= 0


# ================================================================ export
def strip(rows):
    return [{k: v for k, v in r_.items() if not k.startswith("_")} for r_ in rows]


def write_csv(path, rows):
    rows = strip(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r_ in sorted(rows, key=lambda x: json.dumps(x, sort_keys=True)):
            w.writerow(r_)


out = sys.argv[1]
os.makedirs(f"{out}/public", exist_ok=True)
os.makedirs(f"{out}/private", exist_ok=True)
fx_rows = [{"date": k[0].isoformat(), "currency": k[1], "base": BASE, "rate": v}
           for k, v in sorted(fx.items())]
write_csv(f"{out}/public/fx.csv", fx_rows)
write_csv(f"{out}/public/counterparties.csv", list(cp.values()))
write_csv(f"{out}/public/invoices.csv", invoices)
write_csv(f"{out}/public/invoice_lines.csv", lines)
write_csv(f"{out}/public/payments.csv", payments)
write_csv(f"{out}/public/allocations.csv", allocs)
write_csv(f"{out}/public/ledger.csv", ledger)
policy = {
    "base_currency": BASE, "company_utc_offset": "+05:00",
    "period": {"start_local": "2026-09-01", "end_local": "2026-09-30",
               "start_utc": "2026-08-31T19:00:00Z", "end_utc_exclusive": "2026-09-30T19:00:00Z"},
    "ledger_window": {"from": win[0].isoformat(), "to": win[1].isoformat()},
    "posting_date_rule": "company-local date of invoice.issued_at or payment.value_at",
    "fx": {"quote": "base units per 1 unit of currency", "date_rule": "posting_date_rule date",
           "non_business_fallback": "previous business day",
           "non_business_days": sorted(x.isoformat() for x in NON_BUSINESS) + ["weekends"],
           "rounding": "ROUND_HALF_UP to 1 minor unit", "settlement": "clearing rows use the invoice rate, "
           "last allocation clears the base residual, difference to 6250 or 7430"},
    "allocation_rule": "a payment whose remittance_info names exactly one open document group of the same "
                       "counterparty and currency must be allocated to it up to the open amount, "
                       "the rest goes to 3510",
    "duplicate_rule": "two AP invoices of one counterparty with equal currency, total and normalized vendor_ref",
    "vendor_ref_normalization": "uppercase, keep A-Z and 0-9 only",
    "minor_units": {"KZT": 2, "USD": 2, "EUR": 2},
}
with open(f"{out}/public/policy.json", "w") as f:
    json.dump(policy, f, sort_keys=True, indent=2, ensure_ascii=False)
truth = {"episode_id": new_id(), "clean": False,
         "issues": sorted(truth_issues, key=lambda x: x["issue_id"])}
with open(f"{out}/private/truth.json", "w") as f:
    json.dump(truth, f, sort_keys=True, indent=2, ensure_ascii=False)
prov = {"seed": SEED, "generator_version": "golden-0.1", "injections": provenance,
        "twins": sorted(twins, key=lambda x: x["twin_id"])}
with open(f"{out}/private/provenance.json", "w") as f:
    json.dump(prov, f, sort_keys=True, indent=2, ensure_ascii=False)

# expose a few numbers for the spec document
summary = {
    "rates": {k: fx[(rate_date(v), c)] for k, v, c in [
        ("EUR 2026-09-07", dt.date(2026, 9, 7), "EUR"),
        ("USD 2026-09-03", dt.date(2026, 9, 3), "USD"), ("USD 2026-09-16", dt.date(2026, 9, 16), "USD"),
        ("USD 2026-09-14", dt.date(2026, 9, 14), "USD"), ("USD 2026-09-18", dt.date(2026, 9, 18), "USD"),
        ("USD 2026-09-21", dt.date(2026, 9, 21), "USD"), ("USD 2026-09-25", dt.date(2026, 9, 25), "USD"),
        ("USD 2026-09-26 (Sat)", dt.date(2026, 9, 26), "USD"),
        ("USD 2026-09-30", dt.date(2026, 9, 30), "USD"), ("USD 2026-10-01", dt.date(2026, 10, 1), "USD")]},
    "S1_base_each": b1["_base"],
    "S2_je_rows": [(x["account"], x["debit_minor"], x["credit_minor"], x["doc_amount_minor"], x["exchange_rate"])
                   for x in by_je[je2]],
    "S3": {"posted": car3["debit_minor"], "expected": conv(s3["total_minor"], right_r), "diff": fx3},
    "S4_base": car4["debit_minor"],
    "C1_je_rows": [(x["account"], x["debit_minor"], x["credit_minor"], x["doc_amount_minor"], x["exchange_rate"])
                   for x in by_je[jec]],
    "C1_expected_bank": conv(pc1["amount_minor"], good_r),
    "counts": {"invoices": len(invoices), "payments": len(payments), "ledger_rows": len(ledger),
               "issues": len(truth_issues), "twins": len(twins)},
}
with open(f"{out}/private/summary_for_spec.json", "w") as f:
    json.dump(summary, f, sort_keys=True, indent=2)
print(json.dumps(summary, indent=1))
