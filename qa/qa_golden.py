"""QA independent check of a built episode.

Written from policy.json rules only, without reading A's build or verify code paths.
Scans EVERY public record, detects every rule violation, then compares the detected
set with private/truth.json. Also scans public files for leaks and schema gaps.

Usage: python -I qa_golden.py <episode_dir>
Exit code 0 only if there are no CONFIRMED findings.
"""
import csv
import datetime as dt
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

EP = Path(sys.argv[1])
PUB, PRIV = EP / "public", EP / "private"
findings = []  # (severity, code, text)


def report(sev, code, text):
    findings.append((sev, code, text))


def load_csv(name):
    with open(PUB / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


policy = json.loads((PUB / "policy.json").read_text(encoding="utf-8"))
truth = json.loads((PRIV / "truth.json").read_text(encoding="utf-8"))
prov = json.loads((PRIV / "provenance.json").read_text(encoding="utf-8"))
inv = {r["invoice_id"]: r for r in load_csv("invoices.csv")}
pay = {r["payment_id"]: r for r in load_csv("payments.csv")}
led = load_csv("ledger.csv")
alloc = load_csv("allocations.csv")
lines = load_csv("invoice_lines.csv")
fxrows = load_csv("fx.csv")
BASE = policy["base_currency"]
m = re.fullmatch(r"([+-])(\d\d):(\d\d)", policy["company_utc_offset"])
OFF = dt.timedelta(hours=int(m[2]), minutes=int(m[3])) * (1 if m[1] == "+" else -1)
P0 = dt.date.fromisoformat(policy["period"]["start_local"])
P1 = dt.date.fromisoformat(policy["period"]["end_local"])
fx = {(r["date"], r["currency"]): r["rate"] for r in fxrows}
nb_raw = policy["fx"]["non_business_days"]
NB = set()
for x in nb_raw:
    try:
        NB.add(dt.date.fromisoformat(x))
    except ValueError:
        report("low", "POLICY-NB-FORMAT", f"policy.fx.non_business_days holds a non-date value {x!r}")


def local_date(ts):
    t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if t.tzinfo is None:
        report("high", "TS-NAIVE", f"timestamp without offset {ts}")
        return t.date()
    return (t.astimezone(dt.timezone.utc) + OFF).date()


def rate_for(ccy, d):
    if ccy == BASE:
        return "1"
    while d.weekday() >= 5 or d in NB:
        d -= dt.timedelta(days=1)
    return fx.get((d.isoformat(), ccy))


def conv(minor, rate):
    return int((Decimal(minor) * Decimal(rate)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def in_period(d):
    return P0 <= d <= P1


def doc_date(row):
    if row["source_type"] == "invoice":
        return local_date(inv[row["source_id"]]["issued_at"])
    if row["source_type"] == "payment":
        return local_date(pay[row["source_id"]]["value_at"])
    return None


# ------------------------------------------------------------- 1. book integrity
by_je = {}
for r in led:
    by_je.setdefault(r["je_id"], []).append(r)
for je, rs in by_je.items():
    d = sum(int(x["debit_minor"]) for x in rs)
    c = sum(int(x["credit_minor"]) for x in rs)
    if d != c:
        report("critical", "JE-UNBALANCED", f"je {je} debit {d} credit {c}")
    if len({x["posting_date"] for x in rs}) != 1:
        report("high", "JE-MIXED-DATES", f"je {je} has rows on several posting dates")
for r in led:
    if int(r["debit_minor"]) and int(r["credit_minor"]):
        report("medium", "ROW-BOTH-SIDES", f"row {r['row_id']} has both debit and credit")
    if int(r["debit_minor"]) < 0 or int(r["credit_minor"]) < 0:
        report("medium", "ROW-NEGATIVE", f"row {r['row_id']}")
ids_all = [r["row_id"] for r in led] + list(inv) + list(pay) + [a["allocation_id"] for a in alloc] + [x["line_id"] for x in lines]
if len(ids_all) != len(set(ids_all)):
    report("critical", "ID-COLLISION", "ids repeat across tables")
sum_lines = {}
for x in lines:
    sum_lines[x["invoice_id"]] = sum_lines.get(x["invoice_id"], 0) + int(x["amount_minor"])
    if int(x["qty"]) * int(x["unit_price_minor"]) != int(x["amount_minor"]):
        report("medium", "LINE-MATH", f"line {x['line_id']}")
for i, r in inv.items():
    if sum_lines.get(i, 0) != int(r["total_minor"]):
        report("high", "INV-LINES-SUM", f"invoice {i} lines {sum_lines.get(i)} total {r['total_minor']}")
win_from = dt.date.fromisoformat(policy["ledger_window"]["from"])
win_to = dt.date.fromisoformat(policy["ledger_window"]["to"])
for r in led:
    pdt = dt.date.fromisoformat(r["posting_date"])
    if not win_from <= pdt <= win_to:
        report("high", "ROW-OUTSIDE-WINDOW", f"row {r['row_id']} {pdt}")

# ------------------------------------------------------------- 2. detector, rule by rule
detected = []  # dicts with type, key, amount, currency, carrier

# period: posting_date must equal company-local date of the source document
for je, rs in by_je.items():
    right = {doc_date(x) for x in rs if x["source_type"] in ("invoice", "payment")}
    if len(right) != 1:
        report("high", "JE-MIXED-SOURCES", f"je {je}")
        continue
    right = right.pop()
    posted = dt.date.fromisoformat(rs[0]["posting_date"])
    if posted != right:
        if in_period(posted) != in_period(right):
            car = [x for x in rs if x["account"] in ("1210", "3310", "1030") and not x["settles_invoice_id"]
                   and (x["source_type"] == "payment") == (x["account"] == "1030")]
            amt = int(car[0]["debit_minor"]) or int(car[0]["credit_minor"])
            detected.append({"type": "period_cutoff", "je": je, "carrier": car[0]["row_id"], "amount": amt, "currency": BASE})
        else:
            report("critical", "UNLABELED-DATE-SHIFT",
                   f"je {je} posted {posted}, document date {right}, same period, no error type covers this")

# fx: rate column and base amount must follow policy
inv_carrier = {}
for r in led:
    if r["source_type"] == "invoice" and r["account"] in ("1210", "3310"):
        inv_carrier[r["source_id"]] = r
fx_bad_by_je = {}
for r in led:
    ccy = r["doc_currency"]
    if ccy == BASE:
        if r["exchange_rate"] != "1" or int(r["doc_amount_minor"]) != (int(r["debit_minor"]) or int(r["credit_minor"])):
            report("high", "BASE-ROW-RATE", f"row {r['row_id']}")
        continue
    if r["settles_invoice_id"]:
        expected = inv_carrier[r["settles_invoice_id"]]["exchange_rate"]  # clearing rows use the invoice rate
    else:
        expected = rate_for(ccy, doc_date(r))
    if expected is None:
        report("critical", "FX-MISSING", f"no rate for row {r['row_id']}")
        continue
    if r["exchange_rate"] != expected:
        fx_bad_by_je.setdefault(r["je_id"], []).append((r, expected))
    elif not r["settles_invoice_id"]:
        posted = int(r["debit_minor"]) or int(r["credit_minor"])
        if posted != conv(int(r["doc_amount_minor"]), r["exchange_rate"]):
            report("high", "FX-ROUNDING", f"row {r['row_id']} posted {posted}")
for je, bad in fx_bad_by_je.items():
    car = [(r, e) for r, e in bad if r["account"] in ("1210", "3310", "1030")] or bad
    r, exp = car[0]
    posted = int(r["debit_minor"]) or int(r["credit_minor"])
    detected.append({"type": "fx_rate", "je": je, "carrier": r["row_id"],
                     "amount": abs(posted - conv(int(r["doc_amount_minor"]), exp)), "currency": BASE,
                     "rows": sorted(x["row_id"] for x, _ in bad)})

# allocation: a payment naming exactly one open document group must be allocated up to the open amount
pay_cp = {}
for r in led:
    if r["source_type"] == "payment" and r["account"] == "1030":
        pay_cp[r["source_id"]] = r["counterparty_id"]
alloc_by_pay = {}
for a in alloc:
    alloc_by_pay.setdefault(a["payment_id"], []).append(a)
    if a["payment_id"] not in pay or a["invoice_id"] not in inv:
        report("critical", "ALLOC-DANGLING", a["allocation_id"])
    if a["je_id"] not in by_je:
        report("high", "ALLOC-JE-MISSING", a["allocation_id"])
open_doc = {i: int(r["total_minor"]) for i, r in inv.items()}
for pid, p in sorted(pay.items(), key=lambda kv: (kv[1]["value_at"], kv[0])):
    want_dir = "AR" if p["direction"] == "in" else "AP"
    text = norm(p["remittance_info"])
    group = [i for i, r in inv.items() if r["direction"] == want_dir and r["currency"] == p["currency"]
             and r["counterparty_id"] == pay_cp.get(pid)
             and ((r["invoice_number"] and norm(r["invoice_number"]) in text) or (r["vendor_ref"] and norm(r["vendor_ref"]) in text))]
    got = sum(int(a["amount_minor"]) for a in alloc_by_pay.get(pid, []))
    if group:
        refs = {norm(inv[i]["invoice_number"] or inv[i]["vendor_ref"]) for i in group}
        if len(refs) != 1:
            report("high", "ALLOC-AMBIGUOUS", f"payment {pid} names {sorted(refs)}")
            continue
        open_amt = sum(open_doc[i] for i in group)
        want = min(int(p["amount_minor"]), open_amt)
        if got != want:
            detected.append({"type": "payment_allocation", "key": frozenset({pid} | set(group)), "payment": pid,
                             "amount": want - got, "currency": p["currency"]})
    for a in alloc_by_pay.get(pid, []):
        open_doc[a["invoice_id"]] -= int(a["amount_minor"])
        if open_doc[a["invoice_id"]] < 0:
            report("high", "OVER-ALLOCATED", a["invoice_id"])

# duplicates: AP, same counterparty, currency, total, normalized vendor_ref
grp = {}
for i, r in inv.items():
    if r["direction"] == "AP":
        grp.setdefault((r["counterparty_id"], r["currency"], r["total_minor"], norm(r["vendor_ref"])), []).append(i)
for k, ids in grp.items():
    if len(ids) > 1:
        posted = [i for i in ids if i in inv_carrier]
        if len(posted) > 1:
            detected.append({"type": "duplicate_invoice", "key": frozenset(ids), "amount": int(k[2]), "currency": k[1]})

# subledger: a fully allocated invoice must have zero base balance (otherwise reconciliation sees a ghost)
bal = {}
for r in led:
    target = r["settles_invoice_id"] or (r["source_id"] if r["source_type"] == "invoice" and r["account"] in ("1210", "3310") else "")
    if target:
        bal[target] = bal.get(target, 0) + int(r["debit_minor"]) - int(r["credit_minor"])
for i, b in bal.items():
    if open_doc[i] == 0 and b != 0:
        report("high", "SUBLEDGER-RESIDUAL", f"invoice {i} fully paid but base balance {b}")

# ------------------------------------------------------------- 3. compare with truth
row_je = {r["row_id"]: r["je_id"] for r in led}


PREFIXES = ("invoice", "payment", "ledger", "allocation", "je", "doc")


def strip(x):
    # strip only a known table prefix, see P-04. fx/2026-09-14/USD/KZT stays as it is
    head, _, rest = x.partition("/")
    return rest if head in PREFIXES and rest else x


matched, unlabeled = set(), []
for d in detected:
    hit = None
    for t in truth["issues"]:
        if t["type"] != d["type"] or t["issue_id"] in matched:
            continue
        ids = {strip(x) for x in t["affected_ids"]}
        if d["type"] in ("fx_rate", "period_cutoff"):
            ok = d["je"] in ids or d["carrier"] in ids or any(row_je.get(x) == d["je"] for x in ids)
        else:
            ok = d["key"] == ids or (d["type"] == "payment_allocation" and ids <= d["key"])
        if ok:
            hit = t
            break
    if hit is None:
        unlabeled.append(d)
        continue
    matched.add(hit["issue_id"])
    tol = hit.get("tolerance_minor", 0)
    amt = hit.get("expected_amount_minor", hit.get("amount_minor"))
    if abs(amt - d["amount"]) > tol or hit["currency"] != d["currency"]:
        report("critical", "TRUTH-AMOUNT", f"{hit['issue_id']} {hit['type']} truth {amt} {hit['currency']} recomputed {d['amount']} {d['currency']}")
    if amt <= 2 * tol:
        report("high", "AMOUNT-TOO-SMALL", f"{hit['issue_id']} amount {amt} not above two tolerances")
for d in unlabeled:
    report("critical", "UNLABELED-ERROR", f"detector found {d['type']} not in truth: {d}")
for t in truth["issues"]:
    if t["issue_id"] not in matched:
        report("critical", "TRUTH-NOT-DETECTABLE", f"{t['issue_id']} {t['type']} {t.get('scenario_id')}")

# twins must not trigger the detector
twin_ids = {strip(x) for tw in prov.get("twins", []) for x in tw["records"]}
for d in unlabeled:
    keys = set(d.get("key", ())) | {d.get("carrier", "")}
    if keys & twin_ids:
        report("critical", "TWIN-TRIGGERS", str(d))

# ------------------------------------------------------------- 4. evidence and id checks
resolvable = set(inv) | set(pay) | {r["row_id"] for r in led} | {a["allocation_id"] for a in alloc}
for t in truth["issues"]:
    sets = t.get("evidence_sets") or [t.get("evidence", [])]
    for s in sets:
        if len(s) > 10:
            report("high", "EVIDENCE-OVER-10", t["issue_id"])
        bad = [x for x in s if strip(x) not in resolvable]
        if bad:
            report("medium", "EVIDENCE-NOT-A-RECORD", f"{t['issue_id']} {t['type']} ids {bad}")
    bad = [x for x in t["affected_ids"] if strip(x) not in resolvable]
    if bad:
        report("high", "AFFECTED-NOT-READABLE", f"{t['issue_id']} {t['type']} affected_ids {bad}")
    if "expected_amount_minor" not in t:
        report("high", "TRUTH-FIELD", f"{t['issue_id']} has no expected_amount_minor")
    if "evidence_sets" not in t:
        report("high", "TRUTH-FIELD", f"{t['issue_id']} has no evidence_sets")

# ------------------------------------------------------------- 5. leaks in public files
private_words = r"duplicate|dup\b|twin|inject|error|mistake|wrong|truth|scenario|provenance|variant|\bTW-|\bS[1-4]_|\bC[12]_|_key|cutoff|fx_rate|payment_allocation"
private_ids = {t["issue_id"] for t in truth["issues"]} | {truth.get("episode_id", "")}
def json_values(o):
    if isinstance(o, dict):
        return " ".join(json_values(v) for v in o.values())
    if isinstance(o, list):
        return " ".join(json_values(v) for v in o)
    return str(o)


for f in sorted(PUB.iterdir()):
    text = f.read_text(encoding="utf-8")
    if f.name == "policy.json":
        # rule names such as duplicate_rule are public by design and identical in every episode,
        # so only values are scanned here; the env test must still compare policy.json of clean and dirty episodes
        text = json_values(json.loads(text))
    for w in sorted(set(re.findall(private_words, text, flags=re.I))):
        report("critical", "LEAK-WORD", f"{f.name} contains {w!r}")
    for pid in private_ids:
        if pid and re.search(rf"\b{re.escape(pid)}\b", text):
            report("critical", "LEAK-PRIVATE-ID", f"{f.name} contains private id {pid}")
    if f.suffix == ".csv":
        header = text.splitlines()[0].split(",")
        if any(h.startswith("_") for h in header):
            report("critical", "LEAK-COLUMN", f"{f.name} {header}")

# ------------------------------------------------------------- 6. schema against plan v2 (case section 03 as summarized in the plan)
need_files = ["manifest.json", "invoices.jsonl", "payments.jsonl", "ledger.csv", "fx.csv"]
for n in need_files:
    if not (PUB / n).exists():
        report("high", "SCHEMA-FILE", f"missing {n}")
need_led = ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency", "entity_id", "invoice_id", "payment_id"]
hdr = list(led[0].keys())
missing = [c for c in need_led if c not in hdr]
if missing:
    report("high", "SCHEMA-LEDGER", f"ledger.csv lacks {missing}")
elif hdr[:len(need_led)] != need_led:
    report("medium", "SCHEMA-LEDGER-ORDER", f"required columns are not first {hdr[:9]}")
fxh = list(fxrows[0].keys())
if not {"source_currency", "target_currency"} <= set(fxh):
    report("high", "SCHEMA-FX", f"fx.csv columns {fxh}")

# ------------------------------------------------------------- 7. volume
outside = {"before": 0, "after": 0}
for r in led:
    pdt = dt.date.fromisoformat(r["posting_date"])
    if pdt < P0:
        outside["before"] += 1
    elif pdt > P1:
        outside["after"] += 1
if min(outside.values()) < 10:
    report("medium", "INV-03-VOLUME", f"genuine rows outside period {outside}, plan wants at least 10 each side")

# ------------------------------------------------------------- output
print(f"episode {EP}")
print(f"records invoices={len(inv)} payments={len(pay)} ledger_rows={len(led)} allocations={len(alloc)} je={len(by_je)}")
print(f"detector found {len(detected)} issues, truth has {len(truth['issues'])}, matched {len(matched)}")
for d in detected:
    print("  DETECTED", d["type"], d.get("je") or sorted(d["key"]), d["amount"], d["currency"])
order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
for sev, code, text in sorted(findings, key=lambda x: (order[x[0]], x[1], x[2])):
    print(f"[{sev}] {code} {text}")
sys.exit(1 if any(s == "critical" for s, _, _ in findings) else 0)
