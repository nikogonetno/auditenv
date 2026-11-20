"""INV-17: the amount of every issue is recomputed from one evidence set and policy.json alone.

`amount_from_set` sees only the records of the set and the parsed policy, so a set that leans on the
rest of the book cannot pass. `missing_from_set` is the one place that reads the whole book: it checks
that an allocation set holds every clearing row the domain spec A asks for.
"""
from decimal import Decimal

from .book import (PAYABLE, UNAPPLIED, BookError, event_time, is_carrying, is_clearing, minor, normalize_ref,
                   parse_rate, row_base)
from .ids import normalize_id


def _only(items, table, what):
    found = [rec for t, rec in items if t == table]
    if len(found) != 1:
        raise BookError(f"needs exactly one {what}, has {len(found)}")
    return found[0]


def _ledger(items):
    return [rec for t, rec in items if t == "ledger"]


def _document(items):
    docs = [(t, rec) for t, rec in items if t in ("invoices", "payments")]
    if len(docs) != 1:
        raise BookError(f"needs exactly one document, has {len(docs)}")
    return docs[0]


def _carrying(items, table, doc):
    rows = [r for r in _ledger(items) if is_carrying(table, doc, r)]
    if len(rows) != 1:
        raise BookError(f"needs the carrying row of the document, has {len(rows)}")
    return rows[0]


def amount_from_set(issue_type, items, policy):
    """(amount_minor, currency) by the formula of the type, from `items` [(table, record)] and policy."""
    return _FORMULAS[issue_type](items, policy)


def _duplicate(items, policy):
    bills = [rec for t, rec in items if t == "invoices"]
    if len(bills) != 2:
        raise BookError(f"needs both bills, has {len(bills)}")
    a, b = bills
    for bill in bills:
        if bill.get("direction") != "AP":
            raise BookError(f"bill {bill['invoice_id']} is not AP")
        if not any(is_carrying("invoices", bill, r) for r in _ledger(items)):
            raise BookError(f"no {PAYABLE} row of bill {bill['invoice_id']}, posting not shown")
    key_a = (a.get("entity_id"), a.get("currency"), a.get("total_minor"), normalize_ref(a.get("vendor_ref")))
    key_b = (b.get("entity_id"), b.get("currency"), b.get("total_minor"), normalize_ref(b.get("vendor_ref")))
    if key_a != key_b or not key_a[3] or key_a[0] is None:
        raise BookError("bills do not share the duplicate key")
    return minor(a.get("total_minor"), "total_minor"), a.get("currency")


def _allocation(items, policy):
    payment = _only(items, "payments", "payment")
    bill = _only(items, "invoices", "bill")
    pid, iid = payment["payment_id"], bill["invoice_id"]
    if iid not in {normalize_id(r) for r in payment.get("invoice_refs") or [] if type(r) is str}:
        raise BookError(f"payment {pid} does not refer to bill {iid}")
    rows = _ledger(items)
    clearing = [r for r in rows if is_clearing(r) and r["payment_id"] == pid and r["invoice_id"] == iid]
    unapplied = [r for r in rows if r.get("account") == UNAPPLIED and r.get("payment_id") == pid]
    if len(unapplied) != 1:
        raise BookError(f"needs the {UNAPPLIED} row of payment {pid}, has {len(unapplied)}")
    currency = payment.get("currency")
    for r in clearing + unapplied:
        if r.get("doc_currency") != currency:
            raise BookError(f"ledger {r['row_id']} is in {r.get('doc_currency')}, payment in {currency}")
    amount = minor(payment.get("amount_minor"), "amount_minor") - sum(
        minor(r.get("doc_amount_minor"), "doc_amount_minor") for r in clearing)
    on_hold = minor(unapplied[0].get("doc_amount_minor"), "doc_amount_minor")
    if amount != on_hold:
        raise BookError(f"payment minus allocations is {amount}, {UNAPPLIED} row {unapplied[0]['row_id']} holds {on_hold}")
    return amount, currency


def _rate(items, policy):
    table, doc = _document(items)
    row = _carrying(items, table, doc)
    doc_currency = row.get("doc_currency")
    if doc_currency != doc.get("currency"):
        raise BookError(f"carrying row in {doc_currency}, document in {doc.get('currency')}")
    if doc_currency == policy.base:
        right = Decimal(1)
    else:
        quote = _only(items, "fx", "rate row")
        day = policy.rate_date(event_time(table, doc))
        if (quote.get("date"), quote.get("source_currency"), quote.get("target_currency")) != \
                (day.isoformat(), doc_currency, policy.base):
            raise BookError(f"rate row fx/{quote.get('fx_id')} is {quote.get('date')} "
                            f"{quote.get('source_currency')}/{quote.get('target_currency')}, rule asks "
                            f"{day} {doc_currency}/{policy.base}")
        right = parse_rate(quote.get("rate"))
    return abs(row_base(row) - policy.to_base(row.get("doc_amount_minor"), right, doc_currency)), policy.base


def _period(items, policy):
    table, doc = _document(items)
    row = _carrying(items, table, doc)
    posted, happened = policy.local_date(row.get("timestamp")), policy.local_date(event_time(table, doc))
    if posted == happened:
        raise BookError(f"posted on the document date {happened}, nothing to show")
    return row_base(row), policy.base


_FORMULAS = {
    "duplicate_invoice": _duplicate,
    "payment_allocation": _allocation,
    "fx_rate": _rate,
    "period_cutoff": _period,
}


def missing_from_set(issue_type, items, book):
    """Records the type's evidence must hold but the set leaves out, found in the whole book."""
    if issue_type != "payment_allocation":
        return []
    payments = [rec for t, rec in items if t == "payments"]
    bills = [rec for t, rec in items if t == "invoices"]
    if len(payments) != 1 or len(bills) != 1:
        return []                    # amount_from_set already reports the shape
    pid, iid = payments[0]["payment_id"], bills[0]["invoice_id"]
    held = {rec.get("row_id") for t, rec in items if t == "ledger"}
    need = [r["row_id"] for r in book.rows_by_payment.get(pid, [])
            if (is_clearing(r) and r["invoice_id"] == iid) or r.get("account") == UNAPPLIED]
    return sorted(rid for rid in need if rid not in held)

