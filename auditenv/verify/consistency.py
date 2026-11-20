"""Bookkeeping consistency of the public files: INV-04, 05, 06, 08, 09, 10, 12 and one date per entry.

These hold in a clean book and after every error of the four types, so a hit here is a contradiction in
the data, not a hidden error for the agent. The remittance check keeps the allocation rule honest: the
detector reads invoice_refs, and invoice_refs must be what remittance_info names (SPEC §3.2).
"""
import re
from decimal import Decimal

from .book import (BANK, PAYMENT_SIDE, UNAPPLIED, BookError, event_time, is_clearing, minor, normalize_ref,
                   parse_rate, parse_time, row_base)

FX_GAIN, FX_LOSS = "6250", "7430"


def _add(problems, code, detail):
    problems.append({"code": code, "detail": detail})


def check_consistency(book, problems):
    for check in (_documents, _row_bases, _entries, _settlements, _references):
        check(book, problems)


def _documents(book, problems):
    """INV-06: total is the sum of lines, a line is qty x price. Carrying rows copy the document."""
    for invoice in book.invoices:
        iid = invoice["invoice_id"]
        try:
            lines = invoice.get("line_items")
            if type(lines) is not list or not lines:
                raise BookError("no line_items")
            total = 0
            for line in lines:
                qty, price = line.get("qty"), minor(line.get("unit_price_minor"), "unit_price_minor")
                amount = minor(line.get("amount_minor"), "amount_minor")
                if type(qty) is not int or qty <= 0:
                    raise BookError(f"line {line.get('line_id')}: qty {qty!r}")
                if qty * price != amount:
                    _add(problems, "INV-06", f"invoice {iid} line {line.get('line_id')}: {qty} x {price} != {amount}")
                total += amount
            if total != minor(invoice.get("total_minor"), "total_minor"):
                _add(problems, "INV-06", f"invoice {iid}: lines sum to {total}, total_minor {invoice['total_minor']}")
        except (BookError, AttributeError) as exc:
            _add(problems, "SCHEMA", f"invoice {iid}: {exc}")
    for table, doc in book.documents():
        did = book.doc_id(table, doc)
        amount = doc.get("total_minor") if table == "invoices" else doc.get("amount_minor")
        for row in book.carrying_rows(table, doc):
            got = (row.get("doc_currency"), row.get("doc_amount_minor"), row.get("entity_id"))
            want = (doc.get("currency"), amount, doc.get("entity_id"))
            if got != want:
                _add(problems, "INV-08", f"carrying row {row['row_id']} has currency, amount, entity {got}, "
                                         f"document {did} has {want}")


def _payment_time(book, pid):
    payment = book.index["payments"].get(pid)
    return parse_time(payment.get("timestamp")) if payment else None


def _row_bases(book, problems):
    """INV-04: base = round(doc_amount x exchange_rate). The last clearing row of an invoice closes the
    base residual, so it may be off by 1 for every earlier clearing row of that invoice."""
    clearing = {}
    for row in book.ledger:
        if is_clearing(row):
            clearing.setdefault(row["invoice_id"], []).append(row)
    slack = {}
    for iid, rows in clearing.items():
        try:
            rows = sorted(rows, key=lambda r: (_payment_time(book, r["payment_id"]) or parse_time(r.get("timestamp")),
                                               r["row_id"]))
        except BookError:
            continue                 # an unreadable time is reported by the period and window checks
        slack[rows[-1]["row_id"]] = len(rows) - 1
    for row in book.ledger:
        rid = row["row_id"]
        try:
            rate = parse_rate(row.get("exchange_rate"))
            if row.get("doc_currency") == book.base and rate != Decimal(1):
                _add(problems, "INV-04", f"ledger {rid}: base currency row at rate {rate}")
            want = book.policy.to_base(row.get("doc_amount_minor"), rate, row.get("doc_currency"))
            got = row_base(row)
        except BookError as exc:
            _add(problems, "SCHEMA", f"ledger {rid}: {exc}")
            continue
        if abs(got - want) > slack.get(rid, 0):
            _add(problems, "INV-04", f"ledger {rid}: base {got}, {row.get('doc_amount_minor')} x {rate} gives {want}")


def _entries(book, problems):
    """One local date per je_id. INV-05: FX rows only in foreign-currency payment entries, one at most,
    on the side that balances the entry (gain is a credit of 6250, loss a debit of 7430)."""
    entries = {}
    for row in book.ledger:
        if row.get("je_id") is not None:
            entries.setdefault(row["je_id"], []).append(row)
    for je, rows in sorted(entries.items()):
        try:
            days = sorted({book.policy.local_date(r.get("timestamp")).isoformat() for r in rows})
        except BookError:
            days = []                # reported by the window check
        if len(days) > 1:
            _add(problems, "INV-13", f"je {je} rows posted on {days}")
        fx_rows = [r for r in rows if r.get("account") in (FX_GAIN, FX_LOSS)]
        if not fx_rows:
            continue
        pids = {r.get("payment_id") for r in rows if r.get("account") == BANK}
        payment = book.index["payments"].get(next(iter(pids))) if len(pids) == 1 else None
        if payment is None or payment.get("currency") == book.base:
            _add(problems, "INV-05", f"je {je}: FX row outside a foreign-currency payment entry")
            continue
        if len(fx_rows) > 1:
            _add(problems, "INV-05", f"je {je}: {len(fx_rows)} FX rows")
            continue
        fx = fx_rows[0]
        try:
            rest = sum(minor(r.get("debit_minor"), "debit_minor") - minor(r.get("credit_minor"), "credit_minor")
                       for r in rows if r["row_id"] != fx["row_id"])
        except BookError:
            continue                 # reported by INV-01
        want = (FX_GAIN, 0, rest) if rest > 0 else (FX_LOSS, -rest, 0)
        if (fx.get("account"), fx.get("debit_minor"), fx.get("credit_minor")) != want:
            _add(problems, "INV-05", f"je {je}: FX row {fx['row_id']} is {fx.get('account')} "
                                     f"{fx.get('debit_minor')}/{fx.get('credit_minor')}, the entry needs "
                                     f"{want[0]} {want[1]}/{want[2]}")


def _settlements(book, problems):
    """INV-09: clearing in the currency of payment and invoice, never above the invoice.
    INV-10: a receipt is cleared plus 3510, a supplier payment is cleared in full.
    INV-12: an invoice is not paid before it is issued."""
    invoices, per_invoice = book.index["invoices"], {}
    for payment in book.payments:
        pid = payment["payment_id"]
        rows = book.rows_by_payment.get(pid, [])
        try:
            amount = minor(payment.get("amount_minor"), "amount_minor")
            cleared = sum(minor(r.get("doc_amount_minor"), "doc_amount_minor") for r in rows if is_clearing(r))
            held = sum(minor(r.get("doc_amount_minor"), "doc_amount_minor") for r in rows
                       if r.get("account") == UNAPPLIED)
            paid_on = book.policy.local_date(payment.get("timestamp"))
        except BookError as exc:
            _add(problems, "SCHEMA", f"payment {pid}: {exc}")
            continue
        side = PAYMENT_SIDE.get(payment.get("direction"))
        if side is None:
            _add(problems, "SCHEMA", f"payment {pid}: direction {payment.get('direction')!r}")
        elif cleared + held != amount or (side == "AP" and held):
            _add(problems, "INV-10", f"payment {pid} {amount}: cleared {cleared}, on {UNAPPLIED} {held}")
        for row in rows:
            if not is_clearing(row):
                continue
            invoice = invoices.get(row["invoice_id"])
            if invoice is None:
                continue             # reported by the rate check as INV-08
            if not row.get("doc_currency") == invoice.get("currency") == payment.get("currency"):
                _add(problems, "INV-09", f"ledger {row['row_id']}: {row.get('doc_currency')}, invoice "
                                         f"{invoice.get('currency')}, payment {payment.get('currency')}")
            if invoice.get("direction") != side:
                _add(problems, "INV-09", f"ledger {row['row_id']}: {side} payment {pid} clears "
                                         f"{invoice.get('direction')} invoice {invoice['invoice_id']}")
            per_invoice[invoice["invoice_id"]] = per_invoice.get(invoice["invoice_id"], 0) + row["doc_amount_minor"]
            try:
                if paid_on < book.policy.local_date(event_time("invoices", invoice)):
                    _add(problems, "INV-12", f"payment {pid} on {paid_on} clears invoice {invoice['invoice_id']} "
                                             f"issued later")
            except BookError:
                pass                 # reported by the period check
    for iid, total in sorted(per_invoice.items()):
        if type(invoices[iid].get("total_minor")) is int and total > invoices[iid]["total_minor"]:
            _add(problems, "INV-09", f"invoice {iid}: cleared {total} of {invoices[iid].get('total_minor')}")


def _ref_pattern(key):
    """A group reference inside free text: its characters in order, any separators between them."""
    return re.compile(r"(?<![A-Z0-9])" + r"[^A-Z0-9]*".join(map(re.escape, key)) + r"(?![A-Z0-9])")


def _references(book, problems):
    """SPEC §3.2: invoice_refs are the invoices remittance_info points to, sorted, null when none."""
    keys = []
    for invoice in book.invoices:
        ref = invoice.get("invoice_number") if invoice.get("direction") == "AR" else invoice.get("vendor_ref")
        key = normalize_ref(ref)
        if key:
            keys.append((_ref_pattern(key), invoice["invoice_id"]))
    for payment in book.payments:
        text = payment.get("remittance_info")
        text = text.upper() if type(text) is str else ""
        named = sorted({iid for pattern, iid in keys if pattern.search(text)}) or None
        if payment.get("invoice_refs") != named:
            _add(problems, "INV-08", f"payment {payment['payment_id']}: invoice_refs {payment.get('invoice_refs')}, "
                                     f"remittance_info names {named}")
