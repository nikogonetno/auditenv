"""Own detector of the clean-book rules INV-13…INV-16 on the final public files.

One violation is (type, canonical affected ids, amount, currency, note). Affected ids follow SPEC §3.6:
both bills for a duplicate, payment and bill for an allocation, the carrying row for rate and period.
Records that cannot be read go to `problems` and are skipped, the rest of the book is still checked.
"""
from dataclasses import dataclass
from decimal import Decimal

from .book import (PAYMENT_SIDE, BookError, event_time, group_key, is_clearing, minor, normalize_ref,
                   parse_rate, parse_time, row_base)
from .ids import normalize_id


@dataclass(frozen=True)
class Violation:
    type: str
    ids: tuple            # sorted canonical ids
    amount: int
    currency: str
    note: str

    @property
    def key(self):
        return self.type, frozenset(self.ids)


def detect(book, problems):
    """All rule violations of the book, sorted by (type, ids). Unreadable records go to problems."""
    found = []
    for check in (_period_and_rate, _clearing_rates, _duplicates, _allocations):
        found.extend(check(book, problems))
    unique = {}
    for v in found:
        unique.setdefault(v.key, v)
    return sorted(unique.values(), key=lambda v: (v.type, v.ids))


def _problem(problems, code, detail):
    problems.append({"code": code, "detail": detail})


def _period_and_rate(book, problems):
    """INV-13: posting date is the local date of the document event.
    INV-14, first half: the carrying row uses the rate of the rule date, 1 in the base currency."""
    policy = book.policy
    rates = {}
    for fx in book.fx:
        rates.setdefault((fx.get("date"), fx.get("source_currency"), fx.get("target_currency")), []).append(fx)
    out = []
    for table, doc in book.documents():
        did = book.doc_id(table, doc)
        carrying = book.carrying_rows(table, doc)
        if len(carrying) > 1:
            _problem(problems, "INV-13", f"document {did} has {len(carrying)} carrying rows")
        for row in carrying:
            rid = row["row_id"]
            try:
                event = event_time(table, doc)
                posted, happened = policy.local_date(row.get("timestamp")), policy.local_date(event)
                base = row_base(row)
                if posted != happened:
                    out.append(Violation("period_cutoff", (rid,), base, book.base,
                                         f"posted {posted}, document {did} dated {happened}"))
                booked = parse_rate(row.get("exchange_rate"))
                doc_currency = doc.get("currency")
                if doc_currency == book.base:
                    right, where = Decimal(1), "base currency"
                else:
                    day = policy.rate_date(event)
                    quotes = rates.get((day.isoformat(), doc_currency, book.base), [])
                    if len(quotes) != 1:
                        _problem(problems, "INV-14",
                                 f"{len(quotes)} rates {doc_currency}/{book.base} for {day}, carrying row {rid}")
                        continue
                    right, where = parse_rate(quotes[0].get("rate")), f"rate of {day} fx/{quotes[0].get('fx_id')}"
                if booked != right:
                    amount = abs(base - policy.to_base(row.get("doc_amount_minor"), right, row.get("doc_currency")))
                    out.append(Violation("fx_rate", (rid,), amount, book.base,
                                         f"rate {booked}, {where} is {right}, document {did}"))
            except BookError as exc:
                _problem(problems, "SCHEMA", f"carrying row {rid}: {exc}")
    return out


def _clearing_rates(book, problems):
    """INV-14, second half: a clearing row uses the rate the invoice was booked at."""
    out = []
    invoices = book.index["invoices"]
    for row in book.ledger:
        if not is_clearing(row):
            continue
        rid, iid = row["row_id"], row["invoice_id"]
        invoice = invoices.get(iid)
        if invoice is None:
            _problem(problems, "INV-08", f"ledger {rid} clears unknown invoice {iid}")
            continue
        carrying = book.carrying_rows("invoices", invoice)
        if len(carrying) != 1:
            _problem(problems, "INV-14", f"ledger {rid} clears invoice {iid} with {len(carrying)} carrying rows")
            continue
        try:
            booked, invoice_rate = parse_rate(row.get("exchange_rate")), parse_rate(carrying[0].get("exchange_rate"))
            if booked != invoice_rate:
                amount = abs(row_base(row) - book.policy.to_base(row.get("doc_amount_minor"), invoice_rate,
                                                                 row.get("doc_currency")))
                out.append(Violation("fx_rate", (rid,), amount, book.base,
                                     f"clearing rate {booked}, invoice {iid} booked at {invoice_rate}"))
        except BookError as exc:
            _problem(problems, "SCHEMA", f"clearing row {rid}: {exc}")
    return out


def _posted(book, invoice):
    return bool(book.carrying_rows("invoices", invoice))


def _duplicates(book, problems):
    """INV-15: no two posted AP bills of one counterparty with equal currency, total and vendor_ref."""
    groups = {}
    for invoice in book.invoices:
        if invoice.get("direction") != "AP" or invoice.get("entity_id") is None or not _posted(book, invoice):
            continue
        key = normalize_ref(invoice.get("vendor_ref"))
        if key:
            dup_key = (invoice["entity_id"], invoice.get("currency"), invoice.get("total_minor"), key)
            groups.setdefault(dup_key, []).append(invoice["invoice_id"])
    out = []
    for (entity, currency, total, key), ids in sorted(groups.items(), key=lambda kv: sorted(kv[1])):
        if len(ids) > 1:
            if len(ids) > 2:
                _problem(problems, "INV-15", f"{len(ids)} bills share the duplicate key: {sorted(ids)}")
            try:
                amount = minor(total, "total_minor")
            except BookError as exc:
                _problem(problems, "SCHEMA", f"bills {sorted(ids)}: {exc}")
                continue
            out.append(Violation("duplicate_invoice", tuple(sorted(ids)), amount, currency,
                                 f"entity {entity}, vendor_ref {key}, {total} {currency}"))
    return out


def _payment_order(book, problems):
    keyed = []
    for p in book.payments:
        try:
            keyed.append((parse_time(p.get("timestamp")), p["payment_id"], p))
        except BookError as exc:
            _problem(problems, "SCHEMA", f"payment {p['payment_id']}: {exc}")
    keyed.sort(key=lambda k: (k[0], k[1]))
    return [p for _, _, p in keyed]


def _allocations(book, problems):
    """INV-16: a payment whose reference names exactly one open group of the same counterparty and
    currency is allocated to it up to the open amount, the rest goes to 3510. Payments in bank order."""
    invoices = book.index["invoices"]
    groups = {}
    for invoice in book.invoices:
        key = group_key(invoice)
        if key is not None and _posted(book, invoice):
            groups.setdefault(key, []).append(invoice)
    cleared = {}                     # invoice_id -> doc amount cleared by the payments seen so far
    out = []
    for payment in _payment_order(book, problems):
        pid = payment["payment_id"]
        try:
            mine = {}
            for row in book.rows_by_payment.get(pid, []):
                if is_clearing(row):
                    amount = minor(row.get("doc_amount_minor"), "doc_amount_minor")
                    mine[row["invoice_id"]] = mine.get(row["invoice_id"], 0) + amount
        except BookError as exc:
            _problem(problems, "SCHEMA", f"payment {pid}: {exc}")
            continue
        try:
            violation = _allocation(book, payment, mine, cleared, groups, invoices, problems)
        except BookError as exc:
            _problem(problems, "SCHEMA", f"payment {pid}: {exc}")
            violation = None
        for iid, amount in mine.items():
            cleared[iid] = cleared.get(iid, 0) + amount
        if violation is not None:
            out.append(violation)
    return out


def _allocation(book, payment, mine, cleared, groups, invoices, problems):
    pid = payment["payment_id"]
    refs = payment.get("invoice_refs")
    if refs is None:
        return None                  # no reference: the whole amount may lie on 3510
    if type(refs) is not list or not refs:
        raise BookError(f"invoice_refs {refs!r}")
    side = PAYMENT_SIDE.get(payment.get("direction"))
    named = set()
    for ref in refs:
        invoice = invoices.get(normalize_id(ref)) if type(ref) is str else None
        if invoice is None:
            _problem(problems, "INV-08", f"payment {pid} refers to unknown invoice {ref!r}")
            continue
        key = group_key(invoice)
        if (key is not None and key[0] == side and invoice.get("entity_id") == payment.get("entity_id")
                and invoice.get("currency") == payment.get("currency")):
            named.add(key)
    if len(named) != 1:
        return None                  # no group or several: the policy asks for nothing
    group = sorted(groups.get(named.pop(), []), key=lambda i: i["invoice_id"])
    ids = [i["invoice_id"] for i in group]
    open_before = {i["invoice_id"]: minor(i.get("total_minor"), "total_minor") - cleared.get(i["invoice_id"], 0)
                   for i in group}
    amount = minor(payment.get("amount_minor"), "amount_minor")
    required = min(amount, max(sum(open_before.values()), 0))
    allocated = sum(mine.get(iid, 0) for iid in ids)
    if allocated == required:
        return None
    if len(ids) == 1:
        blamed = ids
    elif allocated < required:       # the bills that stay open after this payment
        blamed = [iid for iid in ids if open_before[iid] - mine.get(iid, 0) > 0]
    else:                            # the bills this payment over-cleared
        blamed = [iid for iid in ids if mine.get(iid, 0) > 0]
    note = f"payment {amount}, group open {sum(open_before.values())}, allocated {allocated}, due {required}"
    return Violation("payment_allocation", tuple(sorted([pid, *blamed])), abs(required - allocated),
                     payment.get("currency"), note)
