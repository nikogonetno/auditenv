"""Baseline «Простой алгоритм сверки» (case §06; docs/tasks/B_next.md item 6).

    python3 -m policies.reconcile [<seed, ignored>]

Deliberately naive: a baseline, not a verifier. It pages through ledger, invoices, payments and fx
with list_records (limit 50) until next_cursor is null, and stops reading while two steps are
still left, so the submit always fits in the 100-step budget. Then it applies four direct rules
to what it has read and submits up to 30 findings in case §05 shape. It is deterministic; the
seed argument is accepted and unused.

    a) duplicate_invoice   two AP invoices of one entity_id with the same currency and total_minor
    b) payment_allocation  an invoice in a payment's invoice_refs is still open after its clearing
                           rows, and part of that payment is not cleared against it
    c) fx_rate             a carrier row's exchange_rate differs from fx.csv on the company-local
                           date of the document; with no quote that day it takes the next published
                           quote, never the previous business day the policy asks for
    d) period_cutoff       the document's UTC date and the carrier row's local date fall on
                           different sides of the period

The rules ignore what policy.json says about vendor_ref, partial payments, non-business days and
the company-local posting date, so they fire on correct look-alike records too. That is the point
of this baseline. Money is int minor units; rates go through Decimal with ROUND_HALF_UP.
"""
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from itertools import combinations

from policies.stdio import TABLE_ORDER, is_currency, serve, tables_in_order

PAGE_LIMIT = 50
MAX_STEPS = 100
MIN_STEPS_LEFT = 2      # read only while at least this many steps remain; the last one is the submit
MAX_FINDINGS = 30       # case §05
MAX_EVIDENCE = 10       # case §05
INVOICE_CARRIERS = ("1210", "3310")   # receivable and payable rows of an invoice
BANK = "1030"
ADVANCES = "3510"
PAYABLE = "3310"

EXPLAIN = {
    "duplicate_invoice": "two AP invoices of one counterparty with the same currency and total",
    "payment_allocation": "invoice named by the payment is still open and part of the payment is not cleared to it",
    "fx_rate": "row exchange_rate differs from fx.csv on the document date or the next quoted date",
    "period_cutoff": "document UTC date and posting date fall in different periods",
}


# ---------------------------------------------------------------- value helpers (never raise)
def _text(value):
    return value if isinstance(value, str) and value else None


def _int(value):
    return value if type(value) is int else None


def _decimal(value):
    if not isinstance(value, str):
        return None
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def _round_half_up(number):
    return int(number.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _time(value):
    """datetime from ISO 8601, aware when the text carries an offset; None if it does not parse."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _offset(value):
    """timezone for a fixed offset like "+05:00", else None."""
    if not isinstance(value, str) or len(value) != 6 or value[0] not in "+-" or value[3] != ":":
        return None
    try:
        hours, minutes = int(value[1:3]), int(value[4:6])
    except ValueError:
        return None
    if hours > 23 or minutes > 59:
        return None
    sign = 1 if value[0] == "+" else -1
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def _date(value):
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _local_date(moment, tz):
    """Date of a datetime in the company offset; a naive one keeps its own date."""
    if moment is None:
        return None
    if moment.tzinfo is None or tz is None:
        return moment.date()
    return moment.astimezone(tz).date()


def _utc_date(moment):
    if moment is None or moment.tzinfo is None:
        return None
    return moment.astimezone(timezone.utc).date()


def _base_amount(row):
    """The base-currency amount of a ledger row: its debit, else its credit."""
    debit, credit = _int(row.get("debit_minor")) or 0, _int(row.get("credit_minor")) or 0
    return debit if debit else credit


def _doc_amount(row):
    return _int(row.get("doc_amount_minor")) or 0


# ---------------------------------------------------------------- the book as read
class Book:
    """Indexes over the records read, first record wins on a repeated id."""

    def __init__(self, manifest, tables):
        manifest = manifest if isinstance(manifest, dict) else {}
        self.base = manifest.get("base_currency") if is_currency(manifest.get("base_currency")) else None
        self.tz = _offset(manifest.get("timezone"))
        self.period = (_date(manifest.get("period_start")), _date(manifest.get("period_end")))
        self.invoices = self._index(tables.get("invoices"), "invoice_id")
        self.payments = self._index(tables.get("payments"), "payment_id")
        self.ledger = [r for r in tables.get("ledger") or () if isinstance(r, dict) and _text(r.get("row_id"))]
        self.ledger.sort(key=lambda r: r["row_id"])
        self.fx = {}
        for row in tables.get("fx") or ():
            if not isinstance(row, dict):
                continue
            key = (row.get("date"), row.get("source_currency"), row.get("target_currency"))
            if all(isinstance(k, str) for k in key) and _text(row.get("fx_id")) and key not in self.fx:
                self.fx[key] = row

    @staticmethod
    def _index(records, id_field):
        index = {}
        for record in records or ():
            if isinstance(record, dict) and _text(record.get(id_field)) and record[id_field] not in index:
                index[record[id_field]] = record
        return dict(sorted(index.items()))

    def rows(self, account=None, invoice_id=None, payment_id=None, has_payment=None):
        """Ledger row ids in id order, filtered by account, invoice_id, payment_id and whether payment_id is set."""
        found = []
        for row in self.ledger:
            if account is not None and row.get("account") != account:
                continue
            if invoice_id is not None and row.get("invoice_id") != invoice_id:
                continue
            if payment_id is not None and row.get("payment_id") != payment_id:
                continue
            if has_payment is not None and bool(_text(row.get("payment_id"))) != has_payment:
                continue
            found.append(row)
        return found

    def carriers(self):
        """(row, document ref, document time) of every carrier row, in row id order.

        The 1210 or 3310 row of an invoice (invoice_id set, payment_id empty) carries the invoice;
        the 1030 row of a payment carries the payment.
        """
        for row in self.ledger:
            invoice_id, payment_id = _text(row.get("invoice_id")), _text(row.get("payment_id"))
            if row.get("account") in INVOICE_CARRIERS and invoice_id and not payment_id:
                document = self.invoices.get(invoice_id)
                if document is not None:
                    yield row, f"invoice/{invoice_id}", _time(document.get("issued_at"))
            elif row.get("account") == BANK and payment_id:
                document = self.payments.get(payment_id)
                if document is not None:
                    yield row, f"payment/{payment_id}", _time(document.get("timestamp"))


def _finding(issue_type, affected, amount, currency, evidence):
    if type(amount) is not int or amount <= 0 or not is_currency(currency):
        return None
    unique = []
    for ref in evidence:
        if ref not in unique:
            unique.append(ref)
    return {
        "type": issue_type,
        "affected_ids": sorted(affected),
        "claimed_amount_minor": amount,
        "currency": currency,
        "evidence_ids": unique[:MAX_EVIDENCE],
        "explanation": EXPLAIN[issue_type],
    }


# ---------------------------------------------------------------- the four rules
def duplicate_findings(book):
    """a) Two AP invoices of one entity_id with equal currency and total_minor, every pair of a group."""
    groups = {}
    for invoice_id, invoice in book.invoices.items():
        key = (_text(invoice.get("entity_id")), _text(invoice.get("currency")), _int(invoice.get("total_minor")))
        if invoice.get("direction") == "AP" and None not in key:
            groups.setdefault(key, []).append(invoice_id)
    findings = []
    for (_, currency, total), members in sorted(groups.items()):
        for first, second in combinations(members, 2):
            evidence = [f"invoice/{first}", f"invoice/{second}"]
            for invoice_id in (first, second):
                evidence += [f"ledger/{r['row_id']}" for r in book.rows(PAYABLE, invoice_id, has_payment=False)]
            findings.append(_finding("duplicate_invoice", [f"invoice/{first}", f"invoice/{second}"],
                                     total, currency, evidence))
    return findings


def allocation_findings(book):
    """b) An invoice named by a payment is still open, and the payment is not fully cleared to it."""
    findings = []
    for payment_id, payment in book.payments.items():
        refs = payment.get("invoice_refs")
        amount = _int(payment.get("amount_minor"))
        if not isinstance(refs, list) or amount is None:
            continue
        for invoice_id in sorted({r for r in refs if _text(r)}):
            invoice = book.invoices.get(invoice_id)
            total = _int(invoice.get("total_minor")) if invoice is not None else None
            if total is None:
                continue
            clearing = [r for r in book.rows(invoice_id=invoice_id, has_payment=True)
                        if r.get("account") in INVOICE_CARRIERS]
            if sum(_doc_amount(r) for r in clearing) >= total:
                continue
            own = [r for r in clearing if r.get("payment_id") == payment_id]
            evidence = [f"payment/{payment_id}", f"invoice/{invoice_id}"]
            evidence += [f"ledger/{r['row_id']}" for r in own]
            evidence += [f"ledger/{r['row_id']}" for r in book.rows(ADVANCES, payment_id=payment_id)]
            findings.append(_finding("payment_allocation", [f"payment/{payment_id}", f"invoice/{invoice_id}"],
                                     amount - sum(_doc_amount(r) for r in own), payment.get("currency"), evidence))
    return findings


NEXT_QUOTE_DAYS = 7     # how far rule c) looks forward for a quote


def rate_findings(book):
    """c) A carrier row's rate differs from fx.csv on the document's company-local date. On a day without
    a quote the rule takes the next quoted day, the naive reading of «the rate of the document date»."""
    findings = []
    for row, document_ref, moment in book.carriers():
        currency = _text(row.get("doc_currency"))
        day = _local_date(moment, book.tz)
        row_rate = _decimal(row.get("exchange_rate"))
        if book.base is None or currency in (None, book.base) or day is None or row_rate is None:
            continue
        quote = None
        for ahead in range(NEXT_QUOTE_DAYS + 1):
            quote = book.fx.get(((day + timedelta(days=ahead)).isoformat(), currency, book.base))
            if quote is not None:
                break
        quote_rate = _decimal(quote.get("rate")) if quote is not None else None
        if quote_rate is None or quote_rate == row_rate:
            continue
        amount = abs(_base_amount(row) - _round_half_up(Decimal(_doc_amount(row)) * quote_rate))
        findings.append(_finding("fx_rate", [f"ledger/{row['row_id']}"], amount, book.base,
                                 [document_ref, f"ledger/{row['row_id']}", f"fx/{quote['fx_id']}"]))
    return findings


def period_findings(book):
    """d) The document's UTC date and the carrier row's local date are on different sides of the period."""
    start, end = book.period
    if book.base is None or start is None or end is None:
        return []
    findings = []
    for row, document_ref, moment in book.carriers():
        document_day = _utc_date(moment)
        row_day = _local_date(_time(row.get("timestamp")), book.tz)
        if document_day is None or row_day is None:
            continue
        if (start <= document_day <= end) != (start <= row_day <= end):
            findings.append(_finding("period_cutoff", [f"ledger/{row['row_id']}"], _base_amount(row), book.base,
                                     [document_ref, f"ledger/{row['row_id']}"]))
    return findings


RULES = (duplicate_findings, allocation_findings, rate_findings, period_findings)


# Record content that survives a permutation of rows and a change of ids (case §10 paired checks).
CONTENT = {
    "ledger": ("timestamp", "account", "debit_minor", "credit_minor", "doc_currency", "doc_amount_minor",
               "exchange_rate", "memo"),
    "invoice": ("issued_at", "currency", "total_minor", "invoice_number", "vendor_ref", "po_ref",
                "counterparty_name", "entered_at"),
    "payment": ("timestamp", "currency", "amount_minor", "remittance_info", "counterparty_name_raw", "entered_at"),
}


def _content_key(book, rows, finding):
    """Sort key of a finding from what its records say, never from their ids."""
    parts = []
    for ref in finding["affected_ids"]:
        table, _, rid = ref.partition("/")
        record = {"ledger": rows, "invoice": book.invoices, "payment": book.payments}.get(table, {}).get(rid) or {}
        parts.append(tuple(repr(record.get(c)) for c in CONTENT.get(table, ())))
    return finding["claimed_amount_minor"], finding["currency"], tuple(sorted(parts))


def reconcile(manifest, tables):
    """Findings of the four rules over the records read, in rule order, at most 30. Inside a rule the
    findings go in the order of their records' content, so the same book with other ids and row order
    gets the same findings (the oracle and this policy keep their reward under the paired checks)."""
    book = Book(manifest, tables)
    rows = {r["row_id"]: r for r in book.ledger}
    findings = []
    for rule in RULES:
        found = [f for f in rule(book) if f is not None]
        findings += sorted(found, key=lambda f: _content_key(book, rows, f))
    return findings[:MAX_FINDINGS]


# ---------------------------------------------------------------- the policy process
class ReconcilePolicy:
    def __init__(self):
        self.manifest = None
        self.pending = []       # tables still to page through
        self.cursor = None
        self.steps = 0          # actions sent so far
        self.tables = {table: [] for table in TABLE_ORDER}

    def __call__(self, message):
        observation = message.get("observation") if isinstance(message, dict) else None
        observation = observation if isinstance(observation, dict) else {}
        if self.manifest is None:
            manifest = observation.get("manifest")
            self.manifest = manifest if isinstance(manifest, dict) else {}
            self.pending = [table for table in tables_in_order(observation) if table in TABLE_ORDER]
        else:
            self._absorb(observation)
        if not self.pending or self._steps_left(observation) < MIN_STEPS_LEFT:
            return self._submit()
        self.steps += 1
        action = {"action": "list_records", "table": self.pending[0], "limit": PAGE_LIMIT}
        if self.cursor is not None:
            action["cursor"] = self.cursor
        return action

    def _steps_left(self, observation):
        """The smaller of the env's steps_left and the policy's own count."""
        own = MAX_STEPS - self.steps
        seen = observation.get("steps_left")
        return min(own, seen) if type(seen) is int else own

    def _absorb(self, observation):
        result = observation.get("result")
        table = self.pending[0] if self.pending else None
        if not isinstance(result, dict) or result.get("table") != table:
            if self.pending:  # an error: give up on this table
                self.pending.pop(0)
            self.cursor = None
            return
        records = result.get("records")
        self.tables[table] += [r for r in records if isinstance(r, dict)] if isinstance(records, list) else []
        cursor = result.get("next_cursor")
        self.cursor = cursor if isinstance(cursor, str) else None
        if self.cursor is None:
            self.pending.pop(0)

    def _submit(self):
        return {"action": "submit",
                "findings": {"episode_id": self.manifest.get("episode_id"),
                             "findings": reconcile(self.manifest, self.tables)}}


def main():
    serve(ReconcilePolicy())


if __name__ == "__main__":
    main()
