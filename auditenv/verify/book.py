"""Read-only view of one episode for the verifier: times in the company offset, money, ids and links.

Built only from SPEC §3 and the policy rules of the domain spec A, never from the generator code.
Every value that cannot be read raises BookError; the caller turns it into a problem and goes on.
"""
import datetime as dt
import decimal
import re
from decimal import Decimal

from ..episode import load_episode
from .ids import normalize_id, split_id

ISSUE_TYPES = ("duplicate_invoice", "payment_allocation", "fx_rate", "period_cutoff")
PREFIX_TABLE = {"invoice": "invoices", "payment": "payments", "ledger": "ledger", "fx": "fx"}

BANK, RECEIVABLE, PAYABLE, UNAPPLIED = "1030", "1210", "3310", "3510"
CARRYING_ACCOUNT = {"AR": RECEIVABLE, "AP": PAYABLE}
PAYMENT_SIDE = {"in": "AR", "out": "AP"}

_CTX = decimal.Context(prec=60, rounding=decimal.ROUND_HALF_UP, Emin=-999999, Emax=999999,
                       traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow])
_OFFSET = re.compile(r"([+-])(\d\d):(\d\d)")
_RATE = re.compile(r"[0-9]+(\.[0-9]+)?")
_NOT_ALNUM = re.compile(r"[^A-Z0-9]")


class BookError(ValueError):
    """A value the verifier cannot read. Reported as a problem, never raised to the caller."""


def normalize_ref(text):
    """policy.vendor_ref_normalization: uppercase, keep A-Z and 0-9 only. Also used for invoice_number."""
    if type(text) is not str:
        return ""
    return _NOT_ALNUM.sub("", text.upper())


def parse_offset(text):
    m = _OFFSET.fullmatch(text) if type(text) is str else None
    if not m:
        raise BookError(f"bad utc offset {text!r}")
    minutes = int(m.group(2)) * 60 + int(m.group(3))
    return dt.timezone(dt.timedelta(minutes=-minutes if m.group(1) == "-" else minutes))


def parse_time(text):
    """ISO 8601 with an explicit offset. A naive time has no company date, so it is an error."""
    try:
        value = dt.datetime.fromisoformat(text)
    except (TypeError, ValueError):
        raise BookError(f"bad timestamp {text!r}") from None
    if value.utcoffset() is None:
        raise BookError(f"timestamp without offset {text!r}")
    return value


def parse_date(text):
    try:
        return dt.date.fromisoformat(text)
    except (TypeError, ValueError):
        raise BookError(f"bad date {text!r}") from None


def parse_rate(text):
    """Quotes are decimal strings (case §03). Exponents, signs and spaces are not quotes."""
    if type(text) is not str or not _RATE.fullmatch(text):
        raise BookError(f"bad rate {text!r}")
    return Decimal(text)


def minor(value, what):
    if type(value) is not int or value < 0:
        raise BookError(f"bad {what} {value!r}")
    return value


class Policy:
    """The parts of policy.json the rules need, parsed once."""

    def __init__(self, policy):
        try:
            self.base = policy["base_currency"]
            self.tz = parse_offset(policy["company_utc_offset"])
            self.minor_units = dict(policy["minor_units"])
            fx = policy["fx"]
            self.weekend_days = frozenset(fx["weekend_days"])
            self.holidays = frozenset(parse_date(d) for d in fx["holidays"])
            self.accounts = frozenset(a["code"] for a in policy["accounts"])
        except (KeyError, TypeError, AttributeError) as exc:
            raise BookError(f"policy.json: {type(exc).__name__} {exc}") from None
        if type(self.base) is not str or not all(type(v) is int and v >= 0 for v in self.minor_units.values()):
            raise BookError("policy.json: bad base_currency or minor_units")
        if not self.weekend_days <= frozenset(range(1, 8)) or len(self.weekend_days) == 7:
            raise BookError("policy.json: bad fx.weekend_days")

    def local_date(self, text):
        return parse_time(text).astimezone(self.tz).date()

    def rate_date(self, event_text):
        """Rate date by the policy: the local document date, moved back to the previous business day."""
        day = self.local_date(event_text)
        for _ in range(400):
            if day.isoweekday() not in self.weekend_days and day not in self.holidays:
                return day
            day -= dt.timedelta(days=1)
        raise BookError("policy.json: no business day within 400 days")

    def to_base(self, doc_minor, rate, doc_currency):
        """base_minor = round_half_up(doc_minor * rate), shifted when the minor units differ."""
        try:
            shift = self.minor_units[self.base] - self.minor_units[doc_currency]
        except KeyError:
            raise BookError(f"no minor units for {doc_currency!r}") from None
        value = _CTX.multiply(Decimal(minor(doc_minor, "doc_amount_minor")), rate).scaleb(shift, _CTX)
        return int(value.quantize(Decimal(1), rounding=decimal.ROUND_HALF_UP, context=_CTX))


def row_base(row):
    """Base amount of a ledger row: the one filled side. Both or none filled is an INV-01 problem."""
    debit, credit = minor(row.get("debit_minor"), "debit_minor"), minor(row.get("credit_minor"), "credit_minor")
    if (debit > 0) == (credit > 0):
        raise BookError(f"ledger {row.get('row_id')}: debit and credit both {'filled' if debit else 'empty'}")
    return debit or credit


def event_time(table, doc):
    return doc.get("issued_at") if table == "invoices" else doc.get("timestamp")


def is_carrying(table, doc, row):
    """SPEC §3.6: 1210 of a customer bill, 3310 of a supplier bill, 1030 of a payment."""
    if table == "invoices":
        return (row.get("invoice_id") == doc["invoice_id"] and row.get("payment_id") is None
                and row.get("account") == CARRYING_ACCOUNT.get(doc.get("direction")))
    return row.get("payment_id") == doc["payment_id"] and row.get("account") == BANK


def is_clearing(row):
    """A payment row that settles one invoice: 1210 or 3310 with both payment_id and invoice_id."""
    return (row.get("account") in (RECEIVABLE, PAYABLE) and row.get("payment_id") is not None
            and row.get("invoice_id") is not None)


def group_key(invoice):
    """Document group by the policy: AR by invoice_number, AP by normalized vendor_ref."""
    side = invoice.get("direction")
    ref = invoice.get("invoice_number") if side == "AR" else invoice.get("vendor_ref") if side == "AP" else None
    key = normalize_ref(ref)
    if not key:
        return None
    return side, invoice.get("entity_id"), invoice.get("currency"), key


class Book:
    """One loaded episode with the indexes the checks share."""

    def __init__(self, episode_dir):
        episode = load_episode(episode_dir)
        self.episode = episode
        self.manifest = episode.manifest
        self.raw_policy = episode.policy
        self.policy = Policy(episode.policy)
        self.base = self.policy.base
        self.ledger = episode.tables["ledger"]
        self.invoices = episode.tables["invoices"]
        self.payments = episode.tables["payments"]
        self.fx = episode.tables["fx"]
        self.index = episode.index
        self.aliases = episode.aliases
        self.rows_by_invoice, self.rows_by_payment = {}, {}
        for row in self.ledger:
            if row.get("invoice_id") is not None:
                self.rows_by_invoice.setdefault(row["invoice_id"], []).append(row)
            if row.get("payment_id") is not None:
                self.rows_by_payment.setdefault(row["payment_id"], []).append(row)

    def documents(self):
        """(table, document) for every invoice and payment, in file order."""
        return [("invoices", d) for d in self.invoices] + [("payments", d) for d in self.payments]

    def doc_id(self, table, doc):
        return doc["invoice_id"] if table == "invoices" else doc["payment_id"]

    def carrying_rows(self, table, doc):
        rows = self.rows_by_invoice if table == "invoices" else self.rows_by_payment
        return [r for r in rows.get(self.doc_id(table, doc), ()) if is_carrying(table, doc, r)]

    def canon(self, ref):
        """Canonical record id of a reference: known prefix stripped once, doc_id mapped to its record."""
        if type(ref) is not str:
            raise BookError(f"reference is not a string: {ref!r}")
        rid = normalize_id(ref)
        return self.aliases.get(rid, rid)

    def resolve(self, ref):
        """(table, record) for a reference, or BookError. ID are unique across tables (SPEC §3.3)."""
        prefix, rid = split_id(ref) if type(ref) is str else (None, None)
        if rid is None or ref.startswith("je/"):
            raise BookError(f"unresolvable reference {ref!r}")
        rid = self.aliases.get(rid, rid)
        tables = [PREFIX_TABLE[prefix]] if prefix in PREFIX_TABLE else list(self.index)
        hits = [(t, self.index[t][rid]) for t in tables if rid in self.index[t]]
        if prefix == "doc":
            hits = [h for h in hits if h[0] in ("invoices", "payments")]
        if len(hits) != 1:
            raise BookError(f"reference {ref!r} resolves to {len(hits)} records")
        return hits[0]
