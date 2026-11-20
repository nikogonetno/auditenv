"""Hidden company history and the clean, balanced ledger built from it (role A).

The world is a set of documents, invoices and bank payments, together with the decisions a correct
accountant makes about them: posting date, exchange rate and payment allocation. `post` turns the
documents into ledger rows by the policy rules of docs/DOMAIN_SPEC_A.md. An error (auditenv/inject)
changes one decision or adds one document, and the book is posted again. So the clean and the final
book come from one engine and differ only where an injection acted.

Records carry internal keys (C01, I0001, P0001, ...) until auditenv/generate.py assigns public ids.
Deterministic: randomness only from the random.Random passed in, money in int minor units, rates as
Decimal with ROUND_HALF_UP, no set or dict reaches the output unsorted, no clock, no tzdata.
"""
import copy
import datetime as dt
import functools
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Context, Decimal, DivisionByZero, InvalidOperation, Overflow

BASE = "KZT"
FOREIGN = ("CNY", "EUR", "RUB", "USD")
MINOR_UNITS = {c: 2 for c in (BASE, *FOREIGN)}
WEEKEND_DAYS = (6, 7)                    # ISO weekday numbers, date.isoweekday(): Saturday, Sunday
# Synthetic non-business days. Only those inside the rate table reach policy.json.
HOLIDAYS = tuple(dt.date.fromisoformat(d) for d in (
    "2025-12-16", "2025-12-17", "2026-01-01", "2026-01-02", "2026-01-07", "2026-03-09", "2026-03-23",
    "2026-03-24", "2026-05-01", "2026-05-07", "2026-05-11", "2026-07-06", "2026-08-31", "2026-10-26",
    "2026-12-16", "2026-12-17", "2027-01-01", "2027-01-07"))
WINDOW_DAYS = 7
NEIGHBOUR_INVOICES = 6                   # per side of the period: 12 rows, INV-03 asks for 10; more when
                                         # an error moves documents out of the neighbour count (anchors)
RATE_LEAD_DAYS = 14                      # rates start this many days before the window
RATE_TAIL_DAYS = 14                      # and run past it, so every entered_at date has a rate
# Clean-world shares, domain spec A world table.
BEHAVIOURS = ("on_time", "late", "partial", "split", "overpay", "partial_then_full")
BEHAVIOUR_WEIGHTS = (58, 25, 7, 5, 2, 3)
SUBSCRIPTION_PCT = (20, 40)              # share of vendor bills that repeat monthly with one amount
AP_SHARE_OF_DOCUMENTS = 0.45             # calibrated: the measured monthly share lands in SUBSCRIPTION_PCT
BILLS_PER_CONTRACT = 1.45                # a 44-day window holds one or two bills of a monthly contract
COMBINED_AR_SHARE = 0.05                 # a customer pays two invoices with one payment
COMBINED_AP_SHARE = 0.08                 # we pay two bills of one vendor with one payment
CLERK_RETYPE_SHARE = 0.2                 # bills whose vendor reference the clerk keyed with other separators
CENTS_SHARE_FOREIGN, CENTS_SHARE_BASE = 0.5, 0.2
# A trading company works on business days (PM 21.11 03.15: 2-6% of documents on weekends and holidays).
# Each book draws a quota of OFF_DAY_PERMILLE of its documents; a draw that falls on an off day is kept
# with probability OFF_DAY_KEEP while the quota lasts, otherwise redrawn (documents) or moved to the next
# business day (bank value dates, monthly bills). The planted foreign weekend look-alike uses the quota too.
OFF_DAY_PERMILLE = (25, 55)
OFF_DAY_KEEP = 0.25
_DECIMAL = Context(prec=34, rounding=ROUND_HALF_UP, Emin=-999999, Emax=999999, capitals=1, clamp=0, flags=[],
                   traps=[InvalidOperation, DivisionByZero, Overflow])
MAX_CONTRACTS_PER_VENDOR = 4
REPEATED_FULL_TWINS = (1, 2)              # planted in every book that has room: clean and erroneous alike

# Rate random walk in hundredths of KZT per unit: start level, daily step, floor.
RATE_WALK = {"CNY": (7090, 30, 5000), "EUR": (59800, 240, 40000), "RUB": (562, 4, 300), "USD": (51200, 220, 35000)}
# Document totals in major units, by currency.
TOTAL_RANGE = {"KZT": (40_000, 6_000_000), "USD": (300, 15_000), "EUR": (300, 12_000),
               "RUB": (30_000, 1_200_000), "CNY": (2_000, 90_000)}

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
BANK, RECEIVABLE, PAYABLE, UNAPPLIED, REVENUE, FX_GAIN, FX_LOSS = "1030", "1210", "3310", "3510", "6010", "6250", "7430"
CARRIER = {"AR": RECEIVABLE, "AP": PAYABLE}

CITIES = ("Almaty", "Astana", "Shymkent", "Aktobe", "Atyrau", "Urumqi", "Tashkent", "Hamburg", "Busan",
          "Shanghai", "Duisburg", "Istanbul", "Novosibirsk", "Frankfurt")
INDUSTRIES = {
    "logistics": ("Freight forwarding {a}-{b}", "Container freight {a}-{b} 40ft", "Customs brokerage",
                  "Warehouse handling", "Cross-dock services", "Air freight {a}-{b}", "Project cargo {a}-{b}"),
    "wholesale": ("Steel pipes DN50", "Office chairs, batch", "Canned vegetables, pallet", "Ceramic tiles 60x60",
                  "Cable VVG 3x2.5, drum", "Paint, 20 l buckets", "Copy paper A4, pallet"),
    "it_services": ("Software development, sprint", "Server support", "Cloud migration consulting",
                    "Integration works", "Helpdesk outsourcing", "Database administration", "Security audit"),
}
# Vendor kinds: expense account, line descriptions, billed monthly with one fixed amount.
VENDOR_KINDS = {
    "freight": ("7110", ("Line haul {a}-{b}", "Rail freight {a}-{b}", "Trucking services"), False),
    "warehouse": ("7110", ("Warehouse services", "Pallet storage", "Loading and unloading"), False),
    "office": ("7210", ("Office supplies", "Printer maintenance", "Courier services"), False),
    "hosting": ("7210", ("Hosting subscription", "Backup storage subscription", "Software licences, monthly"), True),
    "rent": ("7210", ("Office rent", "Warehouse rent", "Parking rent"), True),
    "telecom": ("7210", ("Internet and telephony", "Mobile communication"), True),
    "security": ("7210", ("Security services", "Office cleaning"), True),
    "leasing": ("7110", ("Truck leasing", "Forklift rental"), True),
}
# Well-known fictional names only, so no record matches a real company. No digits in any name.
CUSTOMER_NAMES = (
    "Contoso Ltd", "Fabrikam Inc", "Adventure Works LLC", "Wingtip Toys GmbH", "Litware Inc", "Proseware LLC",
    "Fourth Coffee LLP", "Coho Winery Ltd", "Lucerne Publishing", "Trey Research Inc", "Woodgrove Trading LLP",
    "Blue Yonder Airlines", "Southridge Video LLC", "Relecloud Ltd", "VanArsdel Ltd", "Bellows Trading LLP",
    "Datum Corporation", "Lamna Foods LLP", "Fincher Architects", "Munson Pickles Ltd")
# (name, vendor_ref prefix, kinds the name fits). Prefixes are unique, two or three letters, never a lone S
# like the customer invoice numbers, so no reference text can name a bill of another vendor.
VENDOR_NAMES = (
    ("Tailspin Hosting LLP", "TSH", ("hosting",)), ("Litware Cloud Services", "LCS", ("hosting",)),
    ("Blue Yonder Telecom", "BYT", ("telecom",)), ("Alpine Security LLP", "ASL", ("security",)),
    ("Contoso Leasing", "CL", ("leasing",)), ("Lucerne Property LLP", "LP", ("rent",)),
    ("Proseware Facilities", "PWF", ("rent", "security")),
    ("Northwind Freight GmbH", "NWF", ("freight",)), ("Wide World Importers", "WWI", ("freight", "warehouse")),
    ("Consolidated Messenger LLP", "CM", ("freight", "office")), ("Coho Storage LLP", "CHS", ("warehouse",)),
    ("Woodgrove Office Supply", "WOS", ("office",)), ("Northwind Traders", "NWT", ("freight", "warehouse")),
    ("Tailwind Logistics", "TWL", ("freight",)), ("Fabrikam Rail Cargo", "FRC", ("freight",)),
    ("Humongous Haulage", "HH", ("freight",)), ("Margie Travel LLP", "MT", ("office",)),
    ("Graphic Design Institute", "GDI", ("office",)))
MONTHLY_NAMES = tuple(v for v in VENDOR_NAMES if VENDOR_KINDS[v[2][0]][2])
ONE_OFF_NAMES = tuple(v for v in VENDOR_NAMES if not VENDOR_KINDS[v[2][0]][2])
UNKNOWN_PAYERS = ("ALPINE SKI HOUSE", "CITY POWER AND LIGHT", "SCHOOL OF FINE ART", "TREY RESEARCH FUND")
UNKNOWN_TEXTS = ("OPLATA", "OPLATA PO DOGOVORU", "TRANSFER", "PREDOPLATA")
COUNTERPARTY_OFFSETS = ("+00:00", "+01:00", "+02:00", "+03:00", "+05:00", "+06:00", "+08:00")
# How a vendor prints its references: separator after the prefix, separator after the year (None: no
# year), digits of the number. dash NWF-2026-0418, plain NWF20260418, space WWI 07781, short CM-0412.
REF_STYLES = {"dash": ("-", "-", 4), "plain": ("", "", 4), "space": (" ", None, 5), "short": ("-", None, 4)}


class WorldError(RuntimeError):
    """The generator broke one of its own invariants INV-01…INV-16: a generator bug, never bad input."""


# ---------------------------------------------------------------- time, money and rates
def canonical_offset(offset):
    """'-00:00' and '+00:00' are one offset; write it the way isoformat() writes UTC."""
    return "+00:00" if offset == "-00:00" else offset


def tz_of(offset):
    sign = -1 if offset[0] == "-" else 1
    return dt.timezone(sign * dt.timedelta(hours=int(offset[1:3]), minutes=int(offset[4:6])))


def conv(doc_minor, rate):
    """base_minor = round_half_up(doc_minor x rate). Every currency here has 2 minor units."""
    return int(_DECIMAL.multiply(Decimal(doc_minor), Decimal(rate)).quantize(Decimal(1), rounding=ROUND_HALF_UP,
                                                                              context=_DECIMAL))


def norm_ref(text):
    """policy.vendor_ref_normalization: uppercase, keep A-Z and 0-9 only."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


@functools.lru_cache(maxsize=4096)
def ref_pattern(key):
    """A reference inside free text: the key's characters in order, any separators between them."""
    return re.compile(r"(?<![A-Z0-9])" + r"[^A-Z0-9]*".join(map(re.escape, key)) + r"(?![A-Z0-9])")


def month_bounds(year, month):
    first = dt.date(year, month, 1)
    nxt = dt.date(year + month // 12, month % 12 + 1, 1)
    return first, nxt - dt.timedelta(days=1)


class Rates:
    """Synthetic daily quotes, KZT per 1 unit, two decimals, business days only (policy.fx)."""

    def __init__(self, rng, first, last):
        self.first, self.last = first, last
        self.holidays = tuple(d for d in HOLIDAYS if first <= d <= last)
        self.quotes = {}
        level = {}
        for ccy in FOREIGN:
            start, step, _ = RATE_WALK[ccy]
            level[ccy] = start + rng.randint(-25 * step, 25 * step) // 5
        day = first
        while day <= last:
            if self.is_business(day):
                for ccy in FOREIGN:
                    _, step, floor = RATE_WALK[ccy]
                    level[ccy] = max(level[ccy] + rng.randint(-step, step), floor)
                    self.quotes[(day, ccy)] = f"{level[ccy] // 100}.{level[ccy] % 100:02d}"
            day += dt.timedelta(days=1)

    def is_business(self, day):
        return day.isoweekday() not in WEEKEND_DAYS and day not in self.holidays

    def rate_date(self, day):
        """The rate date of a document date: the date itself or the previous business day."""
        while not self.is_business(day):
            day -= dt.timedelta(days=1)
        if day < self.first:
            raise WorldError(f"no rate before {self.first}")
        return day

    def rate(self, currency, day):
        if currency == BASE:
            return "1"
        key = (self.rate_date(day), currency)
        if key not in self.quotes:
            raise WorldError(f"no {currency} rate for {day}")
        return self.quotes[key]


# ---------------------------------------------------------------- records
@dataclass
class Counterparty:
    key: str
    name: str
    role: str                     # customer or vendor
    offset: str                   # where the counterparty dates its own documents
    currency: str
    kind: str = ""                # vendor kind, VENDOR_KINDS
    prefix: str = ""              # vendor_ref prefix
    style: str = ""               # vendor_ref format
    next_number: int = 0


@dataclass
class Line:
    key: str
    description: str
    qty: int
    unit_price_minor: int

    @property
    def amount_minor(self):
        return self.qty * self.unit_price_minor


@dataclass
class Invoice:
    key: str
    direction: str                # AR: we bill a customer; AP: a vendor bills us
    cp: str
    issued_at: dt.datetime        # aware, in the issuer's offset
    entered_at: dt.datetime       # aware, company offset
    terms: int
    currency: str
    lines: list
    number: str = None            # AR invoice_number
    vendor_ref: str = None        # AP reference of the vendor, as our clerk keyed it
    ref_parts: tuple = None       # AP (prefix, year, number) of that reference
    po_ref: str = None
    service_from: str = None
    service_to: str = None
    expense_account: str = None   # AP only
    post_date: dt.date = None     # decision override, None means the policy rule
    rate: str = None              # decision override, None means the policy rule

    @property
    def total_minor(self):
        return sum(line.amount_minor for line in self.lines)


@dataclass
class Payment:
    key: str
    direction: str                # in: from a customer; out: to a vendor
    cp: str                       # None for an unidentified receipt
    raw_name: str
    timestamp: dt.datetime        # aware, UTC, as in the bank statement
    entered_at: dt.datetime
    currency: str
    amount_minor: int
    remittance: str
    alloc: list = field(default_factory=list)    # [(invoice key, doc amount)] in allocation order
    post_date: dt.date = None
    rate: str = None


@dataclass
class Row:
    key: str                      # "<document key>:<role>", stable between the clean and the final book
    je: str                       # the document key: one journal entry per document
    post_date: dt.date
    entered_at: dt.datetime
    account: str
    debit_minor: int
    credit_minor: int
    entity: str
    invoice: str
    payment: str
    doc_currency: str
    doc_amount_minor: int
    rate: str
    memo: str

    @property
    def base(self):
        return self.debit_minor or self.credit_minor


@dataclass
class World:
    period: str
    period_start: dt.date
    period_end: dt.date
    window_start: dt.date
    window_end: dt.date
    offset: str
    industry: str
    foreign_pct: int
    rates: Rates
    counterparties: dict          # key -> Counterparty
    invoices: dict                # key -> Invoice, in creation order
    payments: dict                # key -> Payment, in creation order
    twins: list = field(default_factory=list)   # {kind, mimics, records, feature}
    next_key: dict = field(default_factory=dict)
    anchors: dict = field(default_factory=dict)  # record key -> the error type it was built for
    off_days_left: int = 0                       # documents that may still fall on a weekend or holiday

    @property
    def tz(self):
        return tz_of(self.offset)

    def local_date(self, when):
        return when.astimezone(self.tz).date()

    def new_key(self, prefix):
        n = self.next_key.get(prefix, 0) + 1
        self.next_key[prefix] = n
        return f"{prefix}{n:04d}"

    def copy(self):
        return copy.deepcopy(self)

    def documents(self):
        return len(self.invoices) + len(self.payments)

    def event_date(self, doc):
        return self.local_date(doc.issued_at if isinstance(doc, Invoice) else doc.timestamp)

    def ref_key(self, invoice):
        """Group key text of an invoice: AR invoice_number, AP vendor_ref, normalized."""
        return norm_ref(invoice.number if invoice.direction == "AR" else invoice.vendor_ref)

    def invoice_refs(self, payment):
        """Invoice keys the remittance text names, sorted; None when it names none (SPEC §3.2)."""
        text = (payment.remittance or "").upper()
        found = [i.key for i in self.invoices.values() if self.ref_key(i) and ref_pattern(self.ref_key(i)).search(text)]
        return sorted(found) or None


# ---------------------------------------------------------------- posting engine
@dataclass
class Book:
    rows: list
    by_doc: dict                  # document key -> its rows
    invoice_rate: dict            # invoice key -> rate of its carrier row
    open_doc: dict                # invoice key -> open amount in document currency at the end
    open_base: dict               # invoice key -> open amount in base at the end

    def carrier(self, doc_key):
        rows = [r for r in self.by_doc[doc_key] if r.key.endswith(":carrier")]
        if len(rows) != 1:
            raise WorldError(f"{doc_key}: {len(rows)} carrier rows")
        return rows[0]

    def rows_of(self, doc_key, role_prefix):
        return [r for r in self.by_doc[doc_key] if r.key.split(":", 1)[1].startswith(role_prefix)]


def payment_order(world):
    return sorted(world.payments.values(), key=lambda p: (p.timestamp, p.key))


def post(world):
    """The ledger of the world: one balanced journal entry per document (policy rules of spec A)."""
    rows, by_doc, invoice_rate, open_doc, open_base = [], {}, {}, {}, {}

    def add(doc_key, role, post_date, entered_at, account, debit, credit, entity, invoice, payment,
            currency, amount, rate, memo):
        row = Row(f"{doc_key}:{role}", doc_key, post_date, entered_at, account, debit, credit, entity, invoice,
                  payment, currency, amount, rate, memo)
        rows.append(row)
        by_doc.setdefault(doc_key, []).append(row)
        return row

    for inv in world.invoices.values():
        day = inv.post_date or world.event_date(inv)
        rate = inv.rate or world.rates.rate(inv.currency, world.event_date(inv))
        total = inv.total_minor
        base = conv(total, rate)
        memo = inv.number if inv.direction == "AR" else inv.vendor_ref
        args = (inv.cp, inv.key, None, inv.currency, total, rate, memo)
        if inv.direction == "AR":
            add(inv.key, "carrier", day, inv.entered_at, RECEIVABLE, base, 0, *args)
            add(inv.key, "revenue", day, inv.entered_at, REVENUE, 0, base, *args)
        else:
            add(inv.key, "expense", day, inv.entered_at, inv.expense_account, base, 0, *args)
            add(inv.key, "carrier", day, inv.entered_at, PAYABLE, 0, base, *args)
        invoice_rate[inv.key] = rate
        open_doc[inv.key], open_base[inv.key] = total, base

    for pay in payment_order(world):
        day = pay.post_date or world.event_date(pay)
        rate = pay.rate or world.rates.rate(pay.currency, world.event_date(pay))
        bank = conv(pay.amount_minor, rate)
        common = (pay.cp, None, pay.key)
        if pay.direction == "in":
            add(pay.key, "carrier", day, pay.entered_at, BANK, bank, 0, *common, pay.currency, pay.amount_minor, rate,
                pay.remittance)
        cleared = 0
        for inv_key, amount in pay.alloc:
            if amount == open_doc[inv_key]:
                base = open_base[inv_key]          # the last allocation clears the base residual
            else:
                base = conv(amount, invoice_rate[inv_key])
            open_doc[inv_key] -= amount
            open_base[inv_key] -= base
            cleared += base
            debit, credit = (0, base) if pay.direction == "in" else (base, 0)
            add(pay.key, f"clear:{inv_key}", day, pay.entered_at, RECEIVABLE if pay.direction == "in" else PAYABLE,
                debit, credit, pay.cp, inv_key, pay.key, pay.currency, amount, invoice_rate[inv_key], None)
        rest = pay.amount_minor - sum(a for _, a in pay.alloc)
        if pay.direction == "in":
            held = conv(rest, rate) if rest else 0
            if rest:
                add(pay.key, "unapplied", day, pay.entered_at, UNAPPLIED, 0, held, *common, pay.currency, rest, rate,
                    "unapplied receipt")
            diff = bank - cleared - held
        else:
            add(pay.key, "carrier", day, pay.entered_at, BANK, 0, bank, *common, pay.currency, pay.amount_minor, rate,
                pay.remittance)
            diff = cleared - bank
        if diff > 0:
            add(pay.key, "fx", day, pay.entered_at, FX_GAIN, 0, diff, *common, BASE, diff, "1", "realized FX gain")
        elif diff < 0:
            add(pay.key, "fx", day, pay.entered_at, FX_LOSS, -diff, 0, *common, BASE, -diff, "1", "realized FX loss")
    return Book(rows, by_doc, invoice_rate, open_doc, open_base)


# ---------------------------------------------------------------- rule detector, INV-13…INV-16
def violations(world, book):
    """Breaks of the clean-book rules on a posted book: {(type, frozenset of internal record keys)}.

    Affected records follow SPEC §3.6: both bills, payment and bill, the carrier row for rate and period.
    """
    found = set()
    for doc in [*world.invoices.values(), *world.payments.values()]:
        row = book.carrier(doc.key)
        event = world.event_date(doc)
        if row.post_date != event:
            found.add(("period_cutoff", frozenset({row.key})))
        if row.rate != world.rates.rate(doc.currency, event):
            found.add(("fx_rate", frozenset({row.key})))
    for row in book.rows:
        if row.invoice and row.payment and row.rate != book.invoice_rate[row.invoice]:
            found.add(("fx_rate", frozenset({row.key})))
    groups = {}
    for inv in world.invoices.values():
        if inv.direction == "AP" and inv.cp:
            groups.setdefault((inv.cp, inv.currency, inv.total_minor, world.ref_key(inv)), []).append(inv.key)
    for keys in groups.values():
        if len(keys) > 1:
            found.add(("duplicate_invoice", frozenset(keys)))
    cleared = {}
    side = {"in": "AR", "out": "AP"}
    for pay in payment_order(world):
        mine = {}
        for inv_key, amount in pay.alloc:
            mine[inv_key] = mine.get(inv_key, 0) + amount
        refs = world.invoice_refs(pay) or []
        named = {(world.ref_key(world.invoices[k]), world.invoices[k].currency) for k in refs
                 if world.invoices[k].direction == side[pay.direction] and world.invoices[k].cp == pay.cp
                 and world.invoices[k].currency == pay.currency}
        if len(named) == 1:
            ref, currency = named.pop()
            group = sorted(i.key for i in world.invoices.values()
                           if i.cp == pay.cp and i.currency == currency and i.direction == side[pay.direction]
                           and world.ref_key(i) == ref)
            open_before = {k: world.invoices[k].total_minor - cleared.get(k, 0) for k in group}
            required = min(pay.amount_minor, max(sum(open_before.values()), 0))
            allocated = sum(mine.get(k, 0) for k in group)
            if allocated != required:
                if len(group) == 1:
                    blamed = group
                elif allocated < required:
                    blamed = [k for k in group if open_before[k] - mine.get(k, 0) > 0]
                else:
                    blamed = [k for k in group if mine.get(k, 0) > 0]
                found.add(("payment_allocation", frozenset([pay.key, *blamed])))
        for inv_key, amount in mine.items():
            cleared[inv_key] = cleared.get(inv_key, 0) + amount
    return found


# ---------------------------------------------------------------- book invariants, INV-01…INV-12
def check_book(world, book, expected_violations, moved_docs=()):
    """Raise WorldError unless INV-01…INV-12 hold and the rule breaks are exactly `expected_violations`."""
    def fail(code, detail):
        raise WorldError(f"{code}: {detail}")

    per_je = {}
    for row in book.rows:
        if (row.debit_minor > 0) == (row.credit_minor > 0) or min(row.debit_minor, row.credit_minor) < 0:
            fail("INV-01", f"{row.key}: debit {row.debit_minor} credit {row.credit_minor}")
        d, c = per_je.get(row.je, (0, 0))
        per_je[row.je] = (d + row.debit_minor, c + row.credit_minor)
    for je, (d, c) in per_je.items():
        if d != c:
            fail("INV-01", f"entry {je}: debit {d} credit {c}")
    if sum(r.debit_minor for r in book.rows) != sum(r.credit_minor for r in book.rows):
        fail("INV-02", "trial balance is not zero")

    before = after = 0
    for row in book.rows:
        if not world.window_start <= row.post_date <= world.window_end:
            fail("INV-03", f"{row.key} posted {row.post_date} outside the window")
        if row.je in moved_docs:
            continue
        before += row.post_date < world.period_start
        after += row.post_date > world.period_end
    if min(before, after) < 10:
        fail("INV-03", f"{before} rows before and {after} after the period")

    clearing_rows = {}
    for row in book.rows:
        if row.invoice and row.payment:
            clearing_rows.setdefault(row.invoice, []).append(row)
    slack = {rows[-1].key: len(rows) - 1 for rows in clearing_rows.values()}
    for row in book.rows:
        if abs(row.base - conv(row.doc_amount_minor, row.rate)) > slack.get(row.key, 0):
            fail("INV-04", f"{row.key}: base {row.base}, {row.doc_amount_minor} x {row.rate}")
        if row.doc_currency == BASE and row.rate != "1":
            fail("INV-04", f"{row.key}: base currency at rate {row.rate}")
    for doc_key, rows in book.by_doc.items():
        fx_rows = [r for r in rows if r.account in (FX_GAIN, FX_LOSS)]
        if fx_rows and (doc_key not in world.payments or world.payments[doc_key].currency == BASE or len(fx_rows) > 1):
            fail("INV-05", f"{doc_key}: FX rows {[r.key for r in fx_rows]}")
        if len({r.post_date for r in rows}) != 1:
            fail("INV-13", f"{doc_key}: rows on several dates")

    for inv in world.invoices.values():
        if not inv.lines or any(line.qty <= 0 or line.unit_price_minor <= 0 for line in inv.lines):
            fail("INV-06", f"{inv.key}: bad lines")
        due = world.event_date(inv) + dt.timedelta(days=inv.terms)
        if inv.terms not in (14, 30, 45) or due <= world.event_date(inv):
            fail("INV-07", f"{inv.key}: terms {inv.terms}")
        if inv.cp not in world.counterparties:
            fail("INV-08", f"{inv.key}: counterparty {inv.cp}")
    per_invoice = {}
    for pay in world.payments.values():
        if pay.cp is not None and pay.cp not in world.counterparties:
            fail("INV-08", f"{pay.key}: counterparty {pay.cp}")
        allocated = 0
        for inv_key, amount in pay.alloc:
            inv = world.invoices.get(inv_key)
            if inv is None:
                fail("INV-08", f"{pay.key}: allocation to unknown {inv_key}")
            if amount <= 0 or inv.currency != pay.currency or inv.cp != pay.cp:
                fail("INV-09", f"{pay.key}: allocation {amount} {pay.currency} to {inv_key}")
            if inv.direction != {"in": "AR", "out": "AP"}[pay.direction]:
                fail("INV-09", f"{pay.key}: {pay.direction} payment clears {inv.direction} {inv_key}")
            if world.event_date(pay) < world.event_date(inv):
                fail("INV-12", f"{pay.key} paid before {inv_key} was issued")
            per_invoice[inv_key] = per_invoice.get(inv_key, 0) + amount
            allocated += amount
        if allocated > pay.amount_minor or (pay.direction == "out" and allocated != pay.amount_minor):
            fail("INV-10", f"{pay.key}: allocated {allocated} of {pay.amount_minor}")
        held = sum(r.doc_amount_minor for r in book.rows_of(pay.key, "unapplied"))
        if pay.direction == "in" and allocated + held != pay.amount_minor:
            fail("INV-10", f"{pay.key}: allocated {allocated} + held {held} != {pay.amount_minor}")
    for inv_key, amount in per_invoice.items():
        if amount > world.invoices[inv_key].total_minor:
            fail("INV-09", f"{inv_key}: allocated {amount} above the total")

    balances = {RECEIVABLE: 0, PAYABLE: 0}
    for row in book.rows:
        if row.account in balances:
            balances[row.account] += row.debit_minor - row.credit_minor
    open_ar = sum(b for k, b in book.open_base.items() if world.invoices[k].direction == "AR")
    open_ap = sum(b for k, b in book.open_base.items() if world.invoices[k].direction == "AP")
    if balances[RECEIVABLE] != open_ar or balances[PAYABLE] != -open_ap:
        fail("INV-11", f"balances {balances}, open items AR {open_ar} AP {open_ap}")

    found = violations(world, book)
    if found != set(expected_violations):
        extra = sorted((t, sorted(k)) for t, k in found - set(expected_violations))
        missing = sorted((t, sorted(k)) for t, k in set(expected_violations) - found)
        fail("INV-18", f"rule breaks differ from the errors: extra {extra}, missing {missing}")


# ---------------------------------------------------------------- the clean history
def local_time(rng, world, day, near_midnight_share=0.05):
    """A company-local moment on `day`: business hours, about 5% within half an hour of midnight."""
    if rng.random() < near_midnight_share:
        minute = rng.choice([rng.randint(0, 29), rng.randint(23 * 60 + 30, 23 * 60 + 59)])
    else:
        minute = rng.randint(9 * 60, 18 * 60)
    return dt.datetime.combine(day, dt.time(minute // 60, minute % 60), world.tz)


def entered_time(rng, world, event_local):
    """When a clerk keyed the document: 0 to 10 days later, never before the event. One distribution for
    clean and erroneous records (domain spec A, «Задержка внесения в учёт»)."""
    lag = rng.choices((0, rng.randint(1, 3), rng.randint(4, 10)), weights=(50, 30, 20))[0]
    day = event_local.date() + dt.timedelta(days=lag)
    when = dt.datetime.combine(day, dt.time(rng.randint(9, 18), rng.randint(0, 59)), world.tz)
    if when <= event_local:
        when = event_local + dt.timedelta(minutes=rng.randint(10, 120))
    return when


def _price(rng, currency, major):
    """A unit price in minor units: often with cents in foreign currency, sometimes with tiyn in KZT, so
    that conversions round and the residual rule of the last allocation matters."""
    if rng.random() < (CENTS_SHARE_FOREIGN if currency != BASE else CENTS_SHARE_BASE):
        return max(major, 1) * 100 + rng.randint(1, 99)
    step = 1000 if major >= 100_000 else 10 if major >= 1000 else 1
    return max(major // step * step, 1) * 100


def _amount_lines(rng, world, currency, descriptions, fixed_total=None):
    low, high = TOTAL_RANGE[currency]
    count = 1 if fixed_total else rng.choices((1, 2, 3), weights=(60, 30, 10))[0]
    lines = []
    for _ in range(count):
        if fixed_total:
            qty, unit = 1, fixed_total
        else:
            qty = rng.choice((1, 1, 1, 2, 3, 4, 5, 10))
            unit = _price(rng, currency, rng.randint(low, high) // (qty * count))
        text = rng.choice(descriptions).format(a=rng.choice(CITIES), b=rng.choice(CITIES))
        lines.append(Line(world.new_key("N"), text, qty, unit))
    return lines


def c1_boundary_ok(year, month):
    """True when the month's last day and the next day are business days by weekday and the holiday list."""
    _, last = month_bounds(year, month)
    return all(d.isoweekday() not in WEEKEND_DAYS and d not in HOLIDAYS for d in (last, last + dt.timedelta(days=1)))


def boundary_moment(rng, world):
    """A company-local moment within half an hour of the midnight that ends the period, whose UTC date lies
    on the other side of the period end: just after midnight for a positive offset (UTC is still on the
    period's last day), just before it for a negative one (UTC is already in the next month)."""
    minutes = int(world.tz.utcoffset(None).total_seconds() // 60)
    day1 = world.period_end + dt.timedelta(days=1)
    if minutes > 0:
        minute = rng.randint(0, min(29, minutes - 1))
        return dt.datetime.combine(day1, dt.time(0, minute), world.tz)
    minute = rng.randint(max(30, 60 + minutes), 59)
    return dt.datetime.combine(world.period_end, dt.time(23, minute), world.tz)


def _c1_receipt(rng, world, customers, foreign_customers):
    """C1 anchor: a foreign invoice of the period and its full payment at the period-end midnight, in a
    currency whose rates differ on the two boundary days. Clean: posted on the local date at its rate."""
    end, day1 = world.period_end, world.period_end + dt.timedelta(days=1)
    currencies = [c for c in FOREIGN if world.rates.rate(c, end) != world.rates.rate(c, day1)]
    if not currencies:
        raise WorldError("C1: no currency with two rates on the period boundary")
    payers = [c for c in foreign_customers if c.currency in currencies]
    if payers:
        cp = rng.choice(payers)
    else:
        taken = {c.name for c in world.counterparties.values()}
        cp = Counterparty(world.new_key("C"), rng.choice([n for n in CUSTOMER_NAMES if n not in taken]), "customer",
                          rng.choice(COUNTERPARTY_OFFSETS), rng.choice(currencies))
        world.counterparties[cp.key] = cp
        customers.append(cp)
        foreign_customers.append(cp)
    first = max(world.window_start, world.period_start)
    inv = _add_invoice(rng, world, "AR", cp,
                       local_time(rng, world, business_day(rng, world, first, end - dt.timedelta(days=3), False)))
    moment = boundary_moment(rng, world)
    pay = _add_payment(rng, world, "in", cp, moment, cp.currency, inv.total_minor,
                       _ar_remittance(rng, [inv.number], cp), [(inv.key, inv.total_minor)])
    return inv, pay


def entry_rate_differs(world, inv, tolerance=1):
    """True when the rate of the invoice's entry date gives a base amount more than two tolerances away
    from the rate of its document date (INV-20): the precondition of the rate_of_entry_date error."""
    if inv.currency == BASE:
        return False
    right = world.rates.rate(inv.currency, world.event_date(inv))
    wrong = world.rates.rate(inv.currency, world.local_date(inv.entered_at))
    return abs(conv(inv.total_minor, wrong) - conv(inv.total_minor, right)) > 2 * tolerance


def _keep_off_day(rng, world):
    if world.off_days_left > 0 and rng.random() < OFF_DAY_KEEP:
        world.off_days_left -= 1
        return True
    return False


def business_day(rng, world, first, last, off_days=True):
    """A day in [first, last]: a business day, or while the book's quota lasts sometimes an off day."""
    for _ in range(50):
        day = first + dt.timedelta(days=rng.randint(0, (last - first).days))
        if world.rates.is_business(day) or (off_days and _keep_off_day(rng, world)):
            return day
    return day


def later_day(rng, world, day, most):
    """A business day 0..most days after `day` and inside the window, else `day` itself."""
    later = bank_day(rng, world, day + dt.timedelta(days=rng.randint(0, most)))
    return later if later <= world.window_end else day


def bank_day(rng, world, day):
    """A value date or a scheduled bill on an off day goes to the next business day, sometimes it stays."""
    if world.rates.is_business(day) or _keep_off_day(rng, world):
        return day
    while not world.rates.is_business(day):
        day += dt.timedelta(days=1)
    return day


def _part(rng, total, low_pct, high_pct, currency):
    """A share of `total` strictly between 0 and total: whole tenge in KZT, any minor amount otherwise."""
    part = total * rng.randint(low_pct, high_pct) // 100
    if currency == BASE:
        part = part // 100 * 100
    return min(max(part, 1), total - 1)


# ---------------------------------------------------------------- vendor references
def ref_text(prefix, year, number, after_prefix, after_year, width):
    tail = f"{number:0{width}d}"
    return prefix + after_prefix + (f"{year}{after_year}{tail}" if after_year is not None else tail)


def ref_variants(parts, style):
    """Every way a clerk may key one vendor reference: the same characters, other separators."""
    prefix, year, number = parts
    _, after_year, width = REF_STYLES[style]
    seps = ("-", " ", "")
    return sorted({ref_text(prefix, year, number, a, b if after_year is not None else None, width)
                   for a in seps for b in seps})


def ref_shape(text):
    """Letters as A, digit runs as 9s, separators kept: 'NWF-2026-0418' -> 'A-9999-9999'."""
    return re.sub(r"[0-9]+", lambda m: "9" * len(m.group()), re.sub(r"[A-Za-z]+", "A", text))


def _canonical_ref(cp, parts):
    after_prefix, after_year, width = REF_STYLES[cp.style]
    return ref_text(*parts, after_prefix, after_year, width)


# ---------------------------------------------------------------- documents and payments
def _add_invoice(rng, world, direction, cp, local_dt, numbers=None, fixed_total=None, service=None,
                 description=None):
    """One invoice dated at the company-local moment `local_dt`. Numbers and vendor references are
    placeholders until _number_documents numbers everything by issue time."""
    terms = rng.choice((14, 30, 45))
    if direction == "AR":
        lines = _amount_lines(rng, world, cp.currency, INDUSTRIES[world.industry], fixed_total)
        inv = Invoice(world.new_key("I"), "AR", cp.key, local_dt, entered_time(rng, world, local_dt), terms,
                      cp.currency, lines)
        inv.number = f"@{inv.key}@"
    else:
        account, descriptions, _ = VENDOR_KINDS[cp.kind]
        lines = _amount_lines(rng, world, cp.currency, (description,) if description else descriptions, fixed_total)
        inv = Invoice(world.new_key("I"), "AP", cp.key, local_dt.astimezone(tz_of(cp.offset)),
                      entered_time(rng, world, local_dt), terms, cp.currency, lines, expense_account=account)
        inv.vendor_ref = f"@{inv.key}@"
        if numbers is not None and rng.random() < 0.5:
            inv.po_ref = "@PO@"                  # numbered by issue time in _number_documents
    inv.service_from = inv.service_to = local_dt.date().isoformat()
    if service:
        inv.service_from, inv.service_to = service
    world.invoices[inv.key] = inv
    return inv


def _add_payment(rng, world, direction, cp, local_dt, currency, amount, remittance, alloc, raw_name=None):
    timestamp = local_dt.astimezone(dt.timezone.utc)
    name = raw_name or (cp.name.upper() if rng.random() < 0.5 else cp.name)
    pay = Payment(world.new_key("P"), direction, cp.key if cp else None, name, timestamp,
                  entered_time(rng, world, local_dt), currency, amount, remittance, list(alloc))
    world.payments[pay.key] = pay
    return pay


def _ar_remittance(rng, numbers, cp):
    """Remittance text of a customer payment citing invoice number placeholders."""
    cited = ", ".join(numbers)
    return rng.choice((f"INV {cited} {cp.name.upper()}", cited, f"Payment {cited}", f"Oplata po schetu {cited}"))


class _Numbers:
    """First AR invoice number and the company's purchase orders."""

    def __init__(self, rng):
        self.ar = rng.randint(100, 900)
        self.po = rng.randint(10000, 60000)


def _twin(world, kind, mimics, records, feature):
    world.twins.append({"kind": kind, "mimics": mimics, "records": list(records), "feature": feature})


def _pay_customer(rng, world, inv, budget, behaviour=None):
    """Payments of one customer invoice that fall inside the window, at most `budget` of them.
    Behaviours follow the domain spec A world table, plus a repeated full payment after a partial one."""
    cp = world.counterparties[inv.cp]
    issued = world.event_date(inv)
    total, ccy = inv.total_minor, inv.currency
    behaviour = behaviour or rng.choices(BEHAVIOURS, weights=BEHAVIOUR_WEIGHTS)[0]
    if behaviour == "late":
        day = issued + dt.timedelta(days=inv.terms + rng.randint(1, 20))
    else:
        day = issued + dt.timedelta(days=rng.randint(1, inv.terms))
    if day > world.window_end or budget <= 0:
        return []
    day = bank_day(rng, world, day)
    if day > world.window_end:
        return []
    local = local_time(rng, world, day)
    cites = _ar_remittance(rng, [inv.number], cp)
    fx_note = ", each clearing row at the invoice rate" if ccy != BASE else ""
    if behaviour in ("on_time", "late"):
        return [_add_payment(rng, world, "in", cp, local, ccy, total, cites, [(inv.key, total)])]
    if behaviour == "overpay":
        extra = _part(rng, total, 1, 30, ccy)
        pay = _add_payment(rng, world, "in", cp, local, ccy, total + extra, cites, [(inv.key, total)])
        _twin(world, "overpayment", "payment_allocation", [inv.key, pay.key],
              "payment above the open amount, the excess lies on 3510")
        return [pay]
    if behaviour == "partial":
        part = _part(rng, total, 30, 80, ccy)
        pay = _add_payment(rng, world, "in", cp, local, ccy, part, f"Partial payment {inv.number}", [(inv.key, part)])
        _twin(world, "real_partial_payment", "payment_allocation", [inv.key, pay.key],
              "bank amount below the invoice, the whole payment allocated")
        return [pay]
    first = _part(rng, total, 30, 70, ccy)
    text = f"{inv.number} part 1" if behaviour == "split" else f"Partial payment {inv.number}"
    made = [_add_payment(rng, world, "in", cp, local, ccy, first, text, [(inv.key, first)])]
    second_day = day + dt.timedelta(days=rng.randint(3, 15))
    second_day = bank_day(rng, world, second_day) if second_day <= world.window_end else second_day
    if second_day > world.window_end or budget < 2:
        _twin(world, "real_partial_payment", "payment_allocation", [inv.key, made[0].key],
              "bank amount below the invoice, the whole payment allocated")
        return made
    second_local = local_time(rng, world, second_day)
    if behaviour == "split":
        made.append(_add_payment(rng, world, "in", cp, second_local, ccy, total - first, f"{inv.number} part 2",
                                 [(inv.key, total - first)]))
        _twin(world, "split_payment", "payment_allocation", [inv.key, *(p.key for p in made)],
              "several payments, each allocated in full" + fx_note)
    else:
        made.append(_add_payment(rng, world, "in", cp, second_local, ccy, total, cites, [(inv.key, total - first)]))
        _twin(world, "repeated_full_payment", "payment_allocation", [inv.key, *(p.key for p in made)],
              "the second payment is the full total although part was paid: allocated up to the open amount, "
              "the excess lies on 3510")
    return made


def _repeated_full(rng, world, inv, cp):
    """A partial payment, then the full total again a few days later, both inside the window."""
    first_day = bank_day(rng, world, world.event_date(inv) + dt.timedelta(days=rng.randint(1, 7)))
    second_day = bank_day(rng, world, first_day + dt.timedelta(days=rng.randint(3, 10)))
    total, ccy = inv.total_minor, inv.currency
    first = _part(rng, total, 30, 70, ccy)
    made = [_add_payment(rng, world, "in", cp, local_time(rng, world, first_day), ccy, first,
                         f"Partial payment {inv.number}", [(inv.key, first)]),
            _add_payment(rng, world, "in", cp, local_time(rng, world, second_day), ccy, total,
                         _ar_remittance(rng, [inv.number], cp), [(inv.key, total - first)])]
    _twin(world, "repeated_full_payment", "payment_allocation", [inv.key, *(p.key for p in made)],
          "the second payment is the full total although part was paid: allocated up to the open amount, "
          "the excess lies on 3510")


def _pay_vendor(rng, world, bills, budget):
    """We pay vendor bills in full, on time or up to ten days late, if that falls in the window.
    Two bills of one vendor may go out in one payment that cites both."""
    cp = world.counterparties[bills[0].cp]
    due = min(world.event_date(b) + dt.timedelta(days=b.terms) for b in bills)
    day = due + dt.timedelta(days=rng.choice((-rng.randint(0, 5), rng.randint(1, 10))))
    day = max([day] + [world.event_date(b) + dt.timedelta(days=1) for b in bills])
    if day > world.window_end or budget <= 0:
        return []
    day = bank_day(rng, world, day)
    if day > world.window_end:
        return []
    refs = ", ".join(b.vendor_ref for b in bills)
    word = "invoices" if len(bills) > 1 else "invoice"
    text = rng.choice((f"Payment for {word} {refs}", refs, f"Inv {refs}"))
    amount = sum(b.total_minor for b in bills)
    pay = _add_payment(rng, world, "out", cp, local_time(rng, world, day), bills[0].currency, amount, text,
                       [(b.key, b.total_minor) for b in bills])
    if len(bills) > 1:
        _twin(world, "combined_payment", "duplicate_invoice", [*(b.key for b in bills), pay.key],
              "one payment cites two different bills of the vendor and settles both")
    return [pay]


def _combined_receipt(rng, world, invoices, budget):
    """A customer pays two of its invoices with one payment that cites both."""
    cp = world.counterparties[invoices[0].cp]
    last = max(world.event_date(i) for i in invoices)
    day = last + dt.timedelta(days=rng.randint(1, min(i.terms for i in invoices)))
    if day > world.window_end or budget <= 0:
        return []
    day = bank_day(rng, world, day)
    if day > world.window_end:
        return []
    pay = _add_payment(rng, world, "in", cp, local_time(rng, world, day), invoices[0].currency,
                       sum(i.total_minor for i in invoices), _ar_remittance(rng, [i.number for i in invoices], cp),
                       [(i.key, i.total_minor) for i in invoices])
    _twin(world, "combined_payment", "payment_allocation", [*(i.key for i in invoices), pay.key],
          "one payment cites two invoices of the customer and settles both")
    return [pay]


def build_world(rng, config, reserved_documents=0, anchors=None):
    """The clean company history for a normalized config.

    `reserved_documents` are left for the injectors (a duplicate adds a bill). `anchors` asks for
    records the errors need: they are ordinary clean records, created first so that every requested
    error has a place. full_receipts: customer invoices paid in full; vendor_bills: one-off bills;
    foreign_invoices: foreign-currency invoices whose entry date has another rate; period_bills:
    base-currency bills of the next month's first days; period_moves: extra neighbour invoices per side.
    Whether a config can hold its errors depends on the config only, never on the seed (SPEC §4.4).
    """
    anchors = anchors or {}
    low, high = config["documents"]
    neighbours = NEIGHBOUR_INVOICES + anchors.get("period_moves", 0)
    lookalikes = (config["twins"][1] > 0) + (config["foreign_pct"][1] > 0)
    planned = (2 * anchors.get("full_receipts", 0) + anchors.get("vendor_bills", 0)
               + anchors.get("foreign_invoices", 0) + anchors.get("period_bills", 0) + 2 * neighbours + lookalikes
               + 2 * anchors.get("c1_receipts", 0) + 2 * anchors.get("c2_bills", 0))
    if high < planned + reserved_documents:
        raise ValueError(f"config: documents {low}-{high} cannot hold the requested errors, "
                         f"they need at least {planned + reserved_documents}")
    if anchors.get("c1_receipts"):
        # C1 needs a period whose last day and the next day are both business days, so that the UTC date
        # and the company-local date have two different rates (domain spec A, C1).
        months = [m for m in range(1, 13) if c1_boundary_ok(2026, m)]
        period = config["period"] or f"2026-{rng.choice(months):02d}"
        if int(period[5:]) not in months:
            raise ValueError(f"config: C1 needs a period whose last day and the next day are business days, "
                             f"{period} has none")
    else:
        period = config["period"] or f"2026-{rng.randint(1, 12):02d}"
    year, month = int(period[:4]), int(period[5:])
    period_start, period_end = month_bounds(year, month)
    window_start = period_start - dt.timedelta(days=WINDOW_DAYS)
    window_end = period_end + dt.timedelta(days=WINDOW_DAYS)
    offsets = [canonical_offset(o) for o in config["company_offsets"]]
    if anchors.get("c1_receipts"):
        offsets = [o for o in offsets if o != "+00:00"]   # UTC and local date must be able to differ
        if not offsets:
            raise ValueError("config: C1 needs a company offset other than +00:00")
    offset = rng.choice(offsets)
    industry = rng.choice(sorted(INDUSTRIES))
    foreign_pct = rng.randint(*config["foreign_pct"])
    rates = Rates(rng, window_start - dt.timedelta(days=RATE_LEAD_DAYS), window_end + dt.timedelta(days=RATE_TAIL_DAYS))
    world = World(period, period_start, period_end, window_start, window_end, offset, industry, foreign_pct, rates,
                  {}, {}, {})
    target = max(rng.randint(low, high), planned + reserved_documents) - reserved_documents
    world.off_days_left = (target + reserved_documents) * rng.randint(*OFF_DAY_PERMILLE) // 1000
    subscription_pct = rng.randint(*SUBSCRIPTION_PCT)

    def currency():
        return rng.choice(FOREIGN) if rng.randint(1, 100) <= foreign_pct else BASE

    for name in rng.sample(CUSTOMER_NAMES, rng.randint(6, 12)):
        cp = Counterparty(world.new_key("C"), name, "customer", rng.choice(COUNTERPARTY_OFFSETS), currency())
        world.counterparties[cp.key] = cp
    # Monthly contracts: about subscription_pct of the vendor bills repeat one amount every month
    # (domain spec A world table, «Подписки»). A contract bills one or two times in the window.
    # Customer receipts the errors need are not vendor bills, so they do not count for the share.
    trade = target - 2 * anchors.get("full_receipts", 0)
    contracts = max(1, round(subscription_pct * AP_SHARE_OF_DOCUMENTS * trade / 100 / BILLS_PER_CONTRACT))
    monthly = min(len(MONTHLY_NAMES), max(2, -(-contracts // MAX_CONTRACTS_PER_VENDOR), rng.randint(2, 4)))
    contracts = min(contracts, monthly * MAX_CONTRACTS_PER_VENDOR)
    names = rng.sample(MONTHLY_NAMES, monthly) + rng.sample(ONE_OFF_NAMES, rng.randint(4, 7))
    for name, prefix, fits in names:
        kind = rng.choice(fits)
        cp = Counterparty(world.new_key("C"), name, "vendor", rng.choice(COUNTERPARTY_OFFSETS), currency(), kind,
                          prefix, rng.choice(sorted(REF_STYLES)), rng.randint(10, 900))
        world.counterparties[cp.key] = cp
    customers = [c for c in world.counterparties.values() if c.role == "customer"]
    suppliers = [c for c in world.counterparties.values() if c.role == "vendor"]
    monthly_vendors = [c for c in suppliers if VENDOR_KINDS[c.kind][2]]
    one_off = [c for c in suppliers if not VENDOR_KINDS[c.kind][2]]
    numbers = _Numbers(rng)
    # Conditions the errors and twins need that the drawn shares may lack: an occasional foreign
    # customer, an occasional base-currency vendor. They trade like the others, so they do not stand
    # out as holders of anchors only.
    taken = {c.name for c in world.counterparties.values()}
    foreign_customers = [c for c in customers if c.currency != BASE]
    if not foreign_customers and config["foreign_pct"][1] > 0 and (anchors.get("foreign_invoices") or config["twins"][1]):
        cp = Counterparty(world.new_key("C"), rng.choice([n for n in CUSTOMER_NAMES if n not in taken]), "customer",
                          rng.choice(COUNTERPARTY_OFFSETS), rng.choice(FOREIGN))
        world.counterparties[cp.key] = cp
        foreign_customers = [cp]
        customers.append(cp)
    base_vendors = [c for c in one_off if c.currency == BASE]
    if not base_vendors and anchors.get("period_bills"):
        name, prefix, fits = rng.choice([v for v in ONE_OFF_NAMES if v[0] not in taken])
        cp = Counterparty(world.new_key("C"), name, "vendor", rng.choice(COUNTERPARTY_OFFSETS), BASE, rng.choice(fits),
                          prefix, rng.choice(sorted(REF_STYLES)), rng.randint(10, 900))
        world.counterparties[cp.key] = cp
        base_vendors = [cp]
        one_off.append(cp)

    # The foreign weekend look-alike planted below takes its place in the off-day quota up front.
    if foreign_customers or any(c.currency != BASE for c in one_off):
        world.off_days_left = max(world.off_days_left - 1, 0)

    def room():
        return target - world.documents()

    for n in range(contracts):
        cp = monthly_vendors[n % len(monthly_vendors)]
        description = rng.choice(VENDOR_KINDS[cp.kind][1])
        low_total, high_total = TOTAL_RANGE[cp.currency]
        fixed = _price(rng, cp.currency, rng.randint(low_total, high_total) // 4)
        day_of_month = rng.randint(1, 28)
        bills = []
        y, m = window_start.year, window_start.month
        while dt.date(y, m, 1) <= window_end:
            day = dt.date(y, m, day_of_month)
            day = bank_day(rng, world, day) if window_start <= day <= window_end and room() > planned else day
            if window_start <= day <= window_end and room() > planned:
                first, last = month_bounds(y, m)
                inv = _add_invoice(rng, world, "AP", cp, local_time(rng, world, day, 0), numbers, fixed,
                                   (first.isoformat(), last.isoformat()), description)
                bills.append(inv.key)
                _pay_vendor(rng, world, [inv], room() - planned)
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        if len(bills) > 1:
            _twin(world, "monthly_subscription", "duplicate_invoice", bills,
                  "same vendor and amount, other vendor_ref and service month")

    def window_day(first=None, last=None):
        return business_day(rng, world, first or window_start, last or window_end)

    def new_ar(cp, day):
        return _add_invoice(rng, world, "AR", cp, local_time(rng, world, day), numbers)

    def new_ap(cp, day):
        return _add_invoice(rng, world, "AP", cp, local_time(rng, world, day), numbers)

    # Records the requested errors need, then real rows of both neighbouring periods (INV-03). Each
    # anchor is reserved for its error type, so another injector cannot use it up (seed-independent).
    for _ in range(anchors.get("full_receipts", 0)):
        cp = rng.choice(customers)
        inv = new_ar(cp, window_day(last=window_end - dt.timedelta(days=15)))
        pays = _pay_customer(rng, world, inv, 1, "on_time")
        if not pays:
            day = bank_day(rng, world, world.event_date(inv) + dt.timedelta(days=rng.randint(1, 10)))
            pays = [_add_payment(rng, world, "in", cp, local_time(rng, world, day), inv.currency, inv.total_minor,
                                 _ar_remittance(rng, [inv.number], cp), [(inv.key, inv.total_minor)])]
        world.anchors.update({inv.key: "payment_allocation", pays[0].key: "payment_allocation"})
    for _ in range(anchors.get("vendor_bills", 0)):
        world.anchors[new_ap(rng.choice(one_off), window_day()).key] = "duplicate_invoice"
    for _ in range(anchors.get("foreign_invoices", 0)):
        inv = new_ar(rng.choice(foreign_customers), business_day(rng, world, window_start, window_end, False))
        for _ in range(500):                     # redraw until a business day whose entry date has another rate
            if rates.is_business(world.event_date(inv)) and entry_rate_differs(world, inv):
                break
            local = local_time(rng, world, business_day(rng, world, window_start, window_end, False))
            inv.issued_at, inv.entered_at = local, entered_time(rng, world, local)
            inv.service_from = inv.service_to = local.date().isoformat()
        world.anchors[inv.key] = "fx_rate"
    for _ in range(anchors.get("period_bills", 0)):
        day = window_day(period_end + dt.timedelta(days=1), period_end + dt.timedelta(days=3))
        inv = _add_invoice(rng, world, "AP", rng.choice(base_vendors), local_time(rng, world, day, 0))
        world.anchors[inv.key] = "period_cutoff"
    for _ in range(anchors.get("c1_receipts", 0)):
        inv, pay = _c1_receipt(rng, world, customers, foreign_customers)
        world.anchors.update({inv.key: "C1", pay.key: "C1"})
    for _ in range(anchors.get("c2_bills", 0)):
        bill = new_ap(rng.choice(one_off), window_day(last=window_end - dt.timedelta(days=12)))
        day = bank_day(rng, world, world.event_date(bill) + dt.timedelta(days=rng.randint(3, 10)))
        pay = _add_payment(rng, world, "out", world.counterparties[bill.cp], local_time(rng, world, day), bill.currency,
                           bill.total_minor, f"Payment for invoice {bill.vendor_ref}", [(bill.key, bill.total_minor)])
        world.anchors.update({bill.key: "C2", pay.key: "C2"})
    for first, last in ((window_start, period_start - dt.timedelta(days=1)), (period_end + dt.timedelta(days=1), window_end)):
        for _ in range(neighbours):
            day = window_day(first, last)
            if rng.random() < 0.55:
                new_ar(rng.choice(customers), day)
            else:
                new_ap(rng.choice(one_off), day)

    # Look-alikes of the rate and period errors (docs/tasks/B_stage2.md item 4), in clean and erroneous
    # books alike and counted in `planned`, so small books hold them too. A foreign invoice dated on a
    # non-business day takes the previous business day's rate: every book with foreign documents has one
    # (PM 21.11 03.15), whatever config.twins says. An invoice of the period's last days entered in the
    # next month is posted on its own date.
    days = [period_start + dt.timedelta(days=n) for n in range((period_end - period_start).days + 1)]
    off_days = [d for d in days if not rates.is_business(d)]
    foreign_parties = foreign_customers or [c for c in one_off if c.currency != BASE]
    if foreign_parties and off_days:
        cp, day = rng.choice(foreign_parties), rng.choice(off_days)
        inv = new_ar(cp, day) if cp.role == "customer" else new_ap(cp, day)
        _twin(world, "weekend_document", "fx_rate", [inv.key],
              "dated on a non-business day, rate of the previous business day by the policy")
    if config["twins"][1]:
        inv = new_ar(rng.choice(customers), window_day(period_end - dt.timedelta(days=2), period_end))
        event = inv.issued_at.astimezone(world.tz)
        for _ in range(200):                 # the clean entry delay, drawn until it lands in the next month
            if world.local_date(inv.entered_at) > period_end:
                break
            inv.entered_at = entered_time(rng, world, event)
        else:
            inv.entered_at = dt.datetime.combine(period_end + dt.timedelta(days=1), dt.time(10, 0), world.tz)
        _twin(world, "late_entry", "period_cutoff", [inv.key],
              "entered after the period, posted on the document date")

    unknown = rng.randint(0, 2)
    # A full-total payment after a partial one: legitimately part on 3510 although the amount equals the
    # invoice total. Planted in every book with room, so that rule alone does not single out an error.
    for _ in range(rng.randint(*REPEATED_FULL_TWINS)):
        if room() - unknown >= 3:
            cp = rng.choice(customers)
            inv = new_ar(cp, window_day(last=window_end - dt.timedelta(days=20)))
            _repeated_full(rng, world, inv, cp)

    # Ordinary trade until the document count is reached; a payment that would overshoot is left out.
    while room() > unknown:
        roll = rng.random()
        if roll < COMBINED_AR_SHARE and room() - unknown >= 3:
            cp = rng.choice(customers)
            day = window_day()
            pair = [new_ar(cp, day), new_ar(cp, later_day(rng, world, day, 7))]
            _combined_receipt(rng, world, pair, room() - unknown)
        elif roll < COMBINED_AR_SHARE + COMBINED_AP_SHARE and room() - unknown >= 3:
            cp = rng.choice(one_off)
            day = window_day()
            pair = [new_ap(cp, day), new_ap(cp, later_day(rng, world, day, 10))]
            _pay_vendor(rng, world, pair, room() - unknown)
        elif roll < 0.55:
            inv = new_ar(rng.choice(customers), window_day())
            _pay_customer(rng, world, inv, room() - unknown)
        else:
            inv = new_ap(rng.choice(one_off), window_day())
            _pay_vendor(rng, world, [inv], room() - unknown)
    taken = {i.total_minor for i in world.invoices.values()}
    while room() > 0:
        amount = rng.randint(50_000, 900_000) * 100
        if amount in taken:
            continue
        taken.add(amount)
        pay = _add_payment(rng, world, "in", None, local_time(rng, world, window_day()), BASE, amount,
                           rng.choice(UNKNOWN_TEXTS), [], raw_name=rng.choice(UNKNOWN_PAYERS))
        _twin(world, "unidentified_receipt", "payment_allocation", [pay.key],
              "no reference in the remittance, amount matches no open item")
    _first_in_sequence(world)
    # An ERP allocates a payment to an invoice only once the invoice is keyed.
    for pay in world.payments.values():
        keyed = [world.invoices[k].entered_at for k, _ in pay.alloc]
        if keyed and pay.entered_at <= max(keyed):
            pay.entered_at = max(keyed) + dt.timedelta(minutes=rng.randint(10, 240))
    _number_documents(rng, world, numbers.ar, numbers.po)
    return world


def _first_in_sequence(world):
    """Period anchors come first among their vendor's bills after the period end, so a pulled bill keeps
    its vendor's numbering in order (domain spec A, scenario 4 §8). Anchors have no purchase order."""
    start = dt.datetime.combine(world.period_end + dt.timedelta(days=1), dt.time(), world.tz)
    for key in sorted(k for k, t in world.anchors.items() if t == "period_cutoff"):
        bill = world.invoices[key]
        others = [i.issued_at for i in world.invoices.values()
                  if i.cp == bill.cp and world.anchors.get(i.key) != "period_cutoff" and i.issued_at >= start]
        if others and min(others) <= bill.issued_at:
            low = start + dt.timedelta(hours=7) if min(others) > start + dt.timedelta(hours=7, minutes=30) else start
            moved = low + (min(others) - low) / 2
            bill.issued_at = moved.replace(second=0, microsecond=0).astimezone(bill.issued_at.tzinfo)
            bill.service_from = bill.service_to = world.local_date(bill.issued_at).isoformat()


def _number_documents(rng, world, first_ar, first_po):
    """Number documents by issue time, as an ERP and every vendor do, and replace the placeholders.

    Customer invoices get S-<year>-<n>. A vendor numbers its bills in its own format; our clerk keys the
    reference as printed, or in about CLERK_RETYPE_SHARE of bills with other separators. Remittance
    texts cite the printed reference. Our purchase orders also grow with the bill date.
    """
    final = {}
    ar = sorted((i for i in world.invoices.values() if i.direction == "AR"), key=lambda i: (i.issued_at, i.key))
    for n, inv in enumerate(ar, first_ar):
        final[inv.number] = f"S-{world.event_date(inv).year}-{n:04d}"
        inv.number = final[inv.number]
    for cp in world.counterparties.values():
        if cp.role != "vendor":
            continue
        bills = sorted((i for i in world.invoices.values() if i.cp == cp.key), key=lambda i: (i.issued_at, i.key))
        for n, inv in enumerate(bills, cp.next_number):
            inv.ref_parts = (cp.prefix, inv.issued_at.year, n)
            printed = _canonical_ref(cp, inv.ref_parts)
            final[inv.vendor_ref] = printed
            inv.vendor_ref = printed
            if rng.random() < CLERK_RETYPE_SHARE:
                inv.vendor_ref = rng.choice([v for v in ref_variants(inv.ref_parts, cp.style) if v != printed])
    po = first_po
    for inv in sorted((i for i in world.invoices.values() if i.po_ref), key=lambda i: (i.issued_at, i.key)):
        po += rng.randint(1, 7)
        inv.po_ref = f"PO-{po}"
    for pay in world.payments.values():
        for placeholder in re.findall(r"@I\d+@", pay.remittance):
            pay.remittance = pay.remittance.replace(placeholder, final[placeholder])
        if "@" in pay.remittance:
            raise WorldError(f"{pay.key}: remittance cites an unknown document")
