"""Episode generator (role A, docs/tasks/A_generator.md).

    generate_episode(seed, config, out_dir)

writes out_dir/public/{manifest.json, invoices.jsonl, payments.jsonl, ledger.csv, fx.csv, policy.json}
and out_dir/private/{truth.json, provenance.json} in the layout of SPEC §3 and tests/fixtures/mini_v03.

Steps: build the clean world (auditenv/world.py), post it and check INV-01…INV-16, apply the requested
errors (auditenv/inject), post again and check that the rule breaks are exactly the errors. Then assign
public ids, compute every amount and evidence set from the final book, and take the provenance changes
from the difference between the clean and the final book. Deterministic in (seed, config).
"""
import csv
import datetime as dt
import io
import json
import random
import re
from pathlib import Path

from .config import ISSUE_TYPES, config_sha256, normalize_config
from .inject import inject_errors, plan
from .world import (ACCOUNTS, BASE, MINOR_UNITS, WEEKEND_DAYS, Invoice, WorldError, build_world, check_book, conv,
                    post)

GENERATOR_VERSION = "0.1.0"
SCHEMA_VERSION = "1.0"
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
LEDGER_TIME = "T12:00:00"
DUE_TIME = "T00:00:00"
TOLERANCE_MINOR = 1
LEDGER_COLUMNS = ("row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency", "entity_id",
                  "invoice_id", "payment_id", "je_id", "entered_at", "doc_currency", "doc_amount_minor",
                  "exchange_rate", "memo")
FX_COLUMNS = ("date", "source_currency", "target_currency", "rate", "fx_id")
# INV-23 and CLAUDE.md: never in a public file.
PRIVATE_WORDS = ("truth", "provenance", "inject", "twin", "scenario", "issue_id")


class IdSource:
    """6-character Crockford Base32 ids from the episode generator, unique across all tables."""

    def __init__(self, rng):
        self.rng = rng
        self.taken = set()

    def new(self, avoid=()):
        while True:
            candidate = "".join(self.rng.choice(CROCKFORD) for _ in range(6))
            if candidate not in self.taken and candidate not in avoid:
                self.taken.add(candidate)
                return candidate


def generate_episode(seed: int, config: dict, out_dir: str) -> None:
    """Write one episode for (seed, config) into out_dir. ValueError on a seed or config it cannot make."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an int >= 0")
    config = normalize_config(config)
    rng = random.Random(seed)
    needs = plan(config)
    world = build_world(rng, config, needs.reserved_documents, needs.anchors)
    clean_book = post(world)
    check_book(world, clean_book, set())

    final = world.copy()
    injections = inject_errors(rng, final, config)
    book = post(final)
    expected = {(i.type, frozenset(i.affected)) for inj in injections for i in inj.issues}
    moved = {key.split(":")[0] for t, keys in expected if t == "period_cutoff" for key in keys}
    check_book(final, book, expected, moved)

    files = _render(rng, seed, config, world, clean_book, final, book, injections)
    root = Path(out_dir)
    for rel in sorted(files):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(files[rel])


# ---------------------------------------------------------------- public ids
class _Ids:
    def __init__(self, rng, world, final, clean_book, book):
        source = IdSource(rng)
        self.source = source
        self.episode = source.new()
        self.entity = {k: source.new() for k in final.counterparties}
        self.doc, self.record, self.line, self.je = {}, {}, {}, {}
        for doc in [*final.invoices.values(), *final.payments.values()]:
            self.doc[doc.key], self.record[doc.key], self.je[doc.key] = source.new(), source.new(), source.new()
            for line in getattr(doc, "lines", ()):
                self.line[line.key] = source.new()
        self.row = {r.key: source.new() for r in book.rows}
        for r in clean_book.rows:
            if r.key not in self.row:
                self.row[r.key] = source.new()      # a row an injection removed: only provenance names it
        self.fx = {key: source.new() for key in sorted(final.rates.quotes)}

    def ref(self, key):
        """Public reference of an internal key: invoice/…, payment/…, ledger/… or fx/…."""
        if isinstance(key, tuple):
            return f"fx/{self.fx[key]}"
        if ":" in key:
            return f"ledger/{self.row[key]}"
        return f"{'invoice' if key.startswith('I') else 'payment'}/{self.record[key]}"


# ---------------------------------------------------------------- public records
def _iso(when):
    return when.isoformat()


def _invoice_record(world, inv, ids):
    cp = world.counterparties[inv.cp]
    due = world.event_date(inv) + dt.timedelta(days=inv.terms)
    return {
        "doc_id": ids.doc[inv.key],
        "invoice_id": ids.record[inv.key],
        "entity_id": ids.entity[inv.cp],
        "issued_at": _iso(inv.issued_at),
        "due_at": due.isoformat() + DUE_TIME + world.offset,
        "currency": inv.currency,
        "total_minor": inv.total_minor,
        "line_items": [{"line_id": ids.line[line.key], "description": line.description, "qty": line.qty,
                        "unit_price_minor": line.unit_price_minor, "amount_minor": line.amount_minor}
                       for line in inv.lines],
        "direction": inv.direction,
        "counterparty_name": cp.name,
        "invoice_number": inv.number if inv.direction == "AR" else None,
        "vendor_ref": inv.vendor_ref,
        "po_ref": inv.po_ref,
        "entered_at": _iso(inv.entered_at),
        "service_from": inv.service_from,
        "service_to": inv.service_to,
    }


def _payment_record(world, pay, ids):
    refs = world.invoice_refs(pay)
    return {
        "doc_id": ids.doc[pay.key],
        "payment_id": ids.record[pay.key],
        "timestamp": _iso(pay.timestamp),
        "currency": pay.currency,
        "amount_minor": pay.amount_minor,
        "invoice_refs": sorted(ids.record[k] for k in refs) if refs else None,
        "entity_id": ids.entity[pay.cp] if pay.cp else None,
        "direction": pay.direction,
        "counterparty_name_raw": pay.raw_name,
        "entered_at": _iso(pay.entered_at),
        "remittance_info": pay.remittance,
    }


def _row_record(world, row, ids):
    return {
        "row_id": ids.row[row.key],
        "timestamp": row.post_date.isoformat() + LEDGER_TIME + world.offset,
        "account": row.account,
        "debit_minor": row.debit_minor,
        "credit_minor": row.credit_minor,
        "currency": BASE,
        "entity_id": ids.entity[row.entity] if row.entity else None,
        "invoice_id": ids.record[row.invoice] if row.invoice else None,
        "payment_id": ids.record[row.payment] if row.payment else None,
        "je_id": ids.je[row.je],
        "entered_at": _iso(row.entered_at),
        "doc_currency": row.doc_currency,
        "doc_amount_minor": row.doc_amount_minor,
        "exchange_rate": row.rate,
        "memo": row.memo,
    }


def _tables(world, book, ids):
    invoices = sorted((_invoice_record(world, i, ids) for i in world.invoices.values()), key=lambda r: r["invoice_id"])
    payments = sorted((_payment_record(world, p, ids) for p in world.payments.values()), key=lambda r: r["payment_id"])
    order = {key: n for n, key in enumerate(r.key for r in book.rows)}
    rows = sorted(book.rows, key=lambda r: (ids.je[r.je], order[r.key]))   # entries in id order, lines together
    ledger = [_row_record(world, r, ids) for r in rows]
    fx = [{"date": day.isoformat(), "source_currency": ccy, "target_currency": BASE,
           "rate": world.rates.quotes[(day, ccy)], "fx_id": ids.fx[(day, ccy)]}
          for day, ccy in sorted(world.rates.quotes)]
    return {"ledger": ledger, "invoices": invoices, "payments": payments, "fx": fx}


def _policy(world):
    tz = world.tz
    start = dt.datetime.combine(world.period_start, dt.time(), tz).astimezone(dt.timezone.utc)
    end = dt.datetime.combine(world.period_end + dt.timedelta(days=1), dt.time(), tz).astimezone(dt.timezone.utc)
    return {
        "accounts": [{"code": code, "name": name} for code, name in ACCOUNTS],
        "allocation_rule": "a payment whose remittance_info names exactly one open document group of the same "
                           "counterparty and currency must be allocated to it up to the open amount, "
                           "the rest goes to 3510",
        "base_currency": BASE,
        "company_utc_offset": world.offset,
        "duplicate_rule": "two AP invoices of one counterparty with equal currency, total and normalized vendor_ref",
        "fx": {
            "date_rule": "posting_date_rule date",
            "holidays": [d.isoformat() for d in world.rates.holidays],
            "non_business_fallback": "previous business day",
            "quote": "base units per 1 unit of currency",
            "rounding": "ROUND_HALF_UP to 1 minor unit",
            "settlement": "clearing rows use the invoice rate, last allocation clears the base residual, "
                          "difference to 6250 or 7430",
            "weekend_days": list(WEEKEND_DAYS),
        },
        "ledger_window": {"from": world.window_start.isoformat(), "to": world.window_end.isoformat()},
        "minor_units": dict(sorted(MINOR_UNITS.items())),
        "period": {"end_local": world.period_end.isoformat(), "end_utc_exclusive": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "start_local": world.period_start.isoformat(), "start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ")},
        "posting_date_rule": "company-local date of invoice.issued_at or payment.timestamp",
        "vendor_ref_normalization": "uppercase, keep A-Z and 0-9 only",
    }


def _manifest(world, policy, tables, episode_id):
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "base_currency": BASE,
        "period_start": world.period_start.isoformat(),
        "period_end": world.period_end.isoformat(),
        "timezone": world.offset,
        "counts": {name: len(tables[name]) for name in ("ledger", "invoices", "payments", "fx")},
        "documents": len(tables["invoices"]) + len(tables["payments"]),
        "rounding": {"mode": "ROUND_HALF_UP", "minor_units": policy["minor_units"]},
    }


# ---------------------------------------------------------------- truth
def _row(book, key):
    (row,) = [r for r in book.rows if r.key == key]
    return row


def _issue(final, book, issue, ids):
    """(affected refs, amount, currency, evidence refs) of one error, by the formula of its type, from the
    final book only (SPEC §3.6, domain spec A, «Достаточные доказательства»)."""
    if issue.type == "duplicate_invoice":
        first, second = (final.invoices[k] for k in issue.affected)
        evidence = [first.key, second.key, book.carrier(first.key).key, book.carrier(second.key).key]
        return issue.affected, first.total_minor, first.currency, evidence
    if issue.type == "payment_allocation":
        (pay_key,) = [k for k in issue.affected if k in final.payments]
        (inv_key,) = [k for k in issue.affected if k in final.invoices]
        pay = final.payments[pay_key]
        clearing = [r for r in book.by_doc[pay_key] if r.invoice == inv_key]
        (held,) = book.rows_of(pay_key, "unapplied")
        amount = pay.amount_minor - sum(r.doc_amount_minor for r in clearing)
        if amount != held.doc_amount_minor:
            raise WorldError(f"{pay_key}: payment minus allocations {amount}, 3510 holds {held.doc_amount_minor}")
        return issue.affected, amount, pay.currency, [pay_key, inv_key, *(r.key for r in clearing), held.key]
    (row_key,) = issue.affected
    row = _row(book, row_key)
    doc = final.invoices.get(row.je) or final.payments[row.je]
    if issue.type == "period_cutoff":
        return issue.affected, row.base, BASE, [doc.key, row.key]
    event = final.event_date(doc)
    right = final.rates.rate(doc.currency, event)
    evidence = [doc.key, row.key, (final.rates.rate_date(event), doc.currency)]
    return issue.affected, abs(row.base - conv(row.doc_amount_minor, right)), BASE, evidence


def _truth(final, book, injections, ids, public_text):
    issues, ids_by_injection = [], []
    tokens = set(re.findall(r"[0-9A-Z]{6}", public_text))
    for inj in injections:
        mine = []
        for issue in inj.issues:
            affected, amount, currency, evidence = _issue(final, book, issue, ids)
            if issue.type not in ISSUE_TYPES or amount <= 2 * TOLERANCE_MINOR:
                raise WorldError(f"INV-20: {issue.type} amount {amount}")
            issue_id = ids.source.new(avoid=tokens)
            mine.append(issue_id)
            issues.append({
                "affected_ids": sorted(ids.ref(k) for k in affected),
                "currency": currency,
                "evidence_sets": [sorted(ids.ref(k) for k in evidence)],
                "expected_amount_minor": amount,
                "issue_id": issue_id,
                "scenario_id": inj.scenario,
                "tolerance_minor": TOLERANCE_MINOR,
                "type": issue.type,
            })
        ids_by_injection.append(mine)
    issues.sort(key=lambda i: i["issue_id"])
    return {"clean": not issues, "episode_id": ids.episode, "issues": issues}, ids_by_injection


# ---------------------------------------------------------------- provenance
def _changes(world, clean_book, final, book, ids, injections):
    """Changes per injection: every record of the final book that differs from the clean one, by field.
    A difference no injection claims is a generator bug (domain spec A, provenance)."""
    def records(w, b):
        out = {}
        for inv in w.invoices.values():
            out[inv.key] = (inv.key, _invoice_record(w, inv, ids))
        for pay in w.payments.values():
            out[pay.key] = (pay.key, _payment_record(w, pay, ids))
        for row in b.rows:
            out[row.key] = (row.je, _row_record(w, row, ids))
        return out

    before, after = records(world, clean_book), records(final, book)
    owner = {}
    for n, inj in enumerate(injections):
        for doc in inj.docs:
            if doc in owner:
                raise WorldError(f"{doc} touched by two injections")
            owner[doc] = n
    changes = [[] for _ in injections]
    for key in sorted(set(before) | set(after)):
        doc = (after.get(key) or before.get(key))[0]
        old, new = (before.get(key) or (None, None))[1], (after.get(key) or (None, None))[1]
        if old == new:
            continue
        if doc not in owner:
            raise WorldError(f"{key} changed by no injection")
        ref = ids.ref(key)
        if old is None:
            changes[owner[doc]].append({"record": ref, "field": None, "before": None, "after": "created"})
        elif new is None:
            changes[owner[doc]].append({"record": ref, "field": None, "before": "removed", "after": None})
        else:
            for name in sorted(set(old) | set(new)):
                if old.get(name) != new.get(name):
                    changes[owner[doc]].append({"record": ref, "field": name, "before": old.get(name),
                                                "after": new.get(name)})
    return changes


def _twins(final, ids, targets):
    """Correct records that look like an error to a naive rule (domain spec A, «Корректные двойники»)."""
    found = [dict(t) for t in final.twins]
    tz = final.tz
    for doc in [*final.invoices.values(), *final.payments.values()]:
        if doc.key in targets:
            continue
        when = doc.issued_at if isinstance(doc, Invoice) else doc.timestamp
        local = final.event_date(doc)
        in_period = final.period_start <= local <= final.period_end
        if doc.currency != BASE and not final.rates.is_business(local):
            found.append({"kind": "weekend_document", "mimics": "fx_rate", "records": [doc.key],
                          "feature": "dated on a non-business day, rate of the previous business day by the policy"})
        others = (when.date(), when.astimezone(dt.timezone.utc).date())   # issuer's and UTC calendar date
        if any((final.period_start <= day <= final.period_end) != in_period for day in others):
            found.append({"kind": "midnight_utc", "mimics": "period_cutoff", "records": [doc.key],
                          "feature": "the issuer or UTC date is in another period than the company-local date"})
        if in_period and doc.entered_at.astimezone(tz).date() > final.period_end:
            found.append({"kind": "late_entry", "mimics": "period_cutoff", "records": [doc.key],
                          "feature": "entered after the period, posted on the document date"})
    out = []
    for n, twin in enumerate(sorted(found, key=lambda t: (t["kind"], t["records"])), 1):
        out.append({"twin_id": f"TW-{n:02d}", "kind": twin["kind"], "mimics": twin["mimics"],
                    "records": sorted(ids.ref(k) for k in twin["records"]),
                    "distinguishing_feature": twin["feature"]})
    return out


# ---------------------------------------------------------------- output
def _csv(rows, columns):
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(["" if row[c] is None else row[c] for c in columns])
    return buf.getvalue()


def _jsonl(rows):
    return "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)


def _json(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _render(rng, seed, config, world, clean_book, final, book, injections):
    ids = _Ids(rng, world, final, clean_book, book)
    tables = _tables(final, book, ids)
    policy = _policy(final)
    public = {
        "public/manifest.json": _json(_manifest(final, policy, tables, ids.episode)),
        "public/policy.json": _json(policy),
        "public/ledger.csv": _csv(tables["ledger"], LEDGER_COLUMNS),
        "public/invoices.jsonl": _jsonl(tables["invoices"]),
        "public/payments.jsonl": _jsonl(tables["payments"]),
        "public/fx.csv": _csv(tables["fx"], FX_COLUMNS),
    }
    text = "".join(public[k] for k in sorted(public))
    for word in (*PRIVATE_WORDS, *ISSUE_TYPES):
        if re.search(re.escape(word).replace("_", "[ _-]?"), text, re.IGNORECASE):
            raise WorldError(f"INV-23: {word!r} in a public file")
    truth, issue_ids = _truth(final, book, injections, ids, text)
    changes = _changes(world, clean_book, final, book, ids, injections)
    targets = set().union(*(inj.docs for inj in injections)) if injections else set()
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "generator_version": GENERATOR_VERSION,
        "config": config,
        "config_sha256": config_sha256(config),
        "world": {"period": final.period, "timezone": final.offset, "industry": final.industry,
                  "foreign_pct": final.foreign_pct, "documents": len(tables["invoices"]) + len(tables["payments"])},
        "injections": [{"order": n, "injection_id": f"INJ-{n:02d}", "scenario_id": inj.scenario,
                        "variant": inj.variant, "root_cause": inj.cause, "issue_ids": sorted(issue_ids[n - 1]),
                        "changes": changes[n - 1]} for n, inj in enumerate(injections, 1)],
        "twins": _twins(final, ids, targets),
    }
    return {**public, "private/truth.json": _json(truth), "private/provenance.json": _json(provenance)}
