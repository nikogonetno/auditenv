"""The two coupled scenarios of docs/DOMAIN_SPEC_A.md (docs/tasks/B_night_stage3.md items 3 and 4).

C1. A foreign customer pays an invoice in full just after the midnight that ends the period (company time;
    for a negative offset just before it). The clerk takes the bank's UTC date as the posting date, and the
    system takes that date's rate: period_cutoff and fx_rate on the payment's 1030 row. A second, separate
    cause, a typo in the allocation, leaves part of the receipt on 3510: payment_allocation. Three truth
    issues; period and rate share their affected id, which the matcher pairs by type. Every amount comes
    from the final book by the formula of its type (auditenv/generate.py).

C2. A vendor bill is keyed twice, and the real payment, whose remittance names the bill's reference and so
    both records, is allocated to the copy. The original looks unpaid, but the allocation rule holds: the
    payment names one document group and is allocated in full to it. One truth issue, duplicate_invoice;
    a naive reconciliation flags a false payment_allocation on the original.

Both work on anchors that auditenv/world.py builds for them and reserves.
"""
import datetime as dt

from ..world import BASE
from .duplicate import make_copy

REST_PCT = (20, 60)


def _rest(rng, world, pay, inv_key, amount):
    """The part a typo leaves unallocated: not the total or the open amount of another item of the customer."""
    allocated = {}
    for other in world.payments.values():
        for key, part in other.alloc:
            allocated[key] = allocated.get(key, 0) + part
    lookalikes = set()
    for inv in world.invoices.values():
        if inv.cp == pay.cp and inv.currency == pay.currency and inv.key != inv_key:
            lookalikes |= {inv.total_minor, inv.total_minor - allocated.get(inv.key, 0)}
    step = 100 if pay.currency == BASE else 1
    rest = max(amount * rng.randint(*REST_PCT) // 100 // step * step, step)
    while rest in lookalikes:
        rest += step
    return rest


def c1(rng, world, used):
    """(variant, cause, [(type, affected keys)], documents touched)."""
    pays = [p for p in world.payments.values() if world.anchors.get(p.key) == "C1" and p.key not in used]
    if not pays:
        raise ValueError("config: no C1 receipt left")
    pay = rng.choice(pays)
    inv_key, amount = pay.alloc[0]
    utc_day = pay.timestamp.astimezone(dt.timezone.utc).date()
    pay.post_date = utc_day
    pay.rate = world.rates.rate(pay.currency, utc_day)
    rest = _rest(rng, world, pay, inv_key, amount)
    pay.alloc = [(inv_key, amount - rest)]
    carrier = f"{pay.key}:carrier"
    # Two causes (domain spec A, C1): the first gives two errors, the second the third. Each cause names
    # the indexes of the issues it gives.
    causes = [(f"bank UTC date {utc_day} used as the posting date instead of the company date "
               f"{world.event_date(pay)}, the system took that date's rate", [0, 1]),
              (f"allocation keyed as {amount - rest} instead of {amount}, the rest left on 3510", [2])]
    issues = [("period_cutoff", (carrier,)), ("fx_rate", (carrier,)), ("payment_allocation", (pay.key, inv_key))]
    return "utc_date_as_posting_date+partial_allocation", causes, issues, {pay.key, inv_key}


def c2(rng, world, used):
    """(variant, cause, [(type, affected keys)], documents touched)."""
    bills = [i for i in world.invoices.values() if world.anchors.get(i.key) == "C2" and i.key not in used]
    if not bills:
        raise ValueError("config: no C2 bill left")
    bill = rng.choice(bills)
    (pay,) = [p for p in world.payments.values() if world.anchors.get(p.key) == "C2" and p.alloc[0][0] == bill.key]
    copy = make_copy(rng, world, bill, keyed_before=pay.entered_at)
    pay.alloc = [(copy.key, pay.alloc[0][1])]
    # The trap of C2: the original looks unpaid after a payment that names it. Always listed.
    world.twins.append({"kind": "c2_trap", "mimics": "payment_allocation", "records": [bill.key, pay.key],
                        "feature": "the payment names both records and is allocated in full to the copy; "
                                   "the only error is the duplicate", "always": True})
    cause = "vendor bill keyed twice; the payment, whose reference names both records, was allocated to the copy"
    return "duplicate_then_paid", cause, [("duplicate_invoice", (bill.key, copy.key))], {bill.key, copy.key, pay.key}
