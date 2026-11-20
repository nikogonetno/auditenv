"""payment_allocation, variant partial_allocation (docs/DOMAIN_SPEC_A.md, scenario 2).

A customer pays one invoice in full and cites it, but the clerk keys a smaller allocation. The rest of
the receipt goes to 3510 at the payment rate. The entry stays balanced and the allocation agrees with
its ledger rows: the error shows only against the bank amount and the invoice.
"""
from ..world import BASE

VARIANT = "partial_allocation"
REST_PCT = (5, 60)                # overlaps the excess of clean overpayments, so the share alone tells nothing


def inject(rng, world, used):
    """(variant, cause, [affected keys], documents touched). ValueError when no receipt fits."""
    candidates = []
    for pay in world.payments.values():
        if pay.direction != "in" or pay.cp is None or pay.key in used or len(pay.alloc) != 1:
            continue
        inv_key, amount = pay.alloc[0]
        invoice = world.invoices[inv_key]
        if inv_key in used or amount != pay.amount_minor or amount != invoice.total_minor:
            continue
        if world.invoice_refs(pay) != [inv_key]:
            continue
        candidates.append(pay)
    if not candidates:
        raise ValueError("config: no fully paid customer invoice left for payment_allocation")
    pay = rng.choice(candidates)
    inv_key, amount = pay.alloc[0]
    # The rest must not look like the payment of another item of this customer: neither its total nor
    # what is still open on it (domain spec A, variant table).
    allocated = {}
    for other in world.payments.values():
        for key, part in other.alloc:
            allocated[key] = allocated.get(key, 0) + part
    lookalikes = set()
    for inv in world.invoices.values():
        if inv.cp == pay.cp and inv.currency == pay.currency and inv.key != inv_key:
            lookalikes |= {inv.total_minor, inv.total_minor - allocated.get(inv.key, 0)}
    rest = amount * rng.randint(*REST_PCT) // 100
    step = 100 if pay.currency == BASE else 1
    rest = max(rest // step * step, step)
    while rest in lookalikes:
        rest += step
    if not 2 < rest < amount:
        raise ValueError("config: receipt too small for payment_allocation")
    pay.alloc = [(inv_key, amount - rest)]
    cause = f"allocation keyed as {amount - rest} instead of {amount}, the rest left on 3510"
    return VARIANT, cause, [(pay.key, inv_key)], {pay.key, inv_key}
