"""payment_allocation, variant partial_allocation (docs/DOMAIN_SPEC_A.md, scenario 2).

A customer pays one invoice in full and cites it, but the clerk keys a smaller allocation. The rest of
the receipt goes to 3510 at the payment rate. The entry stays balanced and the allocation agrees with
its ledger rows: the error shows only against the bank amount and the invoice.
"""
VARIANT = "partial_allocation"


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
    # The rest must not look like the payment of another open item of this customer.
    other_totals = {i.total_minor for i in world.invoices.values()
                    if i.cp == pay.cp and i.currency == pay.currency and i.key != inv_key}
    rest = max(amount * rng.randint(20, 60) // 100 // 100 * 100, 100)
    while rest in other_totals:
        rest += 100
    if not 2 < rest < amount:
        raise ValueError("config: receipt too small for payment_allocation")
    pay.alloc = [(inv_key, amount - rest)]
    cause = f"allocation keyed as {amount - rest} instead of {amount}, the rest left on 3510"
    return VARIANT, cause, [(pay.key, inv_key)], {pay.key, inv_key}
