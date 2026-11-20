"""period_cutoff, variant pulled_into_period (docs/DOMAIN_SPEC_A.md, scenario 4; docs/tasks/B_stage2.md item 2).

A base-currency invoice of the first days of the next month is posted on the last day of the period. The
currency is the base one, so the move does not change the rate and causes no second error. Amounts and
entered_at stay as they were.

The numbering must not give the move away. A document is a target only when every document that comes
before it after the period end, in its numbering sequence, is posted in the period too. That sequence is
the customer invoice numbers for an AR invoice, and for an AP bill both the vendor's references and our
purchase orders. The period anchors of auditenv/world.py come first in their vendor's sequence, so a
target always exists; a document out of order is never moved.
"""
import datetime as dt

from ..world import BASE

VARIANT = "pulled_into_period"
FIRST_DAYS = 5


def inject(rng, world, used):
    """(variant, cause, [affected keys], documents touched). ValueError when no document fits."""
    end = world.period_end
    after = sorted((i for i in world.invoices.values() if world.event_date(i) > end),
                   key=lambda i: (i.issued_at, i.key))

    def in_order(inv):
        if inv.direction == "AR":
            earlier = [i for i in after if i.direction == "AR"]
        else:
            earlier = [i for i in after if i.cp == inv.cp or (inv.po_ref and i.po_ref)]
        earlier = earlier[:earlier.index(inv)]
        return all(i.post_date == end for i in earlier)

    pool = [i for i in after if i.currency == BASE and i.key not in used
            and world.event_date(i) <= end + dt.timedelta(days=FIRST_DAYS)]
    candidates = [i for i in pool if in_order(i)]
    if not candidates:
        raise ValueError("config: no base-currency invoice of the next month's first days left for period_cutoff")
    inv = rng.choice(candidates)
    inv.post_date = end
    cause = f"document of {world.event_date(inv)} posted on the last day of the period {end}"
    return VARIANT, cause, [(f"{inv.key}:carrier",)], {inv.key}
