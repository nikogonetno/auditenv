"""fx_rate, variant rate_of_entry_date (docs/DOMAIN_SPEC_A.md, scenario 3; docs/tasks/B_stage2.md item 1).

A foreign-currency invoice is keyed some days after its date, and the system takes the rate of the entry
date instead of the rate of the document date. The posting date stays right. Payments of the invoice clear
at the booked rate, so the error stays in the invoice entry and no second error appears in a payment entry.
The invoice's entry delay is its own clean one: only invoices whose entry date already has another rate
qualify, so the injector changes the rate and nothing else.
"""
from ..world import entry_rate_differs

VARIANT = "rate_of_entry_date"


def inject(rng, world, used):
    """(variant, cause, [affected keys], documents touched). ValueError when no invoice fits.

    Touched documents include the payments that clear the invoice: their clearing rows follow the booked
    rate and their FX difference changes with it.
    """
    clearing = {}
    for pay in world.payments.values():
        for key, _ in pay.alloc:
            clearing.setdefault(key, []).append(pay.key)
    candidates = [i for i in world.invoices.values()
                  if i.key not in used and entry_rate_differs(world, i)
                  and not used.intersection(clearing.get(i.key, ()))]
    if not candidates:
        raise ValueError("config: no foreign-currency invoice with another entry-date rate left for fx_rate")
    inv = rng.choice(candidates)
    entered, dated = world.local_date(inv.entered_at), world.event_date(inv)
    inv.rate = world.rates.rate(inv.currency, entered)
    cause = f"rate of the entry date {entered} used instead of the document date {dated}"
    return VARIANT, cause, [(f"{inv.key}:carrier",)], {inv.key, *clearing.get(inv.key, ())}
