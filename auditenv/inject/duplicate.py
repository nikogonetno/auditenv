"""duplicate_invoice, variant same_vendor_ref_reformatted (docs/DOMAIN_SPEC_A.md, scenario 1).

A vendor bill is keyed a second time by another clerk a few days later, with the vendor's reference
typed in another format. The copy has the same counterparty, currency, lines, service period and
issued_at, and is posted by the normal rules, so its posting date and rate are right.
"""
import copy
import datetime as dt
import re

from ..world import VENDOR_KINDS, norm_ref

VARIANT = "same_vendor_ref_reformatted"


def reformat(rng, ref):
    """The same reference as another clerk types it: equal after normalization, different as text."""
    head = re.match(r"[A-Za-z]+", ref).group(0)
    tail = ref[len(head):].lstrip("- ")
    plain_tail = re.sub(r"[^A-Za-z0-9]", "", tail)
    options = {ref.replace("-", "").replace(" ", ""), ref.replace("-", " "), ref.replace(" ", "-"),
               f"{head}-{plain_tail}", f"{head} {plain_tail}", f"{head}{plain_tail}"}
    options = sorted(o for o in options if o != ref and norm_ref(o) == norm_ref(ref))
    return rng.choice(options)


def inject(rng, world, used):
    """(variant, cause, [affected keys], documents touched). ValueError when no bill can be copied."""
    candidates = [i for i in world.invoices.values()
                  if i.direction == "AP" and i.key not in used and not VENDOR_KINDS[world.counterparties[i.cp].kind][2]]
    if not candidates:
        raise ValueError("config: no vendor bill left for duplicate_invoice")
    bill = rng.choice(candidates)
    twin = copy.deepcopy(bill)
    twin.key = world.new_key("I")
    for line in twin.lines:
        line.key = world.new_key("N")
    twin.vendor_ref = reformat(rng, bill.vendor_ref)
    event = bill.issued_at.astimezone(world.tz)
    lag = (bill.entered_at.date() - event.date()).days
    day = event.date() + dt.timedelta(days=rng.randint(min(lag + 1, 10), 10))
    twin.entered_at = dt.datetime.combine(day, dt.time(rng.randint(9, 18), rng.randint(0, 59)), world.tz)
    world.invoices[twin.key] = twin
    cause = f"vendor bill keyed twice, the copy {(day - event.date()).days} days after issue with the reference retyped"
    return VARIANT, cause, [(bill.key, twin.key)], {bill.key, twin.key}
