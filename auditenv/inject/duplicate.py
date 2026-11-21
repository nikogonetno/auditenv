"""duplicate_invoice, variant same_vendor_ref_reformatted (docs/DOMAIN_SPEC_A.md, scenario 1).

A vendor bill is keyed a second time by another clerk, with the vendor's reference typed with other
separators. The copy has the same counterparty, currency, lines, service period and issued_at, and is
posted by the normal rules, so its posting date and rate are right. Its reference format is one the clean
bills of the episode also show, and its entry delay follows the clean distribution, so neither field alone
gives the copy away (domain spec A: «формат копии встречается и в чистых данных», INV-21).
"""
import copy

from ..world import VENDOR_KINDS, entered_time, ref_shape, ref_variants

VARIANT = "same_vendor_ref_reformatted"


def _options(world, bill):
    """(retyped references whose shape this vendor's other bills show, ... any clean bill shows, all)."""
    style = world.counterparties[bill.cp].style
    options = [v for v in ref_variants(bill.ref_parts, style) if v != bill.vendor_ref]
    others = [i for i in world.invoices.values() if i.direction == "AP" and i.key != bill.key]
    vendor = {ref_shape(i.vendor_ref) for i in others if i.cp == bill.cp}
    seen = {ref_shape(i.vendor_ref) for i in others}
    return ([v for v in options if ref_shape(v) in vendor], [v for v in options if ref_shape(v) in seen], options)


def reformat(rng, world, bill):
    """The bill's reference as another clerk keys it: other separators, preferably a shape this vendor's
    other bills already show, else one any clean bill shows."""
    vendor, seen, options = _options(world, bill)
    return rng.choice(vendor or seen or options)


def make_copy(rng, world, bill, keyed_before=None):
    """A second record of `bill` as another clerk keys it, added to the world. `keyed_before`: a moment the
    copy must be keyed before (a payment allocated to the copy comes after it, C2)."""
    twin = copy.deepcopy(bill)
    twin.key = world.new_key("I")
    for line in twin.lines:
        line.key = world.new_key("N")
    twin.vendor_ref = reformat(rng, world, bill)
    event = bill.issued_at.astimezone(world.tz)
    for _ in range(200):                         # the clean entry delay, redrawn until it fits
        twin.entered_at = entered_time(rng, world, event)
        if keyed_before is None or twin.entered_at < keyed_before:
            break
    else:
        twin.entered_at = event + (keyed_before - event) / 2
    world.invoices[twin.key] = twin
    return twin


def inject(rng, world, used):
    """(variant, cause, [affected keys], documents touched). ValueError when no bill can be copied.

    Touched documents include the payments whose remittance cites the original: the copy has the same
    normalized reference, so it joins their invoice_refs.
    """
    cited_by = {}
    for pay in world.payments.values():
        for key in world.invoice_refs(pay) or ():
            cited_by.setdefault(key, []).append(pay.key)
    candidates = [i for i in world.invoices.values()
                  if i.direction == "AP" and i.key not in used and not VENDOR_KINDS[world.counterparties[i.cp].kind][2]
                  and not used.intersection(cited_by.get(i.key, ()))]
    if not candidates:
        raise ValueError("config: no vendor bill left for duplicate_invoice")
    # Small books: prefer bills whose retyped reference can take a shape the clean bills show.
    bill = rng.choice([c for c in candidates if _options(world, c)[1]] or candidates)
    citing = sorted(cited_by.get(bill.key, ()))
    twin = make_copy(rng, world, bill)
    event = bill.issued_at.astimezone(world.tz)
    days = (twin.entered_at.date() - event.date()).days
    # Neutral about which record came first: the added one is sometimes keyed before the other (PM 21.11 12.50).
    cause = (f"one vendor bill keyed twice; one of the two records has the reference typed with other separators "
             f"and was entered {days} days after issue")
    return VARIANT, cause, [(bill.key, twin.key)], {bill.key, twin.key, *citing}
