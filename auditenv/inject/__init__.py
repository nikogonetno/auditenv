"""Error injectors (role A). Each one changes a decision of the clean world or adds a document, and says
which documents it touched and which truth issues it caused, in internal keys.

Variants follow docs/DOMAIN_SPEC_A.md. Stage 1: duplicate_invoice (same_vendor_ref_reformatted) and
payment_allocation (partial_allocation). Stage 2: fx_rate (rate_of_entry_date) and period_cutoff
(pulled_into_period). A record is the target of one injection at most, and never a record of a twin. Amounts and evidence are computed later from the final book (auditenv/generate.py),
so the order of injections never changes truth.
"""
from dataclasses import dataclass, field

from ..config import ISSUE_TYPES
from . import allocation, duplicate, period, rate, scenarios


@dataclass
class Issue:
    type: str
    affected: tuple               # internal keys: invoice and payment keys, or a row key "<doc>:carrier"
    scenario: str = None


@dataclass
class Injection:
    variant: str
    cause: str
    issues: list
    docs: set                     # documents whose records this injection changed or created
    scenario: str = None


@dataclass
class Plan:
    reserved_documents: int = 0   # documents the injectors add, kept out of the clean world's count
    anchors: dict = field(default_factory=dict)


# type -> (module, documents it adds, anchors it needs). Anchors are ordinary clean records built first
# (auditenv/world.py build_world); period_moves adds one real neighbour invoice per side of the period,
# because the moved document no longer counts for INV-03.
INJECTORS = {
    "duplicate_invoice": (duplicate, 1, {"vendor_bills": 1}),
    "payment_allocation": (allocation, 0, {"full_receipts": 1}),
    "fx_rate": (rate, 0, {"foreign_invoices": 1}),
    "period_cutoff": (period, 0, {"period_bills": 1, "period_moves": 1}),
}
# scenario -> (injector, documents it adds, anchors it needs). C1 moves its payment across the period end.
SCENARIOS = {
    "C1": (scenarios.c1, 0, {"c1_receipts": 1, "period_moves": 1}),
    "C2": (scenarios.c2, 1, {"c2_bills": 1}),
}


def plan(config):
    """What the clean world must hold for the errors of a normalized config. ValueError if one is not
    generated yet."""
    result = Plan()
    for scenario, count in sorted(config["scenarios"].items()):
        _, adds, needs = SCENARIOS[scenario]
        result.reserved_documents += adds * count
        for name, n in needs.items():
            result.anchors[name] = result.anchors.get(name, 0) + n * count
    for kind in ISSUE_TYPES:
        count = config["errors"][kind]
        if not count:
            continue
        if kind not in INJECTORS:
            raise ValueError(f"config: error type {kind} is not generated yet")
        _, adds, needs = INJECTORS[kind]
        result.reserved_documents += adds * count
        for name, n in needs.items():
            result.anchors[name] = result.anchors.get(name, 0) + n * count
    return result


def inject_errors(rng, world, config):
    """Apply every requested error to `world` in place. Returns the injections in order."""
    used = set()
    for twin in world.twins:
        used.update(twin["records"])
    injections = []
    for kind in ISSUE_TYPES:
        module = INJECTORS.get(kind, (None,))[0]
        # Anchors built for another error type are off limits, so capacity never depends on the seed.
        others = {key for key, owner in world.anchors.items() if owner != kind}
        for _ in range(config["errors"][kind]):
            variant, cause, issues, docs = module.inject(rng, world, used | others)
            used.update(docs)
            injections.append(Injection(variant, cause, [Issue(kind, tuple(sorted(a))) for a in issues], set(docs)))
    for name, (injector, _, _) in SCENARIOS.items():
        others = {key for key, owner in world.anchors.items() if owner != name}
        for _ in range(config["scenarios"][name]):
            variant, cause, issues, docs = injector(rng, world, used | others)
            used.update(docs)
            injections.append(Injection(variant, cause, [Issue(t, tuple(sorted(a)), name) for t, a in issues],
                                        set(docs), name))
    return injections
