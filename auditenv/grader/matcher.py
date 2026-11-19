"""Strict matching of submitted findings to truth issues (SPEC.md §5.2, §5.3).

A finding matches an issue only if type, the affected ID set, the amount within
tolerance, the currency and a full evidence set all agree. Matching is a maximum
bipartite matching, so a repeated finding can close at most one issue.
"""
from dataclasses import dataclass

ISSUE_TYPES = frozenset({"duplicate_invoice", "fx_rate", "payment_allocation", "period_cutoff"})
# Only these table prefixes are stripped, once. `je` is deliberately absent:
# journal entries are not addressable records (SPEC.md §3.3).
ID_PREFIXES = frozenset({"allocation", "counterparty", "doc", "fx", "invoice", "ledger", "line", "payment"})
MAX_EVIDENCE = 10
DEFAULT_TOLERANCE_MINOR = 1

# Field names of a submitted finding. Case §04/§05 may rename them: change only here.
F_TYPE = "type"
F_AFFECTED = "affected_ids"
F_AMOUNT = "amount_minor"
F_CURRENCY = "currency"
F_EVIDENCE = "evidence"


@dataclass(frozen=True)
class Finding:
    type: str
    affected: frozenset
    amount_minor: int
    currency: str
    evidence: frozenset


@dataclass(frozen=True)
class Issue:
    issue_id: str
    type: str
    affected: frozenset
    expected_amount_minor: int
    currency: str
    evidence_sets: tuple
    tolerance_minor: int


def normalize_id(ref):
    """`invoice/1TNM24` and `1TNM24` name the same record. Unknown prefixes stay as they are."""
    prefix, sep, rest = ref.partition("/")
    if sep and rest and prefix in ID_PREFIXES:
        return rest
    return ref


# Exact type checks, not isinstance: a subclass could override __len__, __eq__ or
# __hash__ and fake a match when the grader is called in-process (review, robustness).
def _is_id_list(value):
    return type(value) is list and all(type(x) is str for x in value)


def _is_currency(value):
    return type(value) is str and len(value) == 3 and value.isascii() and value.isalpha() and value.isupper()


def _is_minor_amount(value):
    # `type(...) is int` rejects bool, float and numeric strings on purpose (E-28).
    return type(value) is int and value >= 0


def parse_finding(raw):
    """Normalized Finding, or None when the finding is malformed. Malformed findings still count as FP.

    Never raises: anything the agent sends is either a valid finding or an invalid one.
    """
    try:
        return _parse_finding(raw)
    except Exception:
        return None


def _parse_finding(raw):
    if type(raw) is not dict:
        return None
    ftype, affected = raw.get(F_TYPE), raw.get(F_AFFECTED)
    amount, currency, evidence = raw.get(F_AMOUNT), raw.get(F_CURRENCY), raw.get(F_EVIDENCE)
    if type(ftype) is not str or ftype not in ISSUE_TYPES:
        return None
    if not _is_id_list(affected) or not affected:
        return None
    if not _is_minor_amount(amount) or not _is_currency(currency):
        return None
    if not _is_id_list(evidence) or len(evidence) > MAX_EVIDENCE:
        return None
    return Finding(
        type=ftype,
        affected=frozenset(map(normalize_id, affected)),
        amount_minor=amount,
        currency=currency,
        evidence=frozenset(map(normalize_id, evidence)),
    )


def parse_truth(truth):
    """Issues sorted by issue_id. Raises ValueError on a malformed or inconsistent truth file."""
    if not isinstance(truth, dict) or not isinstance(truth.get("issues"), list):
        raise ValueError("truth must be an object with an issues list")
    if truth.get("clean") is not (not truth["issues"]):
        raise ValueError("truth.clean must be true exactly when issues is empty")
    issues = []
    for raw in truth["issues"]:
        try:
            ok_lists = _is_id_list(raw["affected_ids"]) and type(raw["evidence_sets"]) is list
            ok_lists = ok_lists and all(_is_id_list(s) for s in raw["evidence_sets"])
            if not ok_lists or type(raw["issue_id"]) is not str or type(raw["type"]) is not str:
                raise TypeError("issue_id, type and ids must be strings, ids in lists")
            all_ids = raw["affected_ids"] + [x for s in raw["evidence_sets"] for x in s]
            if any(normalize_id(x).startswith("je/") or x.startswith("je/") for x in all_ids):
                raise TypeError("je/ ids are not records (SPEC.md §3.3)")
            issue = Issue(
                issue_id=raw["issue_id"],
                type=raw["type"],
                affected=frozenset(map(normalize_id, raw["affected_ids"])),
                expected_amount_minor=raw["expected_amount_minor"],
                currency=raw["currency"],
                evidence_sets=tuple(frozenset(map(normalize_id, s)) for s in raw["evidence_sets"]),
                tolerance_minor=raw.get("tolerance_minor", DEFAULT_TOLERANCE_MINOR),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"malformed truth issue: {exc!r}") from None
        if issue.type not in ISSUE_TYPES or not issue.affected or not _is_currency(issue.currency):
            raise ValueError(f"malformed truth issue {issue.issue_id!r}")
        if type(issue.expected_amount_minor) is not int or issue.expected_amount_minor <= 0:
            raise ValueError(f"issue {issue.issue_id!r}: expected_amount_minor must be a positive int")
        if type(issue.tolerance_minor) is not int or issue.tolerance_minor < 0:
            raise ValueError(f"issue {issue.issue_id!r}: tolerance_minor must be a non-negative int")
        if issue.expected_amount_minor <= 2 * issue.tolerance_minor:
            raise ValueError(f"issue {issue.issue_id!r}: amount must exceed two tolerances (INV-20)")
        sets_ok = issue.evidence_sets and all(0 < len(s) <= MAX_EVIDENCE for s in issue.evidence_sets)
        if not sets_ok:
            raise ValueError(f"issue {issue.issue_id!r}: every evidence set needs 1..{MAX_EVIDENCE} ids")
        issues.append(issue)
    ids = [i.issue_id for i in issues]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate issue_id in truth")
    return sorted(issues, key=lambda i: i.issue_id)


def eligible(finding, issue):
    """All five conditions of the strict rule (SPEC.md §5.2)."""
    return (
        finding.type == issue.type
        and finding.affected == issue.affected
        and abs(finding.amount_minor - issue.expected_amount_minor) <= issue.tolerance_minor
        and finding.currency == issue.currency
        and any(s <= finding.evidence for s in issue.evidence_sets)
    )


def _max_matching_size(candidates, issue_indexes, used):
    """Kuhn's augmenting paths over the given issues, never touching findings in `used`."""
    owner = {}  # finding index -> issue index

    def augment(i, seen):
        for j in candidates[i]:
            if j in used or j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    return sum(1 for i in issue_indexes if augment(i, set()))


def match(findings, issues):
    """Lexicographically smallest maximum matching as (issue index, finding index) pairs.

    Issues come sorted by issue_id. Each issue in turn takes the lowest finding
    number that still allows a maximum matching of the rest (SPEC.md §5.3).
    `findings` may hold None for malformed ones.
    """
    candidates = [
        [j for j, f in enumerate(findings) if f is not None and eligible(f, issue)]
        for issue in issues
    ]
    order = list(range(len(issues)))
    used = set()
    target = _max_matching_size(candidates, order, used)
    pairs = []
    for pos, i in enumerate(order):
        later = order[pos + 1:]
        for j in candidates[i]:
            if j not in used and 1 + _max_matching_size(candidates, later, used | {j}) == target:
                pairs.append((i, j))
                used.add(j)
                target -= 1
                break
    return pairs
