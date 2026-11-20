"""The verifier's own id normalization, written from SPEC §3.3 and §5.2 only.

Kept apart from the grader's copy on purpose: a bug shared by both would let a broken truth score
1.0 and pass verification. tests/test_verify.py checks the two agree on edge cases.
"""

# SPEC §3.3 and §5.2: the eight table prefixes a reference may carry.
KNOWN_PREFIXES = ("allocation", "counterparty", "doc", "fx", "invoice", "ledger", "line", "payment")


def split_id(ref):
    """'invoice/1TNM24' -> ('invoice', '1TNM24'). A bare id or an unknown prefix -> (None, ref).
    SPEC is silent on 'invoice/' with nothing after the slash: it stays as is, an empty id is no id."""
    head, sep, tail = ref.partition("/")
    if sep and tail and head in KNOWN_PREFIXES:
        return head, tail
    return None, ref


def normalize_id(ref):
    """A known prefix is stripped once. Everything else, je/… and case included, stays as is."""
    return split_id(ref)[1]
