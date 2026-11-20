"""Diagnostic «smart spray» (PM 21.11, E-32; docs/tasks/B_night_stage3.md item 8). Not one of the four baselines.

    python3 -m policies.spray [<seed, ignored>]

It reads the book like the reconcile policy and then submits, as well-formed period_cutoff findings, the
carrier rows posted within SPRAY_DAYS of the period end on either side, nearest to the boundary first, at
most 30: the boundary spray QA modelled (SPEC §8.7). It applies no accounting rule, so it measures what
guessing at the period boundary earns under the case formula. Ties go by record content, never by id.
"""
from datetime import timedelta

from policies.reconcile import MAX_FINDINGS, Book, ReconcilePolicy, _base_amount, _finding, _local_date, _time
from policies.stdio import serve

SPRAY_DAYS = 2


def spray(manifest, tables):
    book = Book(manifest, tables)
    _, end = book.period
    if book.base is None or end is None:
        return []
    first = end + timedelta(days=1)
    candidates = []
    for row, document_ref, _ in book.carriers():
        day = _local_date(_time(row.get("timestamp")), book.tz)
        if day is None:
            continue
        distance = (end - day).days if day <= end else (day - first).days
        if distance < SPRAY_DAYS:
            content = tuple(repr(row.get(c)) for c in ("timestamp", "account", "debit_minor", "credit_minor", "memo"))
            candidates.append((distance, content, row, document_ref))
    candidates.sort(key=lambda c: (c[0], c[1]))
    findings = [_finding("period_cutoff", [f"ledger/{row['row_id']}"], _base_amount(row), book.base,
                         [document_ref, f"ledger/{row['row_id']}"]) for _, _, row, document_ref in candidates]
    return [f for f in findings if f is not None][:MAX_FINDINGS]


class SprayPolicy(ReconcilePolicy):
    def _submit(self):
        return {"action": "submit",
                "findings": {"episode_id": self.manifest.get("episode_id"), "findings": spray(self.manifest, self.tables)}}


def main():
    serve(SprayPolicy())


if __name__ == "__main__":
    main()
