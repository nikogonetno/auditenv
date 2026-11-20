"""Baseline «Атака "пометить всё"» (case §05, §06).

    python3 -m policies.flag_all [<seed, ignored>]

Pages through every table with list_records until next_cursor is null, and stops paging at 90
steps so the submit stays inside the 100-step budget. Then it submits up to 30 findings: for the
records in table order and for each of the four types in turn, a finding that names the record
alone, with the canonical `<prefix>/<id>` form so the attack is as strong as a naive one can be.
The case requires this to score no better than silence.
"""
from policies.stdio import (
    ID_PREFIXES, ISSUE_TYPES, currency_of, minor_amounts, record_id, serve, tables_in_order,
)

PAGE_LIMIT = 50
MAX_PAGING_STEPS = 90
MAX_FINDINGS = 30


class FlagAllPolicy:
    def __init__(self):
        self.manifest = None
        self.pending = []       # tables still to page through
        self.cursor = None
        self.steps = 0
        self.records = []       # (table, record) in table order, then page order

    def __call__(self, message):
        observation = message.get("observation") or {}
        if self.manifest is None:
            self.manifest = observation.get("manifest") or {}
            self.pending = tables_in_order(observation)
        else:
            self._absorb(observation)
        if not self.pending or self.steps >= MAX_PAGING_STEPS:
            return self._submit()
        self.steps += 1
        action = {"action": "list_records", "table": self.pending[0], "limit": PAGE_LIMIT}
        if self.cursor is not None:
            action["cursor"] = self.cursor
        return action

    def _absorb(self, observation):
        result = observation.get("result")
        table = self.pending[0] if self.pending else None
        if not isinstance(result, dict) or result.get("table") != table:
            if self.pending:  # an error: give up on this table
                self.pending.pop(0)
            self.cursor = None
            return
        for record in result.get("records") or []:
            if isinstance(record, dict) and record_id(table, record):
                self.records.append((table, record))
        self.cursor = result.get("next_cursor")
        if self.cursor is None:
            self.pending.pop(0)

    def _submit(self):
        findings = []
        pairs = ((table, record, issue_type) for table, record in self.records for issue_type in ISSUE_TYPES)
        for table, record, issue_type in pairs:
            if len(findings) == MAX_FINDINGS:
                break
            ref = f"{ID_PREFIXES.get(table, table)}/{record_id(table, record)}"
            positive = [n for n in minor_amounts(record) if n > 0]
            findings.append({
                "type": issue_type,
                "affected_ids": [ref],
                "claimed_amount_minor": positive[0] if positive else 0,
                "currency": currency_of(record, self.manifest),
                "evidence_ids": [ref],
                "explanation": "flag all",
            })
        return {"action": "submit", "findings": {"episode_id": self.manifest.get("episode_id"), "findings": findings}}


def main():
    serve(FlagAllPolicy())


if __name__ == "__main__":
    main()
