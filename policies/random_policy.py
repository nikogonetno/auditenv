"""Baseline «Случайный исследователь с фиксированным seed» (case §06).

    python3 -m policies.random_policy <seed>

random.Random(seed) draws a budget of 1..30 read steps. Each step is a random read action:
list_records of a random table (sometimes from the last next_cursor of that table), get_record
of a seen or a made-up id, get_document of a seen doc_id, or calculator on two seen numbers.
Then it submits 0..3 random findings built from seen records. The same seed and the same
observations give the same actions.
"""
import random

from policies.stdio import (
    ISSUE_TYPES, TABLE_ORDER, currency_of, minor_amounts, record_id, seed_from_argv, serve, tables_in_order,
)

MAX_READ_STEPS = 30
MAX_FINDINGS = 3


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.manifest = None
        self.tables = list(TABLE_ORDER)
        self.budget = 0
        self.taken = 0
        self.records = []       # (table, record) in the order first seen
        self._seen = set()
        self.doc_ids = []
        self.numbers = []       # numeric strings seen in records, for the calculator
        self.cursors = {}       # table -> last next_cursor

    def __call__(self, message):
        observation = message.get("observation") or {}
        if self.manifest is None:
            self.manifest = observation.get("manifest") or {}
            self.tables = tables_in_order(observation) or list(TABLE_ORDER)
            self.budget = self.rng.randint(1, MAX_READ_STEPS)
        else:
            self._absorb(observation.get("result"))
        if self.taken >= self.budget:
            return self._submit()
        self.taken += 1
        return self._read_action()

    # ------------------------------------------------------------------ observations
    def _absorb(self, result):
        if not isinstance(result, dict):
            return
        table = result.get("table")
        if "records" in result:
            for record in result.get("records") or []:
                self._remember(table, record)
            self.cursors[table] = result.get("next_cursor")
        for key in ("record", "document"):
            if isinstance(result.get(key), dict):
                self._remember(table, result[key])

    def _remember(self, table, record):
        rid = record_id(table, record) if isinstance(record, dict) else None
        if rid is None or (table, rid) in self._seen:
            return
        self._seen.add((table, rid))
        self.records.append((table, record))
        if isinstance(record.get("doc_id"), str):
            self.doc_ids.append(record["doc_id"])
        self.numbers += [str(n) for n in minor_amounts(record)]
        if isinstance(record.get("rate"), str):
            self.numbers.append(record["rate"])

    # ------------------------------------------------------------------ actions
    def _read_action(self):
        rng = self.rng
        kind = rng.choice(("list_records", "get_record", "get_document", "calculator"))
        if kind == "get_record":
            if self.records and rng.random() < 0.8:
                table, record = rng.choice(self.records)
                return {"action": "get_record", "table": table, "id": record_id(table, record)}
            table = rng.choice(self.tables)
            return {"action": "get_record", "table": table, "id": f"R{rng.randrange(100000):05d}"}
        if kind == "get_document" and self.doc_ids:
            return {"action": "get_document", "doc_id": rng.choice(self.doc_ids)}
        if kind == "calculator" and len(self.numbers) >= 2:
            a, b = rng.choice(self.numbers), rng.choice(self.numbers)
            return {"action": "calculator", "expression": f"{a} {rng.choice('+-*/')} {b}"}
        table = rng.choice(self.tables)
        cursor = self.cursors.get(table)
        action = {"action": "list_records", "table": table, "limit": rng.randint(1, 50)}
        if cursor and rng.random() < 0.5:
            action["cursor"] = cursor
        return action

    def _submit(self):
        rng = self.rng
        findings = []
        for _ in range(rng.randint(0, MAX_FINDINGS) if self.records else 0):
            table, record = rng.choice(self.records)
            rid = record_id(table, record)
            amounts = minor_amounts(record)
            findings.append({
                "type": rng.choice(ISSUE_TYPES),
                "affected_ids": [rid],
                "claimed_amount_minor": rng.choice(amounts) if amounts else 0,
                "currency": currency_of(record, self.manifest),
                "evidence_ids": [rid],
                "explanation": "random",
            })
        return {"action": "submit", "findings": {"episode_id": self.manifest.get("episode_id"), "findings": findings}}


def main():
    serve(RandomPolicy(seed_from_argv()))


if __name__ == "__main__":
    main()
