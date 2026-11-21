"""Rollback of provenance: the book before the errors (docs/tasks/B_afternoon.md item 3; case §10,
«бухгалтерские равенства до внесения ошибки»).

Undo the injections in reverse order, the changes of each one from the last: delete what was created, put
back every removed ledger row from its content, and restore every changed field after checking that it holds
the recorded value. With an empty truth the strict verifier V must find nothing in the rebuilt book. This is
B's own rollback; V builds its own (docs/tasks/V_rollback.md) without reading the generator.
"""
import csv
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from auditenv.generate import _json, _jsonl, generate_episode
from auditenv.transform import permute_and_rename
from auditenv.verify import verify_episode

ROOT = Path(__file__).resolve().parent.parent
# Two configs of V's rehearsal (qa/rehearsal/configs.json, PR #13) whose rollback failed on removed rows
# without content: seed 9008 (C1 twice) and seed 9014 (rate_of_entry_date).
REHEARSAL = {
    9008: {"errors": {}, "scenarios": {"C1": 2}, "documents": [180, 200], "foreign_pct": [50, 100],
           "company_offsets": ["-03:00", "+06:00"], "twins": [20, 20]},
    9014: {"errors": {"duplicate_invoice": 2, "fx_rate": 2}, "documents": [120, 160], "foreign_pct": [40, 45],
           "company_offsets": ["+09:30"], "period": "2026-08", "twins": [20, 20]},
}
KEYS = {"ledger": "row_id", "invoice": "invoice_id", "payment": "payment_id"}


def cell(value):
    """A JSON value of a ledger row as ledger.csv writes it."""
    return "" if value is None else str(value)


def rollback(src, dst):
    """Copy the episode src to dst and undo every injection there. Returns the number of changes undone."""
    shutil.copytree(src, dst)
    public = Path(dst) / "public"
    with open(public / "ledger.csv", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns, ledger = reader.fieldnames, {r["row_id"]: r for r in reader}
    def jsonl(name, key):
        return {r[key]: r for r in map(json.loads, (public / name).read_text(encoding="utf-8").splitlines())}

    invoices, payments = jsonl("invoices.jsonl", "invoice_id"), jsonl("payments.jsonl", "payment_id")
    tables = {"ledger": ledger, "invoice": invoices, "payment": payments}
    provenance = json.loads((Path(dst) / "private" / "provenance.json").read_text(encoding="utf-8"))
    undone = 0
    for injection in sorted(provenance["injections"], key=lambda i: i["order"], reverse=True):
        for change in reversed(injection["changes"]):
            table, _, rid = change["record"].partition("/")
            records = tables[table]
            as_stored = cell if table == "ledger" else (lambda v: v)
            if change["after"] == "created":
                if rid not in records:
                    raise AssertionError(f"created {change['record']} is not in the book")
                del records[rid]
            elif change["before"] == "removed":
                row = change.get("row")
                if not isinstance(row, dict) or row.get(KEYS[table]) != rid or rid in records:
                    raise AssertionError(f"removed {change['record']}: no content, or it is still in the book")
                records[rid] = {k: as_stored(v) for k, v in row.items()}
            else:
                record, field = records[rid], change["field"]
                if record.get(field) != as_stored(change["after"]):
                    raise AssertionError(f"{change['record']}.{field} is not the recorded value after the error")
                record[field] = as_stored(change["before"])
            undone += 1
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in ledger.values():
        writer.writerow([row[c] for c in columns])
    (public / "ledger.csv").write_text(buf.getvalue(), encoding="utf-8")
    (public / "invoices.jsonl").write_text(_jsonl(invoices.values()), encoding="utf-8")
    (public / "payments.jsonl").write_text(_jsonl(payments.values()), encoding="utf-8")
    manifest = json.loads((public / "manifest.json").read_text(encoding="utf-8"))
    manifest["counts"].update(ledger=len(ledger), invoices=len(invoices), payments=len(payments))
    manifest["documents"] = len(invoices) + len(payments)
    (public / "manifest.json").write_text(_json(manifest), encoding="utf-8")
    truth = {"clean": True, "episode_id": manifest["episode_id"], "issues": []}
    (Path(dst) / "private" / "truth.json").write_text(_json(truth), encoding="utf-8")
    return undone


class Rollback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def assert_clean_before_the_errors(self, episode, name):
        undone = rollback(episode, self.root / f"{name}-before")
        self.assertEqual(verify_episode(str(self.root / f"{name}-before")), [], name)
        return undone

    def test_committed_train_and_dev_roll_back_to_a_clean_book(self):
        episodes = sorted((ROOT / "episodes").glob("*/[0-9]*"))
        self.assertEqual(len(episodes), 20)
        undone = 0
        for episode in episodes:
            with self.subTest(episode=f"{episode.parent.name}/{episode.name}"):
                undone += self.assert_clean_before_the_errors(episode, f"{episode.parent.name}-{episode.name}")
        self.assertGreater(undone, 0)

    def test_removed_rows_carry_their_content_and_roll_back(self):
        removed = 0
        for seed, config in REHEARSAL.items():
            out = self.root / str(seed)
            generate_episode(seed, config, str(out))
            provenance = json.loads((out / "private" / "provenance.json").read_text(encoding="utf-8"))
            rows = [c for i in provenance["injections"] for c in i["changes"] if c["before"] == "removed"]
            removed += len(rows)
            with self.subTest(seed=seed):
                self.assertTrue(rows)
                with open(out / "public" / "ledger.csv", encoding="utf-8", newline="") as f:
                    columns = csv.DictReader(f).fieldnames
                for change in rows:
                    self.assertEqual(sorted(change["row"]), sorted(columns))
                    self.assertEqual(f"ledger/{change['row']['row_id']}", change["record"])
                self.assert_clean_before_the_errors(out, f"r{seed}")
                copy = self.root / f"{seed}-copy"
                permute_and_rename(out, copy, "rollback")
                id_map = json.loads((copy / "private" / "id_map.json").read_text(encoding="utf-8"))["ids"]
                renamed = json.loads((copy / "private" / "provenance.json").read_text(encoding="utf-8"))
                for change in (c for i in renamed["injections"] for c in i["changes"] if c["before"] == "removed"):
                    self.assertIn(change["row"]["row_id"], id_map.values())
                    self.assertNotIn(change["row"]["je_id"], id_map)        # every id inside the row is new
                self.assert_clean_before_the_errors(copy, f"r{seed}-copy")
        self.assertGreaterEqual(removed, 3)

    def test_every_stress_config_rolls_back(self):
        configs = json.loads((ROOT / "configs" / "stress_v3.json").read_text(encoding="utf-8"))["configs"]
        for n, config in enumerate(configs):
            out = self.root / f"s{n}"
            generate_episode(700 + n, config, str(out))
            with self.subTest(config=n):
                self.assert_clean_before_the_errors(out, f"s{n}")


if __name__ == "__main__":
    unittest.main()
