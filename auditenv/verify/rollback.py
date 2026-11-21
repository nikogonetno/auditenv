"""The book before the errors (docs/tasks/V_rollback.md, case §10 «бухгалтерские равенства до внесения ошибки»).

private/provenance.json lists every change of every error with its before and after. Undoing them on a
copy gives the book the errors were put into. With an empty truth the strict verifier must find nothing
there: every rule holds, and no error is left that provenance does not describe.

    rollback_episode(src, dst)    copy src to dst and undo the changes there, problems as a list
    verify_rollback(src)          the same in a temporary folder, then verify_episode on the copy

Undo order: errors from the highest `order` down, changes inside one error from the last up.
  after "created"     the record goes
  before "removed"    the record comes back from its `row`; without `row` it cannot come back
  a field             its value must equal `after` and becomes `before`
"""
import csv
import json
import shutil
import tempfile
from pathlib import Path

from ..episode import TABLES
from . import verify_episode
from .book import PREFIX_TABLE
from .ids import split_id

CREATED, REMOVED = "created", "removed"


class RollbackError(ValueError):
    """provenance does not fit the files, so this change cannot be undone."""


class _Table:
    """One public table as typed records in file order, the way auditenv.episode reads it."""

    def __init__(self, path, id_col):
        self.path, self.id_col, self.csv = path, id_col, path.suffix == ".csv"
        if self.csv:
            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                self.columns = list(reader.fieldnames or [])
                self.rows = [{k: _from_cell(k, v) for k, v in row.items()} for row in reader]
        else:
            with open(path, encoding="utf-8") as f:
                self.rows = [json.loads(line) for line in f if line.strip()]
            self.columns = list(self.rows[0]) if self.rows else []

    def find(self, rid):
        for i, row in enumerate(self.rows):
            if row.get(self.id_col) == rid:
                return i
        return None

    def insert(self, row):
        # ledger.csv is ordered by je_id: the row goes after the last row of its entry or before the next one.
        if self.id_col == "row_id":
            key = row.get("je_id") or ""
            at = next((i for i, r in enumerate(self.rows) if (r.get("je_id") or "") > key), len(self.rows))
            self.rows.insert(at, row)
        else:
            self.rows.append(row)

    def write(self):
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            if self.csv:
                writer = csv.DictWriter(f, fieldnames=self.columns, lineterminator="\n")
                writer.writeheader()
                for row in self.rows:
                    writer.writerow({k: _to_cell(row.get(k)) for k in self.columns})
            else:
                for row in self.rows:
                    f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _from_cell(column, value):
    if value == "":
        return None
    return int(value) if column.endswith("_minor") else value


def _to_cell(value):
    return "" if value is None else str(value)


def _undo(tables, change):
    if not isinstance(change, dict) or not {"record", "field", "before", "after"} <= change.keys():
        raise RollbackError(f"change {change!r} lacks record, field, before or after")
    ref = change["record"]
    prefix, rid = split_id(ref) if type(ref) is str else (None, ref)
    table = tables.get(PREFIX_TABLE.get(prefix))
    if table is None:
        raise RollbackError(f"record {ref!r} names no public table")
    at = table.find(rid)
    if change["after"] == CREATED:
        if at is None:
            raise RollbackError(f"{ref} was created, but it is not in {table.path.name}")
        del table.rows[at]
    elif change["before"] == REMOVED:
        row = change.get("row")
        if at is not None:
            raise RollbackError(f"{ref} was removed, but it is still in {table.path.name}")
        if not isinstance(row, dict):
            raise RollbackError(f"{ref} was removed and provenance has no row to bring it back")
        if row.get(table.id_col) != rid or sorted(row) != sorted(table.columns):
            raise RollbackError(f"{ref}: its row has {table.id_col} {row.get(table.id_col)!r} and fields "
                                f"{sorted(set(row) ^ set(table.columns))} differ from {table.path.name}")
        table.insert(dict(row))
    else:
        field = change["field"]
        if at is None:
            raise RollbackError(f"{ref}.{field} was changed, but {ref} is not in {table.path.name}")
        record = table.rows[at]
        if field not in table.columns:
            raise RollbackError(f"{ref}: {table.path.name} has no field {field!r}")
        if record.get(field) != change["after"]:
            raise RollbackError(f"{ref}.{field} is {record.get(field)!r}, provenance says after {change['after']!r}")
        record[field] = change["before"]


def _injections(provenance):
    injections = provenance.get("injections") if isinstance(provenance, dict) else None
    if not isinstance(injections, list) or not all(isinstance(i, dict) for i in injections):
        raise RollbackError("provenance.injections is not a list of objects")
    orders = [i.get("order") for i in injections]
    if any(type(o) is not int for o in orders) or len(set(orders)) != len(orders):
        raise RollbackError(f"provenance orders {orders} are not distinct integers")
    for inj in injections:
        if not isinstance(inj.get("changes"), list):
            raise RollbackError(f"injection order {inj['order']}: changes is not a list")
    return sorted(injections, key=lambda i: i["order"], reverse=True)


def rollback_episode(src, dst) -> list[dict]:
    """Copy the episode src to dst and undo every change of its provenance there.
    Problems as {"code": "ROLLBACK", "detail": ...}. Empty means dst holds the book before the errors
    with an empty truth. A change that cannot be undone is reported and skipped, the others still go."""
    src, dst = Path(src), Path(dst)
    try:
        shutil.copytree(src, dst)
        with open(dst / "private" / "provenance.json", encoding="utf-8") as f:
            provenance = json.load(f)
        public = dst / "public"
        tables = {name: _Table(public / filename, id_col) for name, (filename, id_col) in TABLES.items()}
        injections = _injections(provenance)
    except (OSError, ValueError) as exc:
        return [{"code": "ROLLBACK", "detail": f"{type(exc).__name__}: {exc}"}]

    problems = []
    for inj in injections:
        for change in reversed(inj["changes"]):
            try:
                _undo(tables, change)
            except RollbackError as exc:
                problems.append({"code": "ROLLBACK", "detail": f"order {inj['order']}: {exc}"})
    for table in tables.values():
        table.write()

    manifest_path = public / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"] = {name: len(table.rows) for name, table in tables.items()}
    manifest["documents"] = len(tables["invoices"].rows) + len(tables["payments"].rows)
    truth_path = dst / "private" / "truth.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    for path, data in ((manifest_path, manifest),
                       (truth_path, {"clean": True, "episode_id": truth.get("episode_id"), "issues": []})):
        path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return problems


def verify_rollback(episode_dir, strict_volume: bool = True) -> list[dict]:
    """ROLLBACK problems if the book before the errors cannot be restored, else what the strict
    verifier finds on it with an empty truth. Empty means the accounting held before the errors."""
    with tempfile.TemporaryDirectory(prefix="auditenv-rollback-") as tmp:
        restored = Path(tmp) / "episode"
        try:
            problems = rollback_episode(episode_dir, restored)
        except Exception as exc:  # noqa: BLE001 - the contract is a list, never an exception
            return [{"code": "INTERNAL", "detail": f"rollback: {type(exc).__name__}: {exc}"}]
        if problems:
            return problems
        return verify_episode(str(restored), strict_volume=strict_volume)
