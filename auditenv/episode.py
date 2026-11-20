"""One episode in the case §03 layout, loaded read-only.

    <dir>/public/manifest.json, policy.json, ledger.csv, invoices.jsonl, payments.jsonl, fx.csv
    <dir>/private/truth.json          read only when the episode is graded

Public records are plain JSON values: money columns (*_minor) are ints, empty CSV cells
are null (case §03: missing links are explicit nulls), everything else is a string.
"""
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from .grader.matcher import normalize_id

# table name -> (file, id column). Order is the order list_records and the manifest use.
TABLES = {
    "ledger": ("ledger.csv", "row_id"),
    "invoices": ("invoices.jsonl", "invoice_id"),
    "payments": ("payments.jsonl", "payment_id"),
    "fx": ("fx.csv", "fx_id"),
}
DOCUMENT_TABLES = ("invoices", "payments")


class EpisodeError(ValueError):
    """The episode files do not follow case §03. A generator or packaging bug, never agent input."""


@dataclass
class Episode:
    episode_id: str
    manifest: dict
    policy: dict
    tables: dict                      # table -> list of records, in file order
    index: dict                       # table -> {record id: record}
    documents: dict                   # doc_id -> (table, record)
    aliases: dict                     # doc_id -> invoice_id or payment_id, for the grader
    _truth_path: Path = field(default=None, repr=False)
    _truth: dict = field(default=None, repr=False)

    def load_truth(self):
        """Private truth for the grader. Called only at submit or timeout, never for observations."""
        if self._truth is not None:
            return self._truth
        with open(self._truth_path, encoding="utf-8") as f:
            return json.load(f)


def _read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [{k: _csv_value(k, v) for k, v in row.items()} for row in rows]


def _csv_value(column, value):
    if value == "":
        return None
    if column.endswith("_minor"):
        return int(value)
    return value


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_episode(manifest, policy, tables, truth=None, truth_path=None):
    """Index public tables and check ids. Used by the loader and by in-memory generators."""
    episode_id = manifest.get("episode_id")
    if type(episode_id) is not str or not episode_id:
        raise EpisodeError("manifest.episode_id missing")
    index, documents, aliases = {}, {}, {}
    for name, (_, id_col) in TABLES.items():
        rows = tables.get(name)
        if not isinstance(rows, list):
            raise EpisodeError(f"table {name} missing")
        index[name] = {}
        for row in rows:
            rid = row.get(id_col)
            if type(rid) is not str or not rid or normalize_id(rid) != rid:
                raise EpisodeError(f"{name}: bad {id_col}")
            if rid in index[name]:
                raise EpisodeError(f"{name}: duplicate {id_col}")
            index[name][rid] = row
        if name in DOCUMENT_TABLES:
            for row in rows:
                doc_id = row.get("doc_id")
                if type(doc_id) is not str or not doc_id or doc_id in documents:
                    raise EpisodeError(f"{name}: bad or duplicate doc_id")
                documents[doc_id] = (name, row)
                if doc_id != row[id_col]:
                    aliases[doc_id] = row[id_col]
    return Episode(episode_id, manifest, policy, tables, index, documents, aliases,
                   _truth_path=truth_path, _truth=truth)


def load_episode(directory):
    """Load an episode directory in the case §03 layout. Raises EpisodeError on a broken layout."""
    root = Path(directory)
    public = root / "public"
    try:
        manifest = _read_json(public / "manifest.json")
        policy = _read_json(public / "policy.json")
        tables = {}
        for name, (filename, _) in TABLES.items():
            path = public / filename
            tables[name] = _read_csv(path) if filename.endswith(".csv") else _read_jsonl(path)
    except (OSError, ValueError) as exc:
        raise EpisodeError(f"cannot read episode: {type(exc).__name__}") from None
    truth_path = root / "private" / "truth.json"
    if not truth_path.is_file():
        raise EpisodeError("private/truth.json missing")
    return build_episode(manifest, policy, tables, truth_path=truth_path)
