"""Row permutation and id renaming (docs/tasks/B_night_stage3.md item 2; INV-22; case §06 and §10).

    permute_and_rename(episode_dir, out_dir, salt)

writes a copy of an episode whose public rows (and the line items of every invoice) are in another order
and whose ids are all new: episode_id, doc_id, invoice_id, payment_id, row_id, je_id, entity_id, line_id,
fx_id and issue_id. Business attributes (invoice_number, vendor_ref, po_ref), amounts, dates, rates, texts
and policy.json stay as they are. truth.json and provenance.json follow the new ids, so the copy's truth is
the same truth: same types, amounts and currencies, affected ids and evidence naming the same records.
private/id_map.json maps every old id to its new one.

New ids come from random.Random seeded by the salt, are unique across all tables and never reuse an old id,
so a reference to the original episode resolves to nothing in the copy. Deterministic in (episode, salt).
"""
import csv
import hashlib
import io
import json
import random
import re
from pathlib import Path

from .generate import IdSource, _json, _jsonl

LEDGER_IDS = ("row_id", "entity_id", "invoice_id", "payment_id", "je_id")
INVOICE_IDS = ("doc_id", "invoice_id", "entity_id")
PAYMENT_IDS = ("doc_id", "payment_id", "entity_id")
PREFIXES = ("allocation", "counterparty", "doc", "fx", "invoice", "ledger", "line", "payment")


def _rng(salt):
    digest = hashlib.sha256(f"auditenv-transform:{salt}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def _write_csv(rows, columns):
    """Same layout as the generator: header, then one line per row in column order."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row[c] for c in columns])
    return buf.getvalue()


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _old_ids(manifest, ledger, invoices, payments, fx, truth):
    ids = {manifest["episode_id"]}
    for row in ledger:
        ids.update(row[c] for c in LEDGER_IDS if row[c])
    for inv in invoices:
        ids.update(inv[c] for c in INVOICE_IDS if inv[c])
        ids.update(line["line_id"] for line in inv["line_items"])
    for pay in payments:
        ids.update(pay[c] for c in PAYMENT_IDS if pay[c])
        ids.update(pay["invoice_refs"] or ())
    ids.update(row["fx_id"] for row in fx)
    ids.update(issue["issue_id"] for issue in truth["issues"])
    return ids


def _provenance_ids(value):
    """Ids that only provenance names, such as a row an injection removed: every '<prefix>/<id>' string."""
    if isinstance(value, dict):
        return set().union(*(_provenance_ids(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_provenance_ids(v) for v in value)) if value else set()
    if isinstance(value, str):
        prefix, sep, rest = value.partition("/")
        if sep and prefix in PREFIXES and rest:
            return {rest}
    return set()


def _map_refs(value, id_map):
    """Every id inside a JSON value, bare or as '<prefix>/<id>', replaced by its new id."""
    if isinstance(value, dict):
        return {k: _map_refs(v, id_map) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_refs(v, id_map) for v in value]
    if isinstance(value, str):
        if value in id_map:
            return id_map[value]
        prefix, sep, rest = value.partition("/")
        if sep and prefix in PREFIXES and rest in id_map:
            return f"{prefix}/{id_map[rest]}"
    return value


def permute_and_rename(episode_dir, out_dir, salt):
    """Write the permuted and renamed copy of episode_dir into out_dir (created if needed)."""
    src, dst = Path(episode_dir), Path(out_dir)
    manifest = _read_json(src / "public" / "manifest.json")
    ledger_columns, ledger = _read_csv(src / "public" / "ledger.csv")
    fx_columns, fx = _read_csv(src / "public" / "fx.csv")
    invoices = _read_jsonl(src / "public" / "invoices.jsonl")
    payments = _read_jsonl(src / "public" / "payments.jsonl")
    truth = _read_json(src / "private" / "truth.json")
    provenance_path = src / "private" / "provenance.json"
    provenance = _read_json(provenance_path) if provenance_path.is_file() else None

    rng = _rng(salt)
    old = _old_ids(manifest, ledger, invoices, payments, fx, truth)
    if provenance is not None:
        old |= _provenance_ids(provenance)
    source = IdSource(rng)
    source.taken.update(old)                      # a new id never equals an old one
    id_map = {}
    issue_ids = {issue["issue_id"] for issue in truth["issues"]}
    for old_id in sorted(old - issue_ids):
        id_map[old_id] = source.new()

    def ids(row, columns):
        return {k: (id_map[v] if k in columns and v else v) for k, v in row.items()}

    new_ledger = [ids(row, LEDGER_IDS) for row in ledger]
    new_fx = [ids(row, ("fx_id",)) for row in fx]
    new_invoices = []
    for inv in invoices:
        record = ids(inv, INVOICE_IDS)
        lines = [{**line, "line_id": id_map[line["line_id"]]} for line in inv["line_items"]]
        rng.shuffle(lines)
        record["line_items"] = lines
        new_invoices.append(record)
    new_payments = []
    for pay in payments:
        record = ids(pay, PAYMENT_IDS)
        if pay["invoice_refs"] is not None:
            record["invoice_refs"] = sorted(id_map[r] for r in pay["invoice_refs"])
        new_payments.append(record)
    for table in (new_ledger, new_fx, new_invoices, new_payments):
        rng.shuffle(table)
    new_manifest = {**manifest, "episode_id": id_map[manifest["episode_id"]]}

    public = {
        "public/manifest.json": _json(new_manifest),
        "public/policy.json": (src / "public" / "policy.json").read_text(encoding="utf-8"),
        "public/ledger.csv": _write_csv(new_ledger, ledger_columns),
        "public/invoices.jsonl": _jsonl(new_invoices),
        "public/payments.jsonl": _jsonl(new_payments),
        "public/fx.csv": _write_csv(new_fx, fx_columns),
    }
    # New issue ids are drawn last and never appear as a token of the public text (INV-23).
    tokens = set(re.findall(r"[0-9A-Z]{6}", "".join(public[k] for k in sorted(public))))
    for old_id in sorted(issue_ids):
        id_map[old_id] = source.new(avoid=tokens)

    new_truth = _map_refs(truth, id_map)
    for issue in new_truth["issues"]:
        issue["affected_ids"] = sorted(issue["affected_ids"])
        issue["evidence_sets"] = [sorted(s) for s in issue["evidence_sets"]]
    new_truth["issues"].sort(key=lambda i: i["issue_id"])
    files = {**public, "private/truth.json": _json(new_truth),
             "private/id_map.json": _json({"salt": salt, "source_episode_id": manifest["episode_id"],
                                           "ids": dict(sorted(id_map.items()))})}
    if provenance is not None:
        files["private/provenance.json"] = _json({**_map_refs(provenance, id_map),
                                                  "transform": {"salt": salt,
                                                                "source_episode_id": manifest["episode_id"]}})
    for rel in sorted(files):
        path = dst / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(files[rel])
