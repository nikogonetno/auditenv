"""A tiny episode in the case §03 layout, written to a temp dir, for env, server and policy tests."""
import csv
import json
from pathlib import Path

EPISODE_ID = "EPTEST"
LEDGER_COLUMNS = ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
                  "entity_id", "invoice_id", "payment_id", "je_id", "memo"]


def ledger_row(k, account, debit, credit, invoice_id=None, payment_id=None):
    return {"row_id": f"R{k:05d}", "timestamp": "2026-09-10T12:00:00+05:00", "account": account,
            "debit_minor": debit, "credit_minor": credit, "currency": "KZT", "entity_id": "CP0001",
            "invoice_id": invoice_id, "payment_id": payment_id, "je_id": f"J{k // 2:05d}", "memo": None}


def write_episode(root, ledger_rows=120, truth=None):
    public, private = Path(root) / "public", Path(root) / "private"
    public.mkdir(parents=True)
    private.mkdir()
    invoices = [
        {"doc_id": "DOCINV", "invoice_id": "INV001", "entity_id": "CP0001", "issued_at": "2026-09-10T10:00:00+05:00",
         "due_at": "2026-09-24T00:00:00+05:00", "currency": "KZT", "total_minor": 500000,
         "line_items": [{"line_id": "LN0001", "description": "Freight", "qty": 1, "unit_price_minor": 500000, "amount_minor": 500000}]},
        {"doc_id": "DOCDUP", "invoice_id": "INV002", "entity_id": "CP0001", "issued_at": "2026-09-10T10:00:00+05:00",
         "due_at": "2026-09-24T00:00:00+05:00", "currency": "KZT", "total_minor": 500000,
         "line_items": [{"line_id": "LN0002", "description": "Freight", "qty": 1, "unit_price_minor": 500000, "amount_minor": 500000}]},
    ]
    payments = [{"doc_id": "DOCPAY", "payment_id": "PAY001", "timestamp": "2026-09-12T05:00:00+00:00", "currency": "KZT",
                 "amount_minor": 500000, "invoice_refs": ["INV001"]}]
    rows = [ledger_row(k, "1210" if k % 2 else "6010", 500000 if k % 2 else 0, 0 if k % 2 else 500000,
                       invoice_id="INV001" if k < 2 else None) for k in range(ledger_rows)]
    with open(public / "ledger.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, LEDGER_COLUMNS, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: "" if v is None else v for k, v in row.items()})
    with open(public / "fx.csv", "w", encoding="utf-8", newline="") as f:
        f.write("date,source_currency,target_currency,rate,fx_id\n2026-09-10,USD,KZT,513.31,FX0001\n")
    for name, docs in (("invoices.jsonl", invoices), ("payments.jsonl", payments)):
        (public / name).write_text("".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")
    manifest = {"schema_version": "test", "episode_id": EPISODE_ID, "base_currency": "KZT", "period_start": "2026-09-01",
                "period_end": "2026-09-30", "timezone": "+05:00"}
    (public / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (public / "policy.json").write_text(json.dumps({"base_currency": "KZT"}), encoding="utf-8")
    truth = truth or {"clean": False, "episode_id": EPISODE_ID, "issues": [{
        "issue_id": "ISS001", "type": "duplicate_invoice", "affected_ids": ["invoice/INV001", "invoice/INV002"],
        "expected_amount_minor": 500000, "currency": "KZT", "evidence_sets": [["invoice/INV001", "invoice/INV002"]]}]}
    (private / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    return root


GOOD_FINDING = {"type": "duplicate_invoice", "affected_ids": ["invoice/INV001", "invoice/INV002"],
                "claimed_amount_minor": 500000, "currency": "KZT", "evidence_ids": ["INV001", "INV002"],
                "explanation": "same vendor, amount and lines"}


def submit(findings, episode_id=EPISODE_ID):
    return {"action": "submit", "findings": {"episode_id": episode_id, "findings": findings}}
