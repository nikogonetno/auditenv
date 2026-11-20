"""Golden v0.1 seen through schema v2, the way QA converted it for qa_reward.py.

v0.1 truth uses amount_minor, one evidence list and je/ ids for fx_rate and
period_cutoff. v2 wants expected_amount_minor, evidence_sets and the row_id of
the carrier line: 1030 for a payment, 1210 or 3310 for an invoice. Remove this
module once golden v0.2 from A lands.
"""
import csv
import json
from pathlib import Path

EPISODE = Path(__file__).resolve().parent / "fixtures" / "golden_v0.1" / "episode"
CARRIER_ACCOUNTS = {"payment": {"1030"}, "invoice": {"1210", "3310"}}
RECORD_FILES = {
    "invoice": ("invoices.csv", "invoice_id"),
    "payment": ("payments.csv", "payment_id"),
    "ledger": ("ledger.csv", "row_id"),
    "allocation": ("allocations.csv", "allocation_id"),
}


def read_csv(name):
    with open(EPISODE / "public" / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def carrier_row(je_id):
    rows = [r for r in read_csv("ledger.csv") if r["je_id"] == je_id]
    (row,) = [r for r in rows if r["account"] in CARRIER_ACCOUNTS[r["source_type"]] and not r["settles_invoice_id"]]
    return f"ledger/{row['row_id']}"


def truth_v2():
    old = json.loads((EPISODE / "private" / "truth.json").read_text(encoding="utf-8"))
    issues = []
    for i in old["issues"]:
        affected = [carrier_row(a[3:]) if a.startswith("je/") else a for a in i["affected_ids"]]
        issues.append({
            "issue_id": i["issue_id"],
            "type": i["type"],
            "affected_ids": sorted(affected),
            "expected_amount_minor": i["amount_minor"],
            "currency": i["currency"],
            "evidence_sets": [sorted(i["evidence"])],
            "tolerance_minor": i["tolerance_minor"],
        })
    return {"clean": False, "issues": issues}


def clean_truth():
    return {"clean": True, "issues": []}


def oracle_findings(truth):
    return [
        {
            "type": i["type"],
            "affected_ids": list(i["affected_ids"]),
            "claimed_amount_minor": i["expected_amount_minor"],
            "currency": i["currency"],
            "evidence_ids": list(i["evidence_sets"][0]),
            "explanation": f"oracle: {i['type']}",
        }
        for i in truth["issues"]
    ]


def public_record_ids():
    """All 75 addressable records of the golden book, with table prefix."""
    return [f"{table}/{r[col]}" for table, (name, col) in RECORD_FILES.items() for r in read_csv(name)]
