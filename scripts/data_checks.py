"""Measurements quoted in DESIGN.md and README.md (docs/tasks/B_saturday.md item 5). Reads only, changes nothing.

    python3 scripts/data_checks.py [--seeds 60]

1. Reconcile false positives on episodes/train and episodes/dev, by the look-alike that causes them.
2. Share of documents on weekends and holidays in generated books of usual and of small size.
3. Entry delay of injected fx_rate documents against clean invoices with a delay of at least one day.
"""
import argparse
import collections
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auditenv.episode import load_episode  # noqa: E402
from auditenv.generate import generate_episode  # noqa: E402
from auditenv.grader import score  # noqa: E402
from policies.reconcile import reconcile  # noqa: E402


def company_tz(policy):
    return dt.datetime.fromisoformat("2026-01-01T00:00:00" + policy["company_utc_offset"]).tzinfo


def off_day(policy, when):
    day = dt.datetime.fromisoformat(when).astimezone(company_tz(policy)).date()
    return day.isoweekday() in policy["fx"]["weekend_days"] or day.isoformat() in policy["fx"]["holidays"]


def reconcile_false_positives():
    """{(finding type, look-alike): count} over the committed train and dev episodes."""
    found = collections.Counter()
    for directory in sorted((ROOT / "episodes").glob("*/[0-9]*")):
        episode = load_episode(directory)
        truth = episode.load_truth()
        sent = reconcile(episode.manifest, episode.tables)       # what the policy submits after reading all
        matched = {m["finding_index"] for m in score(sent, truth, 0, episode.aliases)["matched"]}
        invoices, payments, ledger = (episode.index[t] for t in ("invoices", "payments", "ledger"))
        twins = {t["kind"]: {r.split("/")[1] for r in t["records"]}
                 for t in json.loads((directory / "private" / "provenance.json").read_text())["twins"]}
        for n, finding in enumerate(sent):
            if n in matched:
                continue
            ids = [x.split("/")[1] for x in finding["affected_ids"]]
            kind = finding["type"]
            if kind == "duplicate_invoice":
                bill = invoices[ids[0]]
                cause = "monthly subscription" if bill["service_from"] != bill["service_to"] else "other"
            elif kind in ("fx_rate", "period_cutoff"):
                row = ledger[ids[0]]
                doc = invoices[row["invoice_id"]] if row["invoice_id"] else payments[row["payment_id"]]
                when = doc.get("issued_at") or doc["timestamp"]
                if kind == "fx_rate":
                    cause = "document on an off day" if off_day(episode.policy, when) else "other"
                else:
                    cause = "midnight UTC " + ("invoice" if row["invoice_id"] else "payment")
            else:
                pay = next(i for i in ids if i in payments)
                cause = ("C2 trap" if pay in twins.get("c2_trap", ()) else
                         "payment naming both copies of a duplicate" if len(payments[pay]["invoice_refs"] or []) > 1
                         else "other")
            found[(kind, cause)] += 1
    return found


def off_day_shares(config, seeds, start):
    shares = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(start, start + seeds):
            out = Path(tmp) / str(seed)
            generate_episode(seed, config, str(out))
            policy = json.loads((out / "public" / "policy.json").read_text())
            rows = [json.loads(x) for name in ("invoices.jsonl", "payments.jsonl")
                    for x in (out / "public" / name).read_text().splitlines()]
            shares.append(sum(off_day(policy, r.get("issued_at") or r["timestamp"]) for r in rows) / len(rows))
    return shares


def entry_delays(seeds, start):
    """(fx_rate targets, clean invoices with delay >= 1): Counter of '1-3' and '4-10' days."""
    target, clean = collections.Counter(), collections.Counter()
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(start, start + seeds):
            out = Path(tmp) / str(seed)
            generate_episode(seed, {"errors": {"fx_rate": 3}}, str(out))
            policy = json.loads((out / "public" / "policy.json").read_text())
            tz = company_tz(policy)
            truth = json.loads((out / "private" / "truth.json").read_text())
            hit = {x.split("/")[1] for i in truth["issues"] for x in i["evidence_sets"][0] if x.startswith("invoice/")}
            for line in (out / "public" / "invoices.jsonl").read_text().splitlines():
                inv = json.loads(line)
                lag = (dt.datetime.fromisoformat(inv["entered_at"]).astimezone(tz).date()
                       - dt.datetime.fromisoformat(inv["issued_at"]).astimezone(tz).date()).days
                if lag >= 1:
                    (target if inv["invoice_id"] in hit else clean)["1-3" if lag <= 3 else "4-10"] += 1
    return target, clean


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=60)
    args = parser.parse_args(argv)
    print("1. reconcile false positives on episodes/train and episodes/dev, by cause")
    for (kind, cause), count in sorted(reconcile_false_positives().items()):
        print(f"   {kind:20} {cause:45} {count}")
    print("2. documents on weekends and holidays")
    for name, config in (("150-200 documents, default config", {}), ("20-30 documents, clean", {"documents": [20, 30], "errors": {}})):
        shares = off_day_shares(config, args.seeds, 5000)
        outside = sum(not 0.02 <= s <= 0.06 for s in shares)
        print(f"   {name:35} min {min(shares):.1%} max {max(shares):.1%} outside 2-6%: {outside} of {len(shares)}")
    target, clean = entry_delays(args.seeds, 6000)
    print("3. entry delay of fx_rate targets vs clean invoices with a delay of at least one day")
    for name, counts in (("fx_rate targets", target), ("clean invoices", clean)):
        total = sum(counts.values())
        print(f"   {name:16} n={total} 1-3 days {counts['1-3'] / total:.0%} 4-10 days {counts['4-10'] / total:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
