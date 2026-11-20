"""Checks of the book and of truth.json. Each one appends {"code", "detail"} problems and returns nothing
the caller has to trust: a check that cannot read a value reports it and moves on."""
import datetime as dt
import re
from pathlib import Path

from .book import ISSUE_TYPES, UNAPPLIED, BookError, minor, parse_date
from .evidence import amount_from_set, missing_from_set

PUBLIC_FILES = ("fx.csv", "invoices.jsonl", "ledger.csv", "manifest.json", "payments.jsonl", "policy.json")
MAX_EVIDENCE = 10
NEIGHBOUR_ROWS = 10
# CLAUDE.md and INV-23: words that must never reach a public file. Matched case-insensitively.
PRIVATE_WORDS = ("truth", "provenance", "inject", "twin", "scenario", "issue_id")
_CURRENCY = re.compile(r"[A-Z]{3}")


def add(problems, code, detail):
    problems.append({"code": code, "detail": detail})


# --- truth.json --------------------------------------------------------------------------------

def well_formed_issues(truth, episode_id, problems):
    """Issues whose shape the other checks can rely on. Everything else is a TRUTH problem."""
    if type(truth) is not dict or type(truth.get("issues")) is not list:
        add(problems, "TRUTH", "truth.json is not an object with an issues list")
        return []
    issues = truth["issues"]
    if type(truth.get("clean")) is not bool or truth["clean"] != (not issues):
        add(problems, "TRUTH", f"clean is {truth.get('clean')!r} with {len(issues)} issues")
    if truth.get("episode_id") != episode_id:
        add(problems, "TRUTH", f"episode_id {truth.get('episode_id')!r}, manifest has {episode_id!r}")
    seen, ok = set(), []
    for n, issue in enumerate(issues):
        label = issue.get("issue_id") if type(issue) is dict and type(issue.get("issue_id")) is str else f"#{n}"
        bad = _shape_errors(issue)
        if not bad and issue["issue_id"] in seen:
            bad = ["issue_id repeated"]
        if bad:
            add(problems, "TRUTH", f"issue {label}: {', '.join(bad)}")
            continue
        seen.add(issue["issue_id"])
        if issue["affected_ids"] != sorted(set(issue["affected_ids"])):
            add(problems, "TRUTH", f"issue {label}: affected_ids not a sorted set")
        ok.append(issue)
    return sorted(ok, key=lambda i: i["issue_id"])


def _shape_errors(issue):
    if type(issue) is not dict:
        return ["not an object"]
    bad = []
    if type(issue.get("issue_id")) is not str or not issue["issue_id"]:
        bad.append("issue_id")
    if issue.get("type") not in ISSUE_TYPES or type(issue.get("type")) is not str:
        bad.append("type")
    ids = issue.get("affected_ids")
    if type(ids) is not list or not ids or any(type(x) is not str or not x for x in ids):
        bad.append("affected_ids")
    if type(issue.get("expected_amount_minor")) is not int:
        bad.append("expected_amount_minor")
    if type(issue.get("currency")) is not str or not _CURRENCY.fullmatch(issue["currency"]):
        bad.append("currency")
    if type(issue.get("tolerance_minor", 1)) is not int or issue.get("tolerance_minor", 1) < 0:
        bad.append("tolerance_minor")
    sets = issue.get("evidence_sets")
    if type(sets) is not list or not sets or any(
            type(s) is not list or not s or any(type(x) is not str for x in s) for s in sets):
        bad.append("evidence_sets")
    return bad


def tolerance(issue):
    return issue.get("tolerance_minor", 1)


def check_inv20(issues, problems):
    """INV-20: every amount is positive and above two tolerances."""
    for issue in issues:
        amount, tol = issue["expected_amount_minor"], tolerance(issue)
        if amount <= 2 * tol:
            add(problems, "INV-20", f"issue {issue['issue_id']}: amount {amount} is not above 2 x {tol}")


# --- INV-17 and INV-18 -------------------------------------------------------------------------

def check_inv17(issues, book, problems):
    """INV-17: every evidence set alone gives the amount within tolerance, in the issue currency."""
    for issue in issues:
        iid, kind = issue["issue_id"], issue["type"]
        affected = {book.canon(x) for x in issue["affected_ids"]}
        for k, refs in enumerate(issue["evidence_sets"]):
            where = f"issue {iid} {kind} set {k}"
            if len(refs) > MAX_EVIDENCE:
                add(problems, "INV-17", f"{where}: {len(refs)} ids, the limit is {MAX_EVIDENCE}")
            items, broken = [], False
            for ref in refs:
                try:
                    items.append(book.resolve(ref))
                except BookError as exc:
                    add(problems, "INV-17", f"{where}: {exc}")
                    broken = True
            if broken:
                continue
            held = {book.canon(ref) for ref in refs}
            if not affected <= held:
                add(problems, "INV-17", f"{where}: affected {sorted(affected - held)} not in the set")
            try:
                amount, currency = amount_from_set(kind, items, book.policy)
            except BookError as exc:
                add(problems, "INV-17", f"{where}: {exc}")
                continue
            if abs(amount - issue["expected_amount_minor"]) > tolerance(issue):
                add(problems, "INV-17", f"{where}: recomputed {amount}, truth says {issue['expected_amount_minor']} "
                                        f"(tolerance {tolerance(issue)})")
            if currency != issue["currency"]:
                add(problems, "INV-17", f"{where}: recomputed in {currency}, truth says {issue['currency']}")
            missing = missing_from_set(kind, items, book)
            if missing:
                add(problems, "INV-17", f"{where}: leaves out {missing}")


def check_inv18(issues, violations, book, problems):
    """INV-18: rule violations of the public book and truth issues match one to one by type and ids.
    Truth issues are taken in issue_id order. Two issues with one type and affected set (QA D-4, SPEC §6)
    break the one to one match: the later one is reported as a duplicate truth issue."""
    left = {v.key: v for v in violations}
    first = {}
    for issue in issues:
        key = (issue["type"], frozenset(book.canon(x) for x in issue["affected_ids"]))
        if key in first:
            add(problems, "INV-18", f"duplicate truth issue: {first[key]} and {issue['issue_id']} are both "
                                    f"{key[0]} on {sorted(key[1])}")
            continue
        first[key] = issue["issue_id"]
        found = left.pop(key, None)
        if found is None:
            add(problems, "INV-18", f"truth issue {issue['issue_id']} {issue['type']} {sorted(key[1])} "
                                    f"is not a rule violation in the public files")
            continue
        if abs(found.amount - issue["expected_amount_minor"]) > tolerance(issue) or found.currency != issue["currency"]:
            add(problems, "INV-17", f"issue {issue['issue_id']} {issue['type']}: the whole book gives "
                                    f"{found.amount} {found.currency}, truth says "
                                    f"{issue['expected_amount_minor']} {issue['currency']}")
    for v in sorted(left.values(), key=lambda v: (v.type, v.ids)):
        add(problems, "INV-18", f"{v.type} {list(v.ids)} violates the policy but is not in truth: "
                                f"{v.note}, amount {v.amount} {v.currency}")


# --- the book ----------------------------------------------------------------------------------

def check_inv01_02(book, problems):
    """INV-01: every je_id balances, one side per row, base currency. INV-02: the trial balance is zero."""
    totals, debit_all, credit_all = {}, 0, 0
    for row in book.ledger:
        rid = row["row_id"]
        try:
            debit, credit = minor(row.get("debit_minor"), "debit_minor"), minor(row.get("credit_minor"), "credit_minor")
        except BookError as exc:
            add(problems, "INV-01", f"ledger {rid}: {exc}")
            continue
        if (debit > 0) == (credit > 0):
            add(problems, "INV-01", f"ledger {rid}: debit {debit} and credit {credit}, exactly one side must be filled")
        if row.get("currency") != book.base:
            add(problems, "INV-01", f"ledger {rid}: currency {row.get('currency')!r}, base is {book.base}")
        if row.get("account") not in book.policy.accounts:
            add(problems, "INV-08", f"ledger {rid}: account {row.get('account')!r} not in policy.accounts")
        je = row.get("je_id")
        if je is None:
            add(problems, "INV-01", f"ledger {rid}: no je_id")
        else:
            d, c = totals.get(je, (0, 0))
            totals[je] = (d + debit, c + credit)
        debit_all += debit
        credit_all += credit
    for je, (d, c) in sorted(totals.items()):
        if d != c:
            add(problems, "INV-01", f"je {je}: debit {d}, credit {c}")
    if debit_all != credit_all:
        add(problems, "INV-02", f"trial balance: debit {debit_all}, credit {credit_all}")


def check_inv03(book, issues, violations, strict_volume, problems):
    """INV-03: every row inside policy.ledger_window. With strict_volume also at least 10 rows of each
    neighbouring period that are not rows of a period error."""
    try:
        window = book.raw_policy["ledger_window"]
        first, last = parse_date(window["from"]), parse_date(window["to"])
        start, end = parse_date(book.manifest["period_start"]), parse_date(book.manifest["period_end"])
    except (BookError, KeyError, TypeError) as exc:
        add(problems, "SCHEMA", f"ledger window or period: {exc}")
        return
    moved = {v.ids[0] for v in violations if v.type == "period_cutoff"}
    moved |= {book.canon(x) for i in issues if i["type"] == "period_cutoff" for x in i["affected_ids"]}
    moved_je = {book.index["ledger"][rid].get("je_id") for rid in moved if rid in book.index["ledger"]}
    before = after = 0
    for row in book.ledger:
        try:
            day = book.policy.local_date(row.get("timestamp"))
        except BookError as exc:
            add(problems, "SCHEMA", f"ledger {row['row_id']}: {exc}")
            continue
        if not first <= day <= last:
            add(problems, "INV-03", f"ledger {row['row_id']}: posted {day}, window {first}..{last}")
        if row.get("je_id") in moved_je:
            continue
        before += day < start
        after += day > end
    if strict_volume and min(before, after) < NEIGHBOUR_ROWS:
        add(problems, "INV-03", f"{before} rows before and {after} after the period, need {NEIGHBOUR_ROWS} each side")


def check_layout(book, episode_dir, problems):
    """SPEC §3.2: only the six public files; manifest agrees with policy and with the tables."""
    public = Path(episode_dir) / "public"
    extra = sorted(p.name for p in public.iterdir() if p.name not in PUBLIC_FILES)
    if extra:
        add(problems, "SCHEMA", f"unexpected public files {extra}")
    m, p = book.manifest, book.raw_policy
    counts = {"ledger": len(book.ledger), "invoices": len(book.invoices), "payments": len(book.payments),
              "fx": len(book.fx)}
    if m.get("counts") != counts:
        add(problems, "SCHEMA", f"manifest.counts {m.get('counts')}, tables have {counts}")
    if m.get("documents") != counts["invoices"] + counts["payments"]:
        add(problems, "SCHEMA", f"manifest.documents {m.get('documents')}, tables have "
                                f"{counts['invoices'] + counts['payments']}")
    pairs = (("base_currency", m.get("base_currency"), p.get("base_currency")),
             ("timezone", m.get("timezone"), p.get("company_utc_offset")),
             ("period_start", m.get("period_start"), (p.get("period") or {}).get("start_local")),
             ("period_end", m.get("period_end"), (p.get("period") or {}).get("end_local")))
    for name, in_manifest, in_policy in pairs:
        if in_manifest != in_policy:
            add(problems, "SCHEMA", f"manifest.{name} {in_manifest!r}, policy has {in_policy!r}")
    try:
        period = p["period"]
        tz = book.policy.tz
        start = dt.datetime.combine(parse_date(period["start_local"]), dt.time(), tz).astimezone(dt.timezone.utc)
        end = dt.datetime.combine(parse_date(period["end_local"]) + dt.timedelta(days=1), dt.time(), tz)
        for name, want in (("start_utc", start), ("end_utc_exclusive", end.astimezone(dt.timezone.utc))):
            if dt.datetime.fromisoformat(period[name]) != want:
                add(problems, "SCHEMA", f"policy.period.{name} {period[name]!r}, local bounds give {want.isoformat()}")
    except (BookError, KeyError, TypeError, ValueError) as exc:
        add(problems, "SCHEMA", f"policy.period: {type(exc).__name__} {exc}")


def check_inv23(episode_dir, issues, problems):
    """INV-23: no issue_id, type name or private word in any public file."""
    patterns = [(name, re.compile(re.escape(name).replace("_", "[ _-]?"), re.IGNORECASE)) for name in ISSUE_TYPES]
    patterns += [(word, re.compile(re.escape(word), re.IGNORECASE)) for word in PRIVATE_WORDS]
    patterns += [(i["issue_id"], re.compile(rf"(?<![0-9A-Za-z]){re.escape(i['issue_id'])}(?![0-9A-Za-z])"))
                 for i in issues]
    public = Path(episode_dir) / "public"
    for path in sorted(public.iterdir()):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            add(problems, "SCHEMA", f"public/{path.name}: {type(exc).__name__}")
            continue
        for what, pattern in patterns:
            for line_no, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    add(problems, "INV-23", f"public/{path.name} line {line_no} contains {what!r}")
