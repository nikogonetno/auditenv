"""Policy side of the JSON-lines protocol of auditenv.runner (stdlib only, no access to the env).

    read_message() -> the next runner message as a dict, or None on EOF
    send_action(a) -> one action line on stdout, flushed
    serve(fn)      -> loop: read a message, stop at the end of the episode, else send fn(message)
"""
import json
import sys


def read_message():
    """The next message from the runner, or None when stdin is closed."""
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    return json.loads(line.decode("utf-8"))


def send_action(action):
    """Write one action as one JSON line on stdout and flush it."""
    sys.stdout.flush()  # keep order with anything printed through the text layer
    sys.stdout.buffer.write((json.dumps(action, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def episode_over(message):
    return bool(message.get("terminated") or message.get("truncated"))


def serve(policy_fn):
    """Answer every runner message with policy_fn(message) until the episode ends or stdin closes."""
    while True:
        message = read_message()
        if message is None or episode_over(message):
            return
        send_action(policy_fn(message))


def seed_from_argv(argv=None, default=0):
    """`python3 -m policies.<name> <seed>`: the seed argument, or the default when it is missing."""
    argv = sys.argv if argv is None else argv
    return int(argv[1]) if len(argv) > 1 else default


# ---------------------------------------------------------------- public record schema, case §03
TABLE_ORDER = ("ledger", "invoices", "payments", "fx")
ID_FIELDS = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": "fx_id"}
ID_PREFIXES = {"ledger": "ledger", "invoices": "invoice", "payments": "payment", "fx": "fx"}
ISSUE_TYPES = ("duplicate_invoice", "payment_allocation", "fx_rate", "period_cutoff")


def tables_in_order(observation):
    """Tables of the reset observation in the case §03 order, unknown ones after them, sorted."""
    tables = observation.get("tables") or {}
    return [t for t in TABLE_ORDER if t in tables] + sorted(t for t in tables if t not in TABLE_ORDER)


def record_id(table, record):
    value = record.get(ID_FIELDS.get(table, ""))
    return value if isinstance(value, str) and value else None


def minor_amounts(record):
    """Non-negative integer *_minor fields of a record, in key order."""
    return [record[k] for k in sorted(record)
            if k.endswith("_minor") and type(record[k]) is int and record[k] >= 0]


def is_currency(value):
    return isinstance(value, str) and len(value) == 3 and value.isascii() and value.isalpha() and value.isupper()


def currency_of(record, manifest):
    """The record's currency, else the base currency of the manifest."""
    if is_currency(record.get("currency")):
        return record["currency"]
    base = (manifest or {}).get("base_currency")
    return base if is_currency(base) else "XXX"
