"""Episode config v1 (docs/tasks/A_generator.md, SPEC §4.4; role B).

    {"period": "2026-09" or null, "base_currency": "KZT", "documents": [150, 200], "foreign_pct": [0, 60],
     "company_offsets": ["+03:00", ...], "errors": {type: count}, "scenarios": {"C1": n, "C2": n}, "twins": [3, 8]}

normalize_config fills defaults and rejects anything else with ValueError (E-23). The result is a fresh
dict with every key and canonical list order, so key or list order in the input never changes the
episode (E-16). The generator receives only normalized configs.
"""
import copy
import hashlib
import json
import re

ISSUE_TYPES = ("duplicate_invoice", "payment_allocation", "fx_rate", "period_cutoff")
SCENARIO_ERRORS = {"C1": 3, "C2": 1}  # C1: period + fx + allocation in one receipt; C2: duplicate paid on the copy
MAX_ERRORS = 12                       # case §03: 0-12 errors per episode
DOCUMENT_RANGE = (20, 200)            # case §03 suggests 20-200 documents
MAX_TWINS = 20
BASE_CURRENCIES = ("KZT",)            # a second base is out of scope until the final

DEFAULTS = {
    "period": None,
    "base_currency": "KZT",
    "documents": [150, 200],
    "foreign_pct": [0, 60],
    "company_offsets": ["+03:00", "+04:00", "+05:00", "+06:00"],
    "errors": {t: 1 for t in ISSUE_TYPES},
    "scenarios": {s: 0 for s in SCENARIO_ERRORS},
    "twins": [3, 8],
}

_PERIOD = re.compile(r"2026-(0[1-9]|1[0-2])")
_OFFSET = re.compile(r"[+-](0[0-9]|1[0-4]):(00|30|45)", re.ASCII)


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"config.{name} must be a non-negative int")
    return value


def _int_range(value, name, low, high):
    if type(value) is not list or len(value) != 2 or any(type(v) is not int for v in value):
        raise ValueError(f"config.{name} must be [min, max] ints")
    if not low <= value[0] <= value[1] <= high:
        raise ValueError(f"config.{name} must satisfy {low} <= min <= max <= {high}")
    return list(value)


def _counts(value, name, keys):
    if type(value) is not dict or any(type(k) is not str for k in value):
        raise ValueError(f"config.{name} must be an object with string keys")
    unknown = sorted(set(value) - set(keys))
    if unknown:
        raise ValueError(f"unknown config.{name} key: {unknown[0]}")
    # A provided object is the whole choice: types it leaves out get 0.
    return {k: _count(value.get(k, 0), f"{name}.{k}") for k in keys}


def normalize_config(config=None):
    """Defaults filled in and every value checked. ValueError on an unknown key or a bad value."""
    config = {} if config is None else config
    if type(config) is not dict:
        raise ValueError("config must be an object")
    if any(type(k) is not str for k in config):
        raise ValueError("config keys must be strings")
    unknown = sorted(set(config) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown config key: {unknown[0]}")
    merged = copy.deepcopy(DEFAULTS)
    try:
        merged.update(copy.deepcopy(config))
    except Exception:  # noqa: BLE001 - anything json.loads cannot give is a bad config, not a crash
        raise ValueError("config values must be JSON values") from None

    period = merged["period"]
    if period is not None and (type(period) is not str or not _PERIOD.fullmatch(period)):
        raise ValueError("config.period must be null or a month of 2026 like \"2026-09\"")
    if merged["base_currency"] not in BASE_CURRENCIES or type(merged["base_currency"]) is not str:
        raise ValueError("config.base_currency must be \"KZT\"")
    offsets = merged["company_offsets"]
    if type(offsets) is not list or not offsets or any(type(o) is not str or not _OFFSET.fullmatch(o) for o in offsets):
        raise ValueError("config.company_offsets must be a non-empty list of offsets like \"+05:00\"")
    offsets = ["+00:00" if o == "-00:00" else o for o in offsets]   # one offset, one spelling
    if len(set(offsets)) != len(offsets):
        raise ValueError("config.company_offsets has duplicates")

    normalized = {
        "period": period,
        "base_currency": merged["base_currency"],
        "documents": _int_range(merged["documents"], "documents", *DOCUMENT_RANGE),
        "foreign_pct": _int_range(merged["foreign_pct"], "foreign_pct", 0, 100),
        "company_offsets": sorted(offsets),
        "errors": _counts(merged["errors"], "errors", ISSUE_TYPES),
        "scenarios": _counts(merged["scenarios"], "scenarios", tuple(SCENARIO_ERRORS)),
        "twins": _int_range(merged["twins"], "twins", 0, MAX_TWINS),
    }
    if total_errors(normalized) > MAX_ERRORS:
        raise ValueError(f"config asks for more than {MAX_ERRORS} errors (C1 counts 3, C2 counts 1)")
    needs_foreign = normalized["errors"]["fx_rate"] or normalized["scenarios"]["C1"]
    if needs_foreign and normalized["foreign_pct"][1] == 0:
        raise ValueError("config: fx_rate and C1 need foreign_pct max above 0")
    return normalized


def total_errors(normalized):
    """Errors the episode will hold: plain errors plus scenario errors."""
    plain = sum(normalized["errors"].values())
    return plain + sum(SCENARIO_ERRORS[s] * n for s, n in normalized["scenarios"].items())


def is_clean(normalized):
    return total_errors(normalized) == 0


def config_sha256(normalized):
    """Stable hash of a normalized config, for provenance.json and configs/*.json versions."""
    text = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
