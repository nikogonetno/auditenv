"""JSON over stdin/stdout so the policy runs in a separate process from env and grader (role B).

Checkpoint 2. Truth is read only at submit; in eval mode reward goes only to the grader log.
For now this module holds the one piece that does not depend on case §04: turning a raw
policy message into plain JSON values without ever crashing the process (QA risk R-09).
"""
import json
import math

# R-07 proposal in SPEC.md §5.4: 1 MB per message, checked before parsing.
MAX_MESSAGE_BYTES = 1_000_000
# Money is in minor units; 30 digits is far above any amount and far below int() cost cliffs.
MAX_INT_DIGITS = 30
# Protocol messages are shallow (action -> findings -> finding -> id list); deep input only
# serves to blow the recursion limit later, in deepcopy or logging.
MAX_DEPTH = 32


class BadMessage(ValueError):
    """A policy message that cannot be accepted. The env answers with a short error and keeps going."""


def _reject_constant(_name):
    raise BadMessage("non-finite number")


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):  # 1e999 parses to inf without passing parse_constant
        raise BadMessage("non-finite number")
    return value


def _bounded_int(text):
    if len(text.lstrip("-")) > MAX_INT_DIGITS:
        raise BadMessage("integer too long")
    return int(text)


def _encodes_as_utf8(text):
    # Lone surrogates (an escaped "\ud800", or stdin read with surrogateescape) parse fine
    # but crash every later UTF-8 write of logs, trajectories or observations.
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _value_problem(value):
    """Why a parsed value cannot be accepted, or None. Iterative, so deep input cannot recurse."""
    stack = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, str):
            if not _encodes_as_utf8(node):
                return "invalid UTF-8"
        elif isinstance(node, (dict, list)):
            if depth > MAX_DEPTH:
                return "nesting too deep"
            if isinstance(node, dict):
                if not all(_encodes_as_utf8(key) for key in node):
                    return "invalid UTF-8"
                children = node.values()
            else:
                children = node
            stack.extend((child, depth + 1) for child in children)
    return None


def decode_message(raw, max_bytes=MAX_MESSAGE_BYTES):
    """Parse one policy message (str or UTF-8 bytes) into plain JSON values.

    Raises only BadMessage, with a short fixed text that never echoes the input.
    """
    # Exact types, as in the grader (QA D-1): a subclass could lie about len() or encode().
    if type(raw) not in (str, bytes, bytearray):
        raise BadMessage("message must be text")
    # Every code point takes at least one UTF-8 byte, so this rejects oversize text
    # before anything proportional to its size is allocated.
    if len(raw) > max_bytes:
        raise BadMessage("message too large")
    try:
        data = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        if len(data) > max_bytes:
            raise BadMessage("message too large")
        text = data.decode("utf-8")
        value = json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float, parse_int=_bounded_int)
    except BadMessage:
        raise
    except UnicodeError:  # lone surrogates in str input, broken UTF-8 in bytes input
        raise BadMessage("invalid UTF-8") from None
    except RecursionError:
        raise BadMessage("nesting too deep") from None
    except ValueError:  # JSONDecodeError and anything int()/float() reject
        raise BadMessage("invalid JSON") from None
    problem = _value_problem(value)
    if problem:
        raise BadMessage(problem)
    return value
