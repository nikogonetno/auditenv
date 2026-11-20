"""calculator(expression), the arithmetic-only action from case §04 (role B).

A hand-written tokenizer and recursive-descent parser over Decimal: numbers, + - * /, unary signs
and parentheses, nothing else. Every rejection is CalculatorError with one short fixed code.
"""
import decimal
import re
from decimal import Decimal

MAX_LENGTH = 200
MAX_DEPTH = 20
MAX_EXPONENT = 40
PRECISION = 34
FRACTION_DIGITS = 18

TOO_LONG = "expression_too_long"
BAD_EXPRESSION = "bad_expression"
DIVISION_BY_ZERO = "division_by_zero"
TOO_LARGE = "result_too_large"

# ASCII classes on purpose: \d, \s and str.isdigit() also accept other scripts' digits and spaces.
_SPACE = re.compile(r"[ \t\r\n]*")
_TOKEN = re.compile(r"(?P<num>(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE](?P<exp>[+-]?[0-9]+))?)|(?P<op>[-+*/()])")
_QUANTUM = Decimal((0, (1,), -FRACTION_DIGITS))  # 1E-18, built without any context
_NEGATE = "neg"


class CalculatorError(ValueError):
    """A rejected expression. str() is one fixed code and never echoes the input."""


def _context(prec):
    # Every field is explicit: Context() copies missing ones from the mutable decimal.DefaultContext.
    return decimal.Context(prec=prec, rounding=decimal.ROUND_HALF_UP, Emin=-999999, Emax=999999,
                           capitals=1, clamp=0, flags=[],
                           traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow])


def _tokenize(text):
    """Operators as one-character str, numbers as exact Decimal: only operations round."""
    tokens = []
    pos = _SPACE.match(text).end()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None:
            raise CalculatorError(BAD_EXPRESSION)
        if match["op"]:
            tokens.append(match["op"])
        else:
            exponent = match["exp"]
            if exponent is not None and abs(int(exponent)) > MAX_EXPONENT:
                raise CalculatorError(BAD_EXPRESSION)
            tokens.append(Decimal(match["num"]))  # exact, independent of any context
        pos = _SPACE.match(text, match.end()).end()
    return tokens


class _Parser:
    """Checks the whole grammar and emits postfix, so a syntax error wins over 1/0 before it.

    expr := term (('+' | '-') term)*      term := unary (('*' | '/') unary)*
    unary := ('+' | '-')* primary         primary := number | '(' expr ')'
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.depth = 0
        self.out = []

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _op(self):
        token = self._peek()
        return token if type(token) is str else None

    def parse(self):
        self._expr()
        if self.pos != len(self.tokens):
            raise CalculatorError(BAD_EXPRESSION)
        return self.out

    def _binary(self, ops, operand):
        operand()
        while (op := self._op()) in ops:
            self.pos += 1
            operand()
            self.out.append(op)

    def _expr(self):
        self._binary(("+", "-"), self._term)

    def _term(self):
        self._binary(("*", "/"), self._unary)

    def _unary(self):
        # A loop, not recursion: "-" * 199 + "1" fits in the length limit.
        negative = False
        while (op := self._op()) in ("+", "-"):
            negative ^= op == "-"
            self.pos += 1
        self._primary()
        if negative:
            self.out.append(_NEGATE)

    def _primary(self):
        token = self._peek()
        if isinstance(token, Decimal):
            self.pos += 1
            self.out.append(token)
            return
        if token != "(" or self.depth == MAX_DEPTH:
            raise CalculatorError(BAD_EXPRESSION)
        self.pos += 1
        self.depth += 1
        self._expr()
        if self._op() != ")":
            raise CalculatorError(BAD_EXPRESSION)
        self.pos += 1
        self.depth -= 1


def _compute(program, ctx):
    stack = []
    for item in program:
        if isinstance(item, Decimal):
            stack.append(item)
        elif item == _NEGATE:
            stack.append(stack.pop().copy_negate())  # exact, so -x and x agree
        else:
            right = stack.pop()
            left = stack.pop()
            if item == "+":
                stack.append(ctx.add(left, right))
            elif item == "-":
                stack.append(ctx.subtract(left, right))
            elif item == "*":
                stack.append(ctx.multiply(left, right))
            elif right.is_zero():  # covers 0/0 too, which decimal reports as InvalidOperation
                raise CalculatorError(DIVISION_BY_ZERO)
            else:
                stack.append(ctx.divide(left, right))
    return stack.pop()


def _format(value):
    """Fixed-point text with at most FRACTION_DIGITS digits after the point and no trailing zeros."""
    if value.is_zero():  # before adjusted(): 0E+40 reports 40
        return "0"
    if value.adjusted() >= PRECISION:
        raise CalculatorError(TOO_LARGE)
    sign, digits, _ = _context(PRECISION + FRACTION_DIGITS).quantize(value, _QUANTUM).as_tuple()
    text = "".join(map(str, digits)).rjust(FRACTION_DIGITS + 1, "0")
    whole, fraction = text[:-FRACTION_DIGITS], text[-FRACTION_DIGITS:].rstrip("0")
    result = whole + "." + fraction if fraction else whole
    return "-" + result if sign and result != "0" else result


def evaluate(expression):
    """Evaluate one arithmetic expression and return the result as a canonical decimal string.

    Raises only CalculatorError: expression_too_long, bad_expression, division_by_zero or
    result_too_large. Independent of the thread's decimal context, which it never touches.
    """
    # Exact type, as in server.decode_message: a str subclass could lie about len() or slicing.
    if type(expression) is not str:
        raise CalculatorError(BAD_EXPRESSION)
    if len(expression) > MAX_LENGTH:
        raise CalculatorError(TOO_LONG)
    ctx = _context(PRECISION)
    try:
        return _format(_compute(_Parser(_tokenize(expression)).parse(), ctx))
    except decimal.DivisionByZero:
        raise CalculatorError(DIVISION_BY_ZERO) from None
    except decimal.Overflow:
        raise CalculatorError(TOO_LARGE) from None
    except (decimal.InvalidOperation, RecursionError):  # neither is reachable from valid grammar
        raise CalculatorError(BAD_EXPRESSION) from None
