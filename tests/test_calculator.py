"""calculator(expression) from case §04: decimal arithmetic only, one short fixed error code, never a crash."""
import ast
import decimal
import operator
import random
import re
import time
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

from auditenv import calculator
from auditenv.calculator import MAX_DEPTH, MAX_LENGTH, CalculatorError, evaluate

CODES = {"expression_too_long", "bad_expression", "division_by_zero", "result_too_large"}
# Fixed point, no exponent, no trailing zeros, at most 34 integer and 18 fraction digits, never "-0".
CANONICAL = re.compile(r"(?!-0\Z)-?(?:0|[1-9][0-9]{0,33})(?:\.[0-9]{0,17}[1-9])?")


def outcome(expression):
    """The result string, or the error code prefixed with '!'."""
    try:
        return evaluate(expression)
    except CalculatorError as exc:
        return "!" + str(exc)


class Examples(unittest.TestCase):
    def test_contract_examples(self):
        cases = {
            "1+2": "3",
            "10/4": "2.5",
            "1/3": "0.333333333333333333",
            "5000*515.88": "2579400",
            "0.1+0.2": "0.3",
            "-(2-5)": "3",
            "1e3": "1000",
            "2579400-2566700": "12700",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                result = evaluate(expression)
                self.assertIs(type(result), str)
                self.assertEqual(result, expected)


class CalculatorCase(unittest.TestCase):
    def check(self, cases):
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(outcome(expression), expected)


class Grammar(CalculatorCase):
    def test_number_literals(self):
        self.check({
            "7": "7", "12.50": "12.5", ".5": "0.5", "12.": "12", "0": "0", "0.000": "0", "000.500": "0.5",
            "007": "7", "1e3": "1000", "1E3": "1000", "1e+03": "1000", "1E+3": "1000", "1e-0": "1",
            "2.5E-2": "0.025", "2.5e-2": "0.025", ".5e1": "5", "1.e1": "10", "123.456e1": "1234.56",
            "123456789012345678901234567890": "123456789012345678901234567890",
        })

    def test_binary_operators_and_precedence(self):
        self.check({
            "2+3": "5", "2-3": "-1", "2*3": "6", "3/2": "1.5",
            "2+3*4": "14", "2*3+4": "10", "1+2*3-4/2": "5", "2*3/4": "1.5",
            "10-4-3": "3", "100/10/5": "2", "8/2*4": "16", "1-2+3": "2",
        })

    def test_unary_signs(self):
        self.check({
            "-1": "-1", "+1": "1", "--1": "1", "---1": "-1", "+-+-1": "1", "- 1": "-1",
            "2*-3": "-6", "2--3": "5", "2+-3": "-1", "-2*-3": "6", "6/-4": "-1.5", "-2/4": "-0.5",
            "-(1+2)": "-3", "-(-(1))": "1", "-.5": "-0.5", "+(2)": "2",
            "-" * 199 + "1": "-1", "-" * 198 + "1": "1",
        })

    def test_parentheses(self):
        self.check({
            "(1)": "1", "((2))": "2", "(2+3)*4": "20", "2*(3+4)": "14", "(1+2)*(3+4)": "21",
            "((1+2)*3)/(4-1)": "3", "(1)+(2)": "3", "10-(4-3)": "9", "100/(10/5)": "50",
        })

    def test_whitespace(self):
        self.check({
            " 1 + 2 ": "3", "\t1\t*\t2\t": "2", "\n1\r\n+\n2\n": "3", "( 1 + 2 ) * 3": "9",
            "  -  ( 2 - 5 )  ": "3", "1" + " " * 50: "1",
        })

    def test_everything_else_is_bad_expression(self):
        bad = [
            # empty, incomplete or unbalanced
            "", "   ", "\n", "+", "-", "1+", "*1", "1*/2", "()", "(1", "1)", ")(", "(-)", "1 - - -", "(1)(2)", "2(3)",
            # malformed numbers
            ".", "..5", "1..2", "1.2.3", "1e", "1e+", "1E-", "e3", ".e3", "1 e3", "1e 3", "1e3.5", "1e1e1",
            "1 2", "1 000", "1,000.00", "1. 5", "1 .5",
            # operators outside the grammar
            "2**3", "7//2", "7%2", "1<2", "1>=2", "1==1", "1!=2", "1&1", "1|1", "1^1", "~1", "1@2",
            "1 and 1", "1 or 1", "not 1", "1 if 1 else 2", "a=1", "1;2", "1,2", "1:2",
            # names, calls, attributes, subscripts, strings, constants
            "x", "abs(1)", "(1).real", "1 .real", "[1]", "[1][0]", "{1}", "'1'", '"1"', "f'1'", "b'1'",
            "True", "None", "lambda: 1", "__import__('os')", "open('x')", "#1", "1 # c", "$1", "\\",
            # other number spellings
            "1_000", "1j", "1J", "0x10", "0X1F", "0b1", "0o7", "nan", "NaN", "inf", "-inf", "Infinity", "sNaN",
            # non-ASCII digits, spaces and signs, control characters
            "\u0661", "\uff11", "\U0001d7cf", "1\u00a0+1", "1\u2003", "\u22121", "1\x00", "1\f+1", "1\v",
            "\ufeff1",
        ]
        for expression in bad:
            with self.subTest(expression=expression):
                self.assertEqual(outcome(expression), "!bad_expression")

    def test_syntax_errors_win_over_arithmetic_errors(self):
        for expression in ["1/0+", "(1/0", "1/0)", "1/0 1", "1e40 1", "9" * 40 + "+", "1/0*x"]:
            with self.subTest(expression=expression):
                self.assertEqual(outcome(expression), "!bad_expression")

    def test_only_exact_str_is_accepted(self):
        class Text(str):
            pass

        for value in [None, 1, 1.5, True, b"1", bytearray(b"1"), ["1"], ("1",), {"1": 1}, decimal.Decimal("1"),
                      Fraction(1), Text("1+1"), b"1" * 500, ["1"] * 500]:
            with self.subTest(value=type(value).__name__):
                self.assertEqual(outcome(value), "!bad_expression")


class Output(CalculatorCase):
    def test_fixed_point_without_trailing_zeros(self):
        self.check({
            "100.0000": "100", "123.4500": "123.45", "1/8": "0.125", "0.5": "0.5", "1e33": "1" + "0" * 33,
            "1e-18": "0.000000000000000001", "-1e-18": "-0.000000000000000001", "1e2*1e2": "10000",
            "2.50*2": "5", "1/1e-17": "100000000000000000",
        })

    def test_rounding_to_18_fraction_digits_half_up(self):
        self.check({
            "2/3": "0.666666666666666667", "-2/3": "-0.666666666666666667", "1/7": "0.142857142857142857",
            "0.1234567890123456789": "0.123456789012345679",
            "5e-19": "0.000000000000000001", "-5e-19": "-0.000000000000000001",
            "2.5e-18": "0.000000000000000003", "-2.5e-18": "-0.000000000000000003",  # half-even would give 2
            "4.9e-19": "0", "1e-19": "0", "1e-40": "0", "1.0000000000000000005": "1.000000000000000001",
        })

    def test_negative_zero_is_zero(self):
        self.check({"-0": "0", "-0.0": "0", "-(0)": "0", "0*-1": "0", "-1e-19": "0", "-4.9e-19": "0",
                    "1-1": "0", "-1+1": "0", "0e10": "0", "-0e-40": "0", "0/-5": "0"})

    def test_arithmetic_is_34_digits_half_up(self):
        self.check({
            "1e33+0.5": "1" + "0" * 32 + "1",  # half-even would round to ...0
            "(1e33+0.5)-1e33": "1",
            "1e33+0.4": "1" + "0" * 33,
            "1/3*3": "1",
            "1/7*7": "1",
            "9" * 34 + ".4+0": "9" * 34,
            "9" * 34 + ".5+0": "!result_too_large",
            # Literals are exact; only operations round (review of the calculator).
            "9" * 34 + ".4": "9" * 34 + ".4",
            "-" + "9" * 34 + ".5": "-" + "9" * 34 + ".5",
            "1" + "0" * 34 + "1-1" + "0" * 35: "1",
            "12345678901234567.123456789012345678-12345678901234567": "0.123456789012345678",
        })

    def test_decimal_not_float(self):
        self.check({
            "0.1+0.2": "0.3", "1.1*3": "3.3", "0.1*3-0.3": "0", "1-0.9": "0.1",
            "12345678901234567890123456789012+1": "12345678901234567890123456789013",
            "9007199254740993": "9007199254740993",
        })


class Limits(CalculatorCase):
    def test_length_limit_is_inclusive(self):
        padded = "1" + " " * (MAX_LENGTH - 1)
        ones = "+".join(["1"] * 100) + " "
        self.assertEqual((len(padded), len(ones)), (200, 200))
        self.assertEqual(evaluate(padded), "1")
        self.assertEqual(evaluate(ones), "100")
        for expression in [padded + " ", ones + "1", "x" * 201, "1" * 201, "(" * 1000, "1" * 1_000_000]:
            with self.subTest(length=len(expression)):
                self.assertEqual(outcome(expression), "!expression_too_long")

    def test_type_is_checked_before_length(self):
        self.assertEqual(outcome(b"1" * 500), "!bad_expression")

    def test_nesting_depth_limit_is_inclusive(self):
        self.assertEqual(MAX_DEPTH, 20)
        self.assertEqual(evaluate("(" * 20 + "1" + ")" * 20), "1")
        self.assertEqual(evaluate("-(" * 20 + "1" + ")" * 20), "1")
        self.assertEqual(evaluate("(" * 20 + "1" + ")" * 20 + "+" + "(" * 20 + "1" + ")" * 20), "2")
        self.assertEqual(evaluate("+".join(["(1)"] * 40)), "40")
        for expression in ["(" * 21 + "1" + ")" * 21, "-(" * 21 + "1" + ")" * 21, "(" * 21 + "1", "(" * 99]:
            with self.subTest(expression=expression):
                self.assertEqual(outcome(expression), "!bad_expression")

    def test_exponent_limit_is_inclusive(self):
        self.assertEqual(outcome("1e40"), "!result_too_large")  # valid literal, 41-digit integer part
        self.check({
            "1e-40": "0", "1e40/1e40": "1", "1E+40*1e-40": "1", "1e040/1e+040": "1", "2.5e-40*4e40": "10",
            "1e" + "0" * 100 + "5": "100000",
        })
        for expression in ["1e41", "1e-41", "1E+041", "0e41", ".5e-99", "1e" + "0" * 100 + "41", "1e99999999999"]:
            with self.subTest(expression=expression):
                self.assertEqual(outcome(expression), "!bad_expression")

    def test_integer_part_limit_is_34_digits(self):
        self.check({
            "9" * 34: "9" * 34, "-" + "9" * 34: "-" + "9" * 34, "1e33": "1" + "0" * 33,
            "9" * 35: "!result_too_large", "1e34": "!result_too_large", "-1e34": "!result_too_large",
            "9" * 34 + "+1": "!result_too_large", "1" * 199: "!result_too_large",
            "1e40*1e40/1e40": "!result_too_large", "1e30*1e30/1e30": "1" + "0" * 30,
            "1e-40*1e-40": "0", "1/1e-40": "!result_too_large",
        })


class Errors(unittest.TestCase):
    def test_division_by_zero(self):
        for expression in ["1/0", "0/0", "-1/0.000", "1/0e5", "1/-0", "1/(2-2)", "1/(1/3-1/3)", "5/(1e-40*0)",
                           "1e40/0", "(" * 20 + "1/0" + ")" * 20]:
            with self.subTest(expression=expression):
                self.assertEqual(outcome(expression), "!division_by_zero")

    def test_overflow_trap_maps_to_result_too_large(self):
        """Unreachable with Emax 999999 in 200 characters, so shrink Emax to reach the trap."""
        real = calculator._context

        def small_emax(prec):
            ctx = real(prec)
            ctx.Emax = 50
            return ctx

        self.assertEqual(evaluate("1e40*1e40*1e-40*1e-40"), "1")
        with mock.patch.object(calculator, "_context", small_emax):
            self.assertEqual(outcome("1e40*1e40*1e-40*1e-40"), "!result_too_large")
            self.assertEqual(evaluate("1e40/1e39"), "10")

    def test_error_is_a_short_fixed_code(self):
        cases = {"SECRET_ledger/QQ5GHK": "bad_expression", "1/0": "division_by_zero",
                 "9" * 35: "result_too_large", "SECRET" * 40: "expression_too_long", "2**SECRET": "bad_expression"}
        for expression, code in cases.items():
            with self.subTest(code=code):
                with self.assertRaises(CalculatorError) as ctx:
                    evaluate(expression)
                exc = ctx.exception
                self.assertIsInstance(exc, ValueError)
                self.assertEqual(str(exc), code)
                self.assertEqual(exc.args, (code,))
                self.assertIsNone(exc.__cause__)
                self.assertNotIn("SECRET", repr(exc))


SAMPLES = {
    "1/3": "0.333333333333333333", "2/3": "0.666666666666666667", "-2/3": "-0.666666666666666667",
    "1e33+0.5": "1" + "0" * 32 + "1", "(1e33+0.5)-1e33": "1", "5e-19": "0.000000000000000001",
    "-5e-19": "-0.000000000000000001", "5000*515.88": "2579400", "1/7*7": "1", "0.1+0.2": "0.3",
    "123456789.123456789123456789": "123456789.123456789123456789",
    "9" * 34: "9" * 34, "1e40": "!result_too_large", "1/0": "!division_by_zero", "0/0": "!division_by_zero",
    "2**3": "!bad_expression", "9" * 35: "!result_too_large", "1e-40*1e-40*1e40*1e40": "1",
}


def hostile(ctx):
    """Turn a context into one that disagrees with the calculator's on every field and traps everything."""
    ctx.prec = 3
    ctx.rounding = decimal.ROUND_FLOOR
    ctx.Emax = 5
    ctx.Emin = -5
    ctx.capitals = 0
    ctx.clamp = 1
    for signal in ctx.traps:
        ctx.traps[signal] = True


def context_state(ctx):
    return (ctx.prec, ctx.rounding, ctx.Emin, ctx.Emax, ctx.capitals, ctx.clamp, dict(ctx.traps), dict(ctx.flags))


class Determinism(unittest.TestCase):
    def test_samples(self):
        for expression, expected in SAMPLES.items():
            with self.subTest(expression=expression):
                self.assertEqual(outcome(expression), expected)
                self.assertEqual(outcome(expression), outcome(expression))

    def test_independent_of_thread_context(self):
        with decimal.localcontext() as ctx:
            hostile(ctx)
            self.assertEqual({e: outcome(e) for e in SAMPLES}, SAMPLES)

    def test_independent_of_default_context(self):
        default = decimal.DefaultContext
        saved = context_state(default)
        try:
            hostile(default)
            self.assertEqual({e: outcome(e) for e in SAMPLES}, SAMPLES)
        finally:
            default.prec, default.rounding, default.Emin, default.Emax, default.capitals, default.clamp = saved[:6]
            for signal, value in saved[6].items():
                default.traps[signal] = value
            for signal, value in saved[7].items():
                default.flags[signal] = value
        self.assertEqual(context_state(default), saved)

    def test_does_not_touch_thread_context(self):
        with decimal.localcontext() as ctx:
            ctx.clear_flags()
            before = context_state(ctx)
            for expression in SAMPLES:
                outcome(expression)
            self.assertIs(decimal.getcontext(), ctx)
            self.assertEqual(context_state(ctx), before)
            self.assertFalse(any(ctx.flags.values()))


# Leaves with at most 4 significant and 3 fraction digits, divisors whose inverse has at most 3 fraction
# digits: six leaves stay exact in 34 digits and 18 fraction digits, so Fraction is an exact oracle.
LEAVES = ["0", "1", "2", "7", "10", "12.5", "0.25", "3.75", "99.99", ".5", "1e2", "2.5E-1", "007", "0.000", "8."]
DIVISORS = ["1", "2", "4", "5", "8", "10", "20", "25", "0.5", "0.25", "1.25"]
OPERATORS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


def random_expression(rng, leaves):
    """A random valid expression with its exact value."""
    if leaves == 1:
        text = rng.choice(LEAVES)
        value = Fraction(decimal.Decimal(text))
    else:
        op = rng.choice("+-*/")
        if op == "/":
            left, left_value = random_expression(rng, leaves - 1)
            right = rng.choice(DIVISORS)
            right_value = Fraction(right)
        else:
            split = rng.randint(1, leaves - 1)
            left, left_value = random_expression(rng, split)
            right, right_value = random_expression(rng, leaves - split)
        space = rng.choice(["", "", " ", "\t"])
        text = f"({left}{space}{op}{space}{right})"
        value = OPERATORS[op](left_value, right_value)
    if rng.random() < 0.2:
        text, value = "-" + text, -value
    elif rng.random() < 0.05:
        text = "+" + text
    return text, value


ALLOWED = "0123456789.eE+-*/() \t\r\n"
FORBIDDEN = list("xjJabn_%<>=!&|^~[]{},;:'\"\\#@$?`\x00\f\v") + [
    "\u00a0", "\u0661", "\uff11", "\u2212", "\u00e9", "\U0001d7cf", "\ufeff", "\u2003"]


class Robustness(unittest.TestCase):
    def assert_canonical(self, result, expression):
        self.assertIs(type(result), str)
        self.assertIsNotNone(CANONICAL.fullmatch(result), (expression, result))
        self.assertEqual(evaluate(result), result, expression)  # canonical output is a fixed point

    def test_random_valid_expressions_match_exact_arithmetic(self):
        rng = random.Random(20261120)
        for _ in range(1500):
            expression, exact = random_expression(rng, rng.randint(1, 6))
            self.assertLessEqual(len(expression), MAX_LENGTH)
            result = evaluate(expression)
            self.assert_canonical(result, expression)
            self.assertEqual(Fraction(result), exact, expression)

    def test_fuzz_only_calculator_error_escapes(self):
        rng = random.Random(4)
        seen = {}
        for i in range(3000):
            if i % 2:
                text = "".join(rng.choice(ALLOWED) if rng.random() < 0.9 else rng.choice(FORBIDDEN)
                               for _ in range(rng.randint(0, 24)))
            else:  # mutate a valid expression so the fuzz also lands near the grammar
                chars = list(random_expression(rng, rng.randint(1, 4))[0])
                for _ in range(rng.randint(1, 3)):
                    at = rng.randrange(len(chars) + 1)
                    edit = rng.choice(["insert", "delete", "replace"])
                    new = rng.choice(ALLOWED) if rng.random() < 0.8 else rng.choice(FORBIDDEN)
                    if edit == "insert":
                        chars.insert(at, new)
                    elif at < len(chars):
                        chars[at:at + 1] = [] if edit == "delete" else [new]
                text = "".join(chars)
            try:
                result = evaluate(text)
            except CalculatorError as exc:
                self.assertIn(str(exc), CODES - {"expression_too_long"}, repr(text))
                seen[str(exc)] = seen.get(str(exc), 0) + 1
            except Exception as exc:  # noqa: BLE001 - the point of the test
                self.fail(f"{type(exc).__name__} escaped for {text!r}")
            else:
                self.assert_canonical(result, text)
                seen["ok"] = seen.get("ok", 0) + 1
        # The fuzz must exercise both sides of the grammar, not only rejections.
        self.assertGreater(seen.get("ok", 0), 300, seen)
        self.assertGreater(seen.get("bad_expression", 0), 500, seen)
        self.assertGreater(seen.get("division_by_zero", 0), 0, seen)

    def test_worst_case_inputs_are_fast(self):
        worst = [
            "1/3*" * 49 + "1",
            "1/7+" * 49 + "1",
            "-" * 199 + "1",
            "9" * 200,
            "0." + "1" * 198,
            "1e40*" * 39 + "1",
            "1e-40*" * 33 + "1",
            "1/1e-40" + "/1e-40" * 32,
            "1e40+1e-40-" * 18 + "1",
            "(" * 20 + "1/7" + ")" * 20 + "*1/3" * 39,
            "(" * 200,
            "+".join(["(1)"] * 50),
            "1+" * 99 + "1",
            "1" + " " * 199,
            "x" * 200,
            "1" * 10_000,  # rejected on length alone
        ]
        for expression in worst:
            with self.subTest(expression=expression[:20]):
                best = float("inf")
                for _ in range(5):
                    start = time.perf_counter()
                    outcome(expression)
                    best = min(best, time.perf_counter() - start)
                self.assertLess(best, 0.010)

    def test_source_has_no_eval_float_or_global_context(self):
        tree = ast.parse(Path(calculator.__file__).read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertFalse(names & {"eval", "exec", "compile", "float", "__import__", "open", "getattr", "globals"})
        self.assertFalse(attributes & {"getcontext", "setcontext", "localcontext", "literal_eval"})
        self.assertLessEqual(imported, {"decimal", "re"})


if __name__ == "__main__":
    unittest.main()
