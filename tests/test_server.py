"""Policy message decoding (SPEC.md §4.3, QA risk R-09): bad input is a short error, never a crash."""
import json
import random
import unittest

from auditenv.server import MAX_DEPTH, MAX_INT_DIGITS, BadMessage, decode_message


class DecodeMessage(unittest.TestCase):
    def test_plain_messages_parse(self):
        msg = {"action": "submit", "findings": [{"type": "fx_rate", "affected_ids": ["ledger/QQ5GHK"],
                                                 "amount_minor": 1270000, "currency": "KZT", "evidence": []}]}
        text = json.dumps(msg)
        self.assertEqual(decode_message(text), msg)
        self.assertEqual(decode_message(text.encode("utf-8")), msg)
        self.assertEqual(decode_message('{"x": 1.5, "y": -2, "z": "Ünïcode"}'), {"x": 1.5, "y": -2, "z": "Ünïcode"})

    def test_r09_cases_become_bad_message(self):
        bad = {
            "NaN": '{"amount_minor": NaN}',
            "Infinity": '{"amount_minor": Infinity}',
            "-Infinity": '{"amount_minor": -Infinity}',
            "float overflow to inf": '{"amount_minor": 1e999}',
            "int with 5001 digits": '{"amount_minor": 1' + "0" * 5000 + "}",
            "int one digit over the cap": '{"n": ' + "9" * (MAX_INT_DIGITS + 1) + "}",
            "negative int over the cap": '{"n": -' + "9" * (MAX_INT_DIGITS + 1) + "}",
            "200000 nested lists": "[" * 200000 + "]" * 200000,
            "nesting one level over the cap": "[" * (MAX_DEPTH + 1) + "]" * (MAX_DEPTH + 1),
            "broken JSON": '{"action": ',
            "trailing garbage": '{"a": 1} x',
            "empty": "",
            "invalid UTF-8": b'{"a": "\xff"}',
            "not text": 42,
            "str subclass": type("Lying", (str,), {"encode": lambda self, *a: b"[]"})('{"a": 1}'),
            "bytes subclass": type("Lying", (bytes,), {"__len__": lambda self: 0})(b'{"a": 1}'),
        }
        for name, raw in bad.items():
            with self.subTest(case=name):
                with self.assertRaises(BadMessage) as ctx:
                    decode_message(raw)
                self.assertLessEqual(len(str(ctx.exception)), 40)

    def test_lone_surrogates_are_rejected_everywhere(self):
        """Accepted strings must survive a later UTF-8 write (review of R-09)."""
        bad = {
            "escaped surrogate, bytes": b'{"id": "\\ud800"}',
            "escaped surrogate, str": '{"id": "\\udfff"}',
            "escaped surrogate in a key": '{"\\udc00": 1}',
            "raw surrogate in str": '{"id": "ledger/\ud800"}',
            "stdin read with surrogateescape": b'{"a": "\xff"}'.decode("utf-8", "surrogateescape"),
        }
        for name, raw in bad.items():
            with self.subTest(case=name):
                with self.assertRaises(BadMessage) as ctx:
                    decode_message(raw)
                self.assertEqual(str(ctx.exception), "invalid UTF-8")
        value = decode_message('{"emoji": "\\ud83d\\ude00", "cyr": "Счёт"}')
        self.assertEqual(json.dumps(value, ensure_ascii=False).encode("utf-8").decode("utf-8"),
                         json.dumps({"emoji": "😀", "cyr": "Счёт"}, ensure_ascii=False))

    def test_size_counts_utf8_bytes_for_str_and_bytes(self):
        text = '"' + "ж" * 499 + '"'  # 2 + 998 = 1000 UTF-8 bytes, 501 code points
        for raw in (text, text.encode("utf-8")):
            with self.subTest(kind=type(raw).__name__):
                self.assertEqual(decode_message(raw, max_bytes=1000), "ж" * 499)
                with self.assertRaises(BadMessage):
                    decode_message(raw, max_bytes=999)
        with self.assertRaises(BadMessage) as ctx:
            decode_message("x" * 1001, max_bytes=1000)  # rejected on length alone, before encoding
        self.assertEqual(str(ctx.exception), "message too large")

    def test_limits_are_inclusive(self):
        self.assertEqual(decode_message('{"n": ' + "9" * MAX_INT_DIGITS + "}"), {"n": int("9" * MAX_INT_DIGITS)})
        deepest = "[" * MAX_DEPTH + "]" * MAX_DEPTH
        self.assertIsInstance(decode_message(deepest), list)

    def test_size_is_checked_before_parsing(self):
        payload = json.dumps({"pad": "x" * 2000})
        self.assertEqual(decode_message(payload, max_bytes=len(payload))["pad"], "x" * 2000)
        with self.assertRaises(BadMessage) as ctx:
            decode_message(payload, max_bytes=len(payload) - 1)
        self.assertEqual(str(ctx.exception), "message too large")
        with self.assertRaises(BadMessage):
            decode_message("[" * 2_000_000)  # over 1 MB: rejected before json.loads ever recurses

    def test_bad_message_is_a_value_error(self):
        self.assertTrue(issubclass(BadMessage, ValueError))

    def test_random_bytes_never_escape_as_other_errors(self):
        rng = random.Random(0)
        alphabet = b'{}[]",:0123456789.eE+-tfnaNIy \\u\xff\x00'
        for case in range(2000):
            raw = bytes(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
            with self.subTest(case=case):
                try:
                    decode_message(raw)
                except BadMessage:
                    pass


if __name__ == "__main__":
    unittest.main()
