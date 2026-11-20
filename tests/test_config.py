"""Episode config v1 (SPEC §4.4): defaults, E-23 bad input, E-16 order independence, error budget."""
import copy
import unittest

from auditenv.config import DEFAULTS, MAX_ERRORS, config_sha256, is_clean, normalize_config, total_errors


class Defaults(unittest.TestCase):
    def test_empty_and_none_give_the_defaults(self):
        self.assertEqual(normalize_config(None), normalize_config({}))
        cfg = normalize_config({})
        self.assertEqual(set(cfg), set(DEFAULTS))
        self.assertEqual(total_errors(cfg), 4)
        self.assertEqual(cfg["documents"], [150, 200])

    def test_result_is_a_fresh_copy(self):
        cfg = normalize_config({})
        cfg["errors"]["fx_rate"] = 9
        cfg["documents"][0] = 1
        self.assertEqual(normalize_config({})["errors"]["fx_rate"], 1)
        self.assertEqual(DEFAULTS["documents"], [150, 200])

    def test_input_is_not_modified(self):
        config = {"errors": {"fx_rate": 2}, "company_offsets": ["+06:00", "+03:00"]}
        before = copy.deepcopy(config)
        normalize_config(config)
        self.assertEqual(config, before)


class OrderIndependence(unittest.TestCase):
    """E-16: key order and list order of the input do not change the episode."""

    def test_key_and_list_order(self):
        a = {"twins": [2, 5], "errors": {"fx_rate": 2, "duplicate_invoice": 1}, "period": "2026-09",
             "company_offsets": ["+05:00", "+03:00"]}
        b = {"company_offsets": ["+03:00", "+05:00"], "period": "2026-09",
             "errors": {"duplicate_invoice": 1, "fx_rate": 2}, "twins": [2, 5]}
        self.assertEqual(normalize_config(a), normalize_config(b))
        self.assertEqual(config_sha256(normalize_config(a)), config_sha256(normalize_config(b)))
        self.assertNotEqual(config_sha256(normalize_config(a)), config_sha256(normalize_config({})))


class Budget(unittest.TestCase):
    def test_types_are_set_independently_and_missing_ones_are_zero(self):
        cfg = normalize_config({"errors": {"period_cutoff": 3}})
        self.assertEqual(cfg["errors"], {"duplicate_invoice": 0, "payment_allocation": 0, "fx_rate": 0, "period_cutoff": 3})

    def test_twelve_errors_with_scenarios_is_the_limit(self):
        ok = {"errors": {"duplicate_invoice": 2, "payment_allocation": 2, "fx_rate": 2, "period_cutoff": 1},
              "scenarios": {"C1": 1, "C2": 2}}
        self.assertEqual(total_errors(normalize_config(ok)), MAX_ERRORS)
        over = copy.deepcopy(ok)
        over["scenarios"]["C2"] = 3
        with self.assertRaises(ValueError):
            normalize_config(over)

    def test_clean_company(self):
        cfg = normalize_config({"errors": {}, "scenarios": {"C1": 0, "C2": 0}})
        self.assertTrue(is_clean(cfg))
        self.assertFalse(is_clean(normalize_config({})))


class BadInput(unittest.TestCase):
    """E-23: a clear ValueError, never a silent default."""

    def test_rejected(self):
        bad = {
            "not an object": [],
            "unknown key": {"colour": "red"},
            "episode_dir is not a generator key": {"episode_dir": "x"},
            "period of another year": {"period": "2025-09"},
            "period month 13": {"period": "2026-13"},
            "period as int": {"period": 202609},
            "second base currency": {"base_currency": "USD"},
            "documents reversed": {"documents": [200, 150]},
            "documents over 200": {"documents": [150, 201]},
            "documents under 20": {"documents": [10, 50]},
            "documents as floats": {"documents": [150.0, 200.0]},
            "documents with bool": {"documents": [True, 200]},
            "foreign_pct over 100": {"foreign_pct": [0, 101]},
            "offsets empty": {"company_offsets": []},
            "offset without sign": {"company_offsets": ["05:00"]},
            "offset over +14": {"company_offsets": ["+15:00"]},
            "offset duplicated": {"company_offsets": ["+05:00", "+05:00"]},
            "unknown error type": {"errors": {"fraud": 1}},
            "negative error count": {"errors": {"fx_rate": -1}},
            "error count as bool": {"errors": {"fx_rate": True}},
            "unknown scenario": {"scenarios": {"C3": 1}},
            "13 errors": {"errors": {t: 3 for t in ("duplicate_invoice", "payment_allocation", "fx_rate")} | {"period_cutoff": 4}},
            "fx error without foreign documents": {"foreign_pct": [0, 0]},
            "C1 without foreign documents": {"foreign_pct": [0, 0], "errors": {}, "scenarios": {"C1": 1}},
            "twins reversed": {"twins": [8, 3]},
        }
        for name, config in bad.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                normalize_config(config)

    def test_clean_company_may_have_no_foreign_documents(self):
        cfg = normalize_config({"foreign_pct": [0, 0], "errors": {}})
        self.assertEqual(cfg["foreign_pct"], [0, 0])


if __name__ == "__main__":
    unittest.main()
