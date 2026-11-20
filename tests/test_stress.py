"""scripts/stress.py with test doubles for the generator (docs/tasks/B_next.md item 7)."""
import json
import unittest
from pathlib import Path

from tests.support import run_python

ROOT = Path(__file__).resolve().parent.parent
STRESS = ROOT / "scripts" / "stress.py"


def stress(*args):
    result = run_python([STRESS, *args], cwd=ROOT)
    lines = [json.loads(line) for line in result.stdout.splitlines()]
    return result.returncode, lines[:-1], lines[-1]


class Stress(unittest.TestCase):
    def test_deterministic_generator_passes_every_check(self):
        # mini_v03 is a small test book: 2 neighbour-period rows each side, not the 10 of INV-03.
        code, rows, summary = stress("--n", "2", "--generator", "tests.fake_generators:copy_mini", "--no-strict-volume")
        self.assertEqual((code, summary["stress"], summary["failed"]), (0, "OK", 0))
        self.assertEqual([r["seed"] for r in rows], [0, 1])
        for row in rows:
            self.assertEqual((row["failures"], row["oracle_reward"], row["issues"]), ([], 1.0, 8))
            self.assertLess(row["episode_s"], 60)

    def test_verifier_runs_on_every_seed_when_it_is_available(self):
        try:
            import auditenv.verify  # noqa: F401
        except ImportError:
            self.skipTest("auditenv.verify not merged")
        code, rows, summary = stress("--n", "1", "--generator", "tests.fake_generators:copy_mini")
        self.assertEqual((code, summary["verifier"]), (1, "on"))
        self.assertTrue(any(f.startswith("verify:") and "INV-03" in f for f in rows[0]["failures"]))

    def test_nondeterministic_generator_fails(self):
        code, rows, summary = stress("--n", "1", "--generator", "tests.fake_generators:nondeterministic")
        self.assertEqual((code, summary["stress"]), (1, "FAIL"))
        self.assertTrue(any(f.startswith("determinism") for f in rows[0]["failures"]))

    def test_generator_error_is_reported_per_seed(self):
        code, rows, summary = stress("--n", "2", "--generator", "tests.fake_generators:failing")
        self.assertEqual((code, summary["failed"]), (1, 2))
        self.assertTrue(rows[0]["failures"][0].startswith("generate: ValueError"))

    def test_missing_generator_is_a_skip(self):
        code, rows, summary = stress("--n", "3", "--generator", "tests.no_such_module:generate_episode")
        self.assertEqual((code, rows, summary["stress"]), (0, [], "SKIP"))


if __name__ == "__main__":
    unittest.main()
