"""scripts/paired.py: repeat and transform checks of case §10 (docs/tasks/B_next.md item 8)."""
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from auditenv.run import run_policy
from auditenv.runner import run_inprocess_episode
from policies.oracle import make_env_oracle
from tests.support import run_python
from tests.tiny_episode import submit, write_episode

ROOT = Path(__file__).resolve().parent.parent
PAIRED = ROOT / "scripts" / "paired.py"
MINI = ROOT / "tests" / "fixtures" / "mini_v03"


def load_paired():
    spec = importlib.util.spec_from_file_location("paired_script", PAIRED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paired = load_paired()


def paired_cli(*args):
    result = run_python([PAIRED, *args], cwd=ROOT)
    lines = [json.loads(line) for line in result.stdout.splitlines()]
    return result.returncode, lines[:-1], (lines[-1] if lines else None), result.stderr


def noisy_runner(policy, env, seed, config, policy_seed=0, trajectory=None, **kwargs):
    """A non-deterministic policy run: the real run plus a line that differs on every call."""
    result = run_policy(policy, env, seed, config, policy_seed, trajectory=trajectory, **kwargs)
    trajectory.write(json.dumps({"noise": os.urandom(8).hex()}) + "\n")
    return result


class Plan(unittest.TestCase):
    def test_repeat_only_cycles_every_policy_over_every_episode(self):
        checks = paired.plan(16, ["a", "b"], with_transform=False)
        self.assertEqual([c[0] for c in checks], list(range(1, 17)))
        self.assertEqual({c[1] for c in checks}, {"repeat"})
        self.assertEqual(len({(c[2], c[3]) for c in checks}), 8)

    def test_with_transform_the_kinds_alternate(self):
        checks = paired.plan(20, ["a"], with_transform=True)
        self.assertEqual([c[1] for c in checks], ["repeat", "transform"] * 10)
        self.assertEqual([c[3] for c in checks[0:8:2]], list(paired.REPEAT_POLICIES))


class Repeat(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tiny = write_episode(Path(self.tmp.name) / "tiny")

    def tearDown(self):
        self.tmp.cleanup()

    def test_repeat_checks_pass_on_mini_v03_and_a_tiny_episode(self):
        out = io.StringIO()
        rows, summary = paired.run_paired([MINI, self.tiny], n=8, out=out)
        self.assertEqual([r["detail"] for r in rows if not r["ok"]], [])
        self.assertEqual(summary, {"paired": "OK", "passed": 8, "total": 8, "repeat": 8,
                                   "transform": paired.SKIPPED})
        self.assertEqual({(r["episode"], r["policy"]) for r in rows},
                         {(str(e), p) for e in (MINI, self.tiny) for p in paired.REPEAT_POLICIES})
        lines = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(lines, [*rows, summary])
        self.assertEqual(set(rows[0]), {"check", "kind", "episode", "policy", "ok", "detail"})

    def test_a_nondeterministic_policy_is_detected(self):
        rows, summary = paired.run_paired([MINI], n=2, runner=noisy_runner)
        self.assertEqual((summary["paired"], summary["passed"]), ("FAIL", 0))
        for row in rows:
            self.assertFalse(row["ok"])
            self.assertIn("trajectories differ at line", row["detail"])

    def test_a_reward_that_changes_between_runs_is_detected(self):
        """Same (empty) trajectory, different answers: the grader entries must differ."""
        calls = []

        def runner(policy, env, seed, config, policy_seed=0, trajectory=None, **kwargs):
            calls.append(policy)
            first = len(calls) % 2 == 1
            policy_fn = make_env_oracle(env) if first else (lambda message: submit([], "KARPSR"))
            return run_inprocess_episode(policy_fn, env, seed, config)

        rows, _ = paired.run_paired([MINI], n=1, runner=runner)
        self.assertFalse(rows[0]["ok"])
        self.assertIn("grader entries differ in", rows[0]["detail"])
        self.assertIn("reward (1.0 vs", rows[0]["detail"])

    def test_a_runner_error_fails_its_check_and_the_rest_still_run(self):
        def runner(policy, env, seed, config, policy_seed=0, trajectory=None, **kwargs):
            if policy == "random":
                raise RuntimeError("policy crashed")
            return run_policy(policy, env, seed, config, policy_seed, trajectory=trajectory)

        rows, summary = paired.run_paired([MINI], n=2, runner=runner)
        self.assertEqual([r["ok"] for r in rows], [False, True])
        self.assertEqual(rows[0]["detail"], "RuntimeError: policy crashed")
        self.assertEqual((summary["passed"], summary["total"]), (1, 2))


class TransformCli(unittest.TestCase):
    def test_identity_transform_passes(self):
        code, rows, summary, _ = paired_cli(MINI, "--n", "2", "--transform", "tests.fake_transforms:copy_unchanged")
        self.assertEqual(code, 0)
        self.assertEqual([(r["kind"], r["ok"]) for r in rows], [("repeat", True), ("transform", True)])
        self.assertIn("8 issues, oracle 1.0 on both", rows[1]["detail"])
        self.assertEqual((summary["paired"], summary["transform"], summary["repeat"]), ("OK", 1, 1))

    def test_a_dropped_truth_issue_fails_the_transform_check(self):
        code, rows, summary, _ = paired_cli(MINI, "--n", "2", "--transform", "tests.fake_transforms:drop_one_issue")
        self.assertEqual((code, summary["paired"], summary["passed"]), (1, "FAIL", 1))
        self.assertEqual([(r["kind"], r["ok"]) for r in rows], [("repeat", True), ("transform", False)])
        self.assertIn("truth (type, amount, currency) differs", rows[1]["detail"])

    def test_truth_left_on_the_old_ids_fails_the_transform_check(self):
        code, rows, _, _ = paired_cli(MINI, "--n", "2", "--transform", "tests.fake_transforms:rename_without_truth")
        self.assertEqual(code, 1)
        self.assertFalse(rows[1]["ok"])
        self.assertIn("truth of the copy names ids with no record: ['ledger/", rows[1]["detail"])
        self.assertNotIn("truth of the original", rows[1]["detail"])

    def test_missing_transform_module_is_a_skip_not_a_failure(self):
        code, rows, summary, _ = paired_cli(MINI, "--n", "2", "--transform", "tests.no_such_module:permute_and_rename")
        self.assertEqual(code, 0)
        self.assertEqual([(r["kind"], r["ok"]) for r in rows], [("repeat", True), ("repeat", True)])
        self.assertEqual(summary, {"paired": "OK", "passed": 2, "total": 2, "repeat": 2, "transform": paired.SKIPPED})

    def test_missing_function_is_a_skip_too(self):
        self.assertIsNone(paired.load_transform("tests.fake_transforms:no_such_function"))
        self.assertIsNone(paired.load_transform("no_such_package.transform:permute_and_rename"))
        self.assertTrue(callable(paired.load_transform("tests.fake_transforms:copy_unchanged")))

    def test_a_missing_episode_dir_is_a_usage_error(self):
        code, rows, summary, stderr = paired_cli(ROOT / "no_such_episode", "--n", "1")
        self.assertEqual((code, rows, summary), (2, [], None))
        self.assertIn("not a directory", stderr)


if __name__ == "__main__":
    unittest.main()
