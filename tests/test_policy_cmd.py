"""--policy-cmd: an external policy through the CLI (docs/tasks/B_afternoon.md items 1 and 2, QA N-3, N-4).

The minimal policy of README section 4 is examples/my_policy.py. Through the CLI it must get the same
reward and the same trajectory as through the Python API. In the Docker step of CI the runner is root, so
the policy runs as the sandbox user auditpolicy, and these tests check that it did.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from auditenv.env import AuditEnv
from auditenv.run import NO_SANDBOX_WARNING, policy_command
from auditenv.runner import POLICY_USER, run_subprocess_episode, sandbox_user

ROOT = Path(__file__).resolve().parent.parent
POLICY = ROOT / "examples" / "my_policy.py"
EPISODE = ROOT / "tests" / "fixtures" / "mini_v03"
SANDBOXED = sandbox_user() is not None        # root where the user auditpolicy exists: the Docker image


def run_cli(*args, cwd=ROOT):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    return subprocess.run([sys.executable, "-P", "-s", "-m", "auditenv.run", *args], capture_output=True,
                          text=True, cwd=cwd, env={**env, "PYTHONPATH": str(ROOT)}, timeout=120)


class PolicyCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_example_is_the_minimal_policy_of_the_readme(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8").split("## 4. Протокол для внешней политики", 1)[1]
        self.assertEqual(re.search(r"```python\n(.*?)```", readme, re.S).group(1), POLICY.read_text(encoding="utf-8"))

    def test_cli_and_python_api_give_the_same_reward_and_trajectory(self):
        cli_traj, api_traj = self.root / "cli.jsonl", self.root / "api.jsonl"
        proc = run_cli("--policy-cmd", f"{sys.executable} examples/my_policy.py", "--episode-dir", str(EPISODE),
                       "--mode", "train", "--trajectories", str(cli_traj))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        env = AuditEnv("train")
        with open(api_traj, "w", encoding="utf-8", newline="\n") as trajectory:
            result = run_subprocess_episode([sys.executable, str(POLICY)], env, 0, {"episode_dir": str(EPISODE)},
                                            trajectory=trajectory)
        self.assertEqual(report["policy"], "external")
        self.assertEqual((report["steps"], report["terminated"], report["truncated"]), (2, True, False))
        self.assertEqual(report["reward"], env.grader_log[-1]["reward"])
        self.assertEqual(report["steps"], result["steps"])
        self.assertEqual(cli_traj.read_bytes(), api_traj.read_bytes())
        self.assertEqual(report["sandbox"], "uid" if SANDBOXED else "none")

    @unittest.skipUnless(SANDBOXED, f"needs root and the user {POLICY_USER} (the Docker image)")
    def test_in_docker_the_policy_runs_as_the_sandbox_user_and_eval_does_not_warn(self):
        proc = run_cli("--policy-cmd", f"{sys.executable} {POLICY}", "--episode-dir", str(EPISODE),
                       "--mode", "eval", "--sandbox", "required")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["sandbox"], "uid")
        self.assertNotIn("WARNING", proc.stderr)

    @unittest.skipIf(SANDBOXED, "the sandbox is on here")
    def test_eval_without_the_sandbox_warns_in_one_line_and_still_runs(self):
        for policy in (["--policy-cmd", f"{sys.executable} {POLICY}"], ["--policy", "flag_all"]):
            with self.subTest(policy=policy[0]):
                proc = run_cli(*policy, "--episode-dir", str(EPISODE), "--mode", "eval")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stderr, NO_SANDBOX_WARNING + "\n")
                self.assertIn("reward", json.loads(proc.stdout))

    def test_no_warning_in_train_mode_or_for_the_oracle(self):
        for args in (["--policy-cmd", f"{sys.executable} {POLICY}", "--mode", "train"],
                     ["--policy", "oracle", "--mode", "eval"]):
            with self.subTest(args=args):
                proc = run_cli(*args, "--episode-dir", str(EPISODE))
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stderr, "")

    def test_policy_and_policy_cmd_exclude_each_other_and_one_is_required(self):
        both = run_cli("--policy", "random", "--policy-cmd", "python3 x.py", "--episode-dir", str(EPISODE))
        neither = run_cli("--episode-dir", str(EPISODE))
        empty = run_cli("--policy-cmd", "  ", "--episode-dir", str(EPISODE))
        self.assertEqual((both.returncode, neither.returncode, empty.returncode), (2, 2, 2))
        self.assertIn("not allowed with argument --policy", both.stderr)
        self.assertIn("one of the arguments --policy --policy-cmd is required", neither.stderr)
        self.assertIn("--policy-cmd is empty", empty.stderr)

    def test_relative_paths_to_existing_files_become_absolute(self):
        cwd = os.getcwd()
        try:
            os.chdir(ROOT)
            argv = policy_command("python3 examples/my_policy.py --flag missing.txt")
        finally:
            os.chdir(cwd)
        self.assertEqual(argv, ["python3", str(POLICY), "--flag", "missing.txt"])
        self.assertEqual(policy_command("python3 'a b.py'"), ["python3", "a b.py"])


if __name__ == "__main__":
    unittest.main()
