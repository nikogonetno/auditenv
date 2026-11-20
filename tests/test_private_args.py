"""The episode seed and config never appear in a process command line while a policy runs (QA N-1)."""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from auditenv.private_args import split_private
from tests.support import run_python

ROOT = Path(__file__).resolve().parent.parent
MINI = ROOT / "tests" / "fixtures" / "mini_v03"

# Scans every /proc/<pid>/cmdline for the seed. The literal number is not in this source, so the
# probe does not find itself.
PROBE = textwrap.dedent("""
    import json, os, sys
    sys.stdin.buffer.readline()
    marker = str(98000 + 765).encode()
    hits = 0
    for name in os.listdir("/proc"):
        if name.isdigit() and int(name) != os.getpid():
            try:
                with open(f"/proc/{name}/cmdline", "rb") as f:
                    hits += marker in f.read()
            except OSError:
                pass
    sys.stdout.write(json.dumps({"action": "calculator", "expression": json.dumps({"hits": hits})}) + "\\n")
    sys.stdout.flush()
    sys.stdin.buffer.readline()
""")

DRIVER = textwrap.dedent(f"""
    import io, json, sys
    sys.path.insert(0, {str(ROOT)!r})
    from auditenv.env import AuditEnv
    from auditenv.private_args import hide_private_args
    from auditenv.runner import run_subprocess_episode
    argv = sys.argv[1:] if "--no-hide" in sys.argv else hide_private_args()
    seed = int(argv[argv.index("--seed") + 1])
    trajectory = io.StringIO()
    run_subprocess_episode([sys.executable, "-c", {PROBE!r}], AuditEnv("eval"), seed,
                           {{"episode_dir": {str(MINI)!r}}}, time_limit=20, trajectory=trajectory)
    rows = [json.loads(line) for line in trajectory.getvalue().splitlines()]
    print(json.dumps({{"seed": seed, "probe": json.loads(rows[1]["action"]["expression"])}}))
""")


class Split(unittest.TestCase):
    def test_both_option_forms_are_moved_and_others_stay(self):
        public, private = split_private(["--policy", "random", "--seed", "7", "--config={}", "--policy-seed", "3", "--mode", "eval"])
        self.assertEqual(public, ["--policy", "random", "--policy-seed", "3", "--mode", "eval"])
        self.assertEqual(private, ["--seed", "7", "--config={}"])

    def test_flag_without_value_stays_for_argparse(self):
        self.assertEqual(split_private(["--policy", "x", "--seed"]), (["--policy", "x", "--seed"], []))


@unittest.skipUnless(os.path.isdir("/proc") and os.name == "posix", "needs /proc")
class NoSeedInCommandLines(unittest.TestCase):
    def run_driver(self, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            driver = Path(tmp) / "driver.py"
            driver.write_text(DRIVER, encoding="utf-8")
            result = run_python([driver, "--policy", "probe", "--seed", "98765", *extra], cwd=tmp)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_policy_cannot_find_the_seed_in_any_command_line(self):
        out = self.run_driver()
        self.assertEqual((out["seed"], out["probe"]["hits"]), (98765, 0))

    def test_control_without_hiding_the_probe_does_find_it(self):
        self.assertGreaterEqual(self.run_driver("--no-hide")["probe"]["hits"], 1)


def run_cli(*args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    return subprocess.run([sys.executable, "-m", "auditenv.run", *map(str, args)], capture_output=True, text=True,
                          cwd=ROOT, env=env, timeout=120)


class CommandLineStillWorks(unittest.TestCase):
    def test_run_cli_keeps_the_seed_after_re_exec(self):
        result = run_cli("--policy", "random", "--episode-dir", MINI, "--seed", "4242")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["seed"], 4242)

    def test_abbreviated_private_flag_is_refused(self):
        result = run_cli("--policy", "random", "--episode-dir", MINI, "--se", "1")
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
