"""The CLI under signals and with bad log paths, and the truncated count of a set run (QA L16, L17, L18).

SIGTERM or SIGHUP leaves auditenv.run through its finally blocks, as Ctrl-C does: the policy process group is
killed and the temporary folders are removed, also during the check of a set, whose folder holds built
episodes with private/. A log path in a missing folder is a one-line error before anything runs, and an
existing log file becomes 0600.
"""
import glob
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EPISODE = ROOT / "tests" / "fixtures" / "mini_v03"


def cli(*args, tmpdir=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env = {**env, "PYTHONPATH": str(ROOT), **({"TMPDIR": str(tmpdir)} if tmpdir else {})}
    return subprocess.Popen([sys.executable, "-P", "-s", "-m", "auditenv.run", *args], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def wait_for(condition, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class Robustness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.tmpdir = self.root / "tmp"
        self.tmpdir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def left_in_tmpdir(self):
        return sorted(str(p.relative_to(self.tmpdir)) for p in self.tmpdir.rglob("*"))

    def test_sigterm_stops_the_policy_and_removes_its_folder(self):
        report = self.root / "policy.txt"
        policy = self.root / "hang.py"
        policy.write_text(f"import os, sys, time\nopen({str(report)!r}, 'w').write(f'{{os.getpid()}}')\n"
                          "sys.stdin.readline()\ntime.sleep(120)\n", encoding="utf-8")
        proc = cli("--policy-cmd", f"{sys.executable} {policy}", "--episode-dir", str(EPISODE), "--mode", "train",
                   "--sandbox", "off", tmpdir=self.tmpdir)
        self.assertTrue(wait_for(lambda: report.exists() and report.read_text()), "the policy did not start")
        pid = int(report.read_text())
        proc.send_signal(signal.SIGTERM)
        _, err = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 128 + signal.SIGTERM, err)
        self.assertIn("stopped by signal", err)
        self.assertTrue(wait_for(lambda: not alive(pid), 10), "the policy outlived the runner")
        self.assertEqual(self.left_in_tmpdir(), [])

    def test_sigterm_during_the_set_check_leaves_no_built_episode(self):
        traj = self.root / "t.jsonl"
        proc = cli("--policy", "oracle", "--config", "configs/dev_v2.json", "--trajectories", str(traj),
                   tmpdir=self.tmpdir)
        built = os.path.join(str(self.tmpdir), "auditenv-set-check-*", "*", "private")
        self.assertTrue(wait_for(lambda: glob.glob(built)), "the set check did not start")
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 128 + signal.SIGTERM)
        self.assertEqual(self.left_in_tmpdir(), [])
        self.assertFalse(traj.exists())

    def test_a_log_path_that_cannot_be_written_is_a_one_line_error(self):
        for flag, path in (("--trajectories", self.root / "missing" / "t.jsonl"),
                           ("--grader-log", self.root / "missing" / "g.jsonl"), ("--grader-log", self.root)):
            with self.subTest(flag=flag, path=path):
                proc = cli("--policy", "oracle", "--episode-dir", str(EPISODE), flag, str(path))
                out, err = proc.communicate(timeout=60)
                self.assertEqual((proc.returncode, out), (2, ""))
                self.assertEqual(len(err.splitlines()), 1, err)
                self.assertIn(f"auditenv.run: {flag}:", err)

    def test_an_existing_log_file_becomes_private(self):
        log = self.root / "g.jsonl"
        log.write_text("", encoding="utf-8")
        log.chmod(0o644)
        proc = cli("--policy", "oracle", "--episode-dir", str(EPISODE), "--grader-log", str(log))
        _, err = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 0, err)
        self.assertEqual(stat.S_IMODE(log.stat().st_mode), 0o600)
        self.assertEqual(len(log.read_text(encoding="utf-8").splitlines()), 1)

    def test_the_summary_counts_truncated_episodes(self):
        """QA L16: a policy that dies at once on clean companies scores by the case rules, but the summary
        says that no answer came."""
        set_file = self.root / "clean.json"
        set_file.write_text(json.dumps({"version": "clean", "episodes": [{"seed": 1, "config": {"errors": {}}},
                                                                         {"seed": 2, "config": {"errors": {}}}]}))
        policy = self.root / "die.py"
        policy.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
        proc = cli("--policy-cmd", f"{sys.executable} {policy}", "--config", str(set_file), "--mode", "train",
                   "--sandbox", "off")
        out, err = proc.communicate(timeout=120)
        self.assertEqual(proc.returncode, 0, err)
        total = json.loads(out.splitlines()[-1])
        self.assertEqual((total["episodes"], total["truncated"], total["invalid"]), (2, 2, 0))


if __name__ == "__main__":
    unittest.main()
