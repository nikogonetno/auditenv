"""The CLI under signals, a closed stdout and bad or full log paths, and the truncated count of a set run
(QA L16, L17, L18, L19).

Ctrl-C, SIGTERM and SIGHUP leave auditenv.run through its finally blocks with one line on stderr: the policy
process group is killed and the temporary folders are removed, also during the check of a set, whose folder
holds built episodes with private/. A closed stdout stops the run quietly. A log path in a missing folder is a
one-line error before anything runs, a full disk a one-line error during it, and an existing log file becomes
0600.
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


def cli(*args, tmpdir=None, preexec_fn=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env = {**env, "PYTHONPATH": str(ROOT), **({"TMPDIR": str(tmpdir)} if tmpdir else {})}
    return subprocess.Popen([sys.executable, "-P", "-s", "-m", "auditenv.run", *args], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=preexec_fn)


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

    def hanging_policy(self, before_reading="", after_first_line="time.sleep(120)"):
        report = self.root / "policy.txt"
        policy = self.root / "policy.py"
        policy.write_text(f"import json, os, signal, sys, time\n{before_reading}\n"
                          f"open({str(report)!r}, 'w').write(f'{{os.getpid()}}')\n"
                          f"first = json.loads(sys.stdin.readline())\n{after_first_line}\n", encoding="utf-8")
        return policy, report

    def test_sigterm_in_the_grace_wait_after_the_time_limit_still_stops_the_policy(self):
        """The policy ignores the end of its stdin; the runner waits up to 2 s for it after the 1 s limit."""
        policy, report = self.hanging_policy(before_reading="signal.signal(signal.SIGPIPE, signal.SIG_IGN)")
        proc = cli("--policy-cmd", f"{sys.executable} {policy}", "--episode-dir", str(EPISODE), "--mode", "train",
                   "--sandbox", "off", "--time-limit", "1", tmpdir=self.tmpdir)
        self.assertTrue(wait_for(lambda: report.exists() and report.read_text()), "the policy did not start")
        pid = int(report.read_text())
        time.sleep(1.6)                                  # past the 1 s limit, inside the 2 s grace wait
        proc.send_signal(signal.SIGTERM)
        _, err = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 128 + signal.SIGTERM, err)
        self.assertTrue(wait_for(lambda: not alive(pid), 10), "the policy outlived the runner")
        self.assertEqual(self.left_in_tmpdir(), [])

    def test_ctrl_c_stops_with_one_line_and_no_traceback(self):
        """QA L19: Ctrl-C behaves as SIGTERM does: cleanup, one line on stderr, exit 130."""
        policy, report = self.hanging_policy()
        proc = cli("--policy-cmd", f"{sys.executable} {policy}", "--episode-dir", str(EPISODE), "--mode", "train",
                   "--sandbox", "off", tmpdir=self.tmpdir)
        self.assertTrue(wait_for(lambda: report.exists() and report.read_text()), "the policy did not start")
        pid = int(report.read_text())
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=60)
        self.assertEqual((proc.returncode, out), (128 + signal.SIGINT, ""))
        self.assertEqual(err, f"auditenv.run: stopped by signal {int(signal.SIGINT)}\n")
        self.assertTrue(wait_for(lambda: not alive(pid), 10), "the policy outlived the runner")
        self.assertEqual(self.left_in_tmpdir(), [])

    @unittest.skipUnless(os.path.exists("/dev/full"), "needs /dev/full (Linux)")
    def test_a_full_disk_is_a_one_line_error(self):
        """QA L19: a log write that fails says so in one line, with exit 2, and names where the logs stop."""
        for flag in ("--trajectories", "--grader-log"):
            with self.subTest(flag=flag):
                proc = cli("--policy", "oracle", "--episode-dir", str(EPISODE), flag, "/dev/full")
                out, err = proc.communicate(timeout=60)
                self.assertEqual((proc.returncode, out), (2, ""))
                self.assertEqual(len(err.splitlines()), 1, err)
                self.assertIn("cannot write /dev/full: No space left on device", err)

    def test_a_closed_stdout_stops_the_set_quietly(self):
        """QA L19: `| head -1` stops the run as a SIGPIPE would, without a traceback or exit code 120."""
        dev = json.loads((ROOT / "configs" / "dev_v2.json").read_text(encoding="utf-8"))
        set_file = self.root / "three.json"
        set_file.write_text(json.dumps({"version": "three", "episodes": dev["episodes"][:3]}), encoding="utf-8")
        proc = cli("--policy", "oracle", "--config", str(set_file), tmpdir=self.tmpdir)
        self.assertTrue(json.loads(proc.stdout.readline())["episode_id"])
        proc.stdout.close()                              # the reader goes away after the first line
        err = proc.stderr.read()
        proc.wait(timeout=60)
        proc.stderr.close()
        self.assertEqual(proc.returncode, 128 + 13, err)
        self.assertNotIn("Traceback", err)
        self.assertEqual(self.left_in_tmpdir(), [])

    def test_a_closed_or_full_stdout_is_a_one_line_error(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        base = [sys.executable, "-P", "-s", "-m", "auditenv.run", "--policy", "oracle", "--episode-dir", str(EPISODE)]
        cases = [("closed", {"stdout": None, "pass_fds": (), "preexec_fn": lambda: os.close(1)})]
        if os.path.exists("/dev/full"):
            cases.append(("full", {"stdout": open("/dev/full", "w")}))
        for name, extra in cases:
            with self.subTest(stdout=name):
                proc = subprocess.run(base, cwd=ROOT, env={**env, "PYTHONPATH": str(ROOT)}, stderr=subprocess.PIPE,
                                      text=True, timeout=60, **extra)
                if extra.get("stdout") is not None:
                    extra["stdout"].close()
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertEqual(len(proc.stderr.splitlines()), 1, proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)

    def test_a_hangup_ignored_at_start_stays_ignored(self):
        """Under nohup a SIGHUP must not stop the run."""
        submit = ("time.sleep(1)\nprint(json.dumps({'action': 'submit', 'findings': {'episode_id': "
                  "first['observation']['manifest']['episode_id'], 'findings': []}}), flush=True)\n"
                  "sys.stdin.readline()")
        policy, report = self.hanging_policy(after_first_line=submit)
        proc = cli("--policy-cmd", f"{sys.executable} {policy}", "--episode-dir", str(EPISODE), "--mode", "train",
                   "--sandbox", "off", preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
        self.assertTrue(wait_for(lambda: report.exists() and report.read_text()), "the policy did not start")
        proc.send_signal(signal.SIGHUP)
        out, err = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 0, err)
        self.assertTrue(json.loads(out)["terminated"])

    def test_sigterm_with_a_closed_stderr_still_exits_143(self):
        policy, report = self.hanging_policy()
        proc = cli("--policy-cmd", f"{sys.executable} {policy}", "--episode-dir", str(EPISODE), "--mode", "train",
                   "--sandbox", "off", tmpdir=self.tmpdir)
        self.assertTrue(wait_for(lambda: report.exists() and report.read_text()), "the policy did not start")
        proc.stderr.close()                              # as a terminal that hung up
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=60)
        proc.stdout.close()
        self.assertEqual(proc.returncode, 128 + signal.SIGTERM)

    def test_a_device_as_log_keeps_its_mode(self):
        before = stat.S_IMODE(os.stat(os.devnull).st_mode)
        proc = cli("--policy", "oracle", "--episode-dir", str(EPISODE), "--grader-log", os.devnull)
        _, err = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 0, err)
        self.assertEqual(stat.S_IMODE(os.stat(os.devnull).st_mode), before)

    def test_a_removal_cut_short_by_a_signal_runs_once_more(self):
        from unittest import mock
        from auditenv import runner
        folder = self.root / "victim"
        (folder / "private").mkdir(parents=True)
        (folder / "private" / "truth.json").write_text("{}", encoding="utf-8")
        real = runner.shutil.rmtree
        calls = []

        def cut_short(path, ignore_errors=False):
            calls.append(path)
            if len(calls) == 1:
                raise SystemExit(143)                    # what auditenv.run's SIGTERM handler raises
            return real(path, ignore_errors=ignore_errors)

        with mock.patch.object(runner.shutil, "rmtree", cut_short), self.assertRaises(SystemExit):
            runner.remove_tree(folder)
        self.assertEqual(len(calls), 2)
        self.assertFalse(folder.exists())

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
