"""Policy sandbox (QA risk R-03, SPEC §4.3). The real checks need root and the auditpolicy user,
so they run inside the Docker image (make docker-verify, CI) and are skipped elsewhere.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from auditenv.env import AuditEnv
from auditenv.runner import REPO_ROOT, _processes_of, run_subprocess_episode, sandbox_user

from tests.tiny_episode import EPISODE_ID, write_episode

USER = sandbox_user()
FIXTURE_TRUTH = REPO_ROOT / "tests" / "fixtures" / "mini_v03" / "private" / "truth.json"


def probe_policy(body):
    """A policy that runs `body` (setting `report`) and sends it back as a calculator expression."""
    source = textwrap.dedent("""
        import json, os, sys
        sys.stdin.buffer.readline()
        report = {}
    """) + textwrap.dedent(body) + textwrap.dedent("""
        sys.stdout.write(json.dumps({"action": "calculator", "expression": json.dumps(report)}) + "\\n")
        sys.stdout.flush()
        sys.stdin.buffer.readline()
    """)
    return [sys.executable, "-c", source]


class SandboxCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = write_episode(Path(self.tmp.name) / "ep")
        self.config = {"episode_dir": str(self.dir)}

    def tearDown(self):
        self.tmp.cleanup()

    def probe(self, body, sandbox="required"):
        env, trajectory = AuditEnv("eval"), io.StringIO()
        result = run_subprocess_episode(probe_policy(body), env, 1, self.config, time_limit=20,
                                        trajectory=trajectory, sandbox=sandbox)
        rows = [json.loads(line) for line in trajectory.getvalue().splitlines()]
        return result, json.loads(rows[1]["action"]["expression"])


@unittest.skipIf(USER is not None, "sandbox available here; the fallback is tested where it is not")
class WithoutRoot(SandboxCase):
    def test_required_sandbox_refuses_to_run_without_root(self):
        with self.assertRaises(RuntimeError):
            run_subprocess_episode([sys.executable, "-c", "pass"], AuditEnv(), 1, self.config, sandbox="required")

    def test_auto_falls_back_and_says_so(self):
        result, report = self.probe("report['uid'] = os.getuid()", sandbox="auto")
        self.assertEqual((result["sandbox"], report["uid"]), ("none", os.getuid()))


@unittest.skipIf(USER is None, "needs root and the auditpolicy user: runs in the Docker image")
class WithRoot(SandboxCase):
    def test_policy_runs_as_the_sandbox_user(self):
        result, report = self.probe("report['ids'] = [os.getuid(), os.geteuid(), os.getgid(), os.getgroups()]")
        self.assertEqual(result["sandbox"], "uid")
        self.assertEqual(report["ids"], [USER[0], USER[0], USER[1], []])

    def test_policy_cannot_read_private_files(self):
        # An episode the policy can walk into, closed only at private/ the way the Docker image closes
        # every private/ directory. Built here, so the test does not depend on how the repo was copied
        # (QA N-9: as root outside Docker the fixtures keep their 0644 modes).
        open_root = Path(tempfile.mkdtemp(prefix="auditenv-open-"))
        try:
            os.chmod(open_root, 0o755)
            shutil.copytree(FIXTURE_TRUTH.parent.parent, open_root / "ep")
            for path in (open_root / "ep").rglob("*"):
                os.chmod(path, 0o755 if path.is_dir() else 0o644)
            os.chmod(open_root / "ep", 0o755)
            os.chmod(open_root / "ep" / "private", 0o700)
            body = f"""
                for name, path in {{"episode": {str(self.dir / "private" / "truth.json")!r},
                                    "image_like": {str(open_root / "ep" / "private" / "truth.json")!r},
                                    "image_like_public": {str(open_root / "ep" / "public" / "manifest.json")!r}}}.items():
                    try:
                        open(path).read(1)
                        report[name] = "read"
                    except OSError as exc:
                        report[name] = type(exc).__name__
            """
            _, report = self.probe(body)
        finally:
            shutil.rmtree(open_root, ignore_errors=True)
        self.assertEqual(report, {"episode": "PermissionError", "image_like": "PermissionError", "image_like_public": "read"})

    @unittest.skipUnless(os.path.exists("/.dockerenv"), "the image closes private/ directories; only checked inside it")
    def test_image_closes_every_private_directory(self):
        opened = [str(p) for d in REPO_ROOT.rglob("private") if d.is_dir()
                  for p in [d, *d.rglob("*")] if p.stat().st_mode & 0o077]
        self.assertEqual(opened, [])

    def test_policy_cannot_write_the_grader_or_signal_the_runner(self):
        body = f"""
            report["grader_writable"] = os.access({str(REPO_ROOT / "auditenv" / "grader" / "reward.py")!r}, os.W_OK)
            report["repo_writable"] = os.access({str(REPO_ROOT)!r}, os.W_OK)
            try:
                os.kill(os.getppid(), 0)
                report["signal"] = "allowed"
            except OSError as exc:
                report["signal"] = type(exc).__name__
            try:
                report["parent_environ"] = len(open(f"/proc/{{os.getppid()}}/environ", "rb").read())
            except OSError as exc:
                report["parent_environ"] = type(exc).__name__
        """
        _, report = self.probe(body)
        self.assertEqual(report, {"grader_writable": False, "repo_writable": False,
                                  "signal": "PermissionError", "parent_environ": "PermissionError"})

    def test_escaped_grandchild_is_killed_after_the_episode(self):
        body = """
            if os.fork() == 0:
                os.setsid()
                import time
                time.sleep(60)
                os._exit(0)
            report["forked"] = True
        """
        result, report = self.probe(body)
        self.assertTrue(report["forked"])
        self.assertEqual(_processes_of(USER[0]), [])

    def test_honest_episode_still_scores_inside_the_sandbox(self):
        submit = json.dumps({"action": "submit", "findings": {"episode_id": EPISODE_ID, "findings": []}})
        cmd = [sys.executable, "-c", f"import sys; sys.stdin.readline(); print({submit!r}, flush=True); sys.stdin.readline()"]
        env = AuditEnv("eval")
        result = run_subprocess_episode(cmd, env, 1, self.config, time_limit=20, sandbox="required")
        self.assertEqual((result["terminated"], result["sandbox"]), (True, "uid"))
        self.assertEqual(env.grader_log[-1]["reward"], -1.0)


if __name__ == "__main__":
    unittest.main()
