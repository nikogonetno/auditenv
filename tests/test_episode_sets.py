"""Episode sets in --config and the closed run of a whole set (docs/tasks/B_qa_fixes.md item 1; QA M1, M4, L3).

configs/dev_v2.json is a set {version, split, comment, episodes: [{seed, config}]}. With --seed the CLI takes
that episode, without it the whole set, one report line per episode and a summary line. In the Docker image
the test also checks that the sandbox user can read neither the logs of a set run nor the set file.
"""
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from auditenv.runner import POLICY_USER, sandbox_user

ROOT = Path(__file__).resolve().parent.parent
POLICY = ROOT / "examples" / "my_policy.py"
SANDBOXED = sandbox_user() is not None        # root where the user auditpolicy exists: the Docker image


def run_cli(*args, umask=-1):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    return subprocess.run([sys.executable, "-P", "-s", "-m", "auditenv.run", *args], capture_output=True,
                          text=True, cwd=ROOT, env={**env, "PYTHONPATH": str(ROOT)}, timeout=300, umask=umask)


def lines(text):
    return [json.loads(line) for line in text.splitlines()]


class EpisodeSets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        dev = json.loads((ROOT / "configs" / "dev_v2.json").read_text(encoding="utf-8"))
        self.pair = {"version": "jury_test", "split": "test", "episodes": dev["episodes"][:2]}
        self.set_file = self.root / "pair.json"
        self.set_file.write_text(json.dumps(self.pair), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_seed_of_the_dev_set_is_the_committed_dev_episode(self):
        proc = run_cli("--policy", "oracle", "--config", "configs/dev_v2.json", "--seed", "2000")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        (report,) = lines(proc.stdout)
        index = {e["seed"]: e for e in json.loads((ROOT / "episodes" / "dev" / "index.json").read_text())["episodes"]}
        self.assertEqual((report["episode_id"], report["config_sha256"]),
                         (index[2000]["episode_id"], index[2000]["config_sha256"]))
        self.assertEqual((report["seed"], report["episode_set"], report["reward"]), (2000, "dev_v2", 1.0))
        self.assertIn("code_version", report)

    def test_without_seed_every_episode_runs_into_one_log_and_a_summary_line(self):
        traj, log = self.root / "t.jsonl", self.root / "g.jsonl"
        proc = run_cli("--policy", "oracle", "--config", str(self.set_file), "--mode", "train",
                       "--trajectories", str(traj), "--grader-log", str(log))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        first, second, total = lines(proc.stdout)
        self.assertEqual([first["seed"], second["seed"]], [2000, 2001])
        self.assertEqual((total["episode_set"], total["split"], total["episodes"], total["mean_reward"]),
                         ("jury_test", "test", 2, 1.0))
        self.assertEqual(total["tp"], first["tp"] + second["tp"])
        self.assertEqual({"tp", "fp", "fn", "precision", "recall", "f1", "mean_steps", "invalid"} - set(total), set())
        self.assertEqual(len(log.read_text(encoding="utf-8").splitlines()), 2)
        episodes = {json.loads(line)["episode_id"] for line in traj.read_text(encoding="utf-8").splitlines()}
        self.assertEqual(episodes, {first["episode_id"], second["episode_id"]})

    def test_a_seed_not_in_the_set_and_a_broken_set_are_clean_errors(self):
        proc = run_cli("--policy", "oracle", "--config", str(self.set_file), "--seed", "7")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("seed 7 is not in the episode set; its seeds: 2000, 2001", proc.stderr)
        for broken in ({**self.pair, "extra": 1}, {**self.pair, "episodes": []},
                       {**self.pair, "episodes": [{"seed": "1", "config": {}}]},
                       {**self.pair, "episodes": [{"seed": 1, "config": {}}, {"seed": 1, "config": {}}]}):
            with self.subTest(broken=broken):
                self.set_file.write_text(json.dumps(broken), encoding="utf-8")
                proc = run_cli("--policy", "oracle", "--config", str(self.set_file))
                self.assertEqual(proc.returncode, 2, proc.stdout)
                self.assertIn("auditenv.run:", proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)

    def test_a_broken_submit_counts_its_issues_as_missed_in_the_summary(self):
        policy = self.root / "broken.py"
        policy.write_text("import json, sys\nsys.stdin.readline()\n"
                          "print(json.dumps({'action': 'submit', 'findings': {'episode_id': 'WRONG', 'findings': "
                          "[{'type': 'x'}]}}), flush=True)\nsys.stdin.readline()\n", encoding="utf-8")
        # The summary is the point here, not the sandbox: in the image the policy file in this root-only
        # temporary folder would be out of reach for auditpolicy.
        proc = run_cli("--policy-cmd", f"{sys.executable} {policy}", "--config", str(self.set_file), "--mode", "train",
                       "--sandbox", "off")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        first, second, total = lines(proc.stdout)
        self.assertEqual((first["error"], first["tp"]), ("episode_id_mismatch", None))
        issues = sum(e["issues"] for e in json.loads((ROOT / "episodes" / "dev" / "index.json").read_text())["episodes"]
                     if e["seed"] in (2000, 2001))
        self.assertEqual((total["invalid"], total["tp"], total["fp"], total["fn"]), (2, 0, 2, issues))
        self.assertEqual((total["f1"], total["mean_reward"]), (0.0, -1.0))

    def test_broken_submits_on_clean_companies_are_not_a_perfect_answer(self):
        """QA H1: a broken empty submit must not look like the honest empty answer on a clean book."""
        clean = {"version": "clean", "episodes": [{"seed": 1, "config": {"errors": {}}},
                                                  {"seed": 2, "config": {"errors": {}}}]}
        self.set_file.write_text(json.dumps(clean), encoding="utf-8")
        policy = self.root / "broken.py"
        policy.write_text("import json, sys\nsys.stdin.readline()\nprint(json.dumps({'action': 'submit', 'findings': "
                          "{'episode_id': 'WRONG', 'findings': []}}), flush=True)\nsys.stdin.readline()\n",
                          encoding="utf-8")
        proc = run_cli("--policy-cmd", f"{sys.executable} {policy}", "--config", str(self.set_file), "--mode", "train",
                       "--sandbox", "off")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        total = lines(proc.stdout)[-1]
        self.assertEqual((total["invalid"], total["tp"], total["fp"], total["fn"]), (2, 0, 2, 0))
        self.assertEqual((total["f1"], total["mean_reward"]), (0.0, -1.0))

    def test_a_bad_item_stops_the_set_before_any_episode(self):
        """QA M5 and L13: every item is checked first; the message names the item; nothing is written."""
        traj = self.root / "t.jsonl"
        for broken, message in (
                ({**self.pair, "episodes": [*self.pair["episodes"], {"seed": 5, "config": {"errors": {"fraud": 1}}}]},
                 "episode set item 3 (seed 5): unknown config.errors key: fraud"),
                ({"episodes": self.pair["episodes"]}, "an episode set needs a non-empty string version"),
                ({**self.pair, "split": 1}, "episode set key split must be a string")):
            with self.subTest(message=message):
                self.set_file.write_text(json.dumps(broken), encoding="utf-8")
                proc = run_cli("--policy", "oracle", "--config", str(self.set_file), "--trajectories", str(traj))
                self.assertEqual((proc.returncode, proc.stdout), (2, ""))
                self.assertIn(message, proc.stderr)
                self.assertFalse(traj.exists())

    def test_the_mean_reward_is_exact(self):
        """QA L14: no float noise such as 0.39999999999999997 for the rewards 0.1 and 0.7."""
        from auditenv.run import set_summary
        reports = [{"reward": 0.1, "steps": 3, "tp": 1}, {"reward": 0.7, "steps": 4, "tp": 1}]
        total = set_summary({"version": "v", "split": None}, reports, [(1, 0, 0), (1, 0, 0)])
        self.assertEqual((total["mean_reward"], total["mean_steps"]), (0.4, 3.5))

    def test_one_config_still_works_and_reports_its_hash(self):
        proc = run_cli("--policy", "oracle", "--config", "{}", "--seed", "7")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        (report,) = lines(proc.stdout)
        self.assertEqual((report["seed"], len(report["config_sha256"])), (7, 64))
        self.assertNotIn("episode_set", report)

    @unittest.skipIf(SANDBOXED, "the sandbox is on here")
    def test_sandbox_required_without_root_refuses_with_one_message(self):
        proc = run_cli("--policy-cmd", f"{sys.executable} {POLICY}", "--config", "{}", "--seed", "7",
                       "--mode", "eval", "--sandbox", "required")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(len(proc.stderr.splitlines()), 1, proc.stderr)
        self.assertIn("sandbox unavailable", proc.stderr)
        self.assertEqual(proc.stdout, "")

    @unittest.skipUnless(SANDBOXED, f"needs root and the user {POLICY_USER} (the Docker image)")
    def test_in_docker_the_sandbox_user_reads_neither_the_logs_nor_the_set(self):
        os.chmod(self.root, 0o755)                         # the files alone must keep the policy out
        hidden, out = self.root / "in", self.root / "out"
        hidden.mkdir()
        out.mkdir()
        os.chmod(hidden, 0o700)
        os.chmod(out, 0o755)                               # open: only the modes of the files keep the policy out
        os.chown(out, 4242, 4242)                          # as a jury's folder: the runner hands the logs back
        set_file = hidden / "hidden_set.json"
        set_file.write_text(json.dumps(self.pair), encoding="utf-8")
        # umask 022 as in a jury's `docker run ... python3 -m auditenv.run`: the CLI's own umask 077 must hold.
        proc = run_cli("--policy-cmd", f"python3 {POLICY}", "--config", str(set_file), "--mode", "eval",
                       "--sandbox", "required", "--trajectories", str(out / "trajectories.jsonl"),
                       "--grader-log", str(out / "grader_log.jsonl"), umask=0o022)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        reports = lines(proc.stdout)
        self.assertEqual([r.get("sandbox") for r in reports[:2]], ["uid", "uid"])
        self.assertEqual(reports[2]["episodes"], 2)
        uid, gid = sandbox_user()
        for path in (out / "trajectories.jsonl", out / "grader_log.jsonl"):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode) & 0o077, 0, path.name)
            self.assertEqual((path.stat().st_uid, path.stat().st_gid), (4242, 4242), path.name)   # QA L15
        for path in (out / "trajectories.jsonl", out / "grader_log.jsonl", set_file):
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())
                probe = subprocess.run([sys.executable, "-c", f"open({str(path)!r}).read()"], capture_output=True,
                                       user=uid, group=gid, extra_groups=[], timeout=30)
                self.assertNotEqual(probe.returncode, 0)
                self.assertIn(b"PermissionError", probe.stderr)


if __name__ == "__main__":
    unittest.main()
