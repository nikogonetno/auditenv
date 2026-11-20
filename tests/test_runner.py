"""Episode runner, baseline policies and trajectories (case §04, §05, §06, §09; SPEC §4.2, §4.3)."""
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from auditenv.env import AuditEnv
from auditenv.grader import MAX_STEPS
from auditenv.runner import (
    REPO_ROOT, TOO_LARGE, TRAJECTORY_KEYS, policy_environment, run_inprocess_episode, run_subprocess_episode,
)
from auditenv.server import MAX_MESSAGE_BYTES
from policies.oracle import make_env_oracle, make_oracle

from tests.tiny_episode import EPISODE_ID, GOOD_FINDING, submit, write_episode

CLEAN_TRUTH = {"clean": True, "episode_id": EPISODE_ID, "issues": []}
FORBIDDEN_IN_OBSERVATIONS = ("expected_amount_minor", "evidence_sets", "issue_id", "truth")


def policy_cmd(name, seed=0):
    return [sys.executable, "-m", f"policies.{name}", str(seed)]


def fake_policy(source):
    return [sys.executable, "-c", textwrap.dedent(source)]


def silence(_message):
    return submit([])


def read_trajectory(text):
    """Trajectory lines split into episodes at each step 0."""
    episodes = []
    for line in text.splitlines():
        row = json.loads(line)
        if row["step"] == 0:
            episodes.append([])
        episodes[-1].append(row)
    return episodes


class RunnerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dir = write_episode(self.root / "ep")
        self.config = {"episode_dir": str(self.dir)}

    def tearDown(self):
        self.tmp.cleanup()

    def clean_config(self):
        path = self.root / "clean"
        if not path.exists():
            write_episode(path, truth=CLEAN_TRUTH)
        return str(path)

    def run_subprocess(self, cmd, mode="train", time_limit=60.0, config=None):
        env, trajectory = AuditEnv(mode), io.StringIO()
        result = run_subprocess_episode(cmd, env, 1, config or self.config, time_limit=time_limit, trajectory=trajectory)
        return env, result, read_trajectory(trajectory.getvalue())

    def run_inprocess(self, policy_fn, mode="train", config=None):
        env, trajectory = AuditEnv(mode), io.StringIO()
        result = run_inprocess_episode(policy_fn, env, 1, config or self.config, trajectory=trajectory)
        return env, result, read_trajectory(trajectory.getvalue())


class Oracle(RunnerCase):
    def test_oracle_gets_the_maximum_in_train_and_eval(self):
        for mode, seen in (("train", 1.0), ("eval", 0.0)):
            with self.subTest(mode=mode):
                env = AuditEnv(mode)
                result = run_inprocess_episode(make_env_oracle(env), env, 1, self.config)
                self.assertEqual(env.grader_log[-1]["reward"], 1.0)
                self.assertEqual(env.grader_log[-1]["tp"], 1)
                self.assertEqual((result["terminated"], result["truncated"], result["steps"]), (True, False, 1))
                self.assertEqual(result["seen_reward"], seen)
                self.assertEqual(result["episode_id"], EPISODE_ID)

    def test_oracle_on_a_clean_company_submits_nothing(self):
        env = AuditEnv("eval")
        run_inprocess_episode(make_env_oracle(env), env, 1, {"episode_dir": self.clean_config()})
        self.assertEqual((env.grader_log[-1]["reward"], env.grader_log[-1]["fp"]), (1.0, 0))

    def test_make_oracle_takes_episode_id_from_the_manifest(self):
        truth = json.loads((self.dir / "private" / "truth.json").read_text(encoding="utf-8"))
        action = make_oracle(truth)({"observation": {"manifest": {"episode_id": "EPX"}}, "info": {}})
        self.assertEqual(action["findings"]["episode_id"], "EPX")
        self.assertEqual(action["findings"]["findings"][0]["explanation"], "oracle")


class RandomPolicy(RunnerCase):
    def test_finishes_within_the_step_limit(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                env, result, episodes = self.run_subprocess(policy_cmd("random_policy", seed))
                self.assertEqual((result["terminated"], result["truncated"]), (True, False))
                self.assertLessEqual(result["steps"], MAX_STEPS)
                self.assertEqual(episodes[0][-1]["action"]["action"], "submit")
                self.assertEqual(env.grader_log[-1]["end"], "submit")

    def test_same_seed_gives_byte_identical_trajectories(self):
        files = []
        for k in range(2):
            path = self.root / f"traj{k}.jsonl"
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                run_subprocess_episode(policy_cmd("random_policy", 7), AuditEnv("train"), 7, self.config, trajectory=f)
            files.append(path.read_bytes())
        self.assertEqual(files[0], files[1])
        self.assertGreater(len(files[0].splitlines()), 1)

    def test_the_seed_matters(self):
        seen = set()
        for seed in range(5):
            trajectory = io.StringIO()
            run_subprocess_episode(policy_cmd("random_policy", seed), AuditEnv("train"), 1, self.config, trajectory=trajectory)
            seen.add(trajectory.getvalue())
        self.assertGreater(len(seen), 1)


class FlagAll(RunnerCase):
    def test_flag_all_is_not_above_silence(self):
        """Case §05 and E-32: flagging everything never beats an empty answer, on an erroneous and a clean book."""
        for name, config in (("erroneous", self.config), ("clean", {"episode_dir": self.clean_config()})):
            with self.subTest(book=name):
                env, result, _ = self.run_subprocess(policy_cmd("flag_all"), mode="eval", config=config)
                self.assertEqual((result["terminated"], result["truncated"]), (True, False))
                flagged = env.grader_log[-1]
                self.assertEqual(flagged["fp"], 30)
                quiet_env, _, _ = self.run_inprocess(silence, mode="eval", config=config)
                self.assertLessEqual(flagged["reward"], quiet_env.grader_log[-1]["reward"])

    def test_flag_all_pages_every_table(self):
        _, result, episodes = self.run_subprocess(policy_cmd("flag_all"))
        listed = [row["action"]["table"] for row in episodes[0][1:-1]]
        self.assertEqual(listed, ["ledger"] * 3 + ["invoices", "payments", "fx"])
        self.assertEqual(result["steps"], 7)


class HostilePolicies(RunnerCase):
    def test_garbage_and_a_2mb_line_each_spend_one_step(self):
        line = json.dumps(submit([GOOD_FINDING]))
        cmd = fake_policy(f"""
            import sys
            out = sys.stdout.buffer
            out.write(b"this is not json\\n")
            out.write(b"x" * 2_000_000 + b"\\n")
            out.write({line!r}.encode("utf-8") + b"\\n")
            out.flush()
        """)
        env, result, episodes = self.run_subprocess(cmd)
        rows = episodes[0]
        self.assertEqual([row["action"] for row in rows[1:3]], [{"bad_message": "invalid JSON"}, {"bad_message": TOO_LARGE}])
        for row, left in zip(rows[1:3], (MAX_STEPS - 1, MAX_STEPS - 2)):
            self.assertEqual(row["observation"], {"error": "bad_action", "steps_left": left})
        self.assertEqual(rows[3]["action"]["action"], "submit")
        entry = env.grader_log[-1]
        self.assertEqual((entry["end"], entry["tp"], entry["inspection_steps"]), ("submit", 1, 2))
        self.assertEqual((result["steps"], result["terminated"], result["seen_reward"]), (3, True, entry["reward"]))

    def test_the_size_limit_is_inclusive(self):
        pad = MAX_MESSAGE_BYTES - len(json.dumps({"action": "calculator", "expression": "1", "pad": ""}))
        cmd = fake_policy(f"""
            import json, sys
            for extra in (0, 1):
                sys.stdout.write(json.dumps({{"action": "calculator", "expression": "1", "pad": "y" * ({pad} + extra)}}) + "\\n")
            sys.stdout.flush()
        """)
        _, result, episodes = self.run_subprocess(cmd)
        rows = episodes[0]
        self.assertEqual(rows[1]["observation"], {"result": {"value": "1"}, "steps_left": MAX_STEPS - 1})
        self.assertEqual(rows[2]["action"], {"bad_message": TOO_LARGE})
        self.assertEqual(rows[3]["action"], {"truncate": "policy_closed"})
        self.assertEqual((result["steps"], result["truncated"]), (2, True))

    def test_silent_policy_is_truncated_by_the_wall_clock(self):
        started = time.monotonic()
        env, result, episodes = self.run_subprocess(fake_policy("import time; time.sleep(60)"), time_limit=1)
        elapsed = time.monotonic() - started
        self.assertEqual((result["terminated"], result["truncated"], result["steps"]), (False, True, 0))
        self.assertLess(elapsed, 8)
        self.assertGreaterEqual(result["wall_time_s"], 1.0)
        self.assertEqual(env.grader_log[-1]["inspection_steps"], MAX_STEPS)
        self.assertEqual(env.grader_log[-1]["end"], "truncated")
        self.assertEqual(episodes[0][-1]["action"], {"truncate": "time_limit"})

    def test_policy_that_exits_at_once_is_truncated(self):
        env, result, episodes = self.run_subprocess(fake_policy("pass"))
        self.assertEqual((result["terminated"], result["truncated"], result["steps"]), (False, True, 0))
        self.assertEqual(env.grader_log[-1]["inspection_steps"], MAX_STEPS)
        self.assertEqual(episodes[0][-1]["action"], {"truncate": "policy_closed"})

    def test_policy_that_never_reads_cannot_block_the_runner(self):
        """100 full pages (over 1 MB of observations) into a stdin nobody reads, then silence."""
        cmd = fake_policy("""
            import json, sys, time
            for _ in range(150):
                sys.stdout.write(json.dumps({"action": "list_records", "table": "ledger"}) + "\\n")
            sys.stdout.flush()
            time.sleep(60)
        """)
        started = time.monotonic()
        env, result, _ = self.run_subprocess(cmd, time_limit=20)
        self.assertLess(time.monotonic() - started, 12)
        self.assertEqual((result["steps"], result["terminated"], result["truncated"]), (MAX_STEPS, False, True))
        self.assertEqual(env.grader_log[-1]["inspection_steps"], MAX_STEPS)

    def test_policy_process_sees_no_episode_path(self):
        cmd = fake_policy("""
            import json, os, sys
            sys.stdin.buffer.readline()
            probe = {"argv": sys.argv, "cwd": os.getcwd(), "files": os.listdir("."), "env": dict(os.environ)}
            sys.stdout.write(json.dumps({"action": "calculator", "expression": json.dumps(probe)}) + "\\n")
            sys.stdout.flush()
        """)
        _, _, episodes = self.run_subprocess(cmd)
        probe = json.loads(episodes[0][1]["action"]["expression"])
        self.assertEqual(probe["files"], [])
        self.assertLessEqual(set(probe["env"]) - {"LC_CTYPE"}, {"PATH", "LANG", "PYTHONPATH", "PYTHONHASHSEED"})
        self.assertEqual(probe["env"]["PYTHONPATH"], str(REPO_ROOT))
        self.assertNotIn(str(self.dir), json.dumps(probe))
        self.assertNotIn("private", json.dumps(probe))
        self.assertEqual(set(policy_environment()) - {"PYTHONHASHSEED"}, {"PATH", "LANG", "PYTHONPATH"})

    def test_inprocess_policy_returning_non_json_spends_a_step(self):
        actions = iter([{"not", "json"}, float("nan"), submit([])])
        env, result, episodes = self.run_inprocess(lambda _m: next(actions))
        self.assertEqual([row["action"] for row in episodes[0][1:3]],
                         [{"bad_message": "invalid JSON"}, {"bad_message": "non-finite number"}])
        self.assertEqual((result["steps"], env.grader_log[-1]["inspection_steps"]), (3, 2))


class Trajectories(RunnerCase):
    def collect(self, mode):
        trajectory = io.StringIO()
        for policy in ("random_policy", "flag_all"):
            run_subprocess_episode(policy_cmd(policy, 3), AuditEnv(mode), 3, self.config, trajectory=trajectory)
        env = AuditEnv(mode)
        run_inprocess_episode(make_env_oracle(env), env, 3, self.config, trajectory=trajectory)
        run_inprocess_episode(silence, AuditEnv(mode), 3, {"episode_dir": self.clean_config()}, trajectory=trajectory)
        return read_trajectory(trajectory.getvalue())

    def test_shape_steps_and_rewards(self):
        for mode in ("train", "eval"):
            episodes = self.collect(mode)
            self.assertEqual(len(episodes), 4)
            for k, rows in enumerate(episodes):
                with self.subTest(mode=mode, episode=k):
                    self.assertEqual([row["step"] for row in rows], list(range(len(rows))))
                    for row in rows:
                        self.assertEqual(set(row), set(TRAJECTORY_KEYS))
                        self.assertEqual(row["episode_id"], EPISODE_ID)
                    self.assertEqual((rows[0]["action"], rows[0]["reward"], rows[0]["terminated"], rows[0]["truncated"]),
                                     (None, 0.0, False, False))
                    self.assertTrue(rows[-1]["terminated"] or rows[-1]["truncated"])
                    self.assertFalse(any(row["terminated"] or row["truncated"] for row in rows[:-1]))
                    self.assertTrue(all(row["reward"] == 0.0 for row in rows[:-1]))
                    if mode == "eval":
                        self.assertEqual(rows[-1]["reward"], 0.0)
                    else:
                        self.assertNotEqual(rows[-1]["reward"], 0.0)

    def test_observations_carry_nothing_from_truth(self):
        truth = json.loads((self.dir / "private" / "truth.json").read_text(encoding="utf-8"))
        issue_ids = [issue["issue_id"] for issue in truth["issues"]]
        self.assertTrue(issue_ids)
        for mode in ("train", "eval"):
            for rows in self.collect(mode):
                for row in rows:
                    text = json.dumps(row["observation"], ensure_ascii=False)
                    for word in (*issue_ids, *FORBIDDEN_IN_OBSERVATIONS):
                        self.assertNotIn(word.lower(), text.lower(), (mode, row["step"]))
                    self.assertNotIn(str(self.dir), text)

    def test_lines_are_sorted_utf8_json(self):
        trajectory = io.StringIO()
        run_inprocess_episode(silence, AuditEnv("train"), 1, self.config, trajectory=trajectory)
        for line in trajectory.getvalue().splitlines():
            self.assertEqual(line, json.dumps(json.loads(line), sort_keys=True, ensure_ascii=False))


class CommandLine(RunnerCase):
    def run_cli(self, *args):
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        return subprocess.run([sys.executable, "-m", "auditenv.run", *args], capture_output=True, text=True,
                              cwd=REPO_ROOT, env=env, timeout=120)

    def test_eval_run_reports_the_grade_and_appends_logs(self):
        traj, log = self.root / "t.jsonl", self.root / "g.jsonl"
        for policy in ("oracle", "flag_all"):
            proc = self.run_cli("--policy", policy, "--episode-dir", str(self.dir), "--mode", "eval",
                                "--trajectories", str(traj), "--grader-log", str(log))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(proc.stdout)
            self.assertEqual(report["policy"], policy)
            self.assertEqual(report["seen_reward"], 0.0)
            for key in ("tp", "fp", "fn", "precision", "recall", "f1", "reward", "steps", "wall_time_s"):
                self.assertIn(key, report)
        entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["reward"], 1.0)
        self.assertEqual(len(read_trajectory(traj.read_text(encoding="utf-8"))), 2)

    def test_bad_episode_dir_is_a_clean_error(self):
        proc = self.run_cli("--policy", "random", "--episode-dir", str(self.root / "missing"))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("auditenv.run:", proc.stderr)


if __name__ == "__main__":
    unittest.main()


class HostilePolicyRegressions(RunnerCase):
    """Fixes after the independent attack on the runner (SPEC §4.3)."""

    def test_rewriting_truth_on_disk_after_reset_changes_nothing(self):
        tamper = f"""
            import json, sys
            sys.stdin.readline()
            open({str(self.dir / "private" / "truth.json")!r}, "w").write(json.dumps({CLEAN_TRUTH!r}))
            print(json.dumps({{"action": "submit", "findings": {{"episode_id": "{EPISODE_ID}", "findings": []}}}}), flush=True)
            sys.stdin.readline()
        """
        env, _, _ = self.run_subprocess(fake_policy(tamper), mode="eval")
        self.assertEqual((env.grader_log[-1]["fn"], env.grader_log[-1]["reward"]), (1, -1.0))

    def test_policy_that_replaces_its_working_directory_cannot_crash_the_runner(self):
        swap = f"""
            import json, os, sys
            sys.stdin.readline()
            d = os.getcwd()
            os.rename(d, d + ".moved")
            open(d, "w").close()
            print(json.dumps({{"action": "submit", "findings": {{"episode_id": "{EPISODE_ID}", "findings": []}}}}), flush=True)
        """
        env, result, _ = self.run_subprocess(fake_policy(swap))
        # Same user: the swap works and the policy submits. In the sandbox the rename is refused,
        # the policy dies and the episode is truncated. Either way the runner returns.
        self.assertTrue(result["terminated"] or result["truncated"])
        self.assertEqual(env.grader_log[-1]["reward"], -1.0)

    def test_line_separators_inside_strings_keep_one_record_per_line(self):
        env, trajectory = AuditEnv("train"), io.StringIO()
        actions = iter([{"action": "calculator", "expression": "1 +\u00851"}, submit([GOOD_FINDING])])
        run_inprocess_episode(lambda _m: next(actions), env, 1, self.config, trajectory=trajectory)
        text = trajectory.getvalue()
        self.assertEqual(len(text.splitlines()), 3)
        self.assertEqual([json.loads(line)["step"] for line in text.splitlines()], [0, 1, 2])

    def test_oversize_actions_are_recorded_by_size_and_hash(self):
        env, trajectory = AuditEnv("train"), io.StringIO()
        actions = iter([{"action": "calculator", "expression": "1", "pad": "x" * 200_000}, submit([])])
        run_inprocess_episode(lambda _m: next(actions), env, 1, self.config, trajectory=trajectory)
        rows = [json.loads(line) for line in trajectory.getvalue().splitlines()]
        self.assertEqual(set(rows[1]["action"]), {"oversize_action"})
        self.assertLess(len(trajectory.getvalue()), 20_000)
