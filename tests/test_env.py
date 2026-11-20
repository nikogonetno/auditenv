"""reset/step contract of case §04 on a tiny episode written to a temp dir (SPEC §4.2, §4.3)."""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from auditenv.env import PAGE_LIMIT, AuditEnv
from auditenv.grader import MAX_STEPS

from tests.tiny_episode import EPISODE_ID, GOOD_FINDING, submit, write_episode


class EnvCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = write_episode(Path(self.tmp.name) / "ep")
        self.config = {"episode_dir": str(self.dir)}

    def tearDown(self):
        self.tmp.cleanup()

    def env(self, mode="train"):
        env = AuditEnv(mode)
        env.reset(seed=1, config=self.config)
        return env


class ResetAndStep(EnvCase):
    def test_reset_observation(self):
        obs, info = AuditEnv().reset(seed=1, config=self.config)
        self.assertEqual(info, {})
        self.assertEqual(set(obs), {"task", "manifest", "policy", "tables", "actions", "steps_left"})
        self.assertEqual(obs["steps_left"], MAX_STEPS)
        self.assertEqual(obs["manifest"]["episode_id"], EPISODE_ID)
        self.assertEqual(set(obs["actions"]), {"list_records", "get_record", "get_document", "calculator", "submit"})
        self.assertEqual(obs["tables"], {"ledger": 120, "invoices": 2, "payments": 1, "fx": 1})

    def test_step_returns_five_fields_and_zero_reward(self):
        result = self.env().step({"action": "get_record", "table": "invoices", "id": "INV001"})
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertEqual((reward, terminated, truncated, info), (0.0, False, False, {}))
        self.assertEqual(set(obs), {"result", "steps_left"})
        self.assertEqual(obs["steps_left"], MAX_STEPS - 1)

    def test_bad_config(self):
        for config in ({"episode_dir": str(self.dir), "colour": 1}, {"colour": 1}, "x",
                       {"episode_dir": str(self.dir) + "_missing"}, {"episode_dir": str(self.dir), "errors": {}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                AuditEnv().reset(seed=1, config=config)
        with self.assertRaises(ValueError):
            AuditEnv("debug")

    def test_step_before_reset(self):
        obs, reward, terminated, truncated, _ = AuditEnv().step({"action": "calculator", "expression": "1+1"})
        self.assertEqual((obs["error"], reward, terminated, truncated), ("reset_required", 0.0, False, False))


class Actions(EnvCase):
    def test_list_records_pages_with_stable_cursors(self):
        env = self.env()
        seen, cursor, steps = [], None, 0
        while True:
            obs = env.step({"action": "list_records", "table": "ledger", "cursor": cursor})[0]
            steps += 1
            page = obs["result"]
            self.assertLessEqual(len(page["records"]), PAGE_LIMIT)
            seen += [r["row_id"] for r in page["records"]]
            cursor = page["next_cursor"]
            if cursor is None:
                break
        self.assertEqual((len(seen), len(set(seen)), steps), (120, 120, 3))
        again = env.step({"action": "list_records", "table": "ledger", "cursor": "c1.EPTEST.ledger.50", "limit": 3})[0]
        self.assertEqual([r["row_id"] for r in again["result"]["records"]], seen[50:53])

    def test_records_are_plain_json_with_nulls(self):
        rec = self.env().step({"action": "get_record", "table": "ledger", "id": "ledger/R00005"})[0]["result"]["record"]
        self.assertEqual(rec["debit_minor"], 500000)
        self.assertIsNone(rec["invoice_id"])
        json.dumps(rec)

    def test_get_document_by_doc_id(self):
        obs = self.env().step({"action": "get_document", "doc_id": "DOCPAY"})[0]
        self.assertEqual(obs["result"]["table"], "payments")
        self.assertEqual(obs["result"]["document"]["invoice_refs"], ["INV001"])

    def test_calculator(self):
        obs = self.env().step({"action": "calculator", "expression": "500000 * 2 - 1"})[0]
        self.assertEqual(obs["result"], {"value": "999999"})
        obs = self.env().step({"action": "calculator", "expression": "__import__('os')"})[0]
        self.assertIn("error", obs)

    def test_returned_records_cannot_change_the_episode(self):
        env = self.env()
        rec = env.step({"action": "get_record", "table": "invoices", "id": "INV001"})[0]["result"]["record"]
        rec["total_minor"] = 1
        again = env.step({"action": "get_record", "table": "invoices", "id": "INV001"})[0]["result"]["record"]
        self.assertEqual(again["total_minor"], 500000)


class Errors(EnvCase):
    def test_bounded_errors_consume_a_step(self):
        cases = {
            "not an object": (["list_records"], "bad_action"),
            "undecodable line": (None, "bad_action"),
            "unknown action": ({"action": "shell", "cmd": "ls"}, "unknown_action"),
            "unknown table": ({"action": "list_records", "table": "truth"}, "unknown_table"),
            "limit over 50": ({"action": "list_records", "table": "ledger", "limit": 51}, "bad_limit"),
            "limit zero": ({"action": "list_records", "table": "ledger", "limit": 0}, "bad_limit"),
            "limit as bool": ({"action": "list_records", "table": "ledger", "limit": True}, "bad_limit"),
            "negative cursor": ({"action": "list_records", "table": "ledger", "cursor": "c1.EPTEST.ledger.-1"}, "bad_cursor"),
            "cursor of another table": ({"action": "list_records", "table": "fx", "cursor": "c1.EPTEST.ledger.50"}, "bad_cursor"),
            "cursor of another episode": ({"action": "list_records", "table": "ledger", "cursor": "c1.OTHER.ledger.50"}, "bad_cursor"),
            "id missing": ({"action": "get_record", "table": "ledger"}, "bad_arguments"),
            "id as number": ({"action": "get_record", "table": "ledger", "id": 5}, "bad_arguments"),
            "id of 1 MB": ({"action": "get_record", "table": "ledger", "id": "x" * 1_000_000}, "bad_arguments"),
        }
        for name, (action, code) in cases.items():
            with self.subTest(case=name):
                env = self.env()
                obs, reward, terminated, truncated, info = env.step(action)
                self.assertEqual(obs, {"error": code, "steps_left": MAX_STEPS - 1})
                self.assertEqual((reward, terminated, truncated, info), (0.0, False, False, {}))

    def test_not_found_is_identical_for_any_missing_id(self):
        """R-04: a private id, a je id, a path or SQL get the same answer as a random id."""
        answers = set()
        for table, rid in (("ledger", "ZZZZZZ"), ("ledger", "ISS001"), ("ledger", "J00001"), ("invoices", "../private/truth.json"),
                           ("payments", "1; DROP TABLE x"), ("fx", "EPTEST"), ("invoices", "DOCINV")):
            obs = self.env().step({"action": "get_record", "table": table, "id": rid})[0]
            answers.add(json.dumps(obs, sort_keys=True))
        for doc_id in ("ISS001", "INV001", "ZZZZZZ"):
            answers.add(json.dumps(self.env().step({"action": "get_document", "doc_id": doc_id})[0], sort_keys=True))
        self.assertEqual(answers, {json.dumps({"error": "not_found", "steps_left": MAX_STEPS - 1}, sort_keys=True)})


class Submit(EnvCase):
    def test_correct_submit_in_train_returns_reward(self):
        env = self.env("train")
        env.step({"action": "get_document", "doc_id": "DOCINV"})
        obs, reward, terminated, truncated, info = env.step(submit([GOOD_FINDING]))
        self.assertEqual(obs, {"result": {"status": "submitted"}, "steps_left": MAX_STEPS - 2})
        self.assertEqual((terminated, truncated), (True, False))
        self.assertEqual(reward, 1 - 0.002)
        self.assertEqual(info["grading"]["tp"], 1)
        self.assertEqual(env.grader_log[-1]["reward"], reward)

    def test_doc_ids_count_as_invoice_ids(self):
        by_doc = dict(GOOD_FINDING, affected_ids=["DOCINV", "doc/DOCDUP"], evidence_ids=["DOCINV", "DOCDUP"])
        self.assertEqual(self.env().step(submit([by_doc]))[1], 1.0)

    def test_broken_structure_gives_minus_one(self):
        for name, action in {
            "foreign episode_id": submit([GOOD_FINDING], episode_id="OTHER"),
            "31 findings": submit([GOOD_FINDING] * 31),
            "old field names": submit([{k.replace("claimed_", "").replace("_ids", "") if k != "affected_ids" else k: v
                                        for k, v in GOOD_FINDING.items()}]),
            "findings missing": {"action": "submit"},
            "findings as list": {"action": "submit", "findings": [GOOD_FINDING]},
        }.items():
            with self.subTest(case=name):
                env = self.env("train")
                obs, reward, terminated, truncated, info = env.step(action)
                self.assertEqual((reward, terminated, truncated), (-1.0, True, False))
                self.assertIn("error", info["grading"])
                self.assertEqual(env.grader_log[-1]["reward"], -1.0)

    def test_eval_hides_reward_and_diagnostics(self):
        """E-25: in eval the policy sees the same bytes for a right, a wrong and a broken answer."""
        seen = set()
        for findings in ([GOOD_FINDING], [], [dict(GOOD_FINDING, claimed_amount_minor=1)], None):
            env = self.env("eval")
            action = submit(findings) if findings is not None else {"action": "submit"}
            seen.add(json.dumps(env.step(action), sort_keys=True))
        self.assertEqual(len(seen), 1)
        env = self.env("eval")
        env.step(submit([GOOD_FINDING]))
        self.assertEqual(env.grader_log[-1]["reward"], 1.0)

    def test_second_submit_changes_nothing(self):
        env = self.env("train")
        env.step(submit([GOOD_FINDING]))
        log = copy.deepcopy(env.grader_log)
        obs, reward, terminated, truncated, _ = env.step(submit([]))
        self.assertEqual((obs["error"], reward, terminated, truncated), ("episode_over", 0.0, True, False))
        self.assertEqual(env.grader_log, log)


class StepLimit(EnvCase):
    def test_hundredth_step_truncates_with_empty_answer(self):
        env = self.env("train")
        for k in range(MAX_STEPS - 1):
            obs, reward, terminated, truncated, _ = env.step({"action": "calculator", "expression": "1"})
            self.assertEqual((reward, terminated, truncated, obs["steps_left"]), (0.0, False, False, MAX_STEPS - 1 - k))
        obs, reward, terminated, truncated, info = env.step({"action": "calculator", "expression": "2"})
        self.assertEqual((obs["result"], obs["steps_left"], terminated, truncated), ({"value": "2"}, 0, False, True))
        self.assertEqual(reward, -1.0)  # empty answer on a book with one error, charged 100 steps
        self.assertEqual(env.grader_log[-1]["inspection_steps"], MAX_STEPS)
        self.assertEqual(env.step({"action": "calculator", "expression": "1"})[0]["error"], "episode_over")

    def test_submit_as_the_hundredth_step_still_counts(self):
        env = self.env("train")
        for _ in range(MAX_STEPS - 1):
            env.step({"action": "calculator", "expression": "1"})
        _, reward, terminated, truncated, _ = env.step(submit([GOOD_FINDING]))
        self.assertEqual((terminated, truncated), (True, False))
        self.assertAlmostEqual(reward, 1 - 0.002 * (MAX_STEPS - 1))

    def test_wall_clock_truncation_on_a_clean_company(self):
        tmp = Path(self.tmp.name) / "clean"
        write_episode(tmp, truth={"clean": True, "episode_id": EPISODE_ID, "issues": []})
        env = AuditEnv("train")
        env.reset(seed=1, config={"episode_dir": str(tmp)})
        env.step({"action": "calculator", "expression": "1"})
        obs, reward, terminated, truncated, _ = env.truncate()
        self.assertEqual((terminated, truncated, obs["steps_left"]), (False, True, 0))
        self.assertAlmostEqual(reward, 0.8)


class Determinism(EnvCase):
    def test_same_actions_same_transcript(self):
        actions = [{"action": "list_records", "table": "ledger", "limit": 7}, {"action": "get_document", "doc_id": "DOCDUP"},
                   {"action": "calculator", "expression": "1/3"}, {"action": "nope"}, submit([GOOD_FINDING])]
        transcripts = []
        for _ in range(2):
            env = AuditEnv("train")
            out = [env.reset(seed=7, config=self.config)]
            out += [env.step(copy.deepcopy(a)) for a in actions]
            transcripts.append(json.dumps(out, sort_keys=True))
        self.assertEqual(transcripts[0], transcripts[1])

    def test_reset_isolates_episodes(self):
        env = AuditEnv("train")
        env.reset(seed=1, config=self.config)
        env.step(submit([]))
        obs, _ = env.reset(seed=1, config=self.config)
        self.assertEqual(obs["steps_left"], MAX_STEPS)
        self.assertEqual(env.step(submit([GOOD_FINDING]))[1], 1.0)


if __name__ == "__main__":
    unittest.main()


class MiniEpisodeEndToEnd(unittest.TestCase):
    """The case §03 test book built from golden v0.1 (tests/fixtures/mini_v03), driven through the env."""

    def test_oracle_through_the_env_scores_full_marks_minus_steps(self):
        root = Path(__file__).resolve().parent / "fixtures" / "mini_v03"
        truth = json.loads((root / "private" / "truth.json").read_text(encoding="utf-8"))
        env = AuditEnv("eval")
        obs, _ = env.reset(seed=0, config={"episode_dir": str(root)})
        page = env.step({"action": "list_records", "table": "ledger"})[0]["result"]
        self.assertEqual((len(page["records"]), page["next_cursor"]), (48, None))
        findings = [{"type": i["type"], "affected_ids": i["affected_ids"], "claimed_amount_minor": i["expected_amount_minor"],
                     "currency": i["currency"], "evidence_ids": i["evidence_sets"][0], "explanation": "oracle"}
                    for i in truth["issues"]]
        _, reward, terminated, _, info = env.step(submit(findings, episode_id=obs["manifest"]["episode_id"]))
        self.assertEqual((reward, terminated, info), (0.0, True, {}))
        self.assertEqual((env.grader_log[-1]["tp"], env.grader_log[-1]["reward"]), (8, 1 - 0.002))


class TruthIsFixedAtReset(EnvCase):
    def test_truth_file_changed_after_reset_does_not_change_the_grade(self):
        env = self.env("train")
        (Path(self.dir) / "private" / "truth.json").write_text(json.dumps({"clean": True, "issues": []}), encoding="utf-8")
        self.assertEqual(env.step(submit([GOOD_FINDING]))[1], 1.0)

    def test_malformed_truth_fails_at_reset(self):
        (Path(self.dir) / "private" / "truth.json").write_text(json.dumps({"clean": False, "issues": []}), encoding="utf-8")
        with self.assertRaises(ValueError):
            AuditEnv().reset(seed=1, config=self.config)


MINI = Path(__file__).resolve().parent / "fixtures" / "mini_v03"


class GeneratedEpisodes(unittest.TestCase):
    """reset(seed, config) without episode_dir runs the generator (docs/tasks/B_next.md item 5).

    The generator is A's; here a test double copies mini_v03 into the directory it is given.
    """

    def setUp(self):
        self.calls = []

        def fake_generator(seed, config, out_dir):
            self.calls.append((seed, copy.deepcopy(config), out_dir))
            shutil.copytree(MINI, out_dir, dirs_exist_ok=True)

        self.generator = fake_generator

    def test_generated_episode_is_loaded_and_its_files_are_gone(self):
        env = AuditEnv("eval", generator=self.generator)
        obs, info = env.reset(seed=7, config={"errors": {"fx_rate": 2}})
        self.assertEqual((obs["manifest"]["episode_id"], info), ("KARPSR", {}))
        seed, config, out_dir = self.calls[0]
        self.assertEqual((seed, config["errors"]["fx_rate"], config["documents"]), (7, 2, [150, 200]))
        self.assertFalse(Path(out_dir).exists())
        truth = json.loads((MINI / "private" / "truth.json").read_text(encoding="utf-8"))
        findings = [{"type": i["type"], "affected_ids": i["affected_ids"], "claimed_amount_minor": i["expected_amount_minor"],
                     "currency": i["currency"], "evidence_ids": i["evidence_sets"][0], "explanation": "oracle"}
                    for i in truth["issues"]]
        env.step(submit(findings, episode_id="KARPSR"))
        self.assertEqual(env.grader_log[-1]["reward"], 1.0)

    def test_key_order_does_not_reach_the_generator(self):
        env = AuditEnv(generator=self.generator)
        env.reset(seed=1, config={"twins": [2, 4], "period": "2026-09"})
        env.reset(seed=1, config={"period": "2026-09", "twins": [2, 4]})
        self.assertEqual(self.calls[0][:2], self.calls[1][:2])

    def test_bad_seed_or_config_fails_before_generating(self):
        env = AuditEnv(generator=self.generator)
        for seed, config in ((None, {}), (-1, {}), (True, {}), ("1", {}), (1, {"errors": {"fraud": 1}}),
                             (1, {"documents": [5, 10]})):
            with self.subTest(seed=seed, config=config), self.assertRaises(ValueError):
                env.reset(seed=seed, config=config)
        self.assertEqual(self.calls, [])

    def test_broken_generator_output_is_a_value_error_and_leaves_nothing(self):
        dirs = []

        def broken(seed, config, out_dir):
            dirs.append(out_dir)
            (Path(out_dir) / "public").mkdir()

        with self.assertRaises(ValueError):
            AuditEnv(generator=broken).reset(seed=1, config={})
        self.assertFalse(Path(dirs[0]).exists())

    def test_missing_generator_is_a_clear_error(self):
        try:
            import auditenv.generate  # noqa: F401
        except ImportError:
            with self.assertRaises(RuntimeError):
                AuditEnv().reset(seed=1, config={})
        else:
            self.skipTest("auditenv.generate exists")
