"""scripts/metrics.py, the diagnostic table of case §10 (docs/tasks/B_next.md item 11).

One run on the case §03 test book (mini_v03, 8 errors) and a split directory with the tiny episode
and a clean one. The oracle must be perfect, flag_all must get -1 everywhere, reconcile must repeat
its known grade on mini_v03, and the report must repeat itself except for the wall times.
"""
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from auditenv.env import AuditEnv
from auditenv.runner import TRAJECTORY_KEYS, run_inprocess_episode

from tests.support import run_python
from tests.tiny_episode import EPISODE_ID, GOOD_FINDING, submit, write_episode

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "scripts" / "metrics.py"
MINI = ROOT / "tests" / "fixtures" / "mini_v03"
POLICIES = ["random", "reconcile", "flag_all", "oracle"]
TYPES = ("duplicate_invoice", "payment_allocation", "fx_rate", "period_cutoff")
CLEAN_TRUTH = {"clean": True, "episode_id": EPISODE_ID, "issues": []}
SUMMARY, PER_TYPE, PER_EPISODE = "Сводка по политикам", "По типам ошибок", "По эпизодам"
WALL_TIME = "время, с"


def metrics(*args):
    return run_python([METRICS, *args], cwd=ROOT)


def tables(markdown):
    """{section heading: [row as {column header: cell}]} for every table of the report."""
    result, heading, header = {}, None, None
    for line in markdown.splitlines():
        if line.startswith("## "):
            heading, header = line[3:], None
        elif line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if header is None:
                header, result[heading] = cells, []
            elif not set(line) <= set("|-: "):
                result[heading].append(dict(zip(header, cells)))
    return result


def episode_starts(path):
    return sum(json.loads(line)["step"] == 0 for line in path.read_text(encoding="utf-8").splitlines())


class Metrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.split = cls.root / "split"
        write_episode(cls.split / "a_tiny")
        write_episode(cls.split / "b_clean", truth=CLEAN_TRUTH)
        (cls.split / "notes").mkdir()  # neither is an episode: both are skipped
        (cls.split / "README.txt").write_text("not an episode\n", encoding="utf-8")
        cls.args = [MINI, cls.split, "--mode", "eval", "--seed", "0"]
        cls.out, cls.json, cls.traj = cls.root / "docs" / "METRICS.md", cls.root / "rows.jsonl", cls.root / "t.jsonl"
        cls.proc = metrics(*cls.args, "--out", cls.out, "--json", cls.json, "--trajectories", cls.traj)
        if cls.proc.returncode != 0:
            raise AssertionError(cls.proc.stderr)
        cls.markdown = cls.out.read_text(encoding="utf-8")
        cls.tables = tables(cls.markdown)
        cls.rows = [json.loads(line) for line in cls.json.read_text(encoding="utf-8").splitlines()]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def summary(self, policy):
        return next(row for row in self.tables[SUMMARY] if row["политика"] == policy)

    def rows_of(self, policy, source=None):
        return [r for r in self.rows if r["policy"] == policy and source in (None, r["source"])]

    def test_split_directory_is_expanded_in_sorted_order(self):
        sources = [str(MINI), str(self.split / "a_tiny"), str(self.split / "b_clean")]
        self.assertEqual([r["source"] for r in self.rows], [s for s in sources for _ in POLICIES])
        self.assertEqual([r["policy"] for r in self.rows], POLICIES * 3)
        self.assertIn("- Эпизоды: 3, из них чистых: 1", self.markdown)

    def test_oracle_is_perfect(self):
        row = self.summary("oracle")
        self.assertEqual((row["TP"], row["FP"], row["FN"]), ("9", "0", "0"))
        self.assertEqual((row["P"], row["R"], row["F1"], row["возврат"]), ("1.000",) * 4)
        self.assertEqual([r["reward"] for r in self.rows_of("oracle")], [1.0] * 3)

    def test_flag_all_gets_minus_one_on_each_episode(self):
        self.assertEqual([r["reward"] for r in self.rows_of("flag_all")], [-1.0] * 3)
        self.assertEqual(self.summary("flag_all")["возврат"], "-1.000")
        self.assertEqual([row["flag_all"] for row in self.tables[PER_EPISODE]], ["-1.000"] * 3)

    def test_reconcile_on_mini_repeats_its_known_grade(self):
        [row] = self.rows_of("reconcile", str(MINI))
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (7, 4, 1))
        self.assertEqual({kind: counts["fn"] for kind, counts in row["per_type"].items()},
                         {"duplicate_invoice": 0, "payment_allocation": 0, "fx_rate": 0, "period_cutoff": 1})
        self.assertEqual(row["missed"], ["XG1494"])  # the period error of the C1 receipt

    def test_per_type_counts_add_up_to_the_grade(self):
        for row in self.rows:
            with self.subTest(policy=row["policy"], source=row["source"]):
                for key in ("tp", "fp", "fn"):
                    self.assertEqual(sum(c[key] for c in row["per_type"].values()), row[key])
        for policy in POLICIES:
            with self.subTest(policy=policy):
                mine = [r for r in self.tables[PER_TYPE] if r["политика"] == policy]
                self.assertEqual([r["тип"] for r in mine], list(TYPES))
                for key in ("TP", "FP", "FN"):
                    self.assertEqual(sum(int(r[key]) for r in mine), int(self.summary(policy)[key]))

    def test_clean_episode_is_counted(self):
        clean = [r for r in self.rows if r["source"] == str(self.split / "b_clean")]
        self.assertEqual({(r["clean"], r["issues"], r["fn"], r["tp"]) for r in clean}, {(True, 0, 0, 0)})
        self.assertEqual(self.tables[PER_EPISODE][2]["чистый"], "да")
        self.assertEqual(self.tables[PER_EPISODE][2]["ошибок"], "0")
        [flagged] = self.rows_of("flag_all", str(self.split / "b_clean"))
        self.assertGreater(flagged["fp"], 0)  # false accusations on a clean book are counted as FP

    def test_report_has_the_four_policy_rows_and_stdout_is_the_summary(self):
        self.assertEqual([row["политика"] for row in self.tables[SUMMARY]], POLICIES)
        self.assertEqual([row["эпизод"] for row in self.tables[PER_EPISODE]], ["KARPSR", EPISODE_ID, EPISODE_ID])
        self.assertEqual(list(self.tables[PER_EPISODE][0])[3:], POLICIES)
        printed = self.proc.stdout.splitlines()
        self.assertEqual(len(printed), 2 + len(POLICIES))
        self.assertIn("\n".join(printed), self.markdown)
        self.assertTrue(printed[0].startswith("| политика |"))

    def test_trajectories_hold_every_run(self):
        self.assertEqual(episode_starts(self.traj), 3 * len(POLICIES))
        for line in self.traj.read_text(encoding="utf-8").splitlines():
            self.assertEqual(set(json.loads(line)), set(TRAJECTORY_KEYS))

    def test_second_run_repeats_everything_but_wall_time_and_appends_trajectories(self):
        out, rows, traj = self.root / "again.md", self.root / "again.jsonl", self.root / "again_t.jsonl"
        shutil.copy(self.traj, traj)
        proc = metrics(*self.args, "--out", out, "--json", rows, "--trajectories", traj)
        self.assertEqual(proc.returncode, 0, proc.stderr)

        def without_wall_time(markdown):
            text = [line for line in markdown.splitlines() if not line.startswith("|")]
            found = tables(markdown)
            for table in found.values():
                for row in table:
                    row.pop(WALL_TIME, None)
            return text, found

        self.assertEqual(without_wall_time(out.read_text(encoding="utf-8")), without_wall_time(self.markdown))
        first = [{k: v for k, v in r.items() if k != "wall_time_s"} for r in self.rows]
        second = [{k: v for k, v in json.loads(line).items() if k != "wall_time_s"}
                  for line in rows.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(second, first)
        once = self.traj.read_text(encoding="utf-8")
        self.assertEqual(traj.read_text(encoding="utf-8"), once + once)

    def test_reconcile_misses_are_split_into_rule_and_cap(self):
        row = self.rows_of("reconcile", str(MINI))[0]
        self.assertEqual(row["missed_by_rule"], [["period_cutoff", "XG1494"]])   # C1: the UTC date is in period
        self.assertEqual(row["missed_by_cap"], [])
        self.assertIn("## Пропуски сверки", self.markdown)
        self.assertIn("## Сводка: эпизоды с ошибками", self.markdown)
        self.assertIn("## Сводка: чистые эпизоды", self.markdown)

    def test_the_diagnostic_spray_runs_in_process_and_writes_no_trajectory(self):
        out, traj = self.root / "spray.md", self.root / "spray.jsonl"
        proc = metrics(MINI, "--policies", "spray,oracle", "--out", out, "--trajectories", traj)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = tables(out.read_text(encoding="utf-8"))
        spray = next(row for row in report[SUMMARY] if row["политика"] == "spray")
        self.assertEqual((spray["TP"], spray["FP"]), ("2", "2"))   # both boundary period errors of mini_v03
        self.assertEqual(episode_starts(traj), 1)                    # only the oracle run

    def test_policies_option_selects_and_orders_the_rows(self):
        out = self.root / "two.md"
        proc = metrics(self.split / "a_tiny", "--policies", "oracle,flag_all", "--out", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = tables(out.read_text(encoding="utf-8"))
        self.assertEqual([row["политика"] for row in report[SUMMARY]], ["oracle", "flag_all"])
        self.assertEqual(list(report[PER_EPISODE][0])[3:], ["oracle", "flag_all"])

    def test_bad_input_is_a_clean_error(self):
        empty = self.root / "empty_split"
        (empty / "notes").mkdir(parents=True, exist_ok=True)
        out = self.root / "never.md"
        for args in ([empty], [self.root / "missing"], [MINI, "--policies", "oracle,nobody"],
                     [MINI, "--policies", "oracle,oracle"]):
            with self.subTest(args=args):
                proc = metrics(*args, "--out", out)
                self.assertEqual(proc.returncode, 2)
                self.assertTrue(proc.stderr.strip())
                self.assertFalse(out.exists())


def load_script():
    spec = importlib.util.spec_from_file_location("metrics_script", METRICS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RowsBeyondTheBaselines(unittest.TestCase):
    """Endings the four baselines never reach: a broken findings.json, truncation, an oversize submit."""

    @classmethod
    def setUpClass(cls):
        cls.script = load_script()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.config = {"episode_dir": str(write_episode(Path(cls.tmp.name) / "ep"))}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def row(self, policy_fn):
        env, trajectory = AuditEnv("eval"), io.StringIO()
        result = run_inprocess_episode(policy_fn, env, 0, self.config, trajectory=trajectory)
        findings = self.script.submitted_findings(trajectory.getvalue())
        return self.script.grade_row("test", "ep", env, result, findings)

    def test_broken_structure_counts_every_finding_as_false(self):
        wrong_type = dict(GOOD_FINDING, type="fx_rate")
        row = self.row(lambda _m: submit([GOOD_FINDING, wrong_type], episode_id="OTHER"))
        self.assertEqual((row["tp"], row["fp"], row["fn"], row["reward"]), (0, 2, 1, -1.0))
        self.assertIn("error", row)
        self.assertEqual(row["per_type"]["duplicate_invoice"], {"tp": 0, "fp": 1, "fn": 1})
        self.assertEqual(row["per_type"]["fx_rate"]["fp"], 1)

    def test_truncated_episode_has_no_findings(self):
        row = self.row(lambda _m: {"action": "calculator", "expression": "1+1"})
        self.assertEqual((row["end"], row["findings"], row["fp"], row["fn"], row["steps"]), ("truncated", 0, 0, 1, 100))
        self.assertEqual(row["missed"], ["ISS001"])

    def test_false_findings_of_an_oversize_submit_go_to_other(self):
        bogus = dict(GOOD_FINDING, affected_ids=["invoice/INV001"])
        action = submit([GOOD_FINDING, bogus])
        action["findings"]["padding"] = "x" * 70000  # extra keys are ignored; the trajectory keeps a digest
        row = self.row(lambda _m: action)
        self.assertEqual((row["tp"], row["fp"], row["findings"]), (1, 1, 0))
        self.assertEqual(row["per_type"]["other"], {"tp": 0, "fp": 1, "fn": 0})
        self.assertEqual(sum(c["fp"] for c in row["per_type"].values()), row["fp"])


class CommittedReport(unittest.TestCase):
    """`make metrics` gives the committed docs/METRICS.md except the wall-time column (B_saturday.md item 2).

    The recipe is read from the Makefile, so the test needs no make binary (the Docker image has none)."""

    def recipe(self):
        lines = (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
        variables = dict(line.split(" := ", 1) for line in lines if " := " in line)
        body = lines[lines.index("metrics:") + 1].strip()
        for name, value in variables.items():
            body = body.replace(f"$({name})", value)
        return body.split()[1:]          # drop $(PY)

    def test_make_metrics_repeats_the_committed_report(self):
        args = self.recipe()
        self.assertIn("--policies", args)
        with tempfile.TemporaryDirectory() as tmp:
            out, rows = Path(tmp) / "METRICS.md", Path(tmp) / "rows.jsonl"
            args[args.index("--out") + 1], args[args.index("--json") + 1] = str(out), str(rows)
            proc = run_python(args, cwd=ROOT)
            self.assertEqual(proc.returncode, 0, proc.stderr)

            def without_time(markdown):
                kept = [line for line in markdown.splitlines() if not line.startswith("|")]
                found = tables(markdown)
                for table in found.values():
                    for row in table:
                        row.pop(WALL_TIME, None)
                return kept, found

            committed = (ROOT / "docs" / "METRICS.md").read_text(encoding="utf-8")
            self.assertEqual(without_time(out.read_text(encoding="utf-8")), without_time(committed))


if __name__ == "__main__":
    unittest.main()
