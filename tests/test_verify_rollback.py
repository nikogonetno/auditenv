"""V's rollback of provenance (docs/tasks/V_rollback.md): the book before the errors under strict V.
B has its own rollback in tests/test_rollback.py, this one tests auditenv/verify/rollback.py."""
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from auditenv.verify import verify_episode
from auditenv.verify.rollback import rollback_episode, verify_rollback
from tests.support import run_python

ROOT = Path(__file__).resolve().parent.parent
# Train 1000 creates invoices and ledger rows and changes amounts, rates and dates; dev 2004 also
# changes ledger.invoice_id and payment.invoice_refs.
EPISODES = (ROOT / "episodes" / "train" / "1000", ROOT / "episodes" / "dev" / "2004")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def ledger(directory):
    with open(Path(directory) / "public" / "ledger.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class Rollback(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="test-rollback-"))
        self.src = self.tmp / "src"
        shutil.copytree(EPISODES[0], self.src)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def provenance(self):
        return read_json(self.src / "private" / "provenance.json")

    def remove_untouched_row(self, with_row):
        """Take out a ledger row no change names and record it as removed by a new last error."""
        named = {c["record"] for i in self.provenance()["injections"] for c in i["changes"]}
        rows = ledger(self.src)
        gone = next(r for r in rows if f"ledger/{r['row_id']}" not in named)
        with open(self.src / "public" / "ledger.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(r for r in rows if r is not gone)
        manifest = read_json(self.src / "public" / "manifest.json")
        manifest["counts"]["ledger"] -= 1
        write_json(self.src / "public" / "manifest.json", manifest)
        change = {"record": f"ledger/{gone['row_id']}", "field": None, "before": "removed", "after": None}
        if with_row:
            change["row"] = {k: (None if v == "" else int(v) if k.endswith("_minor") else v) for k, v in gone.items()}
        provenance = self.provenance()
        last = max(i["order"] for i in provenance["injections"])
        provenance["injections"].append({"order": last + 1, "changes": [change], "issue_ids": []})
        write_json(self.src / "private" / "provenance.json", provenance)
        return gone

    def edit_change(self, predicate, edit):
        provenance = self.provenance()
        change = next(c for i in provenance["injections"] for c in i["changes"] if predicate(c))
        edit(change)
        write_json(self.src / "private" / "provenance.json", provenance)
        return change

    # --- the two episodes ---------------------------------------------------------------------

    def test_train_and_dev_episodes_roll_back_to_a_clean_book(self):
        for episode in EPISODES:
            with self.subTest(episode=episode.name):
                self.assertEqual(verify_rollback(str(episode)), [])

    def test_the_rollback_is_needed_and_does_something(self):
        for episode in EPISODES:
            with self.subTest(episode=episode.name):
                truth = read_json(episode / "private" / "truth.json")
                final = self.tmp / f"final_{episode.name}"
                shutil.copytree(episode, final)
                write_json(final / "private" / "truth.json",
                           {"clean": True, "episode_id": truth["episode_id"], "issues": []})
                self.assertTrue(verify_episode(str(final)), "the final book with an empty truth must fail")
                restored = self.tmp / f"restored_{episode.name}"
                self.assertEqual(rollback_episode(episode, restored), [])
                self.assertEqual(read_json(restored / "private" / "truth.json"),
                                 {"clean": True, "episode_id": truth["episode_id"], "issues": []})
                self.assertNotEqual(read_json(restored / "public" / "manifest.json")["counts"],
                                    read_json(episode / "public" / "manifest.json")["counts"])
                self.assertEqual(verify_episode(str(restored)), [])

    def test_the_source_episode_is_not_touched(self):
        before = {p.relative_to(self.src): p.read_bytes() for p in self.src.rglob("*") if p.is_file()}
        verify_rollback(str(self.src))
        after = {p.relative_to(self.src): p.read_bytes() for p in self.src.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    # --- removed rows -------------------------------------------------------------------------

    def test_a_removed_row_comes_back_from_its_content(self):
        gone = self.remove_untouched_row(with_row=True)
        self.assertEqual(verify_rollback(str(self.src)), [])
        restored = self.tmp / "restored"
        rollback_episode(self.src, restored)
        rows = ledger(restored)
        self.assertIn(gone, rows)
        self.assertEqual([r["je_id"] for r in rows], sorted(r["je_id"] for r in rows))

    def test_a_removed_row_without_content_cannot_come_back(self):
        gone = self.remove_untouched_row(with_row=False)
        problems = verify_rollback(str(self.src))
        self.assertEqual({p["code"] for p in problems}, {"ROLLBACK"})
        self.assertIn(f"ledger/{gone['row_id']} was removed", problems[0]["detail"])

    def test_a_removed_row_that_is_still_there_is_a_problem(self):
        gone = self.remove_untouched_row(with_row=True)
        shutil.copy(EPISODES[0] / "public" / "ledger.csv", self.src / "public" / "ledger.csv")
        problems = verify_rollback(str(self.src))
        self.assertIn(f"ledger/{gone['row_id']} was removed, but it is still in ledger.csv",
                      [p["detail"].split(": ", 1)[1] for p in problems])

    # --- provenance that does not fit the files -----------------------------------------------

    def test_a_field_whose_value_is_not_after_is_a_problem(self):
        change = self.edit_change(lambda c: c["field"] == "credit_minor", lambda c: c.update(after=c["after"] + 1))
        problems = verify_rollback(str(self.src))
        self.assertEqual({p["code"] for p in problems}, {"ROLLBACK"})
        self.assertIn(f"{change['record']}.credit_minor is", problems[0]["detail"])

    def test_a_created_record_that_is_not_there_is_a_problem(self):
        self.edit_change(lambda c: c["after"] == "created", lambda c: c.update(record="invoice/ZZZZZZ"))
        problems = verify_rollback(str(self.src))
        self.assertIn("invoice/ZZZZZZ was created, but it is not in invoices.jsonl",
                      [p["detail"].split(": ", 1)[1] for p in problems])

    def test_an_unknown_table_or_field_is_a_problem(self):
        self.edit_change(lambda c: c["field"] == "timestamp", lambda c: c.update(field="posted_on"))
        self.assertIn("has no field 'posted_on'", verify_rollback(str(self.src))[0]["detail"])
        self.edit_change(lambda c: c["after"] == "created", lambda c: c.update(record="allocation/ZZZZZZ"))
        self.assertIn("names no public table", " ".join(p["detail"] for p in verify_rollback(str(self.src))))

    def test_broken_or_missing_provenance_is_a_problem_not_an_exception(self):
        provenance = self.provenance()
        provenance["injections"][0]["order"] = provenance["injections"][1]["order"]
        write_json(self.src / "private" / "provenance.json", provenance)
        self.assertEqual([p["code"] for p in verify_rollback(str(self.src))], ["ROLLBACK"])
        (self.src / "private" / "provenance.json").unlink()
        self.assertEqual([p["code"] for p in verify_rollback(str(self.src))], ["ROLLBACK"])

    # --- CLI ------------------------------------------------------------------------------------

    def cli(self, *args):
        code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); from auditenv.verify.__main__ import main; "
                f"sys.exit(main({[*args, str(self.src)]!r}))")
        return run_python(["-c", code])

    def test_cli_rollback_exit_codes_and_json_lines(self):
        ok = self.cli("--rollback")
        self.assertEqual((ok.returncode, ok.stdout), (0, ""), ok.stderr)
        self.remove_untouched_row(with_row=False)
        bad = self.cli("--rollback")
        self.assertEqual(bad.returncode, 1, bad.stderr)
        self.assertEqual({json.loads(line)["code"] for line in bad.stdout.splitlines()}, {"ROLLBACK"})


if __name__ == "__main__":
    unittest.main()
