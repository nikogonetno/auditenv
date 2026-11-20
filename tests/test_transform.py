"""auditenv/transform.py: row permutation and id renaming keep truth (INV-22, docs/tasks/B_night_stage3.md item 2)."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from auditenv.env import AuditEnv
from auditenv.episode import load_episode
from auditenv.generate import generate_episode
from auditenv.grader import parse_truth
from auditenv.runner import run_inprocess_episode
from auditenv.transform import permute_and_rename
from auditenv.verify import verify_episode
from policies.oracle import make_env_oracle
from policies.reconcile import ReconcilePolicy

ROOT = Path(__file__).resolve().parent.parent
MINI = ROOT / "tests" / "fixtures" / "mini_v03"


def digest(root):
    h = hashlib.sha256()
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        h.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def all_ids(directory):
    episode = load_episode(directory)
    ids = {episode.episode_id, *episode.documents}
    for table in episode.index.values():
        ids |= set(table)
    for inv in episode.tables["invoices"]:
        ids |= {line["line_id"] for line in inv["line_items"]}
    ids |= {r["je_id"] for r in episode.tables["ledger"]} | {r["entity_id"] for r in episode.tables["ledger"] if r["entity_id"]}
    ids |= {i["issue_id"] for i in episode.load_truth()["issues"]}
    return ids


def reward(directory, policy):
    env = AuditEnv("train")
    run_inprocess_episode(make_env_oracle(env) if policy == "oracle" else ReconcilePolicy(), env, 0,
                          {"episode_dir": str(directory)})
    return env.grader_log[-1]["reward"]


class Transform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.original = root / "original"
        generate_episode(31, {"errors": {"duplicate_invoice": 2, "payment_allocation": 2, "fx_rate": 2,
                                         "period_cutoff": 2}}, str(cls.original))
        cls.copy = root / "copy"
        permute_and_rename(str(cls.original), str(cls.copy), 5)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def truth(self, directory):
        return json.loads((Path(directory) / "private" / "truth.json").read_text(encoding="utf-8"))

    def test_truth_is_the_same_truth_under_new_ids(self):
        """INV-22: same types, amounts and currencies; every id maps through id_map to the original's."""
        id_map = json.loads((self.copy / "private" / "id_map.json").read_text(encoding="utf-8"))["ids"]
        back = {new: old for old, new in id_map.items()}

        def unmap(ref):
            prefix, _, rid = ref.partition("/")
            return f"{prefix}/{back[rid]}"

        original = {i["issue_id"]: i for i in self.truth(self.original)["issues"]}
        copied = self.truth(self.copy)["issues"]
        self.assertEqual(len(original), len(copied))
        for issue in copied:
            source = original[back[issue["issue_id"]]]
            self.assertEqual((issue["type"], issue["expected_amount_minor"], issue["currency"]),
                             (source["type"], source["expected_amount_minor"], source["currency"]))
            self.assertEqual(sorted(map(unmap, issue["affected_ids"])), source["affected_ids"])
            self.assertEqual([sorted(map(unmap, s)) for s in issue["evidence_sets"]], source["evidence_sets"])
        episode = load_episode(self.copy)
        parse_truth(episode.load_truth(), episode.aliases)

    def test_new_ids_are_unique_and_never_old(self):
        old, new = all_ids(self.original), all_ids(self.copy)
        self.assertFalse(old & new)
        id_map = json.loads((self.copy / "private" / "id_map.json").read_text(encoding="utf-8"))["ids"]
        self.assertEqual(len(set(id_map.values())), len(id_map))
        self.assertEqual(set(id_map), old)

    def test_rows_move_and_business_attributes_stay(self):
        def rows(directory, name):
            return (Path(directory) / "public" / name).read_text(encoding="utf-8").splitlines()

        self.assertNotEqual(rows(self.original, "ledger.csv")[1:6], rows(self.copy, "ledger.csv")[1:6])
        self.assertEqual(rows(self.original, "ledger.csv")[0], rows(self.copy, "ledger.csv")[0])

        def business(directory):
            return sorted(repr((i["invoice_number"], i["vendor_ref"], i["po_ref"], i["total_minor"], i["issued_at"]))
                          for i in map(json.loads, rows(directory, "invoices.jsonl")))

        self.assertEqual(business(self.original), business(self.copy))
        self.assertEqual((self.original / "public" / "policy.json").read_bytes(),
                         (self.copy / "public" / "policy.json").read_bytes())

    def test_strict_verifier_oracle_and_reconcile_on_the_copy(self):
        self.assertEqual(verify_episode(str(self.copy)), [])
        self.assertEqual(reward(self.copy, "oracle"), 1.0)
        self.assertEqual(reward(self.copy, "reconcile"), reward(self.original, "reconcile"))

    def test_same_salt_same_bytes_other_salt_other_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            again, other = Path(tmp) / "again", Path(tmp) / "other"
            permute_and_rename(str(self.original), str(again), 5)
            permute_and_rename(str(self.original), str(other), 6)
            self.assertEqual(digest(again), digest(self.copy))
            self.assertNotEqual(digest(other), digest(self.copy))

    def test_the_mini_fixture_keeps_its_eight_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            permute_and_rename(str(MINI), tmp, 1)
            self.assertEqual(reward(tmp, "oracle"), 1.0)
            self.assertEqual(len(self.truth(tmp)["issues"]), 8)
            self.assertEqual(reward(tmp, "reconcile"), reward(MINI, "reconcile"))


if __name__ == "__main__":
    unittest.main()
