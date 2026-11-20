"""Generator stage 1 (docs/tasks/A_generator.md): clean world, duplicate_invoice and payment_allocation."""
import hashlib
import json
import re
import tempfile
import time
import unittest
from pathlib import Path

from auditenv.config import normalize_config
from auditenv.env import AuditEnv
from auditenv.episode import load_episode
from auditenv.generate import generate_episode
from auditenv.grader import parse_truth
from auditenv.runner import run_inprocess_episode
from auditenv.verify import verify_episode
from auditenv.world import WorldError, build_world, check_book, post
from policies.oracle import make_env_oracle
from tests.support import run_python

ROOT = Path(__file__).resolve().parent.parent
STAGE1 = {"errors": {"duplicate_invoice": 1, "payment_allocation": 1}}
CHILD = (
    "import sys, json\n"
    f"sys.path.insert(0, {str(ROOT)!r})\n"
    "from auditenv.generate import generate_episode\n"
    "generate_episode(int(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3])\n"
)


def digest(root):
    h = hashlib.sha256()
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        h.update(path.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def oracle_reward(directory):
    env = AuditEnv("train")
    run_inprocess_episode(make_env_oracle(env), env, 0, {"episode_dir": str(directory)})
    return env.grader_log[-1]


class Generated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dirs = {}
        for seed, config in ((3, STAGE1), (11, {"errors": {"duplicate_invoice": 2, "payment_allocation": 3}}),
                             (5, {"errors": {}}), (8, {"documents": [20, 20], **STAGE1})):
            out = Path(cls.tmp.name) / f"{seed}"
            started = time.monotonic()
            generate_episode(seed, config, str(out))
            cls.dirs[seed] = (out, normalize_config(config), time.monotonic() - started)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_episode_loads_truth_parses_and_oracle_gets_one(self):
        for seed, (out, config, _) in self.dirs.items():
            with self.subTest(seed=seed):
                episode = load_episode(out)
                truth = episode.load_truth()
                parse_truth(truth, episode.aliases)
                entry = oracle_reward(out)
                self.assertEqual((entry["reward"], entry["fp"], entry["fn"]), (1.0, 0, 0))
                expected = sum(config["errors"].values())
                self.assertEqual((len(truth["issues"]), truth["clean"]), (expected, expected == 0))

    def test_independent_verifier_finds_no_problem(self):
        for seed, (out, _, _) in self.dirs.items():
            with self.subTest(seed=seed):
                self.assertEqual(verify_episode(str(out)), [])

    def test_volume_follows_the_config(self):
        for seed, (out, config, seconds) in self.dirs.items():
            with self.subTest(seed=seed):
                manifest = json.loads((out / "public" / "manifest.json").read_text(encoding="utf-8"))
                low, high = config["documents"]
                self.assertTrue(low <= manifest["documents"] <= high, manifest["documents"])
                self.assertLessEqual(manifest["counts"]["ledger"], 2000)
                self.assertLess(seconds, 5)

    def test_ids_are_unique_across_tables(self):
        out = self.dirs[11][0]
        episode = load_episode(out)
        ids = [episode.episode_id]
        for table, rows in episode.tables.items():
            id_col = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": "fx_id"}[table]
            ids += [r[id_col] for r in rows]
        ids += [r["doc_id"] for r in episode.tables["invoices"] + episode.tables["payments"]]
        ids += [line["line_id"] for r in episode.tables["invoices"] for line in r["line_items"]]
        ids += sorted({r["je_id"] for r in episode.tables["ledger"]})
        ids += sorted({r["entity_id"] for r in episode.tables["invoices"]})
        ids += [i["issue_id"] for i in episode.load_truth()["issues"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{6}", i) for i in ids))

    def test_provenance_names_seed_config_and_every_issue(self):
        out, config, _ = self.dirs[11]
        provenance = json.loads((out / "private" / "provenance.json").read_text(encoding="utf-8"))
        truth = json.loads((out / "private" / "truth.json").read_text(encoding="utf-8"))
        self.assertEqual((provenance["seed"], provenance["config"]), (11, config))
        named = sorted(i for inj in provenance["injections"] for i in inj["issue_ids"])
        self.assertEqual(named, sorted(i["issue_id"] for i in truth["issues"]))
        self.assertTrue(all(inj["changes"] for inj in provenance["injections"]))
        self.assertEqual(json.loads((out / "public" / "policy.json").read_text())["fx"]["weekend_days"], [6, 7])


class Determinism(unittest.TestCase):
    def test_same_seed_same_bytes_under_different_hash_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            digests = []
            for hash_seed in ("0", "1", "random"):
                out = Path(tmp) / hash_seed
                result = run_python(["-c", CHILD, "42", json.dumps(STAGE1), out], hash_seed=hash_seed)
                self.assertEqual(result.returncode, 0, result.stderr)
                digests.append(digest(out))
            generate_episode(43, STAGE1, str(Path(tmp) / "other"))
            self.assertEqual(len(set(digests)), 1)
            self.assertNotEqual(digests[0], digest(Path(tmp) / "other"))


class Refusals(unittest.TestCase):
    def test_bad_seed_and_config_are_value_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed, config in ((-1, STAGE1), (True, STAGE1), (1, {"colour": "red"}),
                                 (1, {"documents": [20, 20], "errors": {"payment_allocation": 12}})):
                with self.subTest(seed=seed, config=config), self.assertRaises(ValueError):
                    generate_episode(seed, config, tmp)

    def test_own_invariant_check_catches_a_broken_book(self):
        import random
        world = build_world(random.Random(1), normalize_config({"errors": {}}))
        book = post(world)
        check_book(world, book, set())
        book.rows[0].debit_minor += book.rows[0].debit_minor and 1
        book.rows[0].credit_minor += book.rows[0].credit_minor and 1
        with self.assertRaises(WorldError):
            check_book(world, book, set())


if __name__ == "__main__":
    unittest.main()
