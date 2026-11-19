"""Golden episode v0.1 from role A, re-checked on our side (checkpoint 0).

Byte determinism under different hash seeds (INV-24), recomputation of every
truth amount from evidence only (INV-17), no private markers in public files
(INV-23), IDs unique across tables (decision v2). The fixture still uses the
pre-v2 file schema; QA findings P-01..P-06 are expected until golden v0.2.
"""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tests.support import run_python

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_v0.1"
EPISODE = GOLDEN / "episode"
HASH_SEEDS = ("0", "1", "12345")
ISSUE_TYPES = ("duplicate_invoice", "payment_allocation", "fx_rate", "period_cutoff")
PRIVATE_WORDS = ("truth", "provenance", "inject", "twin", "scenario")
# build_golden.py also writes this debug file; it is not part of the archived fixture.
BUILD_ONLY_FILES = {"private/summary_for_spec.json"}
ID_COLUMNS = {
    "counterparties.csv": "counterparty_id",
    "invoices.csv": "invoice_id",
    "invoice_lines.csv": "line_id",
    "payments.csv": "payment_id",
    "allocations.csv": "allocation_id",
    "ledger.csv": "row_id",
}


def sha256_tree(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def load_json(rel):
    return json.loads((EPISODE / rel).read_text(encoding="utf-8"))


def read_csv(name):
    with open(EPISODE / "public" / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class HashSeedHarness(unittest.TestCase):
    def test_child_hash_seed_is_really_applied(self):
        code = ["-c", "print(hash('auditenv'))"]
        zero, one = run_python(code, "0").stdout, run_python(code, "1").stdout
        self.assertNotEqual(zero, one, "PYTHONHASHSEED ignored, determinism checks would be vacuous")
        self.assertEqual(zero, run_python(code, "0").stdout)


class GoldenRebuild(unittest.TestCase):
    def test_rebuild_matches_fixture_byte_for_byte(self):
        expected = sha256_tree(EPISODE)
        builds = {}
        with tempfile.TemporaryDirectory() as tmp:
            for seed in HASH_SEEDS:
                out = Path(tmp) / f"hashseed_{seed}"
                res = run_python([GOLDEN / "build_golden.py", out], seed, cwd=tmp)
                self.assertEqual(res.returncode, 0, res.stderr)
                builds[seed] = sha256_tree(out)
        for seed, got in builds.items():
            with self.subTest(hash_seed=seed):
                self.assertLessEqual(set(got) - set(expected), BUILD_ONLY_FILES)
                self.assertEqual({k: got.get(k) for k in expected}, expected)
                self.assertEqual(got, builds[HASH_SEEDS[0]])


class GoldenTruth(unittest.TestCase):
    def test_every_truth_amount_recomputes_from_evidence(self):
        n = len(load_json("private/truth.json")["issues"])
        self.assertEqual(n, 8)
        res = run_python([GOLDEN / "verify_golden.py", EPISODE])
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn(f"{n}/{n} issues recomputed from evidence only", res.stdout)


class GoldenLeaks(unittest.TestCase):
    def test_public_files_hold_no_private_markers(self):
        truth = load_json("private/truth.json")
        provenance = load_json("private/provenance.json")
        markers = set(ISSUE_TYPES) | set(PRIVATE_WORDS) | {truth["episode_id"]}
        markers |= {i["issue_id"] for i in truth["issues"]}
        markers |= {t["twin_id"] for t in provenance["twins"]}
        for path in sorted((EPISODE / "public").iterdir()):
            text = path.read_text(encoding="utf-8").lower()
            for marker in sorted(markers):
                with self.subTest(file=path.name, marker=marker):
                    self.assertNotIn(marker.lower(), text)

    def test_no_underscore_columns_in_public_csv(self):
        for name in sorted(p.name for p in (EPISODE / "public").glob("*.csv")):
            with open(EPISODE / "public" / name, encoding="utf-8", newline="") as f:
                header = next(csv.reader(f))
            with self.subTest(file=name):
                self.assertFalse([c for c in header if c.startswith("_")])


class GoldenIds(unittest.TestCase):
    def test_ids_are_unique_across_tables(self):
        owner = {}
        for name, column in ID_COLUMNS.items():
            for row in read_csv(name):
                rid = row[column]
                self.assertNotIn(rid, owner, f"{rid} in {name} and {owner.get(rid)}")
                owner[rid] = name
        for je in sorted({r["je_id"] for r in read_csv("ledger.csv")}):
            self.assertNotIn(je, owner, f"je {je} collides with {owner.get(je)}")


if __name__ == "__main__":
    unittest.main()
