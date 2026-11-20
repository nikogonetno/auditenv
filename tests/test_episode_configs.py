"""Versioned train and dev configs and the episode builder (docs/tasks/B_next.md items 9-10)."""
import json
import tempfile
import unittest
from pathlib import Path

from auditenv.config import SCENARIO_ERRORS, config_sha256, is_clean
from scripts.build_episodes import SEED_RANGES, build, load_split
from tests.fake_generators import copy_mini


class ConfigVersions(unittest.TestCase):
    def test_both_splits_hold_ten_valid_configs_in_their_seed_range(self):
        seeds = {}
        for split in ("train", "dev"):
            version, rows = load_split(split)
            self.assertEqual(version, f"{split}_v1")
            self.assertEqual(len(rows), 10)
            seeds[split] = {seed for seed, _ in rows}
            self.assertEqual(len(seeds[split]), 10)
            self.assertTrue(all(s in SEED_RANGES[split] for s in seeds[split]))
        self.assertFalse(seeds["train"] & seeds["dev"])

    def test_each_split_mixes_clean_and_coupled_episodes(self):
        for split in ("train", "dev"):
            _, rows = load_split(split)
            configs = [c for _, c in rows]
            with self.subTest(split=split):
                self.assertGreaterEqual(sum(is_clean(c) for c in configs), 2)
                for scenario in SCENARIO_ERRORS:
                    self.assertTrue(any(c["scenarios"][scenario] for c in configs), scenario)


class Builder(unittest.TestCase):
    def test_build_writes_an_index_with_config_hashes(self):
        with tempfile.TemporaryDirectory() as out:
            doc = build("dev", out, copy_mini)
            index = json.loads((Path(out) / "dev" / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index, doc)
            _, rows = load_split("dev")
            self.assertEqual([e["seed"] for e in index["episodes"]], [s for s, _ in rows])
            self.assertEqual([e["config_sha256"] for e in index["episodes"]], [config_sha256(c) for _, c in rows])
            self.assertTrue((Path(out) / "dev" / "2003" / "public" / "manifest.json").is_file())
            build("dev", out, copy_mini)  # a rebuild replaces the episodes in place


if __name__ == "__main__":
    unittest.main()
