"""Jury rehearsal, part A item 7, inside the sandbox (docs/tasks/V_rehearsal.md; role V).

The hostile probe policy of qa/rehearsal/probe_policy.py runs as the unprivileged user auditpolicy and
tries files, /proc, the environment and the runner, then broken lines and actions. Outside the Docker
image the policy runs as the same user, so these tests are skipped there and run in the Docker step of
CI. The checks themselves live in qa/rehearsal/probes.py, the same ones the local rehearsal prints.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qa" / "rehearsal"))

from auditenv.runner import sandbox_user  # noqa: E402

USER = sandbox_user()


@unittest.skipIf(USER is None, "needs root and the auditpolicy user: runs in the Docker image")
class HostileProbeInSandbox(unittest.TestCase):
    def setUp(self):
        import probes
        self.probes = probes
        self._tmp = tempfile.TemporaryDirectory(prefix="rehearsal-sandbox-")
        os.chmod(self._tmp.name, 0o700)
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def assert_isolated(self, config, seed, truth):
        out = self.probes.hostile_run(config, seed, truth, sandbox="required")
        print(json.dumps({"hostile probe": config if "episode_dir" not in config else "episode_dir",
                          "uid": out["uid"], "reach": out["reach"], "error_codes": out["errors"]},
                         sort_keys=True), file=sys.stderr)
        self.assertNotIn(out["uid"], (None, 0))
        self.assertEqual(out["problems"], [])
        self.assertEqual(out["reach"], [])

    def test_prepared_episode(self):
        episode = self.tmp / "mini"
        shutil.copytree(ROOT / "tests" / "fixtures" / "mini_v03", episode)
        truth = json.loads((episode / "private" / "truth.json").read_text(encoding="utf-8"))
        self.assert_isolated({"episode_dir": str(episode)}, None, truth)

    def test_episode_generated_in_reset(self):
        from common import build
        config = {"errors": {"fx_rate": 1}, "scenarios": {"C1": 1, "C2": 1}, "foreign_pct": [30, 70],
                  "company_offsets": ["+05:45"], "period": "2026-11", "twins": [20, 20]}
        truth = json.loads((build(9007, config, self.tmp / "same") / "private" / "truth.json").read_text(encoding="utf-8"))
        self.assert_isolated(config, 9007, truth)


if __name__ == "__main__":
    unittest.main()
