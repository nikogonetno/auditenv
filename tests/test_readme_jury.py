"""README «Для жюри»: the three ways to plug in a policy, run word for word.

The jury checks the README literally, so the Python example must print exactly the output shown under it,
the --policy-cmd command must give the report the text describes, and the closed Docker run must be the
command of README section 4 that the CI step runs. Every code_version in the README is the package version.
"""
import json
import os
import re
import shlex
import subprocess
import sys
import unittest
from pathlib import Path

import auditenv

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / ".github" / "workflows" / "verify.yml"   # not in the Docker image (.dockerignore)
README = (ROOT / "README.md").read_text(encoding="utf-8")
JURY = README.split("\n## Для жюри\n", 1)[1].split("\n## ", 1)[0]
IN_PROCESS, POLICY_CMD, DOCKER = (JURY.split("\n**2. ", 1)[0], JURY.split("\n**2. ", 1)[1].split("\n**3. ", 1)[0],
                                  JURY.split("\n**3. ", 1)[1])


def blocks(text, lang):
    return re.findall(rf"```{lang}\n(.*?)```", text, re.S)


def child_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    return {**env, "PYTHONPATH": str(ROOT)}


class ReadmeForTheJury(unittest.TestCase):
    def test_the_section_comes_before_section_1(self):
        self.assertLess(README.index("\n## Для жюри\n"), README.index("\n## 1. "))

    def test_the_python_example_prints_the_output_shown(self):
        (code,), (shown,) = blocks(IN_PROCESS, "python"), blocks(IN_PROCESS, "text")
        self.assertIn("env.reset(7, {})", code)
        proc = subprocess.run([sys.executable, "-P", "-s", "-"], input=code, capture_output=True, text=True,
                              cwd=ROOT, env=child_env(), timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(proc.stdout, shown)

    def test_the_policy_cmd_example_gives_the_report_described(self):
        (command,) = blocks(POLICY_CMD, "bash")
        argv = shlex.split(command)
        self.assertEqual(argv[:3], ["python3", "-m", "auditenv.run"])
        proc = subprocess.run([sys.executable, "-P", "-s", *argv[1:]], capture_output=True, text=True, cwd=ROOT,
                              env=child_env(), timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        for key in ("policy", "steps", "terminated", "reward"):
            with self.subTest(key=key):
                self.assertIn(f'`"{key}": {json.dumps(report[key])}`', POLICY_CMD)
        self.assertEqual(report["episode_id"], "EGGYPY")
        self.assertEqual(report["code_version"], auditenv.__version__)

    def test_the_closed_run_is_the_one_of_section_4(self):
        (jury,) = blocks(DOCKER, "bash")
        build, *run = jury.splitlines(keepends=True)
        section4 = README.split("\n## 4. ", 1)[1].split("\n## 5. ", 1)[0]
        self.assertIn(f"```bash\n{build}", section4)
        closed = blocks(section4.split("**Закрытый прогон жюри.**", 1)[1], "bash")[0]
        self.assertEqual("".join(run), closed)

    @unittest.skipUnless(CI.exists(), "the CI workflow is not copied into the Docker image")
    def test_the_closed_run_is_the_command_of_the_ci_step(self):
        (jury,) = blocks(DOCKER, "bash")
        run = jury.splitlines()[1:]
        ci = CI.read_text(encoding="utf-8")
        for line in (run[0], run[-1]):                  # the policy folder and the docker run itself
            with self.subTest(line=line[:40]):
                self.assertIn(line, ci)

    def test_every_code_version_in_the_readme_is_the_package_version(self):
        versions = re.findall(r'"code_version": "([^"]*)"', README)
        self.assertTrue(versions)
        self.assertEqual(set(versions), {auditenv.__version__})


if __name__ == "__main__":
    unittest.main()
