"""Shared helpers for tests that spawn a child Python."""
import os
import subprocess
import sys


def run_python(args, hash_seed="0", cwd=None):
    """Run a child interpreter with an explicit hash seed.

    `python -I` would be stricter, but it implies -E and silently ignores
    PYTHONHASHSEED. So we pass -P -s and drop inherited PYTHON* variables.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONHASHSEED"] = hash_seed
    return subprocess.run(
        [sys.executable, "-P", "-s", *map(str, args)],
        capture_output=True, text=True, env=env, cwd=cwd, timeout=120,
    )
