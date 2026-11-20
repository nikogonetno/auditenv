"""Test doubles for A's generator: generate_episode(seed, config, out_dir). Never used outside tests."""
import os
import shutil
from pathlib import Path

MINI = Path(__file__).resolve().parent / "fixtures" / "mini_v03"


def copy_mini(seed, config, out_dir):
    """Deterministic: always the mini_v03 book, whatever the seed and config."""
    shutil.copytree(MINI, out_dir, dirs_exist_ok=True)


def nondeterministic(seed, config, out_dir):
    """mini_v03 plus a file that differs on every call, so the stress determinism check must fail."""
    copy_mini(seed, config, out_dir)
    (Path(out_dir) / "public" / "noise.bin").write_bytes(os.urandom(16))


def failing(seed, config, out_dir):
    raise ValueError("cannot place fx_rate: no foreign documents")
