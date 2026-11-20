"""Stress test of the generator, verifier and oracle (docs/tasks/B_next.md item 7).

    python3 scripts/stress.py --n 300 [--start 0] [--configs configs/stress_v1.json] [--generator module:function]

For each seed it generates an episode, then checks:
  load      the episode loads and its truth parses (auditenv.episode, auditenv.grader)
  verify    the independent verifier V finds no problem (auditenv.verify; skipped while V is not merged)
  oracle    the oracle gets reward 1.0 through the env
  time      generation plus the oracle episode take less than 60 s (case §07)
  determinism  two generations in fresh processes with PYTHONHASHSEED 0 and 1 give the same sha256 of every file

Seeds cycle through the configs of --configs. Prints one JSON line per seed and a summary line; exit 1 on
any failure. Without a generator module it prints SKIP and exits 0, so make verify stays green until A lands.
"""
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auditenv.config import normalize_config  # noqa: E402
from auditenv.env import AuditEnv  # noqa: E402
from auditenv.runner import run_inprocess_episode  # noqa: E402
from policies.oracle import make_env_oracle  # noqa: E402

DEFAULT_GENERATOR = "auditenv.generate:generate_episode"
EPISODE_TIME_LIMIT_S = 60.0
HASH_SEEDS = ("0", "1")


def load_generator(spec):
    module_name, _, func = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return getattr(module, func)


def load_verifier():
    try:
        from auditenv.verify import verify_episode
    except ImportError:
        return None
    return verify_episode


def tree_digest(root):
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def generate_in_fresh_process(spec, seed, config, out_dir, hash_seed):
    """Generate in a child interpreter with an explicit hash seed (-P -s: -I would ignore PYTHONHASHSEED)."""
    module_name, _, func = spec.partition(":")
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        f"from {module_name} import {func} as generate\n"
        f"generate({seed}, json.loads(sys.argv[1]), sys.argv[2])\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONHASHSEED"] = hash_seed
    result = subprocess.run([sys.executable, "-P", "-s", "-c", code, json.dumps(config), str(out_dir)],
                            capture_output=True, text=True, env=env, timeout=EPISODE_TIME_LIMIT_S * 2)
    return result.returncode, result.stderr.strip().splitlines()[-1:] if result.stderr else []


def check_seed(seed, config, generator, generator_spec, verifier, workdir, strict_volume=True):
    row = {"seed": seed, "config": config, "failures": []}
    out = Path(workdir) / f"seed_{seed}"
    started = time.monotonic()
    try:
        generator(seed, config, str(out))
    except Exception as exc:  # report and move on: one bad seed must not hide the others
        row["failures"].append(f"generate: {type(exc).__name__}: {exc}")
        return row
    generated = time.monotonic()

    env = AuditEnv("train")
    try:
        run_inprocess_episode(make_env_oracle(env), env, seed, {"episode_dir": str(out)})
    except Exception as exc:
        row["failures"].append(f"load: {type(exc).__name__}: {exc}")
        return row
    entry = env.grader_log[-1]
    row["issues"] = entry["tp"] + entry["fn"]
    row["oracle_reward"] = entry["reward"]
    if entry["reward"] != 1.0:
        row["failures"].append(f"oracle: reward {entry['reward']} (tp {entry['tp']} fp {entry['fp']} fn {entry['fn']})")
    row["episode_s"] = round(time.monotonic() - started, 3)
    row["generate_s"] = round(generated - started, 3)
    if row["episode_s"] >= EPISODE_TIME_LIMIT_S:
        row["failures"].append(f"time: {row['episode_s']} s")

    if verifier is None:
        row["verify"] = "skipped: auditenv.verify not available"
    else:
        problems = verifier(str(out), strict_volume=strict_volume)
        row["verify"] = len(problems)
        if problems:
            row["failures"].append(f"verify: {json.dumps(problems[:3], ensure_ascii=False)}")

    digests = {"in_process": tree_digest(out)}
    for hash_seed in HASH_SEEDS:
        again = Path(workdir) / f"seed_{seed}_hash_{hash_seed}"
        code, err = generate_in_fresh_process(generator_spec, seed, config, again, hash_seed)
        if code != 0:
            row["failures"].append(f"determinism: generation failed with PYTHONHASHSEED={hash_seed}: {err}")
            return row
        digests[f"hash_{hash_seed}"] = tree_digest(again)
    if len(set(digests.values())) != 1:
        row["failures"].append(f"determinism: sha256 differs {digests}")
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--configs", default=str(ROOT / "configs" / "stress_v1.json"))
    parser.add_argument("--generator", default=DEFAULT_GENERATOR, help="module:function, default A's generator")
    parser.add_argument("--no-strict-volume", action="store_true",
                        help="skip the INV-03 volume check: only for small test books, never for generated episodes")
    args = parser.parse_args(argv)

    generator = load_generator(args.generator)
    if generator is None:
        print(json.dumps({"stress": "SKIP", "reason": f"{args.generator} not available yet"}))
        return 0
    with open(args.configs, encoding="utf-8") as f:
        configs = [normalize_config(c) for c in json.load(f)["configs"]]
    verifier = load_verifier()

    failed = 0
    with tempfile.TemporaryDirectory(prefix="auditenv-stress-") as workdir:
        for k in range(args.n):
            seed = args.start + k
            row = check_seed(seed, configs[k % len(configs)], generator, args.generator, verifier, workdir,
                             strict_volume=not args.no_strict_volume)
            failed += bool(row["failures"])
            print(json.dumps(row, sort_keys=True, ensure_ascii=False), flush=True)
    summary = {"stress": "FAIL" if failed else "OK", "seeds": args.n, "failed": failed,
               "verifier": "on" if verifier else "skipped"}
    print(json.dumps(summary, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
