"""Shared helpers of the jury rehearsal (docs/tasks/V_rehearsal.md, part A).

Only the public interface is used: auditenv.config, auditenv.generate.generate_episode (SPEC §4.4),
auditenv.env.AuditEnv, the CLI `python -m auditenv.run`, auditenv.runner and the transform named in
SPEC §7. The generator code itself is not read. Everything is written under runs/rehearsal/.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = ROOT / "qa" / "rehearsal"
CONFIGS = HERE / "configs.json"
OUT = ROOT / "runs" / "rehearsal"
POLICIES = ("random", "reconcile", "flag_all", "oracle")
POLICY_SEED = 9               # the random explorer's fixed seed (case §06)


def plan():
    """[{"seed", "group", "config"}] from configs.json, in seed order."""
    with open(CONFIGS, encoding="utf-8") as f:
        return sorted(json.load(f)["episodes"], key=lambda e: e["seed"])


def episode_dir(seed, root=None):
    return Path(root or OUT / "episodes") / f"ep_{seed}"


def build(seed, config, directory):
    """Generate one episode on disk the way reset does: normalize, then generate_episode."""
    from auditenv.config import normalize_config
    from auditenv.generate import generate_episode
    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    generate_episode(seed, normalize_config(config), str(directory))
    return directory


def file_digests(directory, parts=("public", "private")):
    """{relative path: sha256} of every file under the given parts of an episode."""
    directory = Path(directory)
    out = {}
    for part in parts:
        for path in sorted((directory / part).rglob("*")):
            if path.is_file():
                out[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def tree_digest(digests):
    text = "".join(f"{name} {digest}\n" for name, digest in sorted(digests.items()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def truth_of(directory):
    with open(Path(directory) / "private" / "truth.json", encoding="utf-8") as f:
        return json.load(f)


def oracle_findings(directory):
    """The full right answer straight from truth: what the grader-side oracle should submit."""
    truth = truth_of(directory)
    with open(Path(directory) / "public" / "manifest.json", encoding="utf-8") as f:
        episode_id = json.load(f)["episode_id"]
    findings = [{"type": i["type"], "affected_ids": list(i["affected_ids"]),
                 "claimed_amount_minor": i["expected_amount_minor"], "currency": i["currency"],
                 "evidence_ids": list(i["evidence_sets"][0]), "explanation": "rehearsal"}
                for i in truth["issues"]]
    return {"episode_id": episode_id, "findings": findings}


def child_env(hashseed=None, extra=None):
    """A clean environment for a child Python: no inherited PYTHON*, the given PYTHONHASHSEED."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    if hashseed is not None:
        env["PYTHONHASHSEED"] = str(hashseed)
    env.update(extra or {})
    return env


def run_child(code, hashseed=None, timeout=600):
    """Run `code` in `python -P -s` (CLAUDE.md: not -I, it drops PYTHONHASHSEED) with ROOT on sys.path."""
    prelude = f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
    return subprocess.run([sys.executable, "-P", "-s", "-c", prelude + code], cwd=ROOT, env=child_env(hashseed),
                          capture_output=True, text=True, timeout=timeout)


def run_cli(args, timeout=300):
    """`python3 -m auditenv.run ...` from the repo root, as the README tells the jury. (summary dict, proc)."""
    proc = subprocess.run([sys.executable, "-m", "auditenv.run", *args], cwd=ROOT, env=child_env(),
                          capture_output=True, text=True, timeout=timeout)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    summary = json.loads(lines[-1]) if proc.returncode == 0 and lines else None
    return summary, proc


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def emit(row, sink=None):
    line = json.dumps(row, sort_keys=True, ensure_ascii=False)
    print(line, flush=True)
    if sink is not None:
        sink.write(line + "\n")
        sink.flush()
