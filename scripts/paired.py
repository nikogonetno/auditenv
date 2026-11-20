"""Paired checks of case §10 «Воспроизводимость и обобщение» (docs/tasks/B_next.md item 8).

    python3 scripts/paired.py EPISODE_DIR [EPISODE_DIR ...] [--n 20] [--transform auditenv.transform:permute_and_rename]

An EPISODE_DIR may also be a split directory (episodes/dev): its episodes are taken in name order.

Two kinds of check alternate, over the given episodes in turn, until n checks are done:
  repeat     one baseline policy (cycling random, reconcile, flag_all, oracle) runs twice on the same
             episode with the same seed and policy seed, each time in a fresh AuditEnv: the trajectories
             must be byte-identical and the grader log entries (reward included) equal
  transform  the transform writes a permuted and renamed copy of the episode (salt = check number):
             the oracle gets 1.0 on the original and on the copy, the reconcile policy gets the same
             reward on both, the copy's truth has the same multiset of (type, expected_amount_minor,
             currency) as the original, and every id in either truth names a record of its episode

The default transform is auditenv.transform.permute_and_rename. Without a transform function every check is
a repeat check and the summary says the transform is skipped. Prints one JSON line per check and a summary line; exit 1 if any
check fails, 0 otherwise (a skip is not a failure), 2 on bad arguments.
"""
import argparse
import importlib
import io
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auditenv.env import AuditEnv  # noqa: E402
from auditenv.episode import load_episode  # noqa: E402
from auditenv.grader import normalize_id  # noqa: E402
from auditenv.run import run_policy  # noqa: E402

DEFAULT_TRANSFORM = "auditenv.transform:permute_and_rename"
REPEAT_POLICIES = ("random", "reconcile", "flag_all", "oracle")
MODE = "train"  # the terminal reward then lands in the trajectory too, so the byte comparison covers it
SKIPPED = "skipped: not available"


def load_transform(spec):
    """The function named by module:function, or None when that module or function does not exist (yet).

    A module that exists but fails to import for another reason raises: a broken transform must fail
    loudly, not turn into a skip.
    """
    module_name, _, func = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and (module_name == exc.name or module_name.startswith(exc.name + ".")):
            return None
        raise
    return getattr(module, func, None) if func else None


def plan(n, episodes, with_transform):
    """[(check number, kind, episode, policy)] for n checks."""
    rounds = math.lcm(len(episodes), len(REPEAT_POLICIES))
    checks = []
    for k in range(n):
        if with_transform:
            kind, j = ("repeat" if k % 2 == 0 else "transform"), k // 2
        else:
            kind, j = "repeat", k
        # The policy cycle shifts by one after each full round, so every episode meets every policy.
        policy = REPEAT_POLICIES[(j + j // rounds) % len(REPEAT_POLICIES)] if kind == "repeat" else "oracle,reconcile"
        checks.append((k + 1, kind, episodes[j % len(episodes)], policy))
    return checks


def run_once(runner, policy, episode_dir, seed):
    """(trajectory text, last grader log entry) of one run in a fresh env."""
    env, trajectory = AuditEnv(MODE), io.StringIO()
    runner(policy, env, seed, {"episode_dir": str(episode_dir)}, seed, trajectory=trajectory)
    return trajectory.getvalue(), (env.grader_log[-1] if env.grader_log else None)


def first_difference(a, b):
    lines_a, lines_b = a.splitlines(), b.splitlines()
    for i, (x, y) in enumerate(zip(lines_a, lines_b)):
        if x != y:
            return f"trajectories differ at line {i + 1} ({len(lines_a)} and {len(lines_b)} lines)"
    return f"trajectories differ in length ({len(lines_a)} and {len(lines_b)} lines)"


def repeat_check(runner, policy, episode_dir, seed):
    """(ok, detail) for two runs of one policy with the same seeds."""
    first, entry_a = run_once(runner, policy, episode_dir, seed)
    second, entry_b = run_once(runner, policy, episode_dir, seed)
    problems = []
    if first != second:
        problems.append(first_difference(first, second))
    if entry_a is None or entry_b is None:
        problems.append("no grader log entry")
    elif entry_a != entry_b:
        keys = sorted(k for k in entry_a.keys() | entry_b.keys() if entry_a.get(k) != entry_b.get(k))
        problems.append("grader entries differ in " + ", ".join(
            f"{k} ({entry_a.get(k)} vs {entry_b.get(k)})" for k in keys))
    if problems:
        return False, "; ".join(problems)
    return True, f"{len(first.splitlines())} identical trajectory lines, reward {entry_a['reward']}"


def truth_signature(episode_dir):
    with open(Path(episode_dir) / "private" / "truth.json", encoding="utf-8") as f:
        truth = json.load(f)
    return Counter((i["type"], i["expected_amount_minor"], i["currency"]) for i in truth.get("issues") or [])


def unknown_truth_ids(episode_dir):
    """Ids in truth (affected and evidence) that name no public record: truth left on the old ids.

    The oracle alone cannot see this, because it submits whatever truth says.
    """
    episode = load_episode(episode_dir)
    known = set(episode.documents).union(*episode.index.values())
    named = set()
    for issue in episode.load_truth().get("issues") or []:
        named.update(issue["affected_ids"])
        for evidence in issue.get("evidence_sets") or []:
            named.update(evidence)
    return sorted(ref for ref in named if normalize_id(ref) not in known)


def transform_check(runner, transform, episode_dir, salt, workdir):
    """(ok, detail) for the original episode and its permuted and renamed copy."""
    copy_dir = Path(workdir) / f"check_{salt}"
    transform(str(episode_dir), str(copy_dir), salt)
    problems = []
    original, copied = truth_signature(episode_dir), truth_signature(copy_dir)
    if original != copied:
        missing, extra = sorted((original - copied).elements()), sorted((copied - original).elements())
        problems.append(f"truth (type, amount, currency) differs: missing {missing}, extra {extra}")
    for name, directory in (("original", episode_dir), ("copy", copy_dir)):
        unknown = unknown_truth_ids(directory)
        if unknown:
            problems.append(f"truth of the {name} names ids with no record: {unknown[:5]}")
    rewards = {}
    for policy in ("oracle", "reconcile"):
        rewards[policy] = []
        for directory in (episode_dir, copy_dir):
            _, entry = run_once(runner, policy, directory, salt)
            rewards[policy].append(entry["reward"] if entry else None)
    if rewards["oracle"] != [1.0, 1.0]:
        problems.append(f"oracle reward {rewards['oracle'][0]} on the original, {rewards['oracle'][1]} on the copy")
    if rewards["reconcile"][0] != rewards["reconcile"][1]:
        problems.append(f"reconcile reward {rewards['reconcile'][0]} on the original, "
                        f"{rewards['reconcile'][1]} on the copy")
    if problems:
        return False, "; ".join(problems)
    return True, (f"salt {salt}, {sum(original.values())} issues, oracle 1.0 on both, "
                  f"reconcile {rewards['reconcile'][0]} on both")


def run_paired(episodes, n=20, transform=None, runner=run_policy, out=None):
    """Run n paired checks. Returns (rows, summary); writes each row and the summary as JSON lines to out.

    transform(episode_dir, out_dir, salt) or None to skip the transform checks.
    runner has the signature of auditenv.run.run_policy; tests inject fakes here.
    """
    rows = []
    with tempfile.TemporaryDirectory(prefix="auditenv-paired-") as workdir:
        for number, kind, episode_dir, policy in plan(n, list(episodes), transform is not None):
            try:
                if kind == "repeat":
                    ok, detail = repeat_check(runner, policy, episode_dir, number)
                else:
                    ok, detail = transform_check(runner, transform, episode_dir, number, workdir)
            except Exception as exc:  # report and go on: one broken check must not hide the others
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            row = {"check": number, "kind": kind, "episode": str(episode_dir), "policy": policy,
                   "ok": ok, "detail": detail}
            rows.append(row)
            if out is not None:
                out.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                out.flush()
    passed = sum(row["ok"] for row in rows)
    summary = {"paired": "OK" if passed == len(rows) else "FAIL", "passed": passed, "total": len(rows),
               "repeat": sum(row["kind"] == "repeat" for row in rows),
               "transform": sum(row["kind"] == "transform" for row in rows) if transform is not None else SKIPPED}
    if out is not None:
        out.write(json.dumps(summary, sort_keys=True, ensure_ascii=False) + "\n")
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("episodes", nargs="+", metavar="EPISODE_DIR")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--transform", default=DEFAULT_TRANSFORM, help="module:function, default A's transform")
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be at least 1")
    episodes = []
    for raw in args.episodes:
        path = Path(raw)
        if not path.is_dir():
            parser.error(f"not a directory: {raw}")
        # A split directory (episodes/dev) stands for its episodes in name order, as in scripts/metrics.py.
        found = [path] if (path / "public").is_dir() else sorted(
            (p for p in path.iterdir() if (p / "public").is_dir()), key=lambda p: p.name)
        if not found:
            parser.error(f"neither an episode nor a directory of episodes: {raw}")
        episodes += found
    _, summary = run_paired(episodes, args.n, load_transform(args.transform), out=sys.stdout)
    return 0 if summary["paired"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
