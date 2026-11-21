"""Run one policy on one episode, or on every episode of a set, and report the grade (case §06, §09).

    python -m auditenv.run (--policy {random,reconcile,flag_all,oracle} | --policy-cmd COMMAND)
                           (--episode-dir DIR | --config JSON) [--mode train|eval] [--seed N] [--policy-seed N]
                           [--time-limit S] [--sandbox auto|required|off] [--trajectories PATH] [--grader-log PATH]

--config takes inline JSON or a file with one generator config, or an episode set like configs/dev_v2.json,
{version, split, comment, episodes: [{seed, config}]}. With --seed the set gives that seed's config; without
it every episode of the set runs in turn, into the same trajectories and grader log, and a last line sums
them up. Only this CLI reads sets: env.reset and its strict config check take one config as before.

random, reconcile and flag_all run as separate processes (`python3 -m policies.<name> <seed>`) and see only
JSON observations. --policy-cmd runs any external policy the same way, with the same limits, sandbox, grader
log and trajectories (protocol: README section 4); a relative path to an existing file in the command is
made absolute, because the policy starts in an empty working directory. oracle is grader-side (case §06)
and runs in-process with truth from the env.
--seed goes to env.reset only; the policy gets its own --policy-seed, because the agent must never
see the episode seed (case §03).
The printed summary line is the evaluator's own report: it is never shown to the policy, so it
carries the grade in eval mode too, with the code version and config_sha256 of the episode.
--trajectories and --grader-log append to their files: the trajectory lines and one grader entry per episode.
"""
import argparse
import io
import json
import math
import os
import shlex
import shutil
import sys
from fractions import Fraction

from . import __version__
from .config import config_sha256, normalize_config
from .env import MODES, AuditEnv
from .grader import f1_score
from .private_args import hide_private_args
from .runner import (SANDBOX_MODES, TIME_LIMIT_S, encode_line, run_inprocess_episode, run_subprocess_episode,
                     sandbox_user)

SUBPROCESS_POLICIES = {"random": "policies.random_policy", "reconcile": "policies.reconcile", "flag_all": "policies.flag_all"}
POLICIES = (*SUBPROCESS_POLICIES, "oracle")
REPORT_KEYS = ("tp", "fp", "fn", "precision", "recall", "f1", "reward")
SET_KEYS = {"version", "split", "comment", "episodes"}   # an episode set, as in configs/*_v2.json
EXTERNAL = "external"     # the policy name in the report of a --policy-cmd run
NO_SANDBOX_WARNING = ("auditenv.run: WARNING: eval run without the sandbox: the policy runs as this user and "
                      "can read private/ (the truth); run the closed evaluation only in Docker, see README "
                      "\"Закрытый прогон жюри\"")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m auditenv.run", description=__doc__.split("\n\n")[0],
                                     allow_abbrev=False)  # see auditenv.private_args
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--policy", choices=POLICIES)
    who.add_argument("--policy-cmd", metavar="COMMAND",
                     help='an external policy process, e.g. "python3 my_policy.py" (protocol: README section 4)')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode-dir", help="a prepared episode in the case §03 layout")
    source.add_argument("--config", help="inline JSON or a file: one generator config (SPEC §4.4), or an episode set "
                                         "{version, split, episodes: [{seed, config}]} like configs/dev_v2.json")
    parser.add_argument("--mode", default="eval", choices=MODES)
    parser.add_argument("--seed", type=int, help="episode seed for env.reset, never shown to the policy (default 0; "
                                                "with an episode set: that episode, default all)")
    parser.add_argument("--policy-seed", type=int, default=0, help="seed argument of the policy process")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_S)
    parser.add_argument("--trajectories")
    parser.add_argument("--grader-log")
    parser.add_argument("--sandbox", default="auto", choices=SANDBOX_MODES,
                        help="drop the policy process to an unprivileged user (needs root, see Dockerfile)")
    return parser.parse_args(argv)


def policy_command(text):
    """argv of a --policy-cmd string. Relative paths to existing files become absolute (the policy starts in
    an empty working directory); the program itself is looked up in PATH unless it names a path."""
    argv = shlex.split(text)
    if not argv:
        raise ValueError("--policy-cmd is empty")
    argv = [os.path.abspath(arg) if (n or os.sep in arg) and not os.path.isabs(arg) and os.path.isfile(arg)
            else arg for n, arg in enumerate(argv)]
    # Checked before reset, so a typo gives a clean error and no half-written trajectory. The policy gets
    # the same PATH (auditenv.runner.policy_environment).
    program = argv[0]
    found = (os.path.isfile(program) and os.access(program, os.X_OK)) if os.sep in program else shutil.which(program)
    if not found:
        raise ValueError(f"cannot start the policy: {program}: not found or not executable")
    return argv


def run_policy(policy, env, seed, config, policy_seed=0, time_limit=TIME_LIMIT_S, trajectory=None, sandbox="auto",
               policy_cmd=None):
    """Runner result for one named baseline policy, or for the external command policy_cmd (an argv list)."""
    if policy == "oracle":
        from policies.oracle import make_env_oracle  # grader-side, in this process only
        return run_inprocess_episode(make_env_oracle(env), env, seed, config, trajectory=trajectory)
    if policy_cmd is None:
        policy_cmd = [sys.executable, "-m", SUBPROCESS_POLICIES[policy], str(policy_seed)]
    return run_subprocess_episode(policy_cmd, env, seed, config, time_limit=time_limit, trajectory=trajectory,
                                  sandbox=sandbox)


def summary(args, seed, config, result, entry):
    report = {"policy": args.policy or EXTERNAL, "mode": args.mode, "seed": seed, "policy_seed": args.policy_seed,
              "code_version": __version__, "config_sha256": episode_config_sha256(config), **result}
    if args.policy_cmd is not None:
        report["policy_cmd"] = args.policy_cmd
    report.update({key: entry.get(key) for key in REPORT_KEYS})
    report["inspection_steps"] = entry.get("inspection_steps")
    if "error" in entry:
        report["error"] = entry["error"]
    return report


def episode_config_sha256(config):
    """sha256 of the normalized generator config, as in provenance.json; for a prepared episode the one its
    provenance.json records, None when it has none."""
    if "episode_dir" not in config:
        return config_sha256(normalize_config(config))
    try:
        with open(os.path.join(config["episode_dir"], "private", "provenance.json"), encoding="utf-8") as f:
            value = json.load(f).get("config_sha256")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def episode_set(data):
    """[(seed, config)] of an episode set, or None when data is one generator config. ValueError if broken."""
    if "episodes" not in data:
        return None
    unknown = sorted(set(data) - SET_KEYS)
    if unknown:
        raise ValueError(f"unknown episode set key: {unknown[0]}")
    items = data["episodes"]
    if type(items) is not list or not items:
        raise ValueError("an episode set needs a non-empty list episodes")
    runs = []
    for item in items:
        if (type(item) is not dict or set(item) != {"seed", "config"} or type(item["seed"]) is not int
                or type(item["config"]) is not dict):
            raise ValueError("every item of episodes must be {\"seed\": int, \"config\": object}")
        runs.append((item["seed"], item["config"]))
    if len({seed for seed, _ in runs}) != len(runs):
        raise ValueError("an episode set repeats a seed")
    return runs


def plan_runs(args):
    """([(seed, config)] to run in order, the set's {version, split} or None)."""
    if args.episode_dir:
        return [(0 if args.seed is None else args.seed, {"episode_dir": args.episode_dir})], None
    data = load_config(args.config)
    runs = episode_set(data)
    if runs is None:
        return [(0 if args.seed is None else args.seed, data)], None
    info = {"version": data.get("version"), "split": data.get("split")}
    if args.seed is None:
        return runs, info
    chosen = [run for run in runs if run[0] == args.seed]
    if not chosen:
        raise ValueError(f"seed {args.seed} is not in the episode set; its seeds: "
                         + ", ".join(str(seed) for seed, _ in sorted(runs)))
    return chosen, info


def submitted_findings(trajectory_text):
    """Findings of the submit that ended the episode; [] after a truncation or an unreadable submit."""
    lines = trajectory_text.splitlines()
    action = json.loads(lines[-1]).get("action") if lines else None
    if not isinstance(action, dict) or action.get("action") != "submit":
        return []
    body = action.get("findings")
    findings = body.get("findings") if isinstance(body, dict) else None
    return findings if isinstance(findings, list) else []


def episode_counts(env, trajectory_text):
    """(tp, fp, fn) of one episode. A findings.json with a broken structure matches nothing: its findings are
    false positives and every truth issue is missed, as in scripts/metrics.py (case §05)."""
    entry = env.grader_log[-1]
    if "tp" in entry:
        return entry["tp"], entry["fp"], entry["fn"]
    return 0, len(submitted_findings(trajectory_text)), len(env._truth.get("issues") or [])  # grader side only


def set_summary(info, reports, counts):
    """The last line of a set run: micro P, R and F1 over the episodes, mean reward and mean steps."""
    tp, fp, fn = (sum(c[k] for c in counts) for k in range(3))
    return {"episode_set": info["version"], "split": info["split"], "episodes": len(reports),
            "code_version": __version__, "invalid": sum(r.get("tp") is None for r in reports),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": float(Fraction(tp, tp + fp)) if tp + fp else 0.0,
            "recall": float(Fraction(tp, tp + fn)) if tp + fn else 0.0,
            "f1": float(f1_score(tp, fp, fn)),
            "mean_reward": math.fsum(r["reward"] for r in reports) / len(reports),
            "mean_steps": sum(r["steps"] for r in reports) / len(reports)}


def load_config(text):
    """A generator config or an episode set, from inline JSON or a JSON file path. ValueError if it is not
    a JSON object."""
    if text.lstrip().startswith("{"):
        source = text
    else:
        try:
            with open(text, encoding="utf-8") as f:
                source = f.read()
        except OSError as exc:
            raise ValueError(f"cannot read config: {exc.strerror}") from None
    config = json.loads(source)  # JSONDecodeError is a ValueError
    if type(config) is not dict:
        raise ValueError("config must be a JSON object")
    return config


def main(argv=None):
    # The seed and config never stay in this process's command line (QA N-1).
    args = parse_args(hide_private_args() if argv is None else argv)
    os.umask(0o077)  # trajectories and the grader log stay unreadable for the sandbox user
    try:
        runs, info = plan_runs(args)
        policy_cmd = policy_command(args.policy_cmd) if args.policy_cmd is not None else None
    except ValueError as exc:
        print(f"auditenv.run: {exc}", file=sys.stderr)
        return 2
    subprocess_policy = args.policy != "oracle"
    if subprocess_policy and args.sandbox == "required" and sandbox_user() is None:
        # One message, before any warning (QA L3).
        print("auditenv.run: sandbox unavailable: run as root where the user auditpolicy exists (the Docker image)",
              file=sys.stderr)
        return 2
    if args.mode == "eval" and subprocess_policy and (args.sandbox == "off" or sandbox_user() is None):
        print(NO_SANDBOX_WARNING, file=sys.stderr, flush=True)  # warn only: our own scripts run like this (QA N-3)
    trajectory = open(args.trajectories, "a", encoding="utf-8", newline="\n") if args.trajectories else None
    reports, counts = [], []
    try:
        for seed, config in runs:
            env, text = AuditEnv(args.mode), io.StringIO()
            try:
                result = run_policy(args.policy, env, seed, config, args.policy_seed, args.time_limit, text,
                                    args.sandbox, policy_cmd)
            except (ValueError, RuntimeError) as exc:  # broken config or episode directory, or no sandbox
                print(f"auditenv.run: {exc}", file=sys.stderr)
                return 2
            except OSError as exc:  # the policy command failed to start, e.g. the sandbox user cannot run it
                print(f"auditenv.run: cannot start the policy: {exc.strerror}: {exc.filename}", file=sys.stderr)
                return 2
            if trajectory is not None:
                trajectory.write(text.getvalue())
                trajectory.flush()
            counts.append(episode_counts(env, text.getvalue()))
            entry = env.grader_log[-1]
            if args.grader_log:
                with open(args.grader_log, "a", encoding="utf-8", newline="\n") as f:
                    f.write(encode_line(entry))
            report = summary(args, seed, config, result, entry)
            if info is not None:
                report["episode_set"] = info["version"]
            reports.append(report)
            sys.stdout.write(encode_line(report))
            sys.stdout.flush()
    finally:
        if trajectory is not None:
            trajectory.close()
    if info is not None and args.seed is None:
        sys.stdout.write(encode_line(set_summary(info, reports, counts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
