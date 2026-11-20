"""Run one baseline policy on one episode and report the grade (case §06, §09).

    python -m auditenv.run --policy {random,reconcile,flag_all,oracle} (--episode-dir DIR | --config JSON)
                           [--mode train|eval] [--seed N] [--policy-seed N] [--time-limit S]
                           [--trajectories PATH] [--grader-log PATH]

random, reconcile and flag_all run as separate processes (`python3 -m policies.<name> <seed>`) and see only
JSON observations. oracle is grader-side (case §06) and runs in-process with truth from the env.
--seed goes to env.reset only; the policy gets its own --policy-seed, because the agent must never
see the episode seed (case §03).
The printed summary line is the evaluator's own report: it is never shown to the policy, so it
carries the grade in eval mode too. --trajectories and --grader-log append one JSON line each.
"""
import argparse
import json
import os
import sys

from .env import MODES, AuditEnv
from .private_args import hide_private_args
from .runner import SANDBOX_MODES, TIME_LIMIT_S, encode_line, run_inprocess_episode, run_subprocess_episode

SUBPROCESS_POLICIES = {"random": "policies.random_policy", "reconcile": "policies.reconcile", "flag_all": "policies.flag_all"}
POLICIES = (*SUBPROCESS_POLICIES, "oracle")
REPORT_KEYS = ("tp", "fp", "fn", "precision", "recall", "f1", "reward")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m auditenv.run", description=__doc__.split("\n\n")[0],
                                     allow_abbrev=False)  # see auditenv.private_args
    parser.add_argument("--policy", required=True, choices=POLICIES)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode-dir", help="a prepared episode in the case §03 layout")
    source.add_argument("--config", help="generator config v1 (SPEC §4.4): a JSON file or inline JSON; uses --seed")
    parser.add_argument("--mode", default="eval", choices=MODES)
    parser.add_argument("--seed", type=int, default=0, help="episode seed for env.reset, never shown to the policy")
    parser.add_argument("--policy-seed", type=int, default=0, help="seed argument of the policy process")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_S)
    parser.add_argument("--trajectories")
    parser.add_argument("--grader-log")
    parser.add_argument("--sandbox", default="auto", choices=SANDBOX_MODES,
                        help="drop the policy process to an unprivileged user (needs root, see Dockerfile)")
    return parser.parse_args(argv)


def run_policy(policy, env, seed, config, policy_seed=0, time_limit=TIME_LIMIT_S, trajectory=None, sandbox="auto"):
    """Runner result for one named baseline policy."""
    if policy == "oracle":
        from policies.oracle import make_env_oracle  # grader-side, in this process only
        return run_inprocess_episode(make_env_oracle(env), env, seed, config, trajectory=trajectory)
    cmd = [sys.executable, "-m", SUBPROCESS_POLICIES[policy], str(policy_seed)]
    return run_subprocess_episode(cmd, env, seed, config, time_limit=time_limit, trajectory=trajectory, sandbox=sandbox)


def summary(args, result, entry):
    report = {"policy": args.policy, "mode": args.mode, "seed": args.seed, "policy_seed": args.policy_seed, **result}
    report.update({key: entry.get(key) for key in REPORT_KEYS})
    report["inspection_steps"] = entry.get("inspection_steps")
    if "error" in entry:
        report["error"] = entry["error"]
    return report


def load_config(text):
    """A generator config from inline JSON or a JSON file path. ValueError if it is not a JSON object."""
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
    env = AuditEnv(args.mode)
    try:
        config = {"episode_dir": args.episode_dir} if args.episode_dir else load_config(args.config)
    except ValueError as exc:
        print(f"auditenv.run: {exc}", file=sys.stderr)
        return 2
    trajectory = open(args.trajectories, "a", encoding="utf-8", newline="\n") if args.trajectories else None
    try:
        result = run_policy(args.policy, env, args.seed, config, args.policy_seed, args.time_limit, trajectory,
                            args.sandbox)
    except (ValueError, RuntimeError) as exc:  # broken config or episode directory, or no sandbox when required
        print(f"auditenv.run: {exc}", file=sys.stderr)
        return 2
    finally:
        if trajectory is not None:
            trajectory.close()
    entry = env.grader_log[-1]
    if args.grader_log:
        with open(args.grader_log, "a", encoding="utf-8", newline="\n") as f:
            f.write(encode_line(entry))
    sys.stdout.write(encode_line(summary(args, result, entry)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
