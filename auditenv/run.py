"""Run one policy on one episode and report the grade (case §06, §09).

    python -m auditenv.run (--policy {random,reconcile,flag_all,oracle} | --policy-cmd COMMAND)
                           (--episode-dir DIR | --config JSON) [--mode train|eval] [--seed N] [--policy-seed N]
                           [--time-limit S] [--sandbox auto|required|off] [--trajectories PATH] [--grader-log PATH]

random, reconcile and flag_all run as separate processes (`python3 -m policies.<name> <seed>`) and see only
JSON observations. --policy-cmd runs any external policy the same way, with the same limits, sandbox, grader
log and trajectories (protocol: README section 4); a relative path to an existing file in the command is
made absolute, because the policy starts in an empty working directory. oracle is grader-side (case §06)
and runs in-process with truth from the env.
--seed goes to env.reset only; the policy gets its own --policy-seed, because the agent must never
see the episode seed (case §03).
The printed summary line is the evaluator's own report: it is never shown to the policy, so it
carries the grade in eval mode too. --trajectories and --grader-log append one JSON line each.
"""
import argparse
import json
import os
import shlex
import shutil
import sys

from .env import MODES, AuditEnv
from .private_args import hide_private_args
from .runner import (SANDBOX_MODES, TIME_LIMIT_S, encode_line, run_inprocess_episode, run_subprocess_episode,
                     sandbox_user)

SUBPROCESS_POLICIES = {"random": "policies.random_policy", "reconcile": "policies.reconcile", "flag_all": "policies.flag_all"}
POLICIES = (*SUBPROCESS_POLICIES, "oracle")
REPORT_KEYS = ("tp", "fp", "fn", "precision", "recall", "f1", "reward")
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


def summary(args, result, entry):
    report = {"policy": args.policy or EXTERNAL, "mode": args.mode, "seed": args.seed, "policy_seed": args.policy_seed,
              **result}
    if args.policy_cmd is not None:
        report["policy_cmd"] = args.policy_cmd
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
        policy_cmd = policy_command(args.policy_cmd) if args.policy_cmd is not None else None
    except ValueError as exc:
        print(f"auditenv.run: {exc}", file=sys.stderr)
        return 2
    if args.mode == "eval" and args.policy != "oracle" and (args.sandbox == "off" or sandbox_user() is None):
        print(NO_SANDBOX_WARNING, file=sys.stderr, flush=True)  # warn only: our own scripts run like this (QA N-3)
    trajectory = open(args.trajectories, "a", encoding="utf-8", newline="\n") if args.trajectories else None
    try:
        result = run_policy(args.policy, env, args.seed, config, args.policy_seed, args.time_limit, trajectory,
                            args.sandbox, policy_cmd)
    except (ValueError, RuntimeError) as exc:  # broken config or episode directory, or no sandbox when required
        print(f"auditenv.run: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:  # the policy command failed to start, e.g. the sandbox user cannot run it
        print(f"auditenv.run: cannot start the policy: {exc.strerror}: {exc.filename}", file=sys.stderr)
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
