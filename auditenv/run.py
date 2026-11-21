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
import os
import shlex
import shutil
import signal
import stat
import sys
from fractions import Fraction

from . import __version__
from .config import config_sha256, normalize_config
from .env import MODES, AuditEnv
from .grader import f1_score
from .private_args import hide_private_args
from .runner import (SANDBOX_MODES, TIME_LIMIT_S, encode_line, remove_tree, run_inprocess_episode,
                     run_subprocess_episode, sandbox_user)

SUBPROCESS_POLICIES = {"random": "policies.random_policy", "reconcile": "policies.reconcile", "flag_all": "policies.flag_all"}
POLICIES = (*SUBPROCESS_POLICIES, "oracle")
REPORT_KEYS = ("tp", "fp", "fn", "precision", "recall", "f1", "reward")
SET_KEYS = {"version", "split", "comment", "episodes"}   # an episode set, as in configs/*_v2.json; none is a
                                                         # generator config key (auditenv.config.DEFAULTS)
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
    """[(seed, config)] of an episode set, or None when data is one generator config. ValueError if broken.

    Every item's config goes through normalize_config here, before any episode runs, so a bad item stops the
    run with its number and nothing written (QA M5); main() also builds the items it will run once, for what
    only the generator rejects. version is required, split and comment are optional. Any set key makes the
    object a set, so a set without episodes is not mistaken for one generator config (QA L13).
    """
    if "configs" in data and type(data["configs"]) is list:
        raise ValueError("this is a list of configs for scripts/stress.py, not an episode set or a generator config")
    if not SET_KEYS & set(data):
        return None
    if "episodes" not in data:
        raise ValueError("an episode set needs a non-empty list episodes")
    unknown = sorted(set(data) - SET_KEYS)
    if unknown:
        raise ValueError(f"unknown episode set key: {unknown[0]}")
    if type(data.get("version")) is not str or not data["version"]:
        raise ValueError("an episode set needs a non-empty string version")
    for key in ("split", "comment"):
        if key in data and type(data[key]) is not str:
            raise ValueError(f"episode set key {key} must be a string")
    items = data["episodes"]
    if type(items) is not list or not items:
        raise ValueError("an episode set needs a non-empty list episodes")
    runs = []
    for n, item in enumerate(items, 1):
        seed = item.get("seed") if type(item) is dict else None
        where = f"episode set item {n} (seed {seed})" if type(seed) is int else f"episode set item {n}"
        if (type(item) is not dict or set(item) != {"seed", "config"} or type(seed) is not int or seed < 0
                or type(item["config"]) is not dict):
            raise ValueError(f"{where}: must be {{\"seed\": int >= 0, \"config\": object}}")
        try:
            normalize_config(item["config"])
        except ValueError as exc:
            raise ValueError(f"episode set item {n} (seed {item['seed']}): {exc}") from None
        runs.append((item["seed"], item["config"]))
    first = {}
    for n, (seed, _) in enumerate(runs, 1):
        if seed in first:
            raise ValueError(f"episode set items {first[seed]} and {n} repeat seed {seed}")
        first[seed] = n
    return runs


def build_check(runs, info):
    """Build every episode of a set run once, before any policy runs and any log is written, so a config only
    the generator rejects (a book too small for its errors, a C1 that does not fit) stops the run at once with
    its item number (QA M5). ValueError naming the item."""
    import tempfile                                    # only the set path needs it
    from .generate import generate_episode
    root = tempfile.mkdtemp(prefix="auditenv-set-check-")
    try:
        for seed, config in runs:
            try:
                generate_episode(seed, config, os.path.join(root, str(seed)))
            except (ValueError, RuntimeError) as exc:  # a config error, or WorldError of the generator
                raise ValueError(f"episode set item {info['item'][seed]} (seed {seed}): {exc}") from None
    finally:
        remove_tree(root)                              # built episodes with private/: never half removed


def plan_runs(args):
    """([(seed, config)] to run in order, the set's {version, split, item number by seed} or None)."""
    if args.episode_dir:
        return [(0 if args.seed is None else args.seed, {"episode_dir": args.episode_dir})], None
    data = load_config(args.config)
    runs = episode_set(data)
    if runs is None:
        return [(0 if args.seed is None else args.seed, data)], None
    info = {"version": data.get("version"), "split": data.get("split"),
            "item": {seed: n for n, (seed, _) in enumerate(runs, 1)}}       # for error messages
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


def broken_submit_counts(listed, issues):
    """(tp, fp, fn) of a findings.json with a broken structure (case §05: reward -1, nothing matched). Its
    `listed` findings are false positives, at least one, so a broken empty answer on a clean company never
    counts as the perfect empty answer (QA H1); every truth issue is missed. scripts/metrics.py uses the same
    rule. The count comes from the grader entry, so an oversize submit recorded as a digest counts in full."""
    return 0, max(listed, 1), len(issues)


def episode_counts(env, trajectory_text):
    """(tp, fp, fn) of one episode, a broken submit counted by broken_submit_counts."""
    entry = env.grader_log[-1]
    if "tp" in entry:
        return entry["tp"], entry["fp"], entry["fn"]
    listed = entry.get("submitted_findings", len(submitted_findings(trajectory_text)))
    return broken_submit_counts(listed, env._truth.get("issues") or [])  # grader side only


def set_summary(info, reports, counts):
    """The last line of a set run: micro P, R and F1 over the episodes, mean reward and mean steps."""
    tp, fp, fn = (sum(c[k] for c in counts) for k in range(3))
    return {"episode_set": info["version"], "split": info["split"], "episodes": len(reports),
            "code_version": __version__, "invalid": sum(r.get("tp") is None for r in reports),
            # Episodes that ended without a submit: time limit, 100 steps or a policy that closed its output
            # (QA L16). They are graded as an empty answer, so on a clean company they still score.
            "truncated": sum(bool(r.get("truncated")) for r in reports),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": float(Fraction(tp, tp + fp)) if tp + fp else 0.0,
            "recall": float(Fraction(tp, tp + fn)) if tp + fn else 0.0,
            "f1": float(f1_score(tp, fp, fn)),
            # Rewards are exact fractions turned into floats by the grader; the mean is taken exactly (QA L14).
            "mean_reward": float(sum(Fraction(r["reward"]).limit_denominator(10 ** 7) for r in reports) / len(reports)),
            "mean_steps": float(Fraction(sum(r["steps"] for r in reports), len(reports)))}


def hand_back(paths):
    """Give log files to the owner of their folder when the runner is root (the Docker image), so the jury
    reads trajectories and the grader log without sudo (QA L15). They stay 0600; never to the sandbox user."""
    if os.name != "posix" or os.geteuid() != 0:
        return
    sandbox = sandbox_user()
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        folder = os.stat(os.path.dirname(os.path.abspath(path)))
        if folder.st_uid != 0 and (sandbox is None or folder.st_uid != sandbox[0]):
            os.chown(path, folder.st_uid, folder.st_gid)


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


def _stop(signum, frame):
    """SIGTERM or SIGHUP: leave through every finally block, as Ctrl-C does, so the policy process group is
    killed and the temporary folders (the policy's, the set check's) are removed (QA L17)."""
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_IGN)   # the cleanup itself must not be cut short
    try:  # raw write: a closed stderr (a hung-up terminal) must not turn the exit code into 120
        os.write(2, f"auditenv.run: stopped by signal {signum}\n".encode())
    except OSError:
        pass
    raise SystemExit(128 + signum)


def open_log(path):
    """A log file opened for appending. A regular file gets mode 0600, also when it existed before, if the
    runner may change it; a terminal, /dev/null or a FIFO keeps its mode."""
    f = open(path, "a", encoding="utf-8", newline="\n")
    try:
        if stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            os.fchmod(f.fileno(), 0o600)
    except OSError:  # someone else's file: appending is allowed, changing its mode is not
        pass
    return f


def log_folder_problem(args):
    """A message when a log path cannot be written, checked before any episode runs (QA L18)."""
    for flag, path in (("--trajectories", args.trajectories), ("--grader-log", args.grader_log)):
        if path:
            folder = os.path.dirname(os.path.abspath(path))
            if not os.path.isdir(folder):
                return f"{flag}: folder {folder} does not exist"
            if os.path.isdir(path):
                return f"{flag}: {path} is a folder"
    return None


def main(argv=None):
    # The seed and config never stay in this process's command line (QA N-1).
    args = parse_args(hide_private_args() if argv is None else argv)
    os.umask(0o077)  # trajectories and the grader log stay unreadable for the sandbox user
    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        # A signal ignored at start stays ignored, as Python does for SIGINT: `nohup` keeps working.
        if sig is not None and signal.getsignal(sig) is not signal.SIG_IGN:
            signal.signal(sig, _stop)
    try:
        runs, info = plan_runs(args)
        policy_cmd = policy_command(args.policy_cmd) if args.policy_cmd is not None else None
    except ValueError as exc:
        print(f"auditenv.run: {exc}", file=sys.stderr)
        return 2
    problem = log_folder_problem(args)
    if problem:
        print(f"auditenv.run: {problem}", file=sys.stderr)
        return 2
    subprocess_policy = args.policy != "oracle"
    if subprocess_policy and args.sandbox == "required" and sandbox_user() is None:
        # One message, before any warning (QA L3).
        print("auditenv.run: sandbox unavailable: run as root where the user auditpolicy exists (the Docker image)",
              file=sys.stderr)
        return 2
    if info is not None:
        try:
            build_check(runs, info)
        except ValueError as exc:
            print(f"auditenv.run: {exc}", file=sys.stderr)
            return 2
    if args.mode == "eval" and subprocess_policy and (args.sandbox == "off" or sandbox_user() is None):
        print(NO_SANDBOX_WARNING, file=sys.stderr, flush=True)  # warn only: our own scripts run like this (QA N-3)
    trajectory = grader = None
    try:
        trajectory = open_log(args.trajectories) if args.trajectories else None
        grader = open_log(args.grader_log) if args.grader_log else None
    except OSError as exc:  # e.g. no permission to write there
        print(f"auditenv.run: cannot open {exc.filename}: {exc.strerror}", file=sys.stderr)
        for f in (trajectory, grader):
            if f is not None:
                f.close()
        return 2
    reports, counts = [], []
    try:
        for seed, config in runs:
            where = f"episode set item {info['item'][seed]} (seed {seed}): " if info else ""
            env, text = AuditEnv(args.mode), io.StringIO()
            try:
                result = run_policy(args.policy, env, seed, config, args.policy_seed, args.time_limit, text,
                                    args.sandbox, policy_cmd)
            except (ValueError, RuntimeError) as exc:  # broken config or episode directory, or no sandbox
                print(f"auditenv.run: {where}{exc}", file=sys.stderr)
                return 2
            except OSError as exc:  # the policy command failed to start, e.g. the sandbox user cannot run it
                print(f"auditenv.run: {where}cannot start the policy: {exc.strerror}: {exc.filename}",
                      file=sys.stderr)
                return 2
            if trajectory is not None:
                trajectory.write(text.getvalue())
                trajectory.flush()
            counts.append(episode_counts(env, text.getvalue()))
            entry = env.grader_log[-1]
            if grader is not None:
                grader.write(encode_line(entry))
                grader.flush()
            report = summary(args, seed, config, result, entry)
            if info is not None:
                report["episode_set"] = info["version"]
            reports.append(report)
            sys.stdout.write(encode_line(report))
            sys.stdout.flush()
    finally:
        for f in (trajectory, grader):
            if f is not None:
                f.close()
        hand_back([args.trajectories, args.grader_log])
    if info is not None and args.seed is None:
        sys.stdout.write(encode_line(set_summary(info, reports, counts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
