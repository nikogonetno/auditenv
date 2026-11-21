"""Jury rehearsal, part A items 6 and 7: limits, leaks and isolation (docs/tasks/V_rehearsal.md).

    python3 qa/rehearsal/probes.py [--steps limits,leaks] [--quick]

limits  qa/rehearsal/probe_policy.py through auditenv.runner.run_subprocess_episode: 100 steps with the
        submit, 60 seconds per episode (a policy that sleeps; --quick uses 3 s), broken findings give -1
        and end the episode, and the case §05 limits 30 findings, 10 evidence ids, 1000 characters
leaks   the hostile probe policy: files, /proc, environment, runner, broken lines, foreign cursors,
        private table names, path-like ids. What the policy saw must hold no truth, provenance, seed
        or path. Without root the policy runs as the same user, so file and /proc reads succeed here
        by design; tests/test_rehearsal_sandbox.py repeats the probe as auditpolicy in the Docker step.

Needs the episodes of `python3 qa/rehearsal/part_a.py --steps build`. Writes runs/rehearsal/probes.json.
"""
import argparse
import io
import json
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, OUT, ROOT, emit, episode_dir, plan, truth_of, write_json  # noqa: E402

PROBE = HERE / "probe_policy.py"
DIRTY_SEED, CLEAN_SEED = 9007, 9002         # C1 + C2 + fx, 5 issues; clean, no twins
ERROR_CODE = re.compile(r"[a-z][a-z_]{0,39}")
PRIVATE = re.compile(r"truth|provenance|inject|twin|scenario|issue_id|evidence_sets|\bseed\b", re.IGNORECASE)
ECHO = ("../private", "1 OR 1=1", "__import__", "/etc/passwd", "probe_report", "PROBE")


def run_probe(mode, config, env_mode="eval", time_limit=None, sandbox="auto", seed=None):
    """(runner result, grader log, trajectory lines) of one probe policy episode."""
    from auditenv.env import AuditEnv
    from auditenv.runner import TIME_LIMIT_S, run_subprocess_episode
    env, trajectory = AuditEnv(env_mode), io.StringIO()
    result = run_subprocess_episode([sys.executable, str(PROBE), mode], env, seed, config,
                                    time_limit=TIME_LIMIT_S if time_limit is None else time_limit,
                                    trajectory=trajectory, sandbox=sandbox)
    return result, list(env.grader_log), [json.loads(line) for line in trajectory.getvalue().splitlines()]


def dir_config(seed):
    return {"episode_dir": str(episode_dir(seed))}


def last(lines):
    return {k: lines[-1][k] for k in ("observation", "reward", "terminated", "truncated")}


# ------------------------------------------------------------------------------------- item 6
def limit_cases(quick):
    """[(name, mode, seed, env_mode, time_limit, check(result, log, lines) -> problems)]."""
    def ends(terminated, truncated, steps):
        def check(result, log, lines):
            p = []
            if (result["terminated"], result["truncated"]) != (terminated, truncated):
                p.append(f"terminated/truncated {result['terminated']}/{result['truncated']}")
            if steps is not None and result["steps"] != steps:
                p.append(f"steps {result['steps']}, want {steps}")
            if len(log) != 1:
                p.append(f"{len(log)} grader entries")
            return p
        return check

    def reward(want, inspection=None, error=False):
        def check(result, log, lines):
            p = []
            entry = log[-1] if log else {}
            if entry.get("reward") != want:
                p.append(f"reward {entry.get('reward')}, want {want}")
            if inspection is not None and entry.get("inspection_steps") != inspection:
                p.append(f"inspection_steps {entry.get('inspection_steps')}, want {inspection}")
            if bool(entry.get("error")) != error:
                p.append(f"structure error {entry.get('error')!r}, want {'one' if error else 'none'}")
            if lines and lines[-1]["reward"] != 0:
                p.append("eval: the policy saw a nonzero reward")
            return p
        return check

    both = lambda *checks: (lambda r, g, t: [x for c in checks for x in c(r, g, t)])
    sleep_limit = 3.0 if quick else None
    cases = [
        ("100 steps, no submit, book with errors", "steps", DIRTY_SEED, "eval", None,
         both(ends(False, True, 100), reward(-1.0, 100))),
        ("100 steps, no submit, clean book: empty answer minus 100 steps", "steps", CLEAN_SEED, "eval", None,
         both(ends(False, True, 100), reward(0.8, 100))),
        ("submit as step 100 counts", "submit_at:100", CLEAN_SEED, "eval", None,
         both(ends(True, False, 100), reward(0.802, 99))),
        ("submit as step 101 is too late", "submit_at:101", CLEAN_SEED, "eval", None,
         both(ends(False, True, 100), reward(0.8, 100))),
        ("policy that sleeps: time limit", "sleep", CLEAN_SEED, "eval", sleep_limit,
         both(ends(False, True, 0), reward(0.8, 100))),
        ("policy closes stdout", "close", CLEAN_SEED, "eval", None, both(ends(False, True, 0), reward(0.8, 100))),
        ("second submit is refused", "repeat_submit", CLEAN_SEED, "eval", None,
         both(ends(True, False, None), reward(1.0, 0))),
    ]
    from probe_policy import BAD, EDGE
    for case in BAD:
        cases.append((f"broken findings: {case}", f"bad:{case}", DIRTY_SEED, "eval", None,
                      both(ends(True, False, 1), reward(-1.0, 0, error=True))))
    for case in EDGE:
        want = 1.0 if case == "empty" else -1.0
        cases.append((f"at the limit: {case}", f"edge:{case}", CLEAN_SEED, "eval", None,
                      both(ends(True, False, 1), reward(want, 0, error=False))))
    cases.append(("broken findings in train: the policy sees -1", "bad:findings_31", DIRTY_SEED, "train", None,
                  lambda r, g, t: [] if t[-1]["reward"] == -1.0 and t[-1]["terminated"] else
                  [f"train: last reward {t[-1]['reward']}"]))
    return cases


def step_limits(quick):
    from auditenv.run import parse_args
    from auditenv.runner import TIME_LIMIT_S
    rows = []
    defaults = {"runner TIME_LIMIT_S": TIME_LIMIT_S,
                "CLI --time-limit default": parse_args(["--policy", "oracle", "--episode-dir", "x"]).time_limit}
    row = {"step": "limits", "case": "60 s is the default limit", "ok": set(defaults.values()) == {60.0},
           "detail": defaults}
    rows.append(row)
    emit(row)
    submit_views = {}
    results = {}

    def run(case):
        name, mode, seed, env_mode, time_limit, check = case
        try:
            result, log, lines = run_probe(mode, dir_config(seed), env_mode, time_limit)
            results[name] = (check(result, log, lines), result, log, lines)
        except Exception as exc:  # report, go on
            results[name] = ([f"{type(exc).__name__}: {exc}"], None, [], [])

    cases = limit_cases(quick)
    sleeper = [c for c in cases if c[1] == "sleep"]
    thread = threading.Thread(target=run, args=(sleeper[0],))
    thread.start()                       # the 60 s sleeper runs beside the others
    for case in cases:
        if case[1] != "sleep":
            run(case)
    thread.join()
    for name, mode, seed, env_mode, time_limit, _ in cases:
        problems, result, log, lines = results[name]
        entry = log[-1] if log else {}
        row = {"step": "limits", "case": name, "mode": mode, "seed": seed, "ok": not problems, "problems": problems,
               "steps": result and result["steps"], "wall_time_s": result and result["wall_time_s"],
               "reward": entry.get("reward"), "error": entry.get("error")}
        rows.append(row)
        emit(row)
        if env_mode == "eval" and (mode.startswith("bad:") or mode.startswith("edge:")) and lines:
            submit_views[mode] = json.dumps(last(lines), sort_keys=True)
    # E-25: in eval the policy sees the same thing for any answer, valid or broken.
    row = {"step": "limits", "case": "eval submit looks the same for every answer",
           "ok": len(set(submit_views.values())) == 1, "detail": sorted(set(submit_views.values()))[:3]}
    rows.append(row)
    emit(row)
    return rows


# ------------------------------------------------------------------------------------- item 7
def observation_problems(lines, truth, seed):
    """What the policy saw must hold no truth, private words, seed, paths or an echo of its input."""
    problems = set()
    issue_ids = {i["issue_id"] for i in truth["issues"]}
    errors, not_found = set(), set()
    for line in lines:
        obs = line["observation"]
        text = json.dumps(obs, ensure_ascii=False)
        if line["reward"] != 0:
            problems.add("nonzero reward")
        for iid in issue_ids:
            if re.search(rf"(?<![A-Z0-9]){re.escape(iid)}(?![A-Z0-9])", text):
                problems.add(f"issue id {iid} in an observation")
        for word in PRIVATE.findall(text):
            problems.add(f"private word {word.lower()!r} in an observation")
        for marker in (str(ROOT), "/tmp", "/app", "private/", "truth.json", *ECHO):
            if marker in text:
                problems.add(f"{marker!r} in an observation")
        if isinstance(obs, dict) and "error" in obs:
            code = obs["error"]
            errors.add(code)
            if type(code) is not str or not ERROR_CODE.fullmatch(code):
                problems.add(f"error is not a short code: {str(code)[:60]!r}")
            if set(obs) - {"error", "steps_left"}:
                problems.add(f"error observation carries more: {sorted(obs)}")
            if code == "not_found":
                not_found.add(json.dumps({k: v for k, v in obs.items() if k != "steps_left"}, sort_keys=True))
        if len(text) > 200_000:
            problems.add("an observation over 200 KB")
    if len(not_found) > 1:
        problems.add(f"not_found differs: {sorted(not_found)}")
    return sorted(problems), sorted(errors)


def probe_report(lines):
    report = seen = None
    for line in lines:
        action = line.get("action") or {}
        if action.get("action") == "probe_report":
            report = action.get("report")
        elif action.get("action") == "probe_seen":
            seen = action
    return report, seen


def reach(report, seed):
    """What the hostile policy could reach: [(what, detail)], empty when isolation holds."""
    if report is None:
        return [("no probe report", "the policy did not get to report")]
    out = []
    for r in report.get("file_reads", []):
        if r["result"] == "readable" and r.get("looks_private"):
            out.append(("private file", r["path"]))
    for key in ("parent_environ", "parent_cmdline"):
        r = report.get(key) or {}
        if r.get("result") == "readable" and key == "parent_environ":
            out.append(("runner environment", "readable"))
        if r.get("result") == "readable" and (r.get("has_seed_flag") or str(seed) in r.get("head", "")):
            out.append(("seed in " + key, r.get("head", "")[:80]))
    for fd in report.get("parent_fds", []):
        if fd.get("result") == "readable":
            out.append(("runner file", fd.get("target")))
    for r in report.get("cmdlines_with_seed_or_config", []):
        if "auditenv" in r.get("head", "") or "rehearsal" in r.get("head", ""):   # not other programs' --config
            out.append(("seed or config in a command line", r.get("head", "")[:80]))
    env = report.get("environ", {})
    for key, value in env.items():
        named = re.search("SEED|CONFIG|AUDITENV|EPISODE", key) and key != "PYTHONHASHSEED"  # str hashing, not the episode
        if named or (seed is not None and str(seed) in value):
            out.append(("environment", key))
    if report.get("write_repo") == "written":
        out.append(("write", "the repository is writable"))
    if report.get("signal_runner") == "allowed":
        out.append(("signal", "the runner can be signalled"))
    return out


def hostile_run(config, seed, truth, sandbox="auto"):
    result, log, lines = run_probe("hostile", config, "eval", sandbox=sandbox,
                                   seed=seed if "episode_dir" not in config else None)
    problems, errors = observation_problems(lines, truth, seed)
    report, seen = probe_report(lines)
    if seen is None:
        problems.append("the probe did not reach its last stage")
    else:
        if seen.get("infos") != ["{}"]:
            problems.append(f"info the policy saw: {seen.get('infos')[:3]}")
        if seen.get("rewards") not in (["0"], ["0.0"], ["0", "0.0"]):
            problems.append(f"rewards the policy saw: {seen.get('rewards')}")
    entry = log[-1] if log else {}
    if entry.get("reward") != -1.0 or not entry.get("error"):
        problems.append(f"broken submit graded {entry.get('reward')} with error {entry.get('error')!r}")
    if not result["terminated"]:
        problems.append("the episode did not end on the submit")
    return {"problems": problems, "errors": errors, "reach": reach(report, seed), "steps": result["steps"],
            "uid": report and report.get("uid")}


def step_leaks():
    rows = []
    entry = {e["seed"]: e for e in plan()}[DIRTY_SEED]
    truth = truth_of(episode_dir(DIRTY_SEED))
    for name, config, seed in (("prepared episode dir", dir_config(DIRTY_SEED), DIRTY_SEED),
                               ("generated in reset from seed and config", entry["config"], DIRTY_SEED)):
        out = hostile_run(config, seed, truth)
        sandboxed = out["uid"] not in (None, 0) and out["uid"] != __import__("os").getuid()
        row = {"step": "leaks", "case": name, "ok": not out["problems"], "problems": out["problems"],
               "error_codes": out["errors"], "steps": out["steps"], "sandboxed": sandboxed,
               "reachable_without_sandbox": [f"{a}: {b}" for a, b in out["reach"]]}
        rows.append(row)
        emit(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", default="limits,leaks")
    parser.add_argument("--quick", action="store_true", help="3 s instead of the real 60 s time limit")
    args = parser.parse_args(argv)
    result, failed = {}, False
    for step in args.steps.split(","):
        rows = step_limits(args.quick) if step == "limits" else step_leaks() if step == "leaks" else None
        if rows is None:
            parser.error(f"unknown step {step}")
        summary = {"step": step, "passed": sum(r["ok"] for r in rows), "total": len(rows)}
        result[step] = {"summary": summary, "rows": rows}
        emit({"summary": summary})
        failed = failed or summary["passed"] != summary["total"]
    write_json(OUT / "probes.json", result)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
