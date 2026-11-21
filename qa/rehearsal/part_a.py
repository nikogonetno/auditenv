"""Jury rehearsal, part A items 1-5 (docs/tasks/V_rehearsal.md; case §06, §10).

    python3 qa/rehearsal/part_a.py [--steps build,policies,verify,rollback,paired,repro]

build     the 20 episodes of qa/rehearsal/configs.json into runs/rehearsal/episodes (item 1)
policies  random (fixed policy seed), reconcile, flag_all and oracle in eval mode through the CLI
          `python -m auditenv.run --config ... --seed ...`, as the jury runs hidden seeds (item 2)
verify    the independent verifier V, strict, on every episode: INV-18 and the rest (item 3)
rollback  the book before the errors (docs/tasks/V_rollback.md): provenance undone on a copy, strict V
          with an empty truth on it, for these 20, their copies after transform, train and dev
paired    10 repeat and 10 transform checks on these episodes (item 4)
repro     sha256 of the files under PYTHONHASHSEED 0, 1, random and random again (item 5)

Prints one JSON line per check and a summary line per step; the full result goes to
runs/rehearsal/part_a.json. Exit 1 if any acceptance condition fails.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (OUT, POLICIES, POLICY_SEED, ROOT, build, child_env, emit, episode_dir,  # noqa: E402
                    file_digests, oracle_findings, plan, run_child, run_cli, text_sha256, tree_digest,
                    truth_of, write_json)

STEPS = ("build", "policies", "verify", "rollback", "paired", "repro")
PRIVATE_WORDS = re.compile(r"truth|provenance|inject|twin|scenario|issue_id", re.IGNORECASE)


def config_file(entry):
    path = OUT / "configs" / f"{entry['seed']}.json"
    write_json(path, entry["config"])
    return path


def silence_reward(truth):
    """An empty answer with no inspection: F1 1 on a clean book, 0 otherwise (case §05)."""
    return 1.0 if not truth["issues"] else -1.0


# ------------------------------------------------------------------------------------- item 1
def step_build(entries):
    rows = []
    for e in entries:
        started = time.perf_counter()
        directory = build(e["seed"], e["config"], episode_dir(e["seed"]))
        manifest = json.loads((directory / "public" / "manifest.json").read_text(encoding="utf-8"))
        truth = truth_of(directory)
        digests = file_digests(directory)
        row = {"step": "build", "seed": e["seed"], "group": e["group"], "episode_id": manifest["episode_id"],
               "documents": manifest["documents"], "ledger_rows": manifest["counts"]["ledger"],
               "timezone": manifest["timezone"], "period": manifest["period_start"][:7],
               "issues": dict(sorted(Counter(i["type"] for i in truth["issues"]).items())),
               "clean": truth["clean"], "public_sha256": tree_digest({k: v for k, v in digests.items()
                                                                       if k.startswith("public/")}),
               "all_sha256": tree_digest(digests), "seconds": round(time.perf_counter() - started, 3)}
        # The group must be what the books say: clean books have no issues, coupled ones have some.
        row["ok"] = (row["clean"] == (e["group"] == "clean")) and (truth["clean"] == (not truth["issues"]))
        rows.append(row)
        emit(row)
    return rows


# ------------------------------------------------------------------------------------- item 2
def scan_observations(trajectory_text, truth, seed):
    """Leaks in what the policy saw: issue ids, private words, the seed as a value, paths."""
    issue_ids = {i["issue_id"] for i in truth["issues"]}
    found = set()
    for line in trajectory_text.splitlines():
        record = json.loads(line)
        text = json.dumps(record["observation"], ensure_ascii=False)
        if record["reward"] != 0:
            found.add("nonzero reward in eval")
        for iid in issue_ids:
            if re.search(rf"(?<![A-Z0-9]){re.escape(iid)}(?![A-Z0-9])", text):
                found.add(f"issue id {iid}")
        for word in PRIVATE_WORDS.findall(text):
            found.add(f"word {word.lower()}")
        if re.search(r"\bseed\b", text, re.IGNORECASE):
            found.add("word seed")
        for marker in (str(ROOT), "/tmp/", "runs/rehearsal", "private/"):
            if marker in text:
                found.add(f"path {marker}")
    return sorted(found)


def step_policies(entries):
    rows = []
    traj_dir = OUT / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    grader_log = OUT / "grader_log_eval.jsonl"
    grader_log.unlink(missing_ok=True)
    for e in entries:
        directory = episode_dir(e["seed"])
        truth = truth_of(directory)
        built_id = json.loads((directory / "public" / "manifest.json").read_text(encoding="utf-8"))["episode_id"]
        for policy in POLICIES:
            traj = traj_dir / f"{e['seed']}_{policy}_eval.jsonl"
            traj.unlink(missing_ok=True)
            summary, proc = run_cli(["--policy", policy, "--mode", "eval", "--config", str(config_file(e)),
                                     "--seed", str(e["seed"]), "--policy-seed", str(POLICY_SEED),
                                     "--trajectories", str(traj), "--grader-log", str(grader_log)])
            row = {"step": "policies", "seed": e["seed"], "group": e["group"], "policy": policy}
            if summary is None:
                row.update(ok=False, error=f"exit {proc.returncode}: {proc.stderr.strip()[-300:]}")
            else:
                text = traj.read_text(encoding="utf-8")
                row.update({k: summary.get(k) for k in ("tp", "fp", "fn", "precision", "recall", "f1", "reward",
                                                        "inspection_steps", "steps", "wall_time_s",
                                                        "terminated", "truncated", "seen_reward", "error")})
                row["silence"] = silence_reward(truth)
                row["same_episode_as_built"] = summary.get("episode_id") == built_id
                row["leaks"] = scan_observations(text, truth, e["seed"])
                row["ok"] = row["same_episode_as_built"] and not row["leaks"] and summary.get("seen_reward") == 0
            rows.append(row)
            emit(row)
    return rows


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def policy_acceptance(rows):
    by = {p: [r for r in rows if r["policy"] == p and "reward" in r] for p in POLICIES}
    means = {p: mean(r["reward"] for r in by[p]) for p in POLICIES}
    checks = {
        "oracle 1.0 on all 20": all(r["reward"] == 1.0 for r in by["oracle"]) and len(by["oracle"]) == 20,
        "flag_all not above silence on every episode": all(r["reward"] <= r["silence"] for r in by["flag_all"]),
        "flag_all -1 on every clean book": all(r["reward"] == -1.0 for r in by["flag_all"] if r["group"] == "clean"),
        "mean oracle > reconcile > random": (means["oracle"] is not None and means["reconcile"] is not None and
                                             means["random"] is not None and
                                             means["oracle"] > means["reconcile"] > means["random"]),
        "eval: policy saw reward 0 and nothing leaked": all(r.get("ok") for r in rows),
    }
    diagnostics = {p: {k: mean(r.get(k) for r in by[p]) for k in
                       ("precision", "recall", "f1", "reward", "inspection_steps", "steps", "wall_time_s")}
                   for p in POLICIES}
    return checks, means, diagnostics


# ------------------------------------------------------------------------------------- item 3
def step_verify(entries):
    from auditenv.verify import verify_episode
    rows = []
    for e in entries:
        problems = verify_episode(str(episode_dir(e["seed"])), strict_volume=True)
        row = {"step": "verify", "seed": e["seed"], "group": e["group"], "ok": not problems,
               "problems": problems[:20], "problem_count": len(problems)}
        rows.append(row)
        emit(row)
    return rows


# ------------------------------------------------------------------------------ V_rollback.md
def step_rollback(entries):
    from auditenv.transform import permute_and_rename
    from auditenv.verify.rollback import verify_rollback
    targets = [("rehearsal", e["seed"], episode_dir(e["seed"])) for e in entries]
    targets += [(d.parent.name, int(d.name), d) for d in sorted(ROOT.glob("episodes/*/[0-9]*")) if d.is_dir()]
    rows = []
    with tempfile.TemporaryDirectory(prefix="rehearsal-rollback-") as work:
        for e in entries:
            copy = Path(work) / f"t{e['seed']}"
            permute_and_rename(str(episode_dir(e["seed"])), str(copy), 800001 + e["seed"])
            targets.append(("transformed", e["seed"], copy))
        for kind, seed, directory in targets:
            problems = verify_rollback(str(directory), strict_volume=True)
            row = {"step": "rollback", "kind": kind, "seed": seed, "ok": not problems,
                   "problems": problems[:20], "problem_count": len(problems)}
            rows.append(row)
            emit(row)
    return rows


# ------------------------------------------------------------------------------------- item 4
def grader_entries(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def repeat_check(e, policy, work):
    """Same seed, config and actions twice: same files, same trajectory bytes, same grade."""
    problems = []
    first, second = (build(e["seed"], e["config"], Path(work) / f"r{e['seed']}_{k}") for k in "ab")
    da, db = file_digests(first), file_digests(second)
    if da != db:
        problems.append(f"files differ: {sorted(k for k in da.keys() | db.keys() if da.get(k) != db.get(k))[:5]}")
    built = tree_digest(file_digests(episode_dir(e["seed"])))
    if tree_digest(da) != built:
        problems.append("files differ from the step-1 build")
    trajectories, entries = [], []
    for k in "ab":
        traj, log = Path(work) / f"t{e['seed']}_{k}.jsonl", Path(work) / f"g{e['seed']}_{k}.jsonl"
        summary, proc = run_cli(["--policy", policy, "--mode", "train", "--config", str(config_file(e)),
                                 "--seed", str(e["seed"]), "--policy-seed", str(POLICY_SEED),
                                 "--trajectories", str(traj), "--grader-log", str(log)])
        if summary is None:
            problems.append(f"run {k} failed: {proc.stderr.strip()[-200:]}")
            return problems, None
        trajectories.append(traj.read_text(encoding="utf-8"))
        entries.append(grader_entries(log)[-1])
    if trajectories[0] != trajectories[1]:
        problems.append("trajectories differ")
    if entries[0] != entries[1]:
        problems.append(f"grader entries differ in {sorted(k for k in entries[0] if entries[0][k] != entries[1].get(k))}")
    detail = {"trajectory_sha256": text_sha256(trajectories[0]), "lines": len(trajectories[0].splitlines()),
              "reward": entries[0].get("reward"), "files_sha256": tree_digest(da)}
    return problems, detail


def record_ids(directory):
    """(every record and document id, ledger rows in file order without their ids)."""
    from auditenv.episode import load_episode
    episode = load_episode(str(directory))
    rows = [(r["timestamp"], r["account"], r["debit_minor"], r["credit_minor"]) for r in episode.tables["ledger"]]
    return set(episode.documents).union(*episode.index.values()), rows


def submit_inprocess(directory, findings):
    """Grade one answer on an episode dir through the env itself (train mode shows the grade)."""
    from auditenv.env import AuditEnv
    env = AuditEnv("train")
    env.reset(config={"episode_dir": str(directory)})
    _, reward, terminated, _, info = env.step({"action": "submit", "findings": findings})
    return reward, info.get("grading")


def transform_check(e, salt, work, transform):
    from auditenv.verify import verify_episode
    original = episode_dir(e["seed"])
    copy = Path(work) / f"copy_{e['seed']}_{salt}"
    transform(str(original), str(copy), salt)
    problems, detail = [], {"salt": salt}
    sig = lambda d: Counter((i["type"], i["expected_amount_minor"], i["currency"]) for i in truth_of(d)["issues"])
    if sig(original) != sig(copy):
        problems.append("truth (type, amount, currency) differs")
    ids_a, order_a = record_ids(original)
    ids_b, order_b = record_ids(copy)
    detail["ids_kept"] = len(ids_a & ids_b)
    detail["ledger_order_changed"] = order_a != order_b
    if sorted(order_a) != sorted(order_b):
        problems.append("the copy's ledger is not a permutation of the original's")
    if ids_a & ids_b:
        problems.append(f"{len(ids_a & ids_b)} record ids survive the renaming")
    v = verify_episode(str(copy), strict_volume=True)
    if v:
        problems.append(f"V on the copy: {v[:3]}")
    rewards = {}
    for policy in ("oracle", "reconcile"):
        for name, directory in (("original", original), ("copy", copy)):
            log = Path(work) / f"g{e['seed']}_{salt}_{policy}_{name}.jsonl"
            summary, proc = run_cli(["--policy", policy, "--mode", "eval", "--episode-dir", str(directory),
                                     "--policy-seed", str(POLICY_SEED), "--grader-log", str(log)])
            entry = grader_entries(log)[-1] if summary is not None else {}
            rewards[(policy, name)] = (entry.get("reward"), entry.get("tp"), entry.get("fp"), entry.get("fn"))
    detail["oracle"] = [rewards[("oracle", "original")][0], rewards[("oracle", "copy")][0]]
    detail["reconcile"] = [rewards[("reconcile", "original")], rewards[("reconcile", "copy")]]
    if detail["oracle"] != [1.0, 1.0]:
        problems.append(f"oracle {detail['oracle']}")
    if rewards[("reconcile", "original")] != rewards[("reconcile", "copy")]:
        problems.append(f"reconcile {detail['reconcile']}")
    # The old answer must not fit the renamed book, or the ids did not really change.
    stale, _ = submit_inprocess(copy, oracle_findings(original) | {"episode_id": oracle_findings(copy)["episode_id"]})
    detail["old_answer_on_copy"] = stale
    if truth_of(original)["issues"] and stale == 1.0:
        problems.append("the answer in old ids still scores 1.0 on the copy")
    return problems, detail


def step_paired(entries):
    from auditenv.transform import permute_and_rename
    rows = []
    with tempfile.TemporaryDirectory(prefix="rehearsal-paired-") as work:
        for k, e in enumerate(entries):
            if k % 2 == 0:
                policy = POLICIES[(k // 2) % len(POLICIES)]
                problems, detail = repeat_check(e, policy, work)
                row = {"step": "paired", "kind": "repeat", "seed": e["seed"], "policy": policy}
            else:
                salt = 700001 + 37 * k      # new salts: scripts/paired.py uses 1..20
                problems, detail = transform_check(e, salt, work, permute_and_rename)
                row = {"step": "paired", "kind": "transform", "seed": e["seed"], "policy": "oracle,reconcile"}
            row.update(ok=not problems, problems=problems, detail=detail)
            rows.append(row)
            emit(row)
    return rows


# ------------------------------------------------------------------------------------- item 5
REPRO_CODE = """
import json, sys, tempfile
sys.path.insert(0, {here!r})
from common import build, file_digests, plan, tree_digest
out = {{}}
with tempfile.TemporaryDirectory() as work:
    for e in plan():
        d = build(e["seed"], e["config"], work + "/ep_" + str(e["seed"]))
        digests = file_digests(d)
        out[str(e["seed"])] = [tree_digest({{k: v for k, v in digests.items() if k.startswith("public/")}}),
                               tree_digest(digests)]
print(json.dumps(out, sort_keys=True))
"""


def step_repro(entries, built_rows):
    runs = {}
    for label, hashseed in (("0", 0), ("1", 1), ("random", "random"), ("random-2", "random")):
        proc = run_child(REPRO_CODE.format(here=str(Path(__file__).resolve().parent)), hashseed=hashseed)
        runs[label] = json.loads(proc.stdout) if proc.returncode == 0 else {"error": proc.stderr[-300:]}
    built = {str(r["seed"]): [r["public_sha256"], r["all_sha256"]] for r in built_rows} if built_rows else None
    rows = []
    for e in entries:
        key = str(e["seed"])
        values = {label: run.get(key) for label, run in runs.items()}
        if built:
            values["step-1 build"] = built[key]
        distinct = {json.dumps(v) for v in values.values()}
        row = {"step": "repro", "seed": e["seed"], "ok": len(distinct) == 1 and None not in values.values(),
               "public_sha256": values["0"][0] if values["0"] else None, "runs": len(values)}
        rows.append(row)
        emit(row)
    # The same policy, same actions, under different hash seeds: identical trajectory bytes.
    for e in entries[:3]:
        digests = {}
        for hashseed in (0, 1, "random"):
            with tempfile.TemporaryDirectory() as work:
                traj = Path(work) / "t.jsonl"
                proc = subprocess.run([sys.executable, "-m", "auditenv.run", "--policy", "reconcile", "--mode",
                                       "train", "--config", str(config_file(e)), "--seed", str(e["seed"]),
                                       "--trajectories", str(traj)], cwd=ROOT, env=child_env(hashseed),
                                      capture_output=True, text=True, timeout=300)
                digests[str(hashseed)] = text_sha256(traj.read_text(encoding="utf-8")) if proc.returncode == 0 else None
        row = {"step": "repro", "kind": "trajectory", "seed": e["seed"], "policy": "reconcile",
               "ok": len(set(digests.values())) == 1 and None not in digests.values(), "sha256": digests}
        rows.append(row)
        emit(row)
    return rows


# ------------------------------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", default=",".join(STEPS))
    args = parser.parse_args(argv)
    steps = [s for s in args.steps.split(",") if s]
    unknown = sorted(set(steps) - set(STEPS))
    if unknown:
        parser.error(f"unknown step {unknown[0]}")
    entries = plan()
    result_path = OUT / "part_a.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    failed = False
    for step in STEPS:
        if step not in steps:
            continue
        started = time.perf_counter()
        if step == "build":
            rows = step_build(entries)
            summary = {"passed": sum(r["ok"] for r in rows), "total": len(rows)}
        elif step == "policies":
            rows = step_policies(entries)
            checks, means, diagnostics = policy_acceptance(rows)
            summary = {"checks": checks, "mean_reward": means, "diagnostics": diagnostics,
                       "passed": sum(checks.values()), "total": len(checks)}
        elif step == "verify":
            rows = step_verify(entries)
            summary = {"passed": sum(r["ok"] for r in rows), "total": len(rows)}
        elif step == "rollback":
            rows = step_rollback(entries)
            kinds = sorted({r["kind"] for r in rows})
            summary = {"passed": sum(r["ok"] for r in rows), "total": len(rows),
                       "by_kind": {k: [sum(r["ok"] for r in rows if r["kind"] == k),
                                       sum(r["kind"] == k for r in rows)] for k in kinds},
                       "failed": [f"{r['kind']} {r['seed']}" for r in rows if not r["ok"]]}
        elif step == "paired":
            rows = step_paired(entries)
            summary = {"passed": sum(r["ok"] for r in rows), "total": len(rows),
                       "repeat": sum(r["kind"] == "repeat" for r in rows),
                       "transform": sum(r["kind"] == "transform" for r in rows)}
        else:
            rows = step_repro(entries, (result.get("build") or {}).get("rows"))
            summary = {"passed": sum(r["ok"] for r in rows), "total": len(rows)}
        summary["seconds"] = round(time.perf_counter() - started, 1)
        summary["step"] = step
        result[step] = {"summary": summary, "rows": rows}
        write_json(result_path, result)
        emit({"summary": summary})
        failed = failed or summary["passed"] != summary["total"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
