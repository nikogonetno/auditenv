"""Diagnostic metrics of the baseline policies (case §10; docs/tasks/B_next.md item 11).

    python3 scripts/metrics.py EPISODE_DIR [EPISODE_DIR ...] [--policies random,reconcile,flag_all,oracle]
                               [--mode eval] [--seed 0] [--out docs/METRICS.md] [--json runs/metrics.json]
                               [--trajectories runs/trajectories.jsonl]

An EPISODE_DIR is an episode in the case §03 layout or a split directory (episodes/train) whose
subdirectories are episodes, taken in sorted order. Every policy runs on every episode through
auditenv.run.run_policy with the same env seed and policy seed 0. The grade comes from the grader log
and truth from the env's grader side: this is the evaluator's report and never reaches a policy.

P, R and F1 are micro over all episodes with the case §05 rules, return is the mean terminal reward.
Per type, TP and FN go by the type of the truth issue and FP by the type of the false finding.
Writes the markdown report to --out, the raw rows as JSON lines to --json (overwritten), appends the
trajectories to --trajectories and prints the summary table. Only the wall times differ between runs.
"""
import argparse
import io
import json
import math
import os
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auditenv.config import ISSUE_TYPES  # noqa: E402
from auditenv.env import MODES, AuditEnv  # noqa: E402
from auditenv.grader import f1_score  # noqa: E402
from auditenv.private_args import hide_private_args  # noqa: E402
from auditenv.run import POLICIES, run_policy  # noqa: E402
from auditenv.runner import encode_line  # noqa: E402

POLICY_SEED = 0
OTHER_TYPE = "other"  # a false finding whose type is none of the four, or that could not be read
WALL_TIME = "время, с"
SUMMARY_HEADER = ("политика", "эпизоды", "TP", "FP", "FN", "P", "R", "F1", "возврат", "шаги",
                  "шаги до submit", "обрезано", "неверная структура", WALL_TIME)


def parse_policies(text):
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in POLICIES]
    if not names or unknown or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"a comma-separated list of distinct names from {', '.join(POLICIES)}")
    return names


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="scripts/metrics.py", description=__doc__.split("\n\n")[0],
                                     allow_abbrev=False)  # see auditenv.private_args
    parser.add_argument("episode_dirs", nargs="+", metavar="EPISODE_DIR", help="an episode or a split directory")
    parser.add_argument("--policies", type=parse_policies, default=list(POLICIES))
    parser.add_argument("--mode", default="eval", choices=MODES)
    parser.add_argument("--seed", type=int, default=0, help="episode seed for env.reset, the same for every run")
    parser.add_argument("--out", default=str(ROOT / "docs" / "METRICS.md"), help="markdown report")
    parser.add_argument("--json", help="raw rows as JSON lines, overwritten")
    parser.add_argument("--trajectories", help="trajectories of every run, appended")
    return parser.parse_args(argv)


def is_episode(path):
    return (path / "public").is_dir()


def expand(paths):
    """Episode directories in argument order. A split directory gives its episodes sorted by name."""
    episodes = []
    for raw in paths:
        path = Path(raw)
        if is_episode(path):
            episodes.append(path)
            continue
        if not path.is_dir():
            raise ValueError(f"{raw}: not a directory")
        found = sorted((p for p in path.iterdir() if p.is_dir() and is_episode(p)), key=lambda p: p.name)
        if not found:
            raise ValueError(f"{raw}: neither an episode nor a directory of episodes")
        episodes.extend(found)
    return episodes


def submitted_findings(trajectory_text):
    """Findings of the submit that ended the episode; [] after a truncation or an unreadable submit."""
    lines = trajectory_text.splitlines()
    action = json.loads(lines[-1]).get("action") if lines else None
    if not isinstance(action, dict) or action.get("action") != "submit":
        return []
    body = action.get("findings")
    findings = body.get("findings") if isinstance(body, dict) else None
    return findings if isinstance(findings, list) else []


def grade_row(policy, source, env, result, findings):
    """One raw row from the grader log entry, the runner result, the submitted findings and truth."""
    entry = env.grader_log[-1]
    truth = env._truth  # grader side only
    issues = truth.get("issues") or []
    if "tp" in entry:
        tp, fp, fn, matched = entry["tp"], entry["fp"], entry["fn"], entry["matched"]
    else:  # a findings.json with a broken structure: reward -1, nothing matched (case §05)
        tp, fp, fn, matched = 0, len(findings), len(issues), []
    matched_issues = {m["issue_id"] for m in matched}
    matched_findings = {m["finding_index"] for m in matched}

    per_type = {kind: {"tp": 0, "fp": 0, "fn": 0} for kind in ISSUE_TYPES}
    for issue in issues:
        per_type[issue["type"]]["tp" if issue["issue_id"] in matched_issues else "fn"] += 1
    unread = fp
    for index, finding in enumerate(findings):
        if index in matched_findings:
            continue
        kind = finding.get("type") if isinstance(finding, dict) else None
        kind = kind if kind in ISSUE_TYPES else OTHER_TYPE
        per_type.setdefault(kind, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
        unread -= 1
    if unread > 0:  # an oversize submit is recorded as a digest: its false findings have no type to read
        per_type.setdefault(OTHER_TYPE, {"tp": 0, "fp": 0, "fn": 0})["fp"] += unread

    row = {
        "episode_id": entry["episode_id"], "source": str(source), "clean": bool(truth.get("clean")),
        "issues": len(issues), "policy": policy, "mode": env.mode, "end": entry["end"],
        "tp": tp, "fp": fp, "fn": fn, "reward": entry["reward"], "inspection_steps": entry["inspection_steps"],
        "steps": result["steps"], "wall_time_s": result["wall_time_s"], "findings": len(findings),
        "missed": sorted(i["issue_id"] for i in issues if i["issue_id"] not in matched_issues),
        "per_type": per_type,
    }
    if "error" in entry:
        row["error"] = entry["error"]
    return row


def run_all(episodes, policies, mode, seed, trajectories=None):
    rows = []
    for source in episodes:
        for policy in policies:
            env, trajectory = AuditEnv(mode), io.StringIO()
            try:
                result = run_policy(policy, env, seed, {"episode_dir": str(source)}, POLICY_SEED,
                                    trajectory=trajectory)
            except (ValueError, RuntimeError) as exc:  # broken episode, or no sandbox when required
                raise ValueError(f"{source}: {exc}") from None
            text = trajectory.getvalue()
            if trajectories is not None:
                trajectories.write(text)
                trajectories.flush()
            rows.append(grade_row(policy, source, env, result, submitted_findings(text)))
    return rows


# ---------------------------------------------------------------------------- report
def fmt(value):
    text = f"{float(value):.3f}"
    return "0.000" if text == "-0.000" else text


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def summary(rows, policy):
    mine = [r for r in rows if r["policy"] == policy]
    tp, fp, fn = (sum(r[key] for r in mine) for key in ("tp", "fp", "fn"))
    return {
        "policy": policy, "episodes": len(mine), "tp": tp, "fp": fp, "fn": fn,
        "precision": Fraction(tp, tp + fp) if tp + fp else Fraction(0),
        "recall": Fraction(tp, tp + fn) if tp + fn else Fraction(0),
        "f1": f1_score(tp, fp, fn),  # 1 when truth and answers are empty everywhere (case §05)
        "return": mean(r["reward"] for r in mine),
        "steps": Fraction(sum(r["steps"] for r in mine), len(mine)),
        "inspection_steps": Fraction(sum(r["inspection_steps"] for r in mine), len(mine)),
        "truncated": sum(r["end"] == "truncated" for r in mine),
        "invalid": sum("error" in r for r in mine),
        "wall_time_s": mean(r["wall_time_s"] for r in mine),
    }


def per_type_totals(rows, policy):
    totals = {kind: {"tp": 0, "fp": 0, "fn": 0} for kind in ISSUE_TYPES}
    for row in rows:
        if row["policy"] != policy:
            continue
        for kind, counts in row["per_type"].items():
            for key, value in counts.items():
                totals.setdefault(kind, {"tp": 0, "fp": 0, "fn": 0})[key] += value
    return totals


def table(header, rows, text_columns=1):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" if k < text_columns else "---:" for k in range(len(header))) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def summary_table(rows, policies):
    body = []
    for policy in policies:
        s = summary(rows, policy)
        body.append([policy, s["episodes"], s["tp"], s["fp"], s["fn"], fmt(s["precision"]), fmt(s["recall"]),
                     fmt(s["f1"]), fmt(s["return"]), fmt(s["steps"]), fmt(s["inspection_steps"]),
                     s["truncated"], s["invalid"], fmt(s["wall_time_s"])])
    return table(SUMMARY_HEADER, body)


def episode_rows(rows, policies):
    """(episode_id, clean, issues, {policy: reward}) per episode run, in run order."""
    blocks = [rows[k:k + len(policies)] for k in range(0, len(rows), len(policies))]
    return [(b[0]["episode_id"], b[0]["clean"], b[0]["issues"], {r["policy"]: r["reward"] for r in b}) for b in blocks]


def report(rows, args, policies):
    episodes = episode_rows(rows, policies)
    clean = sum(1 for e in episodes if e[1])
    lines = [
        "# Метрики базовых политик",
        "",
        "Диагностика по кейсу §10: P, R, F1, возврат, шаги и время политик. Файл пишет `python3 scripts/metrics.py`.",
        "При тех же эпизодах, seed и коде все числа, кроме времени, повторяются.",
        "",
        f"- Эпизоды: {len(episodes)}, из них чистых: {clean}",
        "- Источники: " + ", ".join(f"`{d}`" for d in args.episode_dirs),
        f"- Режим: {args.mode}, seed среды {args.seed}, seed политик {POLICY_SEED}",
        "- Политики: " + ", ".join(policies),
        "",
        "## Сводка по политикам",
        "",
        "P, R и F1 микро по всем эпизодам, правила кейса §05: при нулевом знаменателе P и R равны 0,",
        "F1 равен 1, только если пусты и эталон, и ответы. Возврат это средняя терминальная награда. Шаги это",
        "вызовы `step` вместе с submit, шаги до submit это `inspection_steps`. Обрезано: эпизод кончился по лимиту",
        "шагов или времени. Неверная структура: findings.json отклонён с наградой −1, его выводы считаются FP.",
        "Время это среднее время эпизода, единственное невоспроизводимое число.",
        "",
        *summary_table(rows, policies),
        "",
        "## По типам ошибок",
        "",
        "TP и FN по типу ошибки эталона, FP по типу ложного вывода, чистые эпизоды дают только FP.",
        f"`{OTHER_TYPE}` это ложные выводы, тип которых не из четырёх.",
        "",
    ]
    body = []
    for policy in policies:
        for kind, counts in per_type_totals(rows, policy).items():
            body.append([policy, kind, counts["tp"], counts["fn"], counts["fp"]])
    lines += table(("политика", "тип", "TP", "FN", "FP"), body, text_columns=2)
    lines += ["", "## По эпизодам", "", "Возврат каждой политики на каждом эпизоде, в порядке прогона.", ""]
    body = [[episode_id, "да" if is_clean else "нет", issues, *(fmt(rewards[p]) for p in policies)]
            for episode_id, is_clean, issues, rewards in episodes]
    lines += table(("эпизод", "чистый", "ошибок", *policies), body, text_columns=2)
    return "\n".join(lines) + "\n"


def open_output(path, mode):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return open(path, mode, encoding="utf-8", newline="\n")


def main(argv=None):
    # The seed and config never stay in this process's command line (QA N-1).
    args = parse_args(hide_private_args() if argv is None else argv)
    try:
        episodes = expand(args.episode_dirs)
    except ValueError as exc:
        print(f"metrics.py: {exc}", file=sys.stderr)
        return 2
    # Trajectories (oracle answers among them) and rows with issue ids stay unreadable for the sandbox user.
    umask = os.umask(0o077)
    try:
        trajectories = open_output(args.trajectories, "a") if args.trajectories else None
        try:
            rows = run_all(episodes, args.policies, args.mode, args.seed, trajectories)
        finally:
            if trajectories is not None:
                trajectories.close()
        if args.json:
            with open_output(args.json, "w") as f:
                f.writelines(encode_line(row) for row in rows)
    except ValueError as exc:
        print(f"metrics.py: {exc}", file=sys.stderr)
        return 2
    finally:
        os.umask(umask)  # the markdown report is a public document
    with open_output(args.out, "w") as f:
        f.write(report(rows, args, args.policies))
    sys.stdout.write("\n".join(summary_table(rows, args.policies)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
