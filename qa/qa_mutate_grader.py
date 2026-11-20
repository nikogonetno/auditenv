"""Mutation test of B's test suite. Each mutant breaks one rule of the grader.
A mutant is KILLED when `python -m unittest discover -s tests -t .` fails on it.
SURVIVED means the 98 tests do not notice that rule breaking.

Usage: python -I qa_mutate_grader.py <repo_root>
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
M, R = "auditenv/grader/matcher.py", "auditenv/grader/reward.py"
MUTANTS = [
    ("currency check removed", M, "        and finding.currency == issue.currency\n", ""),
    ("evidence check removed", M, "        and any(s <= finding.evidence for s in issue.evidence_sets)\n", ""),
    ("amount check removed", M, "        and abs(finding.amount_minor - issue.expected_amount_minor) <= issue.tolerance_minor\n", ""),
    ("tolerance strict <", M, "<= issue.tolerance_minor\n", "< issue.tolerance_minor\n"),
    ("affected subset instead of equal", M, "finding.affected == issue.affected", "finding.affected <= issue.affected"),
    ("affected overlap instead of equal", M, "finding.affected == issue.affected", "bool(finding.affected & issue.affected)"),
    ("evidence limit 11", M, "MAX_EVIDENCE = 10", "MAX_EVIDENCE = 11"),
    ("evidence repeats not counted", M, "len(evidence) > MAX_EVIDENCE", "len(set(evidence)) > MAX_EVIDENCE"),
    ("invalid findings not counted as FP", R, "fp = len(findings) - tp", "fp = sum(f is not None for f in parsed) - tp"),
    ("greedy matching", M, "    target = _max_matching_size(candidates, order, used)\n",
     "    target = len({next((j for j in c), None) for c in candidates} - {None})\n"),
    ("ties reversed", M, "        for j in candidates[i]:\n            if j not in used and 1 +",
     "        for j in reversed(candidates[i]):\n            if j not in used and 1 +"),
    ("clip upper bound removed", R, "float(min(Fraction(1), max(Fraction(-1), raw)))", "float(max(Fraction(-1), raw))"),
    ("clip lower bound removed", R, "float(min(Fraction(1), max(Fraction(-1), raw)))", "float(min(Fraction(1), raw))"),
    ("step cost 0.02", R, "STEP_COST = Fraction(2, 1000)", "STEP_COST = Fraction(2, 100)"),
    ("empty and empty gives F1 0", R, "Fraction(1) if denominator == 0", "Fraction(0) if denominator == 0"),
    ("timeout charges 0 steps", R, "return score([], truth, MAX_STEPS)", "return score([], truth, 0)"),
    ("naive prefix strip", M, "    if sep and rest and prefix in ID_PREFIXES:\n", "    if sep and rest:\n"),
    ("bool accepted as amount", M, "return type(value) is int and value >= 0", "return isinstance(value, int) and value >= 0"),
    ("negative amount accepted", M, "return type(value) is int and value >= 0", "return type(value) is int"),
    ("lowercase currency accepted", M, " and value.isupper()", ""),
    ("je/ guard removed", M, '            if any(x.startswith("je/") for x in all_ids):\n                raise TypeError("je/ ids are not records (SPEC.md §3.3)")\n', ""),
    ("INV-20 guard removed", M, "        if issue.expected_amount_minor <= 2 * issue.tolerance_minor:\n", "        if False:\n"),
    ("clean flag guard removed", M, '    if truth.get("clean") is not (not truth["issues"]):\n', "    if False:\n"),
    ("duplicate issue_id guard removed", M, "    if len(set(ids)) != len(ids):\n", "    if False:\n"),
    ("tolerance default 0", M, "DEFAULT_TOLERANCE_MINOR = 1", "DEFAULT_TOLERANCE_MINOR = 0"),
    ("negative tolerance accepted", M, "issue.tolerance_minor < 0", "issue.tolerance_minor < -10"),
    ("empty affected accepted", M, "    if not _is_id_list(affected) or not affected:\n", "    if not _is_id_list(affected):\n"),
    ("parse_finding raises", M, "    try:\n        return _parse_finding(raw)\n    except Exception:\n        return None\n",
     "    return _parse_finding(raw)\n"),
    ("findings list subclass accepted", R, "if type(findings) is not list:", "if not isinstance(findings, list):"),
    ("steps bool accepted", R, "if type(inspection_steps) is not int or", "if not isinstance(inspection_steps, int) or"),
]

work = Path(tempfile.mkdtemp(prefix="qa_grader_mutants_"))
src = work / "pristine"
shutil.copytree(ROOT, src, ignore=shutil.ignore_patterns(".git", "__pycache__", "QA_REVIEW.txt"))
base = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=src,
                      capture_output=True, text=True)
print("pristine suite", "passes" if base.returncode == 0 else "FAILS", "\n")
killed = 0
for name, path, old, new in MUTANTS:
    d = work / f"m{MUTANTS.index((name, path, old, new)):02d}"
    shutil.copytree(src, d)
    f = d / path
    text = f.read_text(encoding="utf-8")
    if text.count(old) != 1:
        print(f"SKIPPED  {name:36s} the code no longer has this line")
        continue
    f.write_text(text.replace(old, new), encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=d,
                       capture_output=True, text=True, timeout=600)
    dead = p.returncode != 0
    killed += dead
    fails = [l for l in p.stderr.splitlines() if l.startswith(("FAIL:", "ERROR:"))]
    print(f"{'KILLED  ' if dead else 'SURVIVED'} {name:36s} {len(fails)} failing tests")
print(f"\n{killed} of {len(MUTANTS)} mutants killed, the rest survived or were skipped")
