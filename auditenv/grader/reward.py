"""F1 and the organizers' terminal reward (SPEC.md §5.4).

reward = clip(2·F1 − 1 − 0.002·inspection_steps, −1, 1). Arithmetic is exact in
Fraction and converted to float once, so 40 steps give exactly 0.92.
"""
from fractions import Fraction

from .matcher import match, parse_finding, parse_truth

STEP_COST = Fraction(2, 1000)
MAX_STEPS = 100


def f1_score(tp, fp, fn):
    """F1 as a Fraction. Empty truth and empty answer give 1; no true positive otherwise gives 0."""
    denominator = 2 * tp + fp + fn
    return Fraction(1) if denominator == 0 else Fraction(2 * tp, denominator)


def terminal_reward(f1, inspection_steps):
    raw = 2 * Fraction(f1) - 1 - STEP_COST * inspection_steps
    return float(min(Fraction(1), max(Fraction(-1), raw)))


def score(findings, truth, inspection_steps):
    """Grade one submit. The result is for the grader log only, never for the agent in eval."""
    if not isinstance(findings, list):
        raise TypeError("findings must be a list")
    if type(inspection_steps) is not int or inspection_steps < 0:
        raise ValueError("inspection_steps must be a non-negative int")
    issues = parse_truth(truth)
    parsed = [parse_finding(f) for f in findings]
    pairs = match(parsed, issues)
    tp = len(pairs)
    fp = len(findings) - tp
    fn = len(issues) - tp
    f1 = f1_score(tp, fp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        # Vacuous cases count as 1: no claims means no false claims, no issues means none missed.
        "precision": float(Fraction(tp, tp + fp)) if tp + fp else 1.0,
        "recall": float(Fraction(tp, tp + fn)) if tp + fn else 1.0,
        "f1": float(f1),
        "reward": terminal_reward(f1, inspection_steps),
        "inspection_steps": inspection_steps,
        "matched": [{"issue_id": issues[i].issue_id, "finding_index": j} for i, j in pairs],
        "invalid_findings": [j for j, f in enumerate(parsed) if f is None],
    }


def timeout_score(truth):
    """Step limit or timeout: an empty answer charged for every step (decision v2, R-02)."""
    return score([], truth, MAX_STEPS)
