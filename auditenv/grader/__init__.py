"""Matcher, terminal reward and the independent episode verifier (role B).

Matching rule and reward formula: SPEC.md, section on grading. The verifier comes at checkpoint 2.
"""
from .matcher import normalize_id, parse_finding, parse_truth
from .reward import MAX_STEPS, f1_score, score, terminal_reward, timeout_score

__all__ = [
    "MAX_STEPS",
    "f1_score",
    "normalize_id",
    "parse_finding",
    "parse_truth",
    "score",
    "terminal_reward",
    "timeout_score",
]
