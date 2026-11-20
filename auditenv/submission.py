"""Structure check of findings.json before grading (case §05).

A submit with a broken structure ends the episode with reward -1 and an error in the
grader log (PM decision 2). The grader still re-checks every finding on its own as a
second layer and never raises.
"""
from .grader.matcher import (
    F_AFFECTED, F_AMOUNT, F_CURRENCY, F_EVIDENCE, F_TYPE, ISSUE_TYPES, MAX_EVIDENCE,
    _is_currency, _is_id_list, _is_minor_amount,
)

F_EPISODE = "episode_id"
F_FINDINGS = "findings"
F_EXPLANATION = "explanation"
MAX_FINDINGS = 30
MAX_EXPLANATION = 1000
REQUIRED = (F_TYPE, F_AFFECTED, F_AMOUNT, F_CURRENCY, F_EVIDENCE, F_EXPLANATION)


class InvalidSubmission(ValueError):
    """findings.json breaks the case §05 structure. str() is a short code for the grader log."""


def check_submission(payload, episode_id):
    """The findings list of a well-formed findings.json; otherwise InvalidSubmission.

    Extra keys are ignored, both at the top level and inside a finding.
    """
    try:
        return _check(payload, episode_id)
    except InvalidSubmission:
        raise
    except Exception:  # hostile in-process objects only; JSON input never gets here
        raise InvalidSubmission("bad_structure") from None


def _check(payload, episode_id):
    if type(payload) is not dict:
        raise InvalidSubmission("submit_not_object")
    if type(payload.get(F_EPISODE)) is not str or payload[F_EPISODE] != episode_id:
        raise InvalidSubmission("episode_id_mismatch")
    findings = payload.get(F_FINDINGS)
    if type(findings) is not list:
        raise InvalidSubmission("findings_not_list")
    if len(findings) > MAX_FINDINGS:
        raise InvalidSubmission("too_many_findings")
    for finding in findings:
        _check_finding(finding)
    return findings


def _check_finding(f):
    if type(f) is not dict:
        raise InvalidSubmission("finding_not_object")
    if any(key not in f for key in REQUIRED):
        raise InvalidSubmission("missing_field")
    if type(f[F_TYPE]) is not str or f[F_TYPE] not in ISSUE_TYPES:
        raise InvalidSubmission("bad_type")
    if not _is_id_list(f[F_AFFECTED]) or not f[F_AFFECTED]:
        raise InvalidSubmission("bad_affected_ids")
    if not _is_minor_amount(f[F_AMOUNT]):
        raise InvalidSubmission("bad_claimed_amount")
    if not _is_currency(f[F_CURRENCY]):
        raise InvalidSubmission("bad_currency")
    if not _is_id_list(f[F_EVIDENCE]):
        raise InvalidSubmission("bad_evidence_ids")
    if len(f[F_EVIDENCE]) > MAX_EVIDENCE:
        raise InvalidSubmission("too_many_evidence_ids")
    if type(f[F_EXPLANATION]) is not str:
        raise InvalidSubmission("bad_explanation")
    if len(f[F_EXPLANATION]) > MAX_EXPLANATION:
        raise InvalidSubmission("explanation_too_long")
