"""Independent verifier of one episode (role V, docs/tasks/V_verifier.md).

Checks that truth.json follows from the public records and policy.json alone: its own detector of the
rules INV-13…INV-16 must find exactly the truth issues (INV-18), every evidence set must give the amount
(INV-17), and the book, truth and public files must keep INV-01…INV-06, 08…10, 12, 20 and 23. Two truth
issues with one type and affected set (QA D-4) are an INV-18 problem, SPEC §6.
Written from SPEC §3, §5, §6, case §03 and §05 and the policy rules of the domain spec A only.
"""
import json

from ..episode import EpisodeError
from . import checks
from .book import Book, BookError
from .consistency import check_consistency
from .rules import detect


def verify_episode(episode_dir: str, strict_volume: bool = True) -> list[dict]:
    """Problems as {"code": "INV-18", "detail": "..."}. An empty list means the episode is correct.
    Never raises: anything unexpected becomes a problem with code LOAD or INTERNAL."""
    problems = []
    try:
        book = Book(episode_dir)
        truth = book.episode.load_truth()
    except (EpisodeError, BookError, OSError, ValueError) as exc:
        return [{"code": "LOAD", "detail": f"{type(exc).__name__}: {exc}"}]
    except Exception as exc:  # noqa: BLE001 - the contract is a list, never an exception
        return [{"code": "INTERNAL", "detail": f"load: {type(exc).__name__}: {exc}"}]

    state = {"issues": [], "violations": []}

    def run(name, fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - one broken check must not hide the others
            problems.append({"code": "INTERNAL", "detail": f"{name}: {type(exc).__name__}: {exc}"})

    def issues():
        state["issues"] = checks.well_formed_issues(truth, book.episode.episode_id, problems)

    def violations():
        state["violations"] = detect(book, problems)

    run("layout", lambda: checks.check_layout(book, episode_dir, problems))
    run("truth", issues)
    run("rules", violations)
    run("INV-01", lambda: checks.check_inv01_02(book, problems))
    run("consistency", lambda: check_consistency(book, problems))
    run("INV-03", lambda: checks.check_inv03(book, state["issues"], state["violations"], strict_volume, problems))
    run("INV-17", lambda: checks.check_inv17(state["issues"], book, problems))
    run("INV-18", lambda: checks.check_inv18(state["issues"], state["violations"], book, problems))
    run("INV-20", lambda: checks.check_inv20(state["issues"], problems))
    run("INV-23", lambda: checks.check_inv23(episode_dir, state["issues"], problems))
    return problems


def format_problem(problem):
    return json.dumps(problem, ensure_ascii=False, sort_keys=True)
