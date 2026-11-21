"""python -m auditenv.verify [--rollback] DIR: one JSON line per problem, exit code 1 if there are any.

With --rollback the episode is checked as it was before the errors: provenance is undone on a temporary
copy and the strict verifier runs there with an empty truth (auditenv/verify/rollback.py)."""
import argparse
import sys

from . import format_problem, verify_episode
from .rollback import verify_rollback


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m auditenv.verify", description=__doc__)
    parser.add_argument("episode_dir", help="episode folder with public/ and private/truth.json")
    parser.add_argument("--no-strict-volume", action="store_true",
                        help="skip the INV-03 rule of 10 neighbouring rows per side (small test books)")
    parser.add_argument("--rollback", action="store_true",
                        help="undo private/provenance.json on a copy and verify the book before the errors")
    args = parser.parse_args(argv)
    check = verify_rollback if args.rollback else verify_episode
    problems = check(args.episode_dir, strict_volume=not args.no_strict_volume)
    for problem in problems:
        print(format_problem(problem))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
