"""python -m auditenv.verify DIR: one JSON line per problem, exit code 1 if there are any."""
import argparse
import sys

from . import format_problem, verify_episode


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m auditenv.verify", description=__doc__)
    parser.add_argument("episode_dir", help="episode folder with public/ and private/truth.json")
    parser.add_argument("--no-strict-volume", action="store_true",
                        help="skip the INV-03 rule of 10 neighbouring rows per side (small test books)")
    args = parser.parse_args(argv)
    problems = verify_episode(args.episode_dir, strict_volume=not args.no_strict_volume)
    for problem in problems:
        print(format_problem(problem))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
