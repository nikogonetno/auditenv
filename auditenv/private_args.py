"""Keep the episode seed and config out of every process command line (QA N-1, case §03).

Any user in the container can read /proc/<pid>/cmdline of any process, but only the owner can read
/proc/<pid>/environ. A command that runs policy processes calls hide_private_args() before anything
else: if its argv holds --seed or --config, the process re-executes itself once with those options moved
into the environment. No process alive while a policy runs then shows them, and the documented commands
keep working unchanged.
"""
import json
import os
import sys

ENV = "AUDITENV_PRIVATE_ARGS"
PRIVATE_FLAGS = ("--seed", "--config")


def split_private(args, flags=PRIVATE_FLAGS):
    """(public args, private args) for the forms "--flag value" and "--flag=value".

    Parsers that use this must set allow_abbrev=False, or "--se 1" would slip through as --seed.
    """
    public, private = [], []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.split("=", 1)[0] in flags:
            if "=" in arg:
                private.append(arg)
                i += 1
                continue
            if i + 1 < len(args):
                private += [arg, args[i + 1]]
                i += 2
                continue
        public.append(arg)  # also a flag without its value: argparse reports it
        i += 1
    return public, private


def hide_private_args(flags=PRIVATE_FLAGS):
    """Arguments for argparse with the private values restored. May re-exec this process (POSIX)."""
    args = sys.argv[1:]
    if ENV in os.environ:
        return args + json.loads(os.environ.pop(ENV))
    public, private = split_private(args, flags)
    if not private or os.name != "posix":
        return args
    prefix = sys.orig_argv[:len(sys.orig_argv) - len(args)]  # interpreter, its options, -m module or script
    os.execve(sys.executable, prefix + public, dict(os.environ, **{ENV: json.dumps(private)}))
