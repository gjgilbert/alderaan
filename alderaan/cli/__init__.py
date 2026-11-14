"""`alderaan` command-line entry point; dispatches to subcommand modules"""

import sys

from alderaan.cli import run

_COMMANDS = {
    "run": run.main,
}

USAGE = """usage: alderaan <command> [options]

commands:
  run    run the pipeline on one target"""


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2

    command, rest = argv[0], argv[1:]
    if command not in _COMMANDS:
        print(f"alderaan: unknown command '{command}'\n\n{USAGE}", file=sys.stderr)
        return 2

    return _COMMANDS[command](rest)
