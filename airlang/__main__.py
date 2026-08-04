"""`ail parse <file.airlang> [--ir]` (also runnable as `python -m airlang parse
...`) -- AirLang-M0's original CLI surface: a debugging tool that prints the
IR a .airlang file parses to, as JSON. This is NOT `ai run file.airlang`
(that's AirLang-M2, backed by the real executor -- AirLang-M1, `airlang/
executor.py`, which does exist and is what actually runs a workflow); this
command never executes anything, it only proves parsing works and lets you
inspect the IR by eye.
"""

from __future__ import annotations

import argparse
import json
import sys

from .lexer import AirLangSyntaxError
from .parser import parse_file


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ail", description="AirLang parser/IR inspector (AirLang-M0).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Parse a .airlang file and print its IR.")
    parse_parser.add_argument("file", help="Path to a .airlang file.")
    parse_parser.add_argument("--ir", action="store_true",
                               help="Print the IR as JSON (currently the only output this command has).")

    args = parser.parse_args(argv)

    try:
        ir = parse_file(args.file)
    except AirLangSyntaxError as exc:
        print(f"ail: {args.file}: {exc}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"ail: no such file: {args.file}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(ir, indent=2))


if __name__ == "__main__":
    main()
