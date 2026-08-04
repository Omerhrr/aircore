from __future__ import annotations

import argparse
import os
import runpy
import sys

from aircore import PolicyViolation, Workflow
from aircore.approval import cli_approval_callback
from aircore.graph import build_execution_graph, render_execution_graph
from airpy import Session

from .html_trace import render_trace_html


def _find_reportable_objects(namespace: dict):
    """Any Workflow that actually ran (journal is populated), and any
    Session that's had at least one turn -- both are "did something real"
    signals, same rule as before: a script that defines one of these but
    never uses it is silently skipped, nothing to report.

    Session support didn't exist when this CLI was first built (it was
    added well after M6/M8); this is what "finishing" it meant -- picking
    up everything that grew a real audit trail worth summarizing/tracing
    since, not adding speculative new subcommands. Workflow and Session
    are kept in separate lists (not one mixed list) because they're
    summarized/traced slightly differently below -- a Session has N
    per-turn journals, a Workflow has exactly one."""
    workflows = []
    sessions = []
    for name, value in namespace.items():
        if isinstance(value, Workflow) and value.journal is not None:
            workflows.append((name, value))
        elif isinstance(value, Session) and value.turn_count > 0:
            sessions.append((name, value))
    return workflows, sessions


def _labeled_journals(workflows, sessions):
    """Flattens workflows/sessions into one (label, Journal) list -- the
    shared unit every trace output (text graph, --json, --html) is built
    from, so the three don't drift on what counts as "a run" or how it's
    labeled. A Session contributes one entry per turn (one Journal per
    turn -- see session.py), a Workflow contributes exactly one."""
    runs = []
    for var_name, wf in workflows:
        runs.append((f"{var_name} ({wf.name})", wf.journal))
    for var_name, session in sessions:
        for turn_number, journal in enumerate(session.journals, start=1):
            runs.append((f"{var_name} ({session.session_id}) turn {turn_number}", journal))
    return runs


def _is_ail(path: str) -> bool:
    return path.endswith(".airlang")


def _run_ail(path: str) -> Workflow:
    """AirLang-M2: `.airlang` files are parsed and executed via `airlang`'s IR
    executor instead of runpy -- there's no Python namespace to inspect
    afterward (unlike a .py script), the executor hands back the one
    Workflow it built and ran directly. Every error this can raise
    (AirLangSyntaxError from a bad .airlang file, AirLangNotYetSupportedError for
    `let`/body-level `approval`/body-level `memory`, AirLangBindingError for an
    unresolved tool/capability/provider reference, or a pre-flight
    PolicyViolation) is reported the same way `ai`'s other errors are --
    one line on stderr, exit 1 -- not a raw traceback.

    `cli_approval_callback` (aircore/approval.py) is always passed through
    to execute_file() -- harmless for a workflow whose policy has no
    `approval <tool>` line (never consulted), and it's what lets a
    workflow that DOES have one actually run interactively, blocking on a
    real y/n prompt, instead of failing the pre-flight PolicyViolation
    every other caller of execute_file() gets by default."""
    # Imported here, not at module level: aircli's .py-script path has no
    # need for `airlang` at all, and keeping the import local means a
    # dependency-free `pip install .` (no `airlang` package needed) still
    # lets every non-.airlang aircli command work -- consistent with how
    # LiteLLMProvider is only imported lazily elsewhere in this project.
    from airlang import AirLangBindingError, AirLangNotYetSupportedError, AirLangSyntaxError, execute_file

    try:
        return execute_file(path, approval_callback=cli_approval_callback)
    except FileNotFoundError:
        print(f"ai: no such file: {path}", file=sys.stderr)
        sys.exit(1)
    except AirLangSyntaxError as exc:
        print(f"ai: {path}: syntax error: {exc}", file=sys.stderr)
        sys.exit(1)
    except AirLangNotYetSupportedError as exc:
        print(f"ai: {path}: {exc}", file=sys.stderr)
        sys.exit(1)
    except AirLangBindingError as exc:
        print(f"ai: {path}: {exc}", file=sys.stderr)
        sys.exit(1)
    except PolicyViolation as exc:
        print(f"ai: {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def _run_script(path: str) -> dict:
    try:
        return runpy.run_path(path, run_name="__main__")
    except FileNotFoundError:
        print(f"ai: no such file: {path}", file=sys.stderr)
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    if _is_ail(args.script):
        workflow = _run_ail(args.script)
        journal = workflow.journal
        duration = f"{journal.duration_ms:.2f}ms" if journal.duration_ms is not None else "-"
        print("\n--- ai run summary ---")
        print(f"workflow ({workflow.name}): {journal.status}  {len(journal.steps)} steps  {duration}")
        return

    namespace = _run_script(args.script)
    workflows, sessions = _find_reportable_objects(namespace)
    if not workflows and not sessions:
        return

    print("\n--- ai run summary ---")
    for var_name, wf in workflows:
        journal = wf.journal
        duration = f"{journal.duration_ms:.2f}ms" if journal.duration_ms is not None else "-"
        print(f"{var_name} ({wf.name}): {journal.status}  {len(journal.steps)} steps  {duration}")

    for var_name, session in sessions:
        last_status = session.journals[-1].status if session.journals else "-"
        state = "closed" if session.ended_at else "open"
        print(f"{var_name} ({session.session_id}, {state}): {session.turn_count} turns, "
              f"last turn {last_status}")


def _default_html_path(script: str) -> str:
    stem, _ = os.path.splitext(script)
    return f"{stem}.trace.html"


def _print_and_write_trace(runs, args: argparse.Namespace) -> None:
    """Shared by both the .py and .airlang paths in cmd_trace -- text/--json
    printing and the optional --html file, once `runs` (a list of
    (label, Journal) pairs) has been assembled, however it was assembled."""
    for label, journal in runs:
        print(f"\n--- {label} ---")
        if args.json:
            print(journal.to_json())
        else:
            graph = build_execution_graph(journal)
            print(render_execution_graph(graph))

    if args.html:
        # A static snapshot, not a live view -- see html_trace.py's
        # docstring for why. Written in addition to (not instead of) the
        # text/--json output above, so --html composes with either.
        html_doc = render_trace_html([(label, journal.to_dict()) for label, journal in runs])
        output_path = args.output or _default_html_path(args.script)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        print(f"\nwrote trace viewer to {output_path}")


def cmd_trace(args: argparse.Namespace) -> None:
    if _is_ail(args.script):
        workflow = _run_ail(args.script)
        _print_and_write_trace([(f"workflow ({workflow.name})", workflow.journal)], args)
        return

    namespace = _run_script(args.script)
    workflows, sessions = _find_reportable_objects(namespace)
    runs = _labeled_journals(workflows, sessions)
    if not runs:
        print("ai trace: no completed Workflow or used Session found in this script's "
              "global scope (define one at module level, then call .run() or .send() "
              "on it).", file=sys.stderr)
        return

    _print_and_write_trace(runs, args)


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai", description="Minimal CLI for aircore workflows and sessions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run a script (.py) or workflow (.airlang) and summarize any Workflow/Session it ran."
    )
    run_parser.add_argument("script", help="Path to a .py script or a .airlang workflow file.")
    run_parser.set_defaults(func=cmd_run)

    trace_parser = subparsers.add_parser(
        "trace", help="Run a script (.py) or workflow (.airlang) and print the execution graph of any Workflow/Session it ran."
    )
    trace_parser.add_argument("script", help="Path to a .py script or a .airlang workflow file.")
    trace_parser.add_argument("--json", action="store_true",
                               help="Print each journal as JSON instead of the rendered graph.")
    trace_parser.add_argument("--html", action="store_true",
                               help="Also write a self-contained, clickable HTML trace viewer.")
    trace_parser.add_argument("--output", default=None,
                               help="Path for the --html file (default: <script>.trace.html).")
    trace_parser.set_defaults(func=cmd_trace)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
