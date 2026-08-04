"""Retry gap closure example: a flaky idempotent tool succeeds on its
3rd attempt because retries=3 was declared; a non-idempotent tool that
fails is never retried, even if it *could* have succeeded on a later try.

Run with: python examples/retries.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool


attempts = {"count": 0}


@tool(idempotent=True, retries=3)
def flaky_read():
    attempts["count"] += 1
    if attempts["count"] < 3:
        raise ConnectionError(f"transient failure on attempt {attempts['count']}")
    return "read succeeded"


@tool  # idempotent=False (default) -- retries=0 is the only legal value
def send_payment():
    raise RuntimeError("payment gateway timeout")


if __name__ == "__main__":
    print("=== Idempotent tool, retries=3: recovers from 2 transient failures ===")
    workflow = Workflow("Flaky")
    workflow.step(flaky_read)
    journal = workflow.run()
    print(journal.pretty())
    print(f"Total attempts made: {attempts['count']}\n")

    print("=== Non-idempotent tool: fails once, never retried, even though retries wasn't set ===")
    workflow2 = Workflow("NonIdempotent")
    workflow2.step(send_payment)
    journal2 = workflow2.run()
    print(f"status: {journal2.status}, retries recorded: {journal2.steps[0].retries}\n")

    print("=== Declaring retries>0 on a non-idempotent tool is rejected at construction time ===")
    try:
        @tool(idempotent=False, retries=3)
        def unsafe():
            return "should never be constructed"
    except ValueError as exc:
        print(f"ValueError: {exc}")
