"""M1 acceptance example: sequential step -> parallel block -> sequential step.

Tools are stubs (no real web/GitHub calls) since this is exercising the
scheduler's structure, not building the actual Research workflow from the
architecture spec yet -- that needs Agent/Policy/Capabilities (M2/M3) to be
a real agent workflow rather than plain functions.

Run with: python examples/research.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool


@tool
def search_web():
    time.sleep(0.05)
    return ["paper A", "paper B"]


@tool
def search_github():
    time.sleep(0.05)
    return ["repo A", "repo B"]


@tool
def merge():
    return "merged report (stub)"


if __name__ == "__main__":
    workflow = Workflow("Research")
    workflow.parallel(search_web, search_github)
    workflow.step(merge)

    journal = workflow.run()

    print(journal.pretty())
