"""M4 acceptance example: metrics and the execution graph are collected
automatically -- no manual instrumentation, same run() call as always.

Run with: python examples/observability.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool, build_execution_graph, render_execution_graph


@tool
def search_web():
    time.sleep(0.03)
    return ["paper A", "paper B"]


@tool
def search_github():
    time.sleep(0.03)
    return ["repo A", "repo B"]


@tool
def merge():
    return "merged report (stub)"


if __name__ == "__main__":
    workflow = Workflow("Research")
    workflow.parallel(search_web, search_github)
    workflow.step(merge)
    workflow.run()  # no metrics= or observer= argument needed

    print("=== Metrics (collected automatically) ===")
    print(workflow.metrics.summary())

    print("\n=== Execution graph (rendered from the journal) ===")
    graph = build_execution_graph(workflow.journal)
    print(render_execution_graph(graph))
