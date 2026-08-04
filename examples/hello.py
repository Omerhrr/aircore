"""M0 acceptance example: run one tool through the scheduler and get a
complete journal + event stream.

Run with: python examples/hello.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool


@tool
def hello():
    return "Hello, World!"


if __name__ == "__main__":
    workflow = Workflow("Hello")
    workflow.step(hello)

    journal = workflow.run()

    print(journal.pretty())
    print()
    print(journal.to_json())
