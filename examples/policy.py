"""M3 acceptance example: development mode vs. production mode.

Run with: python examples/policy.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool, Agent, Policy, PolicyViolation, Network


@tool
def download_file():
    return "downloaded"


if __name__ == "__main__":
    print("=== Development mode: Policy(require_agent=False), anonymous step is fine ===")
    dev = Workflow("Dev", policy=Policy(require_agent=False))
    dev.step(download_file)
    journal = dev.run()
    print(f"status: {journal.status}\n")

    print("=== Production mode: Policy(require_agent=True), anonymous step is rejected pre-flight ===")
    prod = Workflow("Prod", policy=Policy(require_agent=True))
    prod.step(download_file)  # no agent attached
    try:
        prod.run()
    except PolicyViolation as exc:
        print(f"PolicyViolation: {exc}\n")

    print("=== Same workflow, now with an identified agent -- passes pre-flight, runs normally ===")
    bot = Agent("Downloader", capabilities=[Network])
    prod2 = Workflow("Prod", policy=Policy(require_agent=True))
    prod2.step(download_file, agent=bot)
    journal = prod2.run()
    print(f"status: {journal.status}\n")

    print("=== max_parallel: a 3-tool parallel block under a limit of 2 is rejected pre-flight ===")

    @tool
    def a():
        return "a"

    @tool
    def b():
        return "b"

    @tool
    def c():
        return "c"

    capped = Workflow("Capped", policy=Policy(max_parallel=2))
    capped.parallel(a, b, c)
    try:
        capped.run()
    except PolicyViolation as exc:
        print(f"PolicyViolation: {exc}")
