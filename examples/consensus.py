"""M6 acceptance example: three voters, majority agreement; then a tie,
which correctly fails rather than silently picking a winner.

Run with: python examples/consensus.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool, unanimous


@tool
def model_a():
    return "yes"


@tool
def model_b():
    return "yes"


@tool
def model_c():
    return "no"


if __name__ == "__main__":
    print("=== majority (default strategy): 2 vs 1, 'yes' wins ===")
    workflow = Workflow("Vote")
    workflow.consensus(model_a, model_b, model_c)
    journal = workflow.run()
    print(journal.pretty())

    print("\n=== unanimous strategy: not unanimous, fails ===")
    workflow2 = Workflow("StrictVote")
    workflow2.consensus(model_a, model_b, model_c, strategy=unanimous)
    journal2 = workflow2.run()
    print(f"status: {journal2.status}")
    print(f"error: {journal2.steps[-1].error}")

    print("\n=== a genuine tie under majority: fails rather than guessing ===")

    @tool
    def voter_x():
        return "A"

    @tool
    def voter_y():
        return "B"

    workflow3 = Workflow("Tie")
    workflow3.consensus(voter_x, voter_y)  # 1-1, no majority possible
    journal3 = workflow3.run()
    print(f"status: {journal3.status}")
    print(f"error: {journal3.steps[-1].error}")

    print("\n=== if any voter fails, the whole block fails -- no aggregation attempted ===")

    @tool
    def flaky_voter():
        raise RuntimeError("model timeout")

    workflow4 = Workflow("VoterFails")
    workflow4.consensus(model_a, flaky_voter)
    journal4 = workflow4.run()
    print(f"status: {journal4.status}")
    print(f"steps recorded: {[s.tool for s in journal4.steps]}")  # no synthetic 'consensus' step
