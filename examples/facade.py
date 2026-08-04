"""airpy facade acceptance example: the whole point is that this file
never imports aircore. `from airpy import Agent, Workflow` is enough to
build the exact "researcher/reviewer/professor -> consensus" shape from
the roadmap discussion -- aircore still does all the actual scheduling,
capability checks, and journaling underneath, but nothing here needs to
know that.

Compare with examples/airpy_agent.py (predates the facade, imports
ModelAgent/Workflow from aircore/airpy separately) -- same behavior, this
is just the version people should actually write.

Run with: python examples/facade.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import majority  # noqa: the one thing not yet re-exported -- see airpy/__init__.py's facade note

from airpy import Agent, MockProvider, Workflow


if __name__ == "__main__":
    researcher = Agent(
        "researcher",
        MockProvider(response="Q3 revenue grew 12% year over year, driven by cloud."),
        "Summarize Q3 revenue performance.",
    )
    reviewer = Agent(
        "reviewer",
        MockProvider(response="Q3 revenue grew 12% year over year, driven by cloud."),
        "Independently summarize Q3 revenue performance.",
    )
    professor = Agent(
        "professor",
        MockProvider(response="Q3 revenue grew 12% year over year, driven by cloud."),
        "Grade and restate the Q3 revenue summary.",
    )

    workflow = (
        Workflow("Audit")
        .parallel(researcher, reviewer, professor)
        .consensus(strategy=majority)
    )

    journal = workflow.run()

    print(journal.pretty())
    print()
    print(f"agreed answer: {journal.steps[-1].output!r}")
