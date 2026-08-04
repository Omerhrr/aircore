"""M2 acceptance example: an agent without the Email capability gets its
tool call rejected before the tool ever runs.

Run with: python examples/capabilities.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool, Agent, Network, Email


@tool(requires=Network)
def fetch_page():
    return "<html>...</html>"


@tool(requires=Email)
def send_email():
    print("  (this should never print -- the agent has no Email capability)")
    return "sent"


if __name__ == "__main__":
    researcher = Agent("Researcher", capabilities=[Network])

    workflow = Workflow("CapabilityDemo")
    workflow.step(fetch_page, agent=researcher)
    workflow.step(send_email, agent=researcher)  # denied: no Email capability

    journal = workflow.run()
    print(journal.pretty())
