"""Executable/airpy acceptance example: ModelAgent is a drop-in Executable
alongside Tool -- same workflow.step/parallel/consensus, same capability
and policy enforcement, with zero changes to aircore. Uses MockProvider, so
this needs no API key and makes no real network calls.

Run with: python examples/airpy_agent.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool, Agent, Network
from airpy import ModelAgent, MockProvider, ask


if __name__ == "__main__":
    print("=== A Tool and a ModelAgent as siblings in the same sequential workflow ===")

    @tool
    def fetch_document():
        return "Q3 revenue grew 12% year over year."

    researcher = ModelAgent(
        name="summarizer",
        provider=MockProvider(response=lambda req: f"Summary of: {req.prompt[:30]}..."),
        prompt="Q3 revenue grew 12% year over year.",
    )

    workflow = Workflow("Mixed")
    workflow.step(fetch_document)
    workflow.step(researcher)
    journal = workflow.run()
    print(journal.pretty())

    print("\n=== Three ModelAgents voting via consensus -- same primitive Tools use ===")
    model_yes_1 = ModelAgent("gpt", MockProvider(response="approve"), prompt="Should we ship?")
    model_yes_2 = ModelAgent("claude", MockProvider(response="approve"), prompt="Should we ship?")
    model_no = ModelAgent("gemini", MockProvider(response="reject"), prompt="Should we ship?")

    vote = Workflow("ShipDecision")
    vote.consensus(model_yes_1, model_yes_2, model_no)
    vote_journal = vote.run()
    print(f"agreed decision: {vote_journal.steps[-1].output!r}")

    print("\n=== Capability enforcement applies to ModelAgent exactly like Tool ===")
    restricted_agent = ModelAgent("web_researcher", MockProvider(response="fetched"),
                                   prompt="search the web", requires=Network)
    no_network = Agent("Sandboxed", capabilities=[])  # no Network capability
    restricted_workflow = Workflow("Restricted")
    restricted_workflow.step(restricted_agent, agent=no_network)
    restricted_journal = restricted_workflow.run()
    print(f"status: {restricted_journal.status}")
    print(f"error: {restricted_journal.steps[0].error}")

    print("\n=== ask(): quick one-off call, no workflow needed ===")
    quick_agent = ModelAgent("quick", MockProvider(response="42"), prompt="What is the answer?")
    print(f"ask(quick_agent) = {ask(quick_agent)!r}")

    raw_provider = MockProvider(response=lambda req: f"echo: {req.prompt}")
    print(f"ask(raw_provider, prompt='hi') = {ask(raw_provider, prompt='hi')!r}")
