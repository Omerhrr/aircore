"""Live validation against a real provider: DeepSeek, via LiteLLM.

This is deliberately NOT part of the automated test suite (tests/) --
it needs real network access and a real API key, which violates the
"keep unit tests independent of external services" rule the M8 design
settled on. Run it manually, on your own machine (not in a sandboxed
environment with restricted network access -- that's why this couldn't
be run from within the assistant's sandbox).

Setup:
    pip install litellm
    export DEEPSEEK_API_KEY=sk-...           (macOS/Linux)
    setx DEEPSEEK_API_KEY "sk-..."            (Windows, then reopen terminal)

Run:
    python examples/live_deepseek.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not os.environ.get("DEEPSEEK_API_KEY"):
    print("DEEPSEEK_API_KEY is not set. Set it and re-run:")
    print("  export DEEPSEEK_API_KEY=sk-...")
    sys.exit(1)

from aircore import Workflow, Policy, tool
from airpy import ModelAgent, LiteLLMProvider


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


if __name__ == "__main__":
    provider = LiteLLMProvider(model="deepseek/deepseek-chat")

    section("1. Single-shot call through a Workflow -- real usage/cost in the journal")
    researcher = ModelAgent("researcher", provider, prompt="In one sentence, what is a Bloom filter?")
    workflow = Workflow("LiveSingleShot")
    workflow.step(researcher)
    journal = workflow.run()
    print(journal.pretty())
    print(workflow.metrics.summary())

    section("2. Tool-calling loop -- does DeepSeek actually decide to call a tool?")

    @tool(description="Look up the current population of a city from a small fixed dataset")
    def get_population(city: str):
        data = {"Lagos": 15_400_000, "Tokyo": 13_960_000, "Nairobi": 4_400_000}
        return data.get(city, "unknown city")

    agent = ModelAgent(
        "geo_bot", provider,
        prompt="What is the population of Lagos? Use the get_population tool to find out, "
               "don't guess.",
        tools=[get_population],
    )
    answer = agent.execute()
    print(f"Final answer: {answer!r}")
    print(f"Tool calls DeepSeek actually made: "
          f"{[(r.name, r.arguments, r.result) for r in agent.tool_call_log]}")
    if not agent.tool_call_log:
        print("NOTE: the model answered without calling the tool -- worth checking whether "
              "deepseek-chat reliably uses function calling for this kind of prompt, or "
              "whether the prompt needs to be more directive.")

    section("3. Policy.max_cost against a real cost figure")
    budgeted = Workflow("LiveBudget", policy=Policy(max_cost=0.10))
    budgeted.step(ModelAgent("a1", provider, prompt="Say hello in French."))
    budgeted.step(ModelAgent("a2", provider, prompt="Say hello in German."))
    budgeted_journal = budgeted.run()
    print(f"status: {budgeted_journal.status}")
    print(f"steps that ran: {[s.tool for s in budgeted_journal.steps]}")
    print(f"total real cost: ${budgeted.metrics.usage_totals.get('cost_usd', 0):.6f}")
    if "cost_usd" not in budgeted.metrics.usage_totals:
        print("NOTE: DeepSeek's response didn't yield a cost figure via litellm.completion_cost() "
              "-- LiteLLMProvider._extract_cost() returned None, which is a real, expected "
              "possibility documented in litellm_provider.py, not a bug.")
