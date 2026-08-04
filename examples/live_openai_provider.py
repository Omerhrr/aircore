"""Live validation of OpenAIProvider (airpy/openai_provider.py) -- the
first native (non-LiteLLM) ModelProvider adapter -- against a real API.

Uses DeepSeek's OpenAI-compatible endpoint (https://api.deepseek.com)
rather than real OpenAI, so this reuses the DEEPSEEK_API_KEY this project
already has live-tested credentials for -- no new API key needed to prove
OpenAIProvider actually works against a real network call, not just the
faked-`openai`-module offline tests in tests/test_openai_provider.py.

This is deliberately NOT part of the automated test suite (tests/), same
reason examples/production_readiness.py and examples/live_deepseek.py
aren't: needs real network access and a real API key, run manually on
your own machine.

Setup:
    pip install openai
    export DEEPSEEK_API_KEY=sk-...           (macOS/Linux)
    setx DEEPSEEK_API_KEY "sk-..."            (Windows, then reopen terminal)

Run:
    python examples/live_openai_provider.py

What this proves:
  1. OpenAIProvider.generate() actually completes a real chat completion
     through the `openai` SDK's client, talking to DeepSeek's
     OpenAI-compatible endpoint via base_url= -- not litellm underneath.
  2. Real usage (prompt/completion/total tokens) comes back from a real
     response and flows into ModelAgent.usage() the same generic way
     LiteLLMProvider's does -- except cost_usd is honestly None (the
     openai SDK has no pricing lookup; see openai_provider.py's
     docstring), not a guessed number.
  3. Tool-calling works end to end through this adapter -- a second run
     asks a question that requires calling a real aircore Tool, proving the
     wire-format round-trip (assistant tool_calls fed back correctly)
     works against a real API, not just the offline fake in
     tests/test_openai_provider.py.
"""

import os
import sys

if not os.environ.get("DEEPSEEK_API_KEY"):
    print("DEEPSEEK_API_KEY is not set. Set it and re-run:")
    print('  export DEEPSEEK_API_KEY=sk-...        (macOS/Linux)')
    print('  setx DEEPSEEK_API_KEY "sk-..."         (Windows, then reopen terminal)')
    sys.exit(1)

from aircore import Workflow, tool
from airpy import ModelAgent, OpenAIProvider

provider = OpenAIProvider(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


def part1_plain_completion():
    print("--- Part 1: plain completion via OpenAIProvider -> DeepSeek ---")
    agent = ModelAgent("bot", provider, "In one sentence, what is a Bloom filter?")
    workflow = Workflow("PlainCompletion").step(agent)
    journal = workflow.run()

    print(f"status: {journal.status}")
    print(f"output: {journal.steps[0].output}")
    usage = agent.usage()
    print(f"usage: {usage}")
    assert journal.status == "success"
    assert usage is not None
    assert usage["tokens_in"] is not None and usage["tokens_out"] is not None
    # usage() (model_agent.py) drops any None value entirely rather than
    # reporting it -- so the honest "no pricing lookup in the openai SDK"
    # gap (see openai_provider.py's docstring) shows up as cost_usd being
    # *absent* from this dict, not present as None. Checking the
    # underlying ModelResponse.usage.cost_usd directly is what actually
    # proves it's None rather than a guessed number.
    assert "cost_usd" not in usage
    assert agent.last_response.usage.cost_usd is None
    print("OK: real completion + real token usage, honestly-absent cost_usd\n")


@tool
def get_capital(country: str) -> str:
    capitals = {"Nigeria": "Abuja", "France": "Paris", "Japan": "Tokyo"}
    return capitals.get(country, "unknown")


def part2_tool_calling():
    print("--- Part 2: tool-calling round trip via OpenAIProvider -> DeepSeek ---")
    agent = ModelAgent(
        "geo_bot", provider,
        prompt="What is the capital of Nigeria? Use the get_capital tool.",
        tools=[get_capital],
    )
    workflow = Workflow("ToolCalling").step(agent)
    journal = workflow.run()

    print(f"status: {journal.status}")
    print(f"output: {journal.steps[0].output}")
    print(f"tool calls made: {[c.name for c in agent.tool_call_log]}")
    assert journal.status == "success"
    assert len(agent.tool_call_log) >= 1
    assert agent.tool_call_log[0].name == "get_capital"
    assert agent.tool_call_log[0].result == "Abuja"
    print("OK: real tool-call round trip through OpenAIProvider\n")


if __name__ == "__main__":
    part1_plain_completion()
    part2_tool_calling()
    print("All live OpenAIProvider checks passed.")
