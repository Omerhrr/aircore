"""ModelAgent.stream(): text chunks as they arrive, instead of blocking
for the whole completion. Runs offline against MockProvider (which
chunks its fixed response the same way a real streaming API would);
LiteLLMProvider.stream() does real token streaming against a live model.

Note what this is NOT: a workflow step. stream() bypasses aircore's
Scheduler the same way ask() does -- see model_agent.py's docstring for
why token-level streaming doesn't fit atomic per-step journaling.

Run with: python examples/streaming.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airpy import ModelAgent, MockProvider


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


if __name__ == "__main__":
    section("Iterating chunks directly")

    agent = ModelAgent(
        "assistant", MockProvider(response="A bloom filter is a space-efficient probabilistic structure."),
        prompt="Explain a bloom filter in one sentence.",
    )
    for chunk in agent.stream():
        print(chunk, end="", flush=True)
    print()
    print(f"(last_response.content == stream output: {agent.last_response.content!r})")

    section("Using on_token instead of iterating")

    collected = []
    agent2 = ModelAgent("assistant", MockProvider(response="Streaming works either way."), prompt="Say something.")
    # Consume the generator (on_token fires as a side effect of iterating
    # it -- the generator still has to be driven to completion).
    for _ in agent2.stream(on_token=collected.append):
        pass
    print("collected via on_token:", collected)

    section("Streaming is not available once tools= is set")

    from aircore import Tool

    lookup = Tool(lambda: "42", name="lookup")
    agent3 = ModelAgent("assistant", MockProvider(), prompt="use the tool", tools=[lookup])
    try:
        agent3.stream()
    except NotImplementedError as exc:
        print(f"raised immediately, as expected: {exc}")
