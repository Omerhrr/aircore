"""M8 acceptance example: LiteLLMProvider populates real usage/cost, which
flows into the journal and Metrics automatically, and Policy.max_cost
actually stops a workflow before it overspends.

This example injects a fake `litellm` module so it runs with no API key
and no network call -- see tests/test_m8.py for why. To use a real
provider, just remove the fake-module setup below and make sure `pip
install litellm` has been run and your provider's API key is set as an
environment variable (e.g. OPENAI_API_KEY).

Run with: python examples/litellm_provider.py
"""

import sys
import os
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_fake_litellm():
    fake = types.ModuleType("litellm")

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, content):
            self.message = FakeMessage(content)
            self.finish_reason = "stop"

    class FakeUsage:
        def __init__(self):
            self.prompt_tokens = 120
            self.completion_tokens = 40
            self.total_tokens = 160

    class FakeResponse:
        def __init__(self, content, model):
            self.choices = [FakeChoice(content)]
            self.usage = FakeUsage()
            self.model = model

    def fake_completion(model, messages, **kwargs):
        prompt = messages[0]["content"]
        return FakeResponse(f"(simulated) answer to: {prompt}", model)

    def fake_completion_cost(completion_response=None):
        return 0.003

    fake.completion = fake_completion
    fake.completion_cost = fake_completion_cost
    sys.modules["litellm"] = fake


_install_fake_litellm()

from aircore import Workflow, Policy  # noqa: E402
from airpy import ModelAgent, LiteLLMProvider  # noqa: E402


if __name__ == "__main__":
    print("=== Real usage flows into the journal and Metrics automatically ===")
    researcher = ModelAgent("researcher", LiteLLMProvider(model="gpt-4o-mini"),
                             prompt="What are the main risks in this contract?")
    workflow = Workflow("Research")
    workflow.step(researcher)
    journal = workflow.run()
    print(journal.pretty())
    print(workflow.metrics.summary())

    print("\n=== Policy.max_cost stops a workflow before it overspends ===")
    agent1 = ModelAgent("call1", LiteLLMProvider(), prompt="first call")
    agent2 = ModelAgent("call2", LiteLLMProvider(), prompt="second call")
    agent3 = ModelAgent("call3", LiteLLMProvider(), prompt="third call")

    budgeted = Workflow("Budgeted", policy=Policy(max_cost=0.005))  # ~1.6 calls at $0.003 each
    budgeted.step(agent1)
    budgeted.step(agent2)
    budgeted.step(agent3)
    budgeted_journal = budgeted.run()
    print(f"status: {budgeted_journal.status}")
    print(f"steps that actually ran: {[s.tool for s in budgeted_journal.steps]}")
    print(f"total cost spent: ${budgeted.metrics.usage_totals.get('cost_usd', 0):.3f}")
