"""M8: real provider integration + usage flowing into Metrics/Policy.

LiteLLMProvider is tested by injecting a fake `litellm` module into
sys.modules before construction -- this needs no API key, no network call,
and no real litellm install, per the "keep unit tests independent of
external services" rule. It also means these tests run the same whether
or not the real litellm package happens to be installed.
"""

import sys
import os
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, Policy, PolicyViolation
from airpy import ModelAgent, MockProvider


def _install_fake_litellm(content="mocked completion", prompt_tokens=10, completion_tokens=5, cost=0.001):
    """Builds a fake litellm module shaped enough like the real one for
    LiteLLMProvider to parse, and installs it into sys.modules so `import
    litellm` inside LiteLLMProvider resolves to this instead of the real
    package (which isn't installed in this environment, and shouldn't be
    required for tests to pass regardless)."""
    fake = types.ModuleType("litellm")

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, content):
            self.message = FakeMessage(content)
            self.finish_reason = "stop"

    class FakeUsage:
        def __init__(self, pt, ct):
            self.prompt_tokens = pt
            self.completion_tokens = ct
            self.total_tokens = pt + ct

    class FakeResponse:
        def __init__(self, content, model):
            self.choices = [FakeChoice(content)]
            self.usage = FakeUsage(prompt_tokens, completion_tokens)
            self.model = model

    def fake_completion(model, messages, **kwargs):
        return FakeResponse(content, model)

    def fake_completion_cost(completion_response=None):
        return cost

    fake.completion = fake_completion
    fake.completion_cost = fake_completion_cost
    sys.modules["litellm"] = fake
    return fake


def _uninstall_fake_litellm():
    sys.modules.pop("litellm", None)


def test_litellm_provider_generates_response_from_fake_module():
    _install_fake_litellm(content="hello from a fake model")
    try:
        from airpy import LiteLLMProvider
        provider = LiteLLMProvider(model="gpt-4o-mini")
        from airpy.providers import ModelRequest
        response = provider.generate(ModelRequest(prompt="hi"))

        assert response.content == "hello from a fake model"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15
        assert response.usage.cost_usd == 0.001
        assert response.finish_reason == "stop"
        assert response.raw is not None
    finally:
        _uninstall_fake_litellm()


def test_litellm_provider_missing_package_raises_clear_error():
    _uninstall_fake_litellm()  # make sure it's really absent
    # Force a real ImportError by ensuring litellm truly isn't importable
    # in this environment (it isn't -- confirmed not installed).
    from airpy import LiteLLMProvider
    try:
        LiteLLMProvider()
        # If litellm happens to be installed in some environment this runs
        # in, we can't assert failure -- skip the assertion in that case.
        import importlib
        if importlib.util.find_spec("litellm") is None:
            assert False, "expected ImportError"
    except ImportError as exc:
        assert "pip install litellm" in str(exc)


def test_model_agent_with_litellm_reports_usage_via_scheduler():
    _install_fake_litellm(content="answer", prompt_tokens=100, completion_tokens=50, cost=0.01)
    try:
        from airpy import LiteLLMProvider
        agent = ModelAgent("researcher", LiteLLMProvider(model="gpt-4o-mini"), prompt="research this")

        workflow = Workflow("Real")
        workflow.step(agent)
        journal = workflow.run()

        assert journal.status == "success"
        step = journal.steps[0]
        assert step.output == "answer"
        assert step.usage is not None
        assert step.usage["tokens_in"] == 100
        assert step.usage["tokens_out"] == 50
        assert step.usage["cost_usd"] == 0.01

        assert workflow.metrics.usage_totals["cost_usd"] == 0.01
        assert workflow.metrics.usage_totals["tokens_in"] == 100
        assert workflow.metrics.by_tool["researcher"].usage_totals["cost_usd"] == 0.01
    finally:
        _uninstall_fake_litellm()


def test_mock_provider_reports_no_usage():
    """MockProvider never sets usage fields -- ModelAgent.usage() should
    return None, and the scheduler should never emit UsageReported, so
    Metrics.usage_totals stays empty. Confirms the mock/real distinction
    the M8 design insisted on: don't invent numbers."""
    agent = ModelAgent("m", MockProvider(response="x"), prompt="p")
    workflow = Workflow("Mocked")
    workflow.step(agent)
    journal = workflow.run()

    assert journal.steps[0].usage is None
    assert workflow.metrics.usage_totals == {}


def test_policy_max_cost_stops_workflow_before_exceeding_budget():
    _install_fake_litellm(content="expensive answer", cost=10.0)
    try:
        from airpy import LiteLLMProvider

        agent1 = ModelAgent("a1", LiteLLMProvider(), prompt="p1")
        agent2 = ModelAgent("a2", LiteLLMProvider(), prompt="p2")

        workflow = Workflow("Budgeted", policy=Policy(max_cost=5.0))
        workflow.step(agent1)  # costs 10.0, already exceeds the 5.0 budget
        workflow.step(agent2)  # should never run
        journal = workflow.run()

        assert journal.status == "failed"
        assert len(journal.steps) == 1  # agent2's step never started
        assert journal.steps[0].tool == "a1"
    finally:
        _uninstall_fake_litellm()


def test_policy_max_cost_none_means_unbounded():
    _install_fake_litellm(content="x", cost=1000.0)
    try:
        from airpy import LiteLLMProvider
        agent = ModelAgent("a", LiteLLMProvider(), prompt="p")
        workflow = Workflow("Unbounded", policy=Policy())  # max_cost=None
        workflow.step(agent)
        journal = workflow.run()
        assert journal.status == "success"
    finally:
        _uninstall_fake_litellm()


def test_metrics_usage_totals_empty_when_nothing_reports():
    from aircore import tool

    @tool
    def plain():
        return "ok"

    workflow = Workflow("Plain")
    workflow.step(plain)
    workflow.run()
    assert workflow.metrics.usage_totals == {}


if __name__ == "__main__":
    test_litellm_provider_generates_response_from_fake_module()
    test_litellm_provider_missing_package_raises_clear_error()
    test_model_agent_with_litellm_reports_usage_via_scheduler()
    test_mock_provider_reports_no_usage()
    test_policy_max_cost_stops_workflow_before_exceeding_budget()
    test_policy_max_cost_none_means_unbounded()
    test_metrics_usage_totals_empty_when_nothing_reports()
    print("All M8 tests passed.")
