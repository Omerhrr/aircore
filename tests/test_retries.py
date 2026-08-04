import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool


def test_non_idempotent_tool_cannot_declare_retries():
    try:
        @tool(idempotent=False, retries=2)
        def unsafe():
            return "x"
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "idempotent" in str(exc)


def test_idempotent_tool_retries_default_zero_means_no_retry():
    calls = []

    @tool(idempotent=True)  # retries defaults to 0
    def fails():
        calls.append(1)
        raise RuntimeError("boom")

    workflow = Workflow("NoRetryByDefault")
    workflow.step(fails)
    journal = workflow.run()

    assert journal.status == "failed"
    assert len(calls) == 1  # never retried despite idempotent=True
    assert journal.steps[0].retries == 0


def test_idempotent_tool_recovers_within_retry_budget():
    calls = {"n": 0}

    @tool(idempotent=True, retries=3)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError(f"fail #{calls['n']}")
        return "ok"

    workflow = Workflow("Recovers")
    workflow.step(flaky)
    journal = workflow.run()

    assert journal.status == "success"
    assert calls["n"] == 3
    assert journal.steps[0].retries == 2
    assert len(journal.steps[0].retry_errors) == 2
    assert journal.steps[0].output == "ok"


def test_idempotent_tool_exhausts_retry_budget_and_fails():
    calls = {"n": 0}

    @tool(idempotent=True, retries=2)
    def always_fails():
        calls["n"] += 1
        raise RuntimeError(f"fail #{calls['n']}")

    workflow = Workflow("Exhausted")
    workflow.step(always_fails)
    journal = workflow.run()

    assert journal.status == "failed"
    assert calls["n"] == 3  # 1 initial + 2 retries
    assert journal.steps[0].retries == 2
    assert "fail #3" in journal.steps[0].error


def test_metrics_count_retries():
    calls = {"n": 0}

    @tool(idempotent=True, retries=5)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("once")
        return "ok"

    workflow = Workflow("MetricsRetries")
    workflow.step(flaky)
    workflow.run()

    assert workflow.metrics.retries_total == 1
    assert workflow.metrics.by_tool["flaky"].retries == 1


def test_capability_denial_is_never_retried_even_with_retries_set():
    from aircore import Agent, Email

    calls = []

    @tool(idempotent=True, retries=5, requires=Email)
    def send():
        calls.append(1)
        return "sent"

    agent = Agent("Bot", capabilities=[])  # lacks Email
    workflow = Workflow("DeniedNotRetried")
    workflow.step(send, agent=agent)
    journal = workflow.run()

    assert journal.status == "failed"
    assert calls == [], "tool body ran despite capability denial"
    assert journal.steps[0].retries == 0
    assert "CapabilityDenied" in journal.steps[0].error


if __name__ == "__main__":
    test_non_idempotent_tool_cannot_declare_retries()
    test_idempotent_tool_retries_default_zero_means_no_retry()
    test_idempotent_tool_recovers_within_retry_budget()
    test_idempotent_tool_exhausts_retry_budget_and_fails()
    test_metrics_count_retries()
    test_capability_denial_is_never_retried_even_with_retries_set()
    print("All retry tests passed.")
