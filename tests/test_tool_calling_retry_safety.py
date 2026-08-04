"""Proves the retry/tool-replay gap flagged after the tool-calling loop
shipped is actually fixed: a ModelAgent with tools + retries never lets
the Scheduler blindly re-run the whole loop (which would replay
already-succeeded, possibly side-effecting tool calls), but transient
model-call failures within a single turn are still safely retried.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool
from airpy import ModelAgent, MockProvider, ModelResponse, ToolCallRequest


def _tool_call_response(name, arguments, call_id="c1"):
    return ModelResponse(content="", tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments)])


def test_scheduler_visible_idempotent_and_retries_forced_when_tools_present():
    @tool
    def noop():
        return "x"

    agent = ModelAgent("m", MockProvider(response="ok"), prompt="p",
                        idempotent=True, retries=5, tools=[noop])

    # what the Scheduler actually sees and would use to decide whether to
    # retry the whole execute() call
    assert agent.idempotent is False
    assert agent.retries == 0
    # the developer's actual values are preserved, just repurposed
    assert agent._loop_call_idempotent is True
    assert agent._loop_call_retries == 5


def test_no_tools_retry_semantics_completely_unchanged():
    agent = ModelAgent("m", MockProvider(response="ok"), prompt="p", idempotent=True, retries=3)
    assert agent.idempotent is True
    assert agent.retries == 3


def test_scheduler_never_replays_already_executed_tool_call_on_failure():
    """The core bug: turn 1's tool call succeeds, turn 2's model call fails
    with a hard exception. Before the fix, a Scheduler-level retry would
    call execute() again from scratch, re-invoking the turn-1 tool. After
    the fix, the Scheduler doesn't retry the whole loop at all (idempotent
    is forced False), so the failure just surfaces once and the tool was
    called exactly one time."""
    calls = {"count": 0}

    @tool
    def side_effecting_tool():
        calls["count"] += 1
        return f"call #{calls['count']}"

    class FlakyAfterFirstToolCall(MockProvider):
        def __init__(self):
            super().__init__()
            self._turn = 0

        def generate(self, request):
            self._turn += 1
            if self._turn == 1:
                return _tool_call_response("side_effecting_tool", {})
            raise ConnectionError("network blip on turn 2")

    agent = ModelAgent("m", FlakyAfterFirstToolCall(), prompt="p",
                        idempotent=True, retries=3, tools=[side_effecting_tool])

    workflow = Workflow("ReplaySafety")
    workflow.step(agent)
    journal = workflow.run()

    assert journal.status == "failed"
    # the crucial assertion: the side-effecting tool was called exactly
    # once, not replayed by a Scheduler-level retry of the whole loop
    assert calls["count"] == 1
    assert len(agent.tool_call_log) == 1


def test_transient_model_call_failure_within_a_turn_is_retried_safely():
    """A failure that happens *before* any tool call for that turn (i.e.
    the model call itself fails) is safe to retry, since nothing side-
    effecting happened yet -- this should still work, not be blocked by
    the fix above."""
    call_attempts = {"count": 0}

    class FlakyThenRecovers(MockProvider):
        def generate(self, request):
            call_attempts["count"] += 1
            if call_attempts["count"] < 3:
                raise ConnectionError("transient")
            return ModelResponse(content="final answer, no tool needed", model=request.model)

    @tool
    def unused_tool():
        return "never called"

    agent = ModelAgent("m", FlakyThenRecovers(), prompt="p",
                        idempotent=True, retries=5, tools=[unused_tool])

    result = agent.execute()

    assert result == "final answer, no tool needed"
    assert call_attempts["count"] == 3  # 2 failures + 1 success, all within one turn


def test_transient_failure_exhausts_retry_budget_and_raises():
    class AlwaysFails(MockProvider):
        def generate(self, request):
            raise ConnectionError("always broken")

    @tool
    def unused_tool():
        return "x"

    agent = ModelAgent("m", AlwaysFails(), prompt="p", idempotent=True, retries=2, tools=[unused_tool])

    try:
        agent.execute()
        assert False, "expected ConnectionError to propagate"
    except ConnectionError:
        pass


def test_workflow_level_retries_kwarg_on_step_does_not_resurrect_the_bug():
    """Sanity check at the Workflow level: even if a developer explicitly
    asks for retries via the agent's constructor, running it inside a
    workflow doesn't somehow let the whole loop get replayed."""
    calls = {"count": 0}

    @tool
    def paying_tool():
        calls["count"] += 1
        return "charged"

    class FailsOnSecondCall(MockProvider):
        def __init__(self):
            super().__init__()
            self._turn = 0

        def generate(self, request):
            self._turn += 1
            if self._turn == 1:
                return _tool_call_response("paying_tool", {})
            raise TimeoutError("provider timeout")

    agent = ModelAgent("billing_bot", FailsOnSecondCall(), prompt="charge the customer",
                        idempotent=True, retries=10, tools=[paying_tool])

    workflow = Workflow("Billing")
    workflow.step(agent)
    workflow.run()

    assert calls["count"] == 1, "a side-effecting tool was replayed by a retry"


if __name__ == "__main__":
    test_scheduler_visible_idempotent_and_retries_forced_when_tools_present()
    test_no_tools_retry_semantics_completely_unchanged()
    test_scheduler_never_replays_already_executed_tool_call_on_failure()
    test_transient_model_call_failure_within_a_turn_is_retried_safely()
    test_transient_failure_exhausts_retry_budget_and_raises()
    test_workflow_level_retries_kwarg_on_step_does_not_resurrect_the_bug()
    print("All retry-safety tests passed.")
