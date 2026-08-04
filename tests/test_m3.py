import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool, Agent, Policy, PolicyViolation, Network


def test_require_agent_false_allows_anonymous_step():
    @tool
    def noop():
        return "ok"

    workflow = Workflow("Dev", policy=Policy(require_agent=False))
    workflow.step(noop)
    journal = workflow.run()
    assert journal.status == "success"


def test_require_agent_true_rejects_anonymous_step_preflight():
    @tool
    def noop():
        return "ok"

    workflow = Workflow("Prod", policy=Policy(require_agent=True))
    workflow.step(noop)

    try:
        workflow.run()
        assert False, "expected PolicyViolation"
    except PolicyViolation as exc:
        assert "noop" in str(exc)
        assert "require_agent" in str(exc)


def test_require_agent_true_nothing_executes_on_violation():
    calls = []

    @tool
    def should_not_run():
        calls.append("ran")
        return "ok"

    workflow = Workflow("Prod", policy=Policy(require_agent=True))
    workflow.step(should_not_run)
    try:
        workflow.run()
    except PolicyViolation:
        pass
    assert calls == [], "tool executed despite pre-flight PolicyViolation"


def test_require_agent_true_passes_with_agent():
    @tool
    def noop():
        return "ok"

    bot = Agent("Bot", capabilities=[Network])
    workflow = Workflow("Prod", policy=Policy(require_agent=True))
    workflow.step(noop, agent=bot)
    journal = workflow.run()
    assert journal.status == "success"


def test_require_agent_checks_parallel_group_members_too():
    @tool
    def a():
        return "a"

    @tool
    def b():
        return "b"

    bot = Agent("Bot")
    workflow = Workflow("Prod", policy=Policy(require_agent=True))
    workflow.parallel(a, b)  # no agent
    try:
        workflow.run()
        assert False, "expected PolicyViolation"
    except PolicyViolation:
        pass


def test_max_parallel_rejects_oversized_group_preflight():
    @tool
    def a():
        return "a"

    @tool
    def b():
        return "b"

    @tool
    def c():
        return "c"

    workflow = Workflow("Capped", policy=Policy(max_parallel=2))
    workflow.parallel(a, b, c)
    try:
        workflow.run()
        assert False, "expected PolicyViolation"
    except PolicyViolation as exc:
        assert "max_parallel" in str(exc)


def test_max_parallel_allows_group_at_the_limit():
    @tool
    def a():
        return "a"

    @tool
    def b():
        return "b"

    workflow = Workflow("AtLimit", policy=Policy(max_parallel=2))
    workflow.parallel(a, b)
    journal = workflow.run()
    assert journal.status == "success"


def test_max_runtime_stops_subsequent_steps():
    @tool
    def slow():
        time.sleep(0.1)
        return "slow-done"

    @tool
    def fast():
        return "fast-done"

    workflow = Workflow("Timeout", policy=Policy(max_runtime=0.05))
    workflow.step(slow)
    workflow.step(fast)
    journal = workflow.run()

    assert journal.status == "failed"
    # the first (slow) step still ran to completion -- it wasn't interrupted
    assert journal.steps[0].status == "success"
    # the second step never started because the deadline had already passed
    assert len(journal.steps) == 1


def test_no_policy_means_fully_permissive_default():
    @tool
    def noop():
        return "ok"

    workflow = Workflow("NoPolicySpecified")  # default Policy()
    workflow.step(noop)
    journal = workflow.run()
    assert journal.status == "success"


if __name__ == "__main__":
    test_require_agent_false_allows_anonymous_step()
    test_require_agent_true_rejects_anonymous_step_preflight()
    test_require_agent_true_nothing_executes_on_violation()
    test_require_agent_true_passes_with_agent()
    test_require_agent_checks_parallel_group_members_too()
    test_max_parallel_rejects_oversized_group_preflight()
    test_max_parallel_allows_group_at_the_limit()
    test_max_runtime_stops_subsequent_steps()
    test_no_policy_means_fully_permissive_default()
    print("All M3 tests passed.")
