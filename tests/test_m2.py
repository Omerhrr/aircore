import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool, Agent, Capability, Network, Email, CapabilityDenied


def test_agent_with_capability_can_call_tool():
    @tool(requires=Network)
    def fetch():
        return "ok"

    agent = Agent("Bot", capabilities=[Network])
    workflow = Workflow("Allowed")
    workflow.step(fetch, agent=agent)
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.steps[0].output == "ok"


def test_agent_without_capability_is_denied():
    @tool(requires=Email)
    def send():
        return "sent"

    agent = Agent("Bot", capabilities=[Network])  # no Email
    workflow = Workflow("Denied")
    workflow.step(send, agent=agent)
    journal = workflow.run()

    assert journal.status == "failed"
    assert journal.steps[0].status == "failed"
    assert "CapabilityDenied" in journal.steps[0].error
    assert "Email" in journal.steps[0].error


def test_tool_never_called_when_denied():
    calls = []

    @tool(requires=Email)
    def send():
        calls.append("called")
        return "sent"

    agent = Agent("Bot", capabilities=[])
    workflow = Workflow("NeverCalled")
    workflow.step(send, agent=agent)
    workflow.run()

    assert calls == [], "tool body ran despite missing capability"


def test_no_agent_means_unrestricted():
    @tool(requires=Email)
    def send():
        return "sent"

    workflow = Workflow("Unrestricted")
    workflow.step(send)  # no agent given at all
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.steps[0].output == "sent"


def test_tool_without_capability_requirement_always_allowed():
    @tool  # no capability declared
    def noop():
        return "fine"

    agent = Agent("Bot", capabilities=[])  # holds nothing
    workflow = Workflow("NoRequirement")
    workflow.step(noop, agent=agent)
    journal = workflow.run()

    assert journal.status == "success"


def test_capability_enforced_inside_parallel_group():
    @tool(requires=Network)
    def allowed():
        return "a"

    @tool(requires=Email)
    def denied():
        return "b"

    agent = Agent("Bot", capabilities=[Network])
    workflow = Workflow("ParallelCap")
    workflow.parallel(allowed, denied, agent=agent)
    journal = workflow.run()

    assert journal.status == "failed"
    assert journal.groups[0].status == "failed"
    statuses = {s.tool: s.status for s in journal.steps}
    assert statuses["allowed"] == "success"
    assert statuses["denied"] == "failed"


def test_custom_capability_by_name_equality():
    GPU = Capability("GPU")
    same_gpu = Capability("GPU")
    assert GPU == same_gpu
    assert hash(GPU) == hash(same_gpu)

    agent = Agent("Bot", capabilities=[same_gpu])
    assert agent.grants(GPU)


if __name__ == "__main__":
    test_agent_with_capability_can_call_tool()
    test_agent_without_capability_is_denied()
    test_tool_never_called_when_denied()
    test_no_agent_means_unrestricted()
    test_tool_without_capability_requirement_always_allowed()
    test_capability_enforced_inside_parallel_group()
    test_custom_capability_by_name_equality()
    print("All M2 tests passed.")
