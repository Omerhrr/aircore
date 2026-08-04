"""ModelAgent(memory=..., conversation_id=...): reusing aircore's existing
Memory primitive (MemoryScope.get/set) so repeated execute() calls build a
real multi-turn conversation instead of starting blank every time. No
aircore changes were needed for this -- ModelAgent just uses the same
get/set contract every Tool already can.
"""

import pytest

from aircore import Memory, Workflow
from airpy import MockProvider, ModelAgent
from airpy.providers import ModelResponse, ToolCallRequest


def test_without_memory_behavior_is_unchanged():
    provider = MockProvider(response="hello")
    agent = ModelAgent("a", provider, prompt="hi")
    assert agent.execute() == "hello"
    assert agent.execute() == "hello"  # still stateless, no memory= given
    assert agent.conversation_history() == []


def test_memory_without_conversation_id_is_rejected_at_construction():
    memory = Memory()
    with pytest.raises(ValueError):
        ModelAgent("a", MockProvider(), prompt="hi", memory=memory.session)


def test_memory_must_look_like_a_memory_scope():
    with pytest.raises(TypeError):
        ModelAgent("a", MockProvider(), prompt="hi", memory=object(), conversation_id="c1")


def test_second_call_sends_the_first_turn_as_history():
    seen_requests = []

    def capture(request):
        seen_requests.append(request)
        return f"reply {len(seen_requests)}"

    memory = Memory()
    provider = MockProvider(response=capture)
    agent = ModelAgent("a", provider, prompt="what is your name?",
                        memory=memory.session, conversation_id="user-1")

    first = agent.execute()
    assert first == "reply 1"
    assert seen_requests[0].messages == [{"role": "user", "content": "what is your name?"}]

    second = agent.execute()
    assert second == "reply 2"
    assert seen_requests[1].messages == [
        {"role": "user", "content": "what is your name?"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "what is your name?"},
    ]


def test_conversation_history_is_readable_after_a_call():
    memory = Memory()
    provider = MockProvider(response="hi there")
    agent = ModelAgent("a", provider, prompt="hello", memory=memory.session, conversation_id="c1")

    agent.execute()

    assert agent.conversation_history() == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_two_agents_sharing_the_same_scope_and_conversation_id_see_each_others_turns():
    memory = Memory()
    provider = MockProvider(response="shared reply")

    agent1 = ModelAgent("a1", provider, prompt="turn one", memory=memory.session, conversation_id="shared")
    agent1.execute()

    seen_requests = []

    def capture(request):
        seen_requests.append(request)
        return "turn two reply"

    agent2 = ModelAgent("a2", MockProvider(response=capture), prompt="turn two",
                         memory=memory.session, conversation_id="shared")
    agent2.execute()

    assert seen_requests[0].messages == [
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "shared reply"},
        {"role": "user", "content": "turn two"},
    ]


def test_different_conversation_ids_on_the_same_scope_stay_isolated():
    memory = Memory()
    agent_a = ModelAgent("a", MockProvider(response="reply a"), prompt="hi",
                          memory=memory.session, conversation_id="conv-a")
    agent_b = ModelAgent("b", MockProvider(response="reply b"), prompt="hi",
                          memory=memory.session, conversation_id="conv-b")

    agent_a.execute()
    agent_b.execute()

    assert agent_a.conversation_history() == [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "reply a"},
    ]
    assert agent_b.conversation_history() == [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "reply b"},
    ]


def test_project_scope_persists_across_separate_memory_instances():
    provider = MockProvider(response="remembered")
    agent1 = ModelAgent("a", provider, prompt="hi", memory=Memory(project="proj-1").project, conversation_id="c1")
    agent1.execute()

    # A brand new Memory("proj-1") elsewhere in the process shares the same
    # underlying project scope (see aircore/memory.py's process-wide
    # registry) -- this is what lets two different agents/workflows in the
    # same project continue one conversation.
    agent2 = ModelAgent("b", MockProvider(response="ignored"), prompt="hi",
                         memory=Memory(project="proj-1").project, conversation_id="c1")

    assert agent2.conversation_history() == [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "remembered"},
    ]


def test_workflow_temporary_scope_is_cleared_after_run_so_history_does_not_survive():
    memory = Memory()
    provider = MockProvider(response="hi")
    agent = ModelAgent("a", provider, prompt="hello", memory=memory.temporary, conversation_id="c1")

    workflow = Workflow("temp-memory", memory=memory)
    workflow.step(agent)
    workflow.run()

    # Workflow.run() clears `temporary` in a finally block -- memory=
    # accepts it, but it defeats the purpose for cross-run continuity, per
    # the module docstring. This is the documented tradeoff, not a bug.
    assert agent.conversation_history() == []


def test_memory_works_inside_the_tool_calling_loop_and_does_not_persist_scratch_messages():
    lookup = None
    from aircore import Tool
    lookup = Tool(lambda item: {"widget": 3}.get(item, 0), name="lookup")

    responses = [
        ModelResponse(content="", tool_calls=[ToolCallRequest(id="1", name="lookup", arguments={"item": "widget"})]),
        "there are 3 widgets",
    ]
    memory = Memory()
    provider = MockProvider(responses=responses)
    agent = ModelAgent("a", provider, prompt="how many widgets?", tools=[lookup],
                        memory=memory.session, conversation_id="c1")

    result = agent.execute()

    assert result == "there are 3 widgets"
    # Only the outward-facing prompt/answer persisted -- no "tool_calls" or
    # role:"tool" scratch messages leaked into the conversation history.
    assert agent.conversation_history() == [
        {"role": "user", "content": "how many widgets?"},
        {"role": "assistant", "content": "there are 3 widgets"},
    ]


def test_memory_combines_with_output_schema():
    schema = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
    memory = Memory()
    provider = MockProvider(response='{"count": 3}')
    agent = ModelAgent("a", provider, prompt="how many?", memory=memory.session,
                        conversation_id="c1", output_schema=schema)

    result = agent.execute()

    assert result == {"count": 3}
    # The raw text (not the parsed dict) is what's stored as history --
    # keeps the persisted conversation wire-compatible for a future turn.
    assert agent.conversation_history()[-1] == {"role": "assistant", "content": '{"count": 3}'}
