"""FileMemoryScope (aircore/persistent_memory.py) -- the persistent-across-
restarts option for ModelAgent(memory=...)/Session(memory=...), closing
the gap memory.py's own docstring names ("every scope vanishes when the
process exits"). Covers the plain get/set/delete/clear/__contains__/
snapshot contract, that it's a genuine drop-in for ModelAgent's
memory-backed conversations, that it actually survives a simulated
process restart (a fresh FileMemoryScope pointed at the same path), and
the JSON-serializability constraint.
"""

import pytest

from airpy import Agent, MockProvider
from aircore import FileMemoryScope, Tool, Workflow


def test_basic_get_set_delete_contains_clear(tmp_path):
    scope = FileMemoryScope(str(tmp_path / "mem.json"))
    assert "k" not in scope
    assert scope.get("k") is None
    assert scope.get("k", "default") == "default"

    scope.set("k", {"a": 1})
    assert "k" in scope
    assert scope.get("k") == {"a": 1}
    assert scope.snapshot() == {"k": {"a": 1}}

    scope.delete("k")
    assert "k" not in scope
    scope.delete("does_not_exist")  # no error

    scope.set("x", 1)
    scope.set("y", 2)
    scope.clear()
    assert scope.snapshot() == {}


def test_a_non_json_serializable_value_is_rejected_immediately():
    scope = FileMemoryScope("/tmp/should_not_be_written.json")

    class NotSerializable:
        pass

    with pytest.raises(TypeError, match="JSON-serializable"):
        scope.set("bad", NotSerializable())


def test_survives_a_simulated_process_restart(tmp_path):
    path = str(tmp_path / "mem.json")
    scope1 = FileMemoryScope(path)
    scope1.set("history", [{"role": "user", "content": "hi"}])

    # A brand new FileMemoryScope instance at the same path -- nothing in
    # memory survived, only what's on disk, same as a real restart.
    scope2 = FileMemoryScope(path)
    assert scope2.get("history") == [{"role": "user", "content": "hi"}]


def test_is_a_genuine_drop_in_for_model_agent_memory(tmp_path):
    path = str(tmp_path / "conversation.json")
    scope = FileMemoryScope(path)

    seen_messages = []

    def fake_response(req):
        seen_messages.append(list(req.messages or []))
        return "an answer"

    provider = MockProvider(response=fake_response)
    agent1 = Agent("bot", provider, "first message", memory=scope, conversation_id="conv-1")
    workflow1 = Workflow("Turn1")
    workflow1.step(agent1)
    workflow1.run()

    # A second, independent ModelAgent/Workflow, using a *new*
    # FileMemoryScope object at the same path -- simulating the process
    # having restarted between turns.
    scope2 = FileMemoryScope(path)
    agent2 = Agent("bot", provider, "second message", memory=scope2, conversation_id="conv-1")
    workflow2 = Workflow("Turn2")
    workflow2.step(agent2)
    workflow2.run()

    # The second call's request should have seen the first turn's history
    last_request_messages = seen_messages[-1]
    contents = [m["content"] for m in last_request_messages]
    assert "first message" in contents
    assert "an answer" in contents
    assert "second message" in contents


def test_a_tool_can_use_file_memory_scope_directly_via_closure(tmp_path):
    path = str(tmp_path / "counter.json")
    scope = FileMemoryScope(path)

    def increment():
        current = scope.get("count", 0)
        scope.set("count", current + 1)
        return current + 1

    workflow = Workflow("W")
    workflow.step(Tool(increment, name="increment"))
    journal = workflow.run()
    assert journal.status == "success"
    assert journal.steps[0].output == 1

    # Persisted -- a fresh scope object at the same path sees it too.
    scope2 = FileMemoryScope(path)
    assert scope2.get("count") == 1
