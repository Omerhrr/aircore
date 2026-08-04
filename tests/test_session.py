"""Session: a long-running conversation built from ModelAgent + Memory +
a real per-turn Workflow. Composes existing primitives (no aircore
changes) -- see airpy/session.py's docstring for what this adds beyond
ModelAgent(memory=...) alone.
"""

import pytest

from aircore import Agent, Capability, Policy, PolicyViolation, CapabilityDenied
from airpy import MockProvider, Session, SessionClosed, SessionTurnFailed


def test_send_returns_the_agents_answer():
    session = Session("assistant", MockProvider(response="hello!"))
    assert session.send("hi") == "hello!"


def test_send_persists_history_across_calls():
    seen_requests = []

    def capture(request):
        seen_requests.append(request)
        return "reply"

    session = Session("assistant", MockProvider(response=capture))
    session.send("turn one")
    session.send("turn two")

    assert seen_requests[1].messages == [
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "turn two"},
    ]


def test_history_property_matches_underlying_memory():
    session = Session("assistant", MockProvider(response="hi"))
    session.send("hello")
    assert session.history == [
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"},
    ]


def test_session_metadata_tracked():
    session = Session("assistant", MockProvider(response="hi"))
    assert session.turn_count == 0
    assert session.last_active_at is None
    assert session.ended_at is None

    session.send("hello")

    assert session.turn_count == 1
    assert session.last_active_at is not None


def test_each_send_produces_a_journal():
    session = Session("assistant", MockProvider(response="hi"))
    session.send("one")
    session.send("two")

    assert len(session.journals) == 2
    assert all(j.status == "success" for j in session.journals)
    assert session.journals[0].steps[0].tool == "assistant"


def test_close_prevents_further_sends():
    session = Session("assistant", MockProvider(response="hi"))
    session.send("hello")
    session.close()

    assert session.ended_at is not None
    with pytest.raises(SessionClosed):
        session.send("after close")


def test_history_still_readable_after_close():
    session = Session("assistant", MockProvider(response="hi"))
    session.send("hello")
    session.close()
    assert session.history == [
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"},
    ]


def test_context_manager_closes_on_exit():
    with Session("assistant", MockProvider(response="hi")) as session:
        session.send("hello")
        assert session.ended_at is None
    assert session.ended_at is not None


def test_two_sessions_have_independent_history_by_default():
    session_a = Session("assistant", MockProvider(response="from a"))
    session_b = Session("assistant", MockProvider(response="from b"))

    session_a.send("hi")
    session_b.send("hi")

    assert session_a.history[-1]["content"] == "from a"
    assert session_b.history[-1]["content"] == "from b"
    assert session_a.session_id != session_b.session_id


def test_max_history_turns_trims_the_oldest_turns():
    session = Session("assistant", MockProvider(response="ok"), max_history_turns=2)
    session.send("one")
    session.send("two")
    session.send("three")

    # 2 turns max => at most 4 messages (2 user + 2 assistant), and it's
    # the most recent ones that survive.
    assert len(session.history) == 4
    assert session.history[0] == {"role": "user", "content": "two"}


def test_max_history_turns_rejects_less_than_one():
    with pytest.raises(ValueError):
        Session("assistant", MockProvider(), max_history_turns=0)


def test_turn_failure_raises_session_turn_failed_with_the_journal_attached():
    def blow_up(request):
        raise ConnectionError("simulated provider failure")

    session = Session("assistant", MockProvider(response=blow_up))

    with pytest.raises(SessionTurnFailed) as excinfo:
        session.send("hello")

    assert excinfo.value.journal.status == "failed"
    assert "simulated provider failure" in excinfo.value.journal.steps[0].error


def test_requires_enforces_capability_on_the_send_side():
    net = Capability("Network")
    session = Session("assistant", MockProvider(response="ok"), requires=net)

    bot_without_network = Agent("bot")
    with pytest.raises(SessionTurnFailed) as excinfo:
        session.send("hello", agent=bot_without_network)
    assert "CapabilityDenied" in excinfo.value.journal.steps[0].error

    bot_with_network = Agent("bot", capabilities=[net])
    assert session.send("hello", agent=bot_with_network) == "ok"


def test_policy_is_enforced_per_turn():
    session = Session("assistant", MockProvider(response="ok"), policy=Policy(require_agent=True))

    with pytest.raises(PolicyViolation):
        session.send("hello")  # no agent= given, but Policy requires one


def test_output_schema_returns_a_structured_value():
    schema = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
    session = Session("assistant", MockProvider(response='{"count": 3}'), output_schema=schema)

    assert session.send("how many?") == {"count": 3}


def test_shared_memory_scope_lets_two_sessions_see_the_same_conversation():
    from aircore import Memory

    memory = Memory()
    session_a = Session("a", MockProvider(response="from a"), memory=memory.session, conversation_id="shared")
    session_a.send("hello")

    session_b = Session("b", MockProvider(response="from b"), memory=memory.session, conversation_id="shared")
    assert session_b.history == session_a.history


def test_repr_reflects_open_closed_and_turn_count():
    session = Session("assistant", MockProvider(response="hi"))
    assert "open" in repr(session)
    session.send("hello")
    assert "turns=1" in repr(session)
    session.close()
    assert "closed" in repr(session)
