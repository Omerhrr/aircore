"""ModelAgent.stream() / ModelProvider.stream(): yields text chunks
instead of blocking for the whole completion. Lives entirely in airpy and
bypasses the Scheduler, the same way ask() already does -- there's no
aircore change here at all, see model_agent.py's stream() docstring for why
true token streaming doesn't fit atomic per-step journaling.
"""

import pytest

from aircore import Memory
from airpy import ModelAgent, MockProvider
from airpy.providers import ModelRequest


def test_mock_provider_stream_yields_chunks_that_join_back_to_the_full_response():
    provider = MockProvider(response="hello there friend")
    chunks = list(provider.stream(ModelRequest(prompt="hi")))
    assert "".join(chunks) == "hello there friend"
    assert len(chunks) > 1  # actually chunked, not one giant blob


def test_base_provider_stream_raises_not_implemented_by_default():
    from airpy.providers import ModelProvider, ModelResponse

    class NoStreamProvider(ModelProvider):
        def generate(self, request):
            return ModelResponse(content="ok")

    with pytest.raises(NotImplementedError):
        list(NoStreamProvider().stream(ModelRequest(prompt="hi")))


def test_agent_stream_yields_chunks_and_updates_last_response():
    agent = ModelAgent("a", MockProvider(response="hello there friend"), prompt="hi")

    chunks = list(agent.stream())

    assert "".join(chunks) == "hello there friend"
    assert agent.last_response.content == "hello there friend"


def test_agent_stream_calls_on_token_for_each_chunk():
    seen = []
    agent = ModelAgent("a", MockProvider(response="a b c"), prompt="hi")

    list(agent.stream(on_token=seen.append))

    assert "".join(seen) == "a b c"
    assert len(seen) > 1


def test_agent_stream_raises_immediately_not_lazily_when_tools_are_set():
    from aircore import Tool

    tool = Tool(lambda: 1, name="t")
    agent = ModelAgent("a", MockProvider(), prompt="hi", tools=[tool])

    # Must raise on the .stream() call itself, before any iteration --
    # a naive `def` wrapping `yield` would defer this until the first
    # next(), which is exactly the bug this test guards against.
    with pytest.raises(NotImplementedError):
        agent.stream()


def test_agent_stream_updates_memory_after_full_consumption():
    memory = Memory()
    agent = ModelAgent("a", MockProvider(response="streamed reply"), prompt="hello",
                        memory=memory.session, conversation_id="c1")

    list(agent.stream())

    assert agent.conversation_history() == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "streamed reply"},
    ]


def test_agent_stream_second_call_sees_first_as_history():
    memory = Memory()
    seen_requests = []

    def capture(request):
        seen_requests.append(request)
        return "reply"

    agent1 = ModelAgent("a", MockProvider(response=capture), prompt="turn one",
                         memory=memory.session, conversation_id="c1")
    list(agent1.stream())

    agent2 = ModelAgent("a", MockProvider(response=capture), prompt="turn two",
                         memory=memory.session, conversation_id="c1")
    list(agent2.stream())

    assert seen_requests[1].messages == [
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "turn two"},
    ]


def test_agent_stream_validates_structured_output_after_full_consumption():
    schema = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
    agent = ModelAgent("a", MockProvider(response='{"count": 3}'), prompt="how many?", output_schema=schema)

    list(agent.stream())  # consumes fully; validates at the end, no error

    from airpy.structured_output import StructuredOutputError

    bad_agent = ModelAgent("b", MockProvider(response="not json"), prompt="how many?", output_schema=schema)
    with pytest.raises(StructuredOutputError):
        list(bad_agent.stream())


def test_agent_stream_partial_consumption_does_not_update_state():
    # If a caller stops iterating early (e.g. `break`), the accumulation
    # code after the loop never runs -- last_response/memory reflect only
    # a fully-consumed stream, same as any generator's lazy semantics.
    agent = ModelAgent("a", MockProvider(response="one two three four"), prompt="hi")

    gen = agent.stream()
    next(gen)  # only the first chunk

    assert agent.last_response is None
