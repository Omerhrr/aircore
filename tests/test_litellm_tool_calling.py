"""Confirms LiteLLMProvider parses real tool_calls responses correctly and
that the full loop works end to end through a real (fake-backed) provider,
not just MockProvider. No API key or network call -- litellm is faked the
same way as test_m8.py.
"""

import sys
import os
import types
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import tool
from airpy import ModelAgent


def _install_fake_litellm_with_tool_calls():
    """First completion() call returns a tool_calls response; every call
    after that returns a plain final answer -- mimics a real multi-turn
    tool-calling exchange."""
    fake = types.ModuleType("litellm")
    state = {"call_count": 0}

    class FakeFunction:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments  # JSON string, same as the real SDK

    class FakeToolCall:
        def __init__(self, id_, name, arguments):
            self.id = id_
            self.function = FakeFunction(name, arguments)

    class FakeMessage:
        def __init__(self, content, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class FakeChoice:
        def __init__(self, message):
            self.message = message
            self.finish_reason = "tool_calls" if message.tool_calls else "stop"

    class FakeUsage:
        def __init__(self):
            self.prompt_tokens = 50
            self.completion_tokens = 20
            self.total_tokens = 70

    class FakeResponse:
        def __init__(self, message, model):
            self.choices = [FakeChoice(message)]
            self.usage = FakeUsage()
            self.model = model

    def fake_completion(model, messages, **kwargs):
        state["call_count"] += 1
        if state["call_count"] == 1:
            # first call: model asks to call get_capital(country="Nigeria")
            tool_call = FakeToolCall("call_abc", "get_capital", json.dumps({"country": "Nigeria"}))
            return FakeResponse(FakeMessage(None, tool_calls=[tool_call]), model)

        # second call: strictly validate the wire format of everything
        # sent back, the way a real API does -- this is exactly the check
        # that was missing before the live DeepSeek run caught the bug
        # (raw ToolCallRequest objects being put straight into `messages`
        # instead of being converted back to {id, type, function: {name,
        # arguments-as-a-JSON-string}}).
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        assert assistant_messages, "no assistant message in history"
        assistant_tool_calls = assistant_messages[-1].get("tool_calls")
        assert isinstance(assistant_tool_calls, list) and len(assistant_tool_calls) >= 1, (
            f"assistant tool_calls must be a non-empty list of wire-format dicts, "
            f"got {assistant_tool_calls!r}"
        )
        for call in assistant_tool_calls:
            assert isinstance(call, dict), f"tool_call must be a dict, got {type(call).__name__}"
            assert call.get("type") == "function"
            assert isinstance(call.get("id"), str) and call["id"]
            function = call.get("function")
            assert isinstance(function, dict) and "name" in function
            # arguments must be a JSON *string*, not a dict -- that's the
            # actual shape a real chat completions API requires
            assert isinstance(function.get("arguments"), str)
            json.loads(function["arguments"])  # must round-trip as valid JSON

        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert tool_messages, "tool result was not fed back"

        return FakeResponse(FakeMessage("The capital of Nigeria is Abuja."), model)

    def fake_completion_cost(completion_response=None):
        return 0.002

    fake.completion = fake_completion
    fake.completion_cost = fake_completion_cost
    sys.modules["litellm"] = fake
    return state


def _uninstall_fake_litellm():
    sys.modules.pop("litellm", None)


def test_litellm_provider_full_tool_calling_round_trip():
    state = _install_fake_litellm_with_tool_calls()
    try:
        from airpy import LiteLLMProvider

        @tool
        def get_capital(country: str):
            capitals = {"Nigeria": "Abuja"}
            return capitals.get(country, "unknown")

        agent = ModelAgent("geo_bot", LiteLLMProvider(model="gpt-4o-mini"),
                            prompt="what is the capital of Nigeria?", tools=[get_capital])

        result = agent.execute()

        assert result == "The capital of Nigeria is Abuja."
        assert state["call_count"] == 2
        assert len(agent.tool_call_log) == 1
        assert agent.tool_call_log[0].name == "get_capital"
        assert agent.tool_call_log[0].arguments == {"country": "Nigeria"}
        assert agent.tool_call_log[0].result == "Abuja"

        # usage from the *second* (final) response is what's reported --
        # confirms the loop's last_response is the final turn's, not the first
        assert agent.last_response.content == "The capital of Nigeria is Abuja."
        assert agent.usage() == {"tokens_in": 50, "tokens_out": 20, "cost_usd": 0.002}
    finally:
        _uninstall_fake_litellm()


if __name__ == "__main__":
    test_litellm_provider_full_tool_calling_round_trip()
    print("All LiteLLM tool-calling tests passed.")
