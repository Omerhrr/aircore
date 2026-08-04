"""OpenAIProvider (airpy/openai_provider.py) -- the first native
(non-LiteLLM) ModelProvider adapter. No API key or network call: the
`openai` package is faked the same way `litellm` is faked in
tests/test_litellm_tool_calling.py, mirroring the real SDK's actual
response shape (response.choices[0].message.content,
response.choices[0].finish_reason, response.usage.prompt_tokens/
completion_tokens/total_tokens) so these tests would catch a real wire-
format mismatch, not just a mismatch against our own fake.
"""

import sys
import os
import types
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import tool
from airpy import ModelAgent


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments  # JSON string, same as the real SDK


class _FakeToolCall:
    def __init__(self, id_, name, arguments):
        self.id = id_
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=5, total_tokens=15):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeResponse:
    def __init__(self, message, model, finish_reason="stop"):
        self.choices = [_FakeChoice(message, finish_reason)]
        self.usage = _FakeUsage()
        self.model = model


def _install_fake_openai(create_fn):
    fake = types.ModuleType("openai")

    class _FakeCompletions:
        def create(self, **kwargs):
            return create_fn(**kwargs)

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, **kwargs):
            self.api_key = api_key
            self.base_url = base_url
            self.chat = _FakeChat()

    fake.OpenAI = _FakeOpenAI
    sys.modules["openai"] = fake


def _uninstall_fake_openai():
    sys.modules.pop("openai", None)


def test_openai_provider_basic_generate():
    def create_fn(model, messages, **kwargs):
        assert model == "gpt-4o-mini"
        assert messages == [{"role": "user", "content": "hello"}]
        return _FakeResponse(_FakeMessage("hi there"), model)

    _install_fake_openai(create_fn)
    try:
        from airpy import OpenAIProvider

        provider = OpenAIProvider(model="gpt-4o-mini", api_key="fake-key")
        agent = ModelAgent("bot", provider, "hello")
        result = agent.execute()

        assert result == "hi there"
        # usage() (see model_agent.py) filters out None values entirely --
        # cost_usd is always None for OpenAIProvider (no pricing lookup in
        # the openai SDK), so it's correctly absent here, not present as
        # None. test_openai_provider_no_cost_lookup_ever_reports_none_not_zero
        # below checks the underlying Usage.cost_usd directly instead.
        assert agent.usage() == {"tokens_in": 10, "tokens_out": 5}
    finally:
        _uninstall_fake_openai()


def test_openai_provider_no_cost_lookup_ever_reports_none_not_zero():
    """The honest gap this module's docstring documents: openai's SDK has
    no pricing lookup, so cost_usd must always be None (unknown), never
    0.0 (a false claim of "free")."""
    _install_fake_openai(lambda model, messages, **kwargs: _FakeResponse(_FakeMessage("x"), model))
    try:
        from airpy import OpenAIProvider

        provider = OpenAIProvider()
        from airpy.providers import ModelRequest
        result = provider.generate(ModelRequest(prompt="hi"))
        assert result.usage.cost_usd is None
    finally:
        _uninstall_fake_openai()


def test_openai_provider_base_url_is_forwarded_for_openai_compatible_endpoints():
    """This is what makes live-testing against DeepSeek's OpenAI-compatible
    endpoint possible without a new adapter or API key -- base_url= must
    actually reach the underlying openai.OpenAI() client construction."""
    _install_fake_openai(lambda model, messages, **kwargs: _FakeResponse(_FakeMessage("ok"), model))
    try:
        from airpy import OpenAIProvider

        provider = OpenAIProvider(model="deepseek-chat", api_key="fake",
                                    base_url="https://api.deepseek.com")
        assert provider._client.base_url == "https://api.deepseek.com"
    finally:
        _uninstall_fake_openai()


def test_openai_provider_full_tool_calling_round_trip():
    state = {"call_count": 0}

    def create_fn(model, messages, **kwargs):
        state["call_count"] += 1
        if state["call_count"] == 1:
            assert kwargs.get("tools"), "first call should offer the tool schema"
            tool_call = _FakeToolCall("call_1", "get_capital", json.dumps({"country": "Nigeria"}))
            return _FakeResponse(_FakeMessage(None, tool_calls=[tool_call]), model,
                                  finish_reason="tool_calls")

        # Second call: verify the wire format fed back is what a real
        # chat completions API requires -- same check
        # test_litellm_tool_calling.py does for LiteLLMProvider, since
        # this is a genuinely different adapter and could get it wrong
        # independently.
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        assert assistant_messages
        assistant_tool_calls = assistant_messages[-1].get("tool_calls")
        assert isinstance(assistant_tool_calls, list) and assistant_tool_calls
        for call in assistant_tool_calls:
            assert call.get("type") == "function"
            assert isinstance(call.get("id"), str) and call["id"]
            function = call.get("function")
            assert isinstance(function.get("arguments"), str)
            json.loads(function["arguments"])
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert tool_messages

        return _FakeResponse(_FakeMessage("The capital of Nigeria is Abuja."), model)

    _install_fake_openai(create_fn)
    try:
        from airpy import OpenAIProvider

        @tool
        def get_capital(country: str):
            return {"Nigeria": "Abuja"}.get(country, "unknown")

        agent = ModelAgent("geo_bot", OpenAIProvider(model="gpt-4o-mini", api_key="fake"),
                            prompt="what is the capital of Nigeria?", tools=[get_capital])
        result = agent.execute()

        assert result == "The capital of Nigeria is Abuja."
        assert state["call_count"] == 2
        assert agent.tool_call_log[0].name == "get_capital"
        assert agent.tool_call_log[0].result == "Abuja"
    finally:
        _uninstall_fake_openai()


def test_openai_provider_requires_the_openai_package():
    _uninstall_fake_openai()  # make sure it's really absent
    sys.modules.pop("openai", None)
    import builtins
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("no module named openai")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = blocking_import
    try:
        from airpy import OpenAIProvider
        try:
            OpenAIProvider()
            assert False, "expected ImportError"
        except ImportError as exc:
            assert "pip install openai" in str(exc)
    finally:
        builtins.__import__ = real_import


if __name__ == "__main__":
    test_openai_provider_basic_generate()
    test_openai_provider_no_cost_lookup_ever_reports_none_not_zero()
    test_openai_provider_base_url_is_forwarded_for_openai_compatible_endpoints()
    test_openai_provider_full_tool_calling_round_trip()
    test_openai_provider_requires_the_openai_package()
    print("All OpenAIProvider tests passed.")
