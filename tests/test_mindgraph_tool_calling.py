"""Offline tests for MindGraph-M2: ModelAgent(tools=[...],
use_mindgraph=True) -- the actual token-reduction integration into the
tool-calling loop. See aircore/mindgraph.py and model_agent.py's "Token
reduction" docstring section."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aircore import MindGraph, Tool, tool, estimate_tokens
from airpy import ModelAgent, MockProvider, ModelResponse, ToolCallRequest


def _tool_call_response(name, arguments, call_id="call_1"):
    return ModelResponse(
        content="",
        tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments)],
    )


def _candles(n=200):
    return [
        {"time": i, "open": 1.10 + i * 0.0001, "high": 1.101 + i * 0.0001,
         "low": 1.099 + i * 0.0001, "close": 1.1005 + i * 0.0001}
        for i in range(n)
    ]


# --- default (use_mindgraph=False) behavior is completely unaffected ------

def test_default_behavior_unchanged_without_use_mindgraph():
    @tool
    def get_candles():
        return _candles(5)

    provider = MockProvider(responses=[
        _tool_call_response("get_candles", {}),
        "done",
    ])
    agent = ModelAgent("a", provider, prompt="p", tools=[get_candles])
    result = agent.execute()

    assert result == "done"
    assert agent.mindgraph_savings == []
    assert len(agent.mindgraph) == 0


# --- summarization actually shrinks what the model sees --------------------

def test_tool_result_message_is_summarized_when_use_mindgraph():
    seen_messages = []

    @tool
    def get_candles():
        return _candles(200)

    def first(request):
        seen_messages.append(list(request.messages or []))
        return ModelResponse(content="", tool_calls=[
            ToolCallRequest(id="1", name="get_candles", arguments={}),
        ])

    def second(request):
        seen_messages.append(list(request.messages or []))
        return ModelResponse(content="done")

    provider = MockProvider(responses=[first, second])
    agent = ModelAgent("a", provider, prompt="p", tools=[get_candles], use_mindgraph=True)
    result = agent.execute()

    assert result == "done"
    # the SECOND generate() call's messages include the tool result that
    # was fed back -- that's what must be small.
    second_call_messages = seen_messages[1]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    tool_content = tool_messages[0]["content"]

    raw_json_tokens = estimate_tokens(json.dumps(_candles(200)))
    summarized_tokens = estimate_tokens(tool_content)
    assert summarized_tokens < raw_json_tokens / 10

    # tool_call_log is NOT affected -- still the full real result.
    assert agent.tool_call_log[0].result == str(_candles(200))


def test_mindgraph_savings_are_recorded():
    @tool
    def get_candles():
        return _candles(200)

    provider = MockProvider(responses=[
        _tool_call_response("get_candles", {}),
        "done",
    ])
    agent = ModelAgent("a", provider, prompt="p", tools=[get_candles], use_mindgraph=True)
    agent.execute()

    assert len(agent.mindgraph_savings) == 1
    entry = agent.mindgraph_savings[0]
    assert entry["tool"] == "get_candles"
    assert entry["summary_tokens"] < entry["raw_tokens"] / 10


def test_mindgraph_gains_a_node_per_tool_call():
    @tool
    def get_price():
        return 1.1042

    provider = MockProvider(responses=[
        _tool_call_response("get_price", {}),
        "done",
    ])
    agent = ModelAgent("a", provider, prompt="p", tools=[get_price], use_mindgraph=True)
    agent.execute()

    assert len(agent.mindgraph) == 1
    node = agent.mindgraph.all_nodes()[0]
    assert node.kind == "tool_result"
    assert node.metadata["tool"] == "get_price"
    assert node.full_ref == 1.1042


# --- expand_node escape hatch -----------------------------------------------

def test_expand_node_tool_is_auto_injected_and_returns_full_value():
    @tool
    def get_candles():
        return _candles(3)

    def first(request):
        return ModelResponse(content="", tool_calls=[
            ToolCallRequest(id="1", name="get_candles", arguments={}),
        ])

    def second(request):
        # find the node id the summary exposed, in the last tool message
        tool_msg = [m for m in request.messages if m.get("role") == "tool"][-1]
        node_id = tool_msg["content"].split("]", 1)[0].lstrip("[")
        return ModelResponse(content="", tool_calls=[
            ToolCallRequest(id="2", name="expand_node", arguments={"node_id": node_id}),
        ])

    def third(request):
        return ModelResponse(content="final answer")

    provider = MockProvider(responses=[first, second, third])
    agent = ModelAgent("a", provider, prompt="p", tools=[get_candles], use_mindgraph=True)
    result = agent.execute()

    assert result == "final answer"
    # the expand_node call itself is logged and returned the real data
    expand_record = [r for r in agent.tool_call_log if r.name == "expand_node"][0]
    assert expand_record.error is None
    returned = json.loads(expand_record.result)
    assert returned == _candles(3)


def test_expand_node_unknown_id_returns_error_text_not_a_crash():
    @tool
    def noop():
        return "ok"

    provider = MockProvider(responses=[
        ModelResponse(content="", tool_calls=[ToolCallRequest(id="1", name="noop", arguments={})]),
        ModelResponse(content="", tool_calls=[
            ToolCallRequest(id="2", name="expand_node", arguments={"node_id": "does-not-exist"}),
        ]),
        "done",
    ])
    agent = ModelAgent("a", provider, prompt="p", tools=[noop], use_mindgraph=True)
    result = agent.execute()

    assert result == "done"
    expand_record = [r for r in agent.tool_call_log if r.name == "expand_node"][0]
    assert "no such node" in expand_record.result


def test_expand_node_not_injected_when_use_mindgraph_false():
    @tool
    def noop():
        return "ok"

    agent = ModelAgent("a", MockProvider(response="done"), prompt="p", tools=[noop])
    # use_mindgraph defaults False -- expand_node should not be callable
    provider = MockProvider(responses=[
        ModelResponse(content="", tool_calls=[
            ToolCallRequest(id="1", name="expand_node", arguments={"node_id": "n1"}),
        ]),
        "done",
    ])
    agent = ModelAgent("a", provider, prompt="p", tools=[noop])
    agent.execute()
    assert "no such tool" in agent.tool_call_log[0].error


# --- error results are not mangled ------------------------------------------

def test_tool_error_is_not_summarized_away():
    @tool
    def boom():
        raise RuntimeError("kaboom")

    provider = MockProvider(responses=[
        _tool_call_response("boom", {}),
        "handled",
    ])
    agent = ModelAgent("a", provider, prompt="p", tools=[boom], use_mindgraph=True)
    result = agent.execute()

    assert result == "handled"
    assert "kaboom" in agent.tool_call_log[0].error
    node = agent.mindgraph.all_nodes()[0]
    assert node.kind == "tool_error"
    assert "kaboom" in node.summary


# --- custom summarizers ------------------------------------------------------

def test_custom_summarizer_is_used_when_registered():
    @tool
    def get_candles():
        return _candles(10)

    def my_summarizer(candles):
        return f"custom summary of {len(candles)} candles"

    provider = MockProvider(responses=[
        _tool_call_response("get_candles", {}),
        "done",
    ])
    agent = ModelAgent("a", provider, prompt="p", tools=[get_candles],
                        use_mindgraph=True, summarizers={"get_candles": my_summarizer})
    agent.execute()

    node = agent.mindgraph.all_nodes()[0]
    assert node.summary == "custom summary of 10 candles"


# --- shared mindgraph across agents (groundwork for M3) --------------------

def test_shared_mindgraph_passed_in_is_reused_not_replaced():
    shared = MindGraph()

    @tool
    def get_price():
        return 1.1

    provider = MockProvider(responses=[
        _tool_call_response("get_price", {}),
        "done",
    ])
    agent = ModelAgent("a", provider, prompt="p", tools=[get_price],
                        use_mindgraph=True, mindgraph=shared)
    agent.execute()

    assert agent.mindgraph is shared
    assert len(shared) == 1
