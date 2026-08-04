import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool, Agent, Network
from airpy import (
    ModelAgent, MockProvider, ModelResponse, ToolCallRequest,
    ModelAgentToolLoopExceeded,
)


def _tool_call_response(name, arguments, call_id="call_1"):
    return ModelResponse(
        content="",
        tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments)],
    )


def test_agent_without_tools_is_unchanged_single_shot():
    agent = ModelAgent("m", MockProvider(response="plain answer"), prompt="p")
    assert agent.execute() == "plain answer"
    assert agent.tool_call_log == []


def test_agent_calls_a_tool_and_returns_final_answer():
    @tool
    def get_weather(city: str):
        return f"sunny in {city}"

    provider = MockProvider(responses=[
        _tool_call_response("get_weather", {"city": "Lagos"}),
        "The weather in Lagos is sunny.",
    ])
    agent = ModelAgent("weather_bot", provider, prompt="what's the weather in Lagos?",
                        tools=[get_weather])

    result = agent.execute()

    assert result == "The weather in Lagos is sunny."
    assert len(agent.tool_call_log) == 1
    record = agent.tool_call_log[0]
    assert record.name == "get_weather"
    assert record.arguments == {"city": "Lagos"}
    assert record.result == "sunny in Lagos"
    assert record.error is None


def test_agent_handles_multiple_tool_calls_in_one_turn():
    @tool
    def add(a: int, b: int):
        return a + b

    @tool
    def multiply(a: int, b: int):
        return a * b

    provider = MockProvider(responses=[
        ModelResponse(content="", tool_calls=[
            ToolCallRequest(id="1", name="add", arguments={"a": 2, "b": 3}),
            ToolCallRequest(id="2", name="multiply", arguments={"a": 4, "b": 5}),
        ]),
        "2+3=5 and 4*5=20",
    ])
    agent = ModelAgent("math_bot", provider, prompt="compute things", tools=[add, multiply])

    result = agent.execute()

    assert result == "2+3=5 and 4*5=20"
    assert len(agent.tool_call_log) == 2
    assert agent.tool_call_log[0].result == "5"
    assert agent.tool_call_log[1].result == "20"


def test_agent_loops_across_multiple_turns():
    @tool
    def step_one():
        return "result_one"

    @tool
    def step_two():
        return "result_two"

    provider = MockProvider(responses=[
        _tool_call_response("step_one", {}),
        _tool_call_response("step_two", {}),
        "done after two steps",
    ])
    agent = ModelAgent("multi_turn", provider, prompt="do both steps", tools=[step_one, step_two])

    result = agent.execute()

    assert result == "done after two steps"
    assert [r.name for r in agent.tool_call_log] == ["step_one", "step_two"]


def test_unknown_tool_name_is_fed_back_as_error_not_a_crash():
    @tool
    def real_tool():
        return "ok"

    provider = MockProvider(responses=[
        _tool_call_response("nonexistent_tool", {}),
        "I couldn't find that tool.",
    ])
    agent = ModelAgent("m", provider, prompt="p", tools=[real_tool])

    result = agent.execute()

    assert result == "I couldn't find that tool."
    assert agent.tool_call_log[0].error is not None
    assert "no such tool" in agent.tool_call_log[0].error


def test_tool_exception_is_fed_back_as_error_not_a_crash():
    @tool
    def broken_tool():
        raise RuntimeError("boom")

    provider = MockProvider(responses=[
        _tool_call_response("broken_tool", {}),
        "The tool failed, so I can't answer.",
    ])
    agent = ModelAgent("m", provider, prompt="p", tools=[broken_tool])

    result = agent.execute()

    assert result == "The tool failed, so I can't answer."
    assert agent.tool_call_log[0].error is not None
    assert "boom" in agent.tool_call_log[0].error


def test_max_turns_exceeded_raises():
    @tool
    def infinite_tool():
        return "still going"

    # every response requests another tool call -- never a final answer
    provider = MockProvider(response=lambda req: _tool_call_response("infinite_tool", {}))

    class LoopingProvider(MockProvider):
        def generate(self, request):
            return _tool_call_response("infinite_tool", {})

    agent = ModelAgent("looper", LoopingProvider(), prompt="p", tools=[infinite_tool], max_turns=3)

    try:
        agent.execute()
        assert False, "expected ModelAgentToolLoopExceeded"
    except ModelAgentToolLoopExceeded as exc:
        assert "max_turns=3" in str(exc)
    assert len(agent.tool_call_log) == 3


def test_identity_blocks_tool_call_missing_capability():
    @tool(requires=Network)
    def fetch_url(url: str):
        return f"fetched {url}"

    provider = MockProvider(responses=[
        _tool_call_response("fetch_url", {"url": "http://example.com"}),
        "I don't have permission to fetch that.",
    ])
    sandboxed_identity = Agent("Sandboxed", capabilities=[])
    agent = ModelAgent("web_bot", provider, prompt="fetch something",
                        tools=[fetch_url], identity=sandboxed_identity)

    result = agent.execute()

    assert result == "I don't have permission to fetch that."
    assert agent.tool_call_log[0].error is not None
    assert "capability" in agent.tool_call_log[0].error
    assert "Network" in agent.tool_call_log[0].error


def test_identity_allows_tool_call_with_capability():
    @tool(requires=Network)
    def fetch_url(url: str):
        return f"fetched {url}"

    provider = MockProvider(responses=[
        _tool_call_response("fetch_url", {"url": "http://example.com"}),
        "Here's what I found.",
    ])
    networked_identity = Agent("Networked", capabilities=[Network])
    agent = ModelAgent("web_bot", provider, prompt="fetch something",
                        tools=[fetch_url], identity=networked_identity)

    result = agent.execute()

    assert result == "Here's what I found."
    assert agent.tool_call_log[0].error is None
    assert agent.tool_call_log[0].result == "fetched http://example.com"


def test_tool_calling_agent_works_as_a_workflow_step():
    @tool
    def lookup(term: str):
        return f"definition of {term}"

    provider = MockProvider(responses=[
        _tool_call_response("lookup", {"term": "aircore"}),
        "aircore is an execution runtime.",
    ])
    agent = ModelAgent("dictionary_bot", provider, prompt="define aircore", tools=[lookup])

    workflow = Workflow("Lookup")
    workflow.step(agent)
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.steps[0].output == "aircore is an execution runtime."
    # internal tool call is invisible to the journal -- only one step recorded
    assert len(journal.steps) == 1


if __name__ == "__main__":
    test_agent_without_tools_is_unchanged_single_shot()
    test_agent_calls_a_tool_and_returns_final_answer()
    test_agent_handles_multiple_tool_calls_in_one_turn()
    test_agent_loops_across_multiple_turns()
    test_unknown_tool_name_is_fed_back_as_error_not_a_crash()
    test_tool_exception_is_fed_back_as_error_not_a_crash()
    test_max_turns_exceeded_raises()
    test_identity_blocks_tool_call_missing_capability()
    test_identity_allows_tool_call_with_capability()
    test_tool_calling_agent_works_as_a_workflow_step()
    print("All tool-calling tests passed.")
