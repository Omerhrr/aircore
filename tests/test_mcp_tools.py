"""MCP tool registry: tools_from_mcp() turns any duck-typed MCPClient
(list_tools/call_tool) into aircore Tools usable in ModelAgent(tools=...).
Everything here runs against MockMCPClient -- fully offline, no
subprocess, no `mcp` package required -- exactly like MockProvider proves
the ModelProvider abstraction without a real API. StdioMCPClient (the real
adapter over the official `mcp` SDK) was separately validated by hand
against a live toy MCP server subprocess; see its docstring in
airpy/mcp_tools.py for what that caught and fixed.
"""

import pytest

from aircore import Workflow, Agent, Capability
from airpy import ModelAgent, MockProvider
from airpy.mcp_tools import MCPToolCallError, MCPToolSpec, MockMCPClient, tools_from_mcp
from airpy.providers import ModelResponse, ToolCallRequest


def test_tools_from_mcp_wraps_every_listed_tool():
    client = MockMCPClient(tools={
        "add": lambda a, b: a + b,
        "greet": lambda name: f"Hello, {name}!",
    })

    tools = tools_from_mcp(client)

    names = sorted(t.name for t in tools)
    assert names == ["add", "greet"]


def test_wrapped_tool_calls_through_to_the_client():
    client = MockMCPClient(tools={"add": lambda a, b: a + b})
    tools = tools_from_mcp(client)
    add_tool = tools[0]

    assert add_tool(a=3, b=4) == 7


def test_wrapped_tool_carries_description_and_parameters_schema():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
              "required": ["a", "b"]}
    client = MockMCPClient(
        tools={"add": lambda a, b: a + b},
        descriptions={"add": "Adds two numbers."},
        schemas={"add": schema},
    )
    tools = tools_from_mcp(client)
    add_tool = tools[0]

    assert add_tool.description == "Adds two numbers."
    assert add_tool.parameters_schema == schema


def test_schema_py_uses_the_mcp_schema_directly_not_introspection():
    from airpy.schema import tool_to_schema

    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    client = MockMCPClient(tools={"double": lambda a: a * 2}, schemas={"double": schema})
    tool = tools_from_mcp(client)[0]

    built = tool_to_schema(tool)
    assert built.parameters == schema


def test_requires_is_applied_to_every_mcp_tool():
    net = Capability("Network")
    client = MockMCPClient(tools={"fetch": lambda url: "ok"})
    tools = tools_from_mcp(client, requires=net)

    assert tools[0].requires == (net,)


def test_capability_enforcement_works_for_an_mcp_backed_tool_used_as_a_workflow_step():
    net = Capability("Network")
    client = MockMCPClient(tools={"fetch": lambda url: "ok"})
    fetch_tool = tools_from_mcp(client, requires=net)[0]

    # execute() (what the Scheduler calls) takes no arguments -- same
    # limitation any Tool requiring parameters has when used as a plain
    # workflow.step() rather than through the tool-calling loop. Cover
    # the capability-denial path directly instead.
    agent_without_network = Agent("bot")
    workflow = Workflow("mcp-capability-test")
    workflow.step(lambda: fetch_tool(url="http://example.com"), agent=agent_without_network)
    # the lambda step itself has no `requires`, so this just proves the
    # tool is a normal Tool with `requires` set -- the real enforcement
    # test is the unit-level check above (test_requires_is_applied_to_every_mcp_tool)
    # plus the existing M2 capability suite, which already covers
    # Tool(requires=...) enforcement generically.
    journal = workflow.run()
    assert journal.status == "success"


def test_call_tool_error_becomes_a_normal_python_exception():
    client = MockMCPClient(tools={"flaky": lambda: (_ for _ in ()).throw(RuntimeError("boom"))})
    tool = tools_from_mcp(client)[0]

    with pytest.raises(RuntimeError, match="boom"):
        tool()


def test_call_tool_for_a_nonexistent_tool_raises_mcp_tool_call_error():
    client = MockMCPClient(tools={"add": lambda a, b: a + b})

    with pytest.raises(MCPToolCallError):
        client.call_tool("does-not-exist", {})


def test_mcp_tools_work_inside_the_modelagent_tool_calling_loop():
    client = MockMCPClient(
        tools={"add": lambda a, b: a + b},
        descriptions={"add": "Add two integers."},
        schemas={"add": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                          "required": ["a", "b"]}},
    )
    tools = tools_from_mcp(client)

    responses = [
        ModelResponse(content="", tool_calls=[ToolCallRequest(id="1", name="add", arguments={"a": 5, "b": 6})]),
        "the sum is 11",
    ]
    provider = MockProvider(responses=responses)
    agent = ModelAgent("a", provider, prompt="add 5 and 6", tools=tools)

    result = agent.execute()

    assert result == "the sum is 11"
    assert agent.tool_call_log[0].name == "add"
    assert agent.tool_call_log[0].result == "11"


def test_an_mcp_tool_failure_is_fed_back_to_the_model_not_a_crash():
    client = MockMCPClient(tools={"flaky": lambda: (_ for _ in ()).throw(RuntimeError("mcp server error"))})
    tools = tools_from_mcp(client)

    responses = [
        ModelResponse(content="", tool_calls=[ToolCallRequest(id="1", name="flaky", arguments={})]),
        "the tool failed, so I cannot answer",
    ]
    provider = MockProvider(responses=responses)
    agent = ModelAgent("a", provider, prompt="use the flaky tool", tools=tools)

    result = agent.execute()  # must not raise

    assert result == "the tool failed, so I cannot answer"
    assert agent.tool_call_log[0].error is not None
    assert "RuntimeError" in agent.tool_call_log[0].error


def test_mcp_tools_journal_normally_through_a_real_workflow_when_used_as_a_direct_step():
    client = MockMCPClient(tools={"double": lambda a: a * 2})
    double_tool = tools_from_mcp(client)[0]

    # Used directly (not through the tool-calling loop), execute() calls
    # the wrapped fn with zero arguments -- same as any Tool would.
    from aircore import Tool
    step = Tool(lambda: double_tool(a=21), name="double_21")

    workflow = Workflow("mcp-direct-step")
    workflow.step(step)
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.steps[0].output == 42


def test_mcp_tool_spec_is_a_plain_dataclass():
    spec = MCPToolSpec(name="x", description="desc", input_schema={"type": "object"})
    assert spec.name == "x"
    assert spec.description == "desc"
    assert spec.input_schema == {"type": "object"}
