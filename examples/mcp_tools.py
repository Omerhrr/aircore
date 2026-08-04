"""MCP tool registry: a model calling tools that live on an external MCP
server, not just local Python functions. Runs entirely offline against
MockMCPClient (no subprocess, no `mcp` package needed) -- see
examples/mcp_live_server.py + examples/mcp_live_client.py for the same
thing against a real MCP server subprocess.

Run with: python examples/mcp_tools.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, Tool
from airpy import ModelAgent, MockProvider
from airpy.mcp_tools import MockMCPClient, tools_from_mcp
from airpy.providers import ModelResponse, ToolCallRequest


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


if __name__ == "__main__":
    section("Wrapping an MCP server's tools as plain aircore Tools")

    # A real MCP client (StdioMCPClient) would list these from an actual
    # server subprocess -- MockMCPClient stands in for that here, the same
    # way MockProvider stands in for a real model.
    mcp_client = MockMCPClient(
        tools={
            "add": lambda a, b: a + b,
            "weather": lambda city: f"sunny in {city}",
        },
        descriptions={
            "add": "Add two integers.",
            "weather": "Look up the current weather for a city.",
        },
        schemas={
            "add": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            "weather": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    )

    mcp_tools = tools_from_mcp(mcp_client)
    for t in mcp_tools:
        print(f"  {t.name}: {t.description}  schema={t.parameters_schema}")

    section("Calling one directly (no model involved)")
    add_tool = next(t for t in mcp_tools if t.name == "add")
    print(f"add(3, 4) = {add_tool(a=3, b=4)}")

    section("A model using an MCP tool through the tool-calling loop")

    # Scripted so this runs offline/deterministically -- a real provider
    # would decide on its own whether/which tool to call.
    scripted_responses = [
        ModelResponse(content="", tool_calls=[
            ToolCallRequest(id="1", name="weather", arguments={"city": "Lisbon"}),
        ]),
        "It's sunny in Lisbon today.",
    ]
    provider = MockProvider(responses=scripted_responses)
    agent = ModelAgent("assistant", provider, prompt="What's the weather in Lisbon?", tools=mcp_tools)
    answer = agent.execute()

    print(f"agent answer: {answer!r}")
    print(f"tool_call_log: {agent.tool_call_log}")

    section("MCP tools work as ordinary workflow steps too")

    # Used directly (not through the tool-calling loop), it's just a Tool
    # -- capability requires=, retries, journaling, all unchanged.
    step = Tool(lambda: add_tool(a=10, b=32), name="compute")
    workflow = Workflow("MCPDirectStep")
    workflow.step(step)
    journal = workflow.run()
    print(f"workflow step output: {journal.steps[0].output!r}")
