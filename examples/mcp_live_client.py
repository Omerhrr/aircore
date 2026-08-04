"""Live validation of StdioMCPClient against a real MCP server subprocess
(examples/mcp_live_server.py) -- not a mock. This is what was actually run
by hand to catch and fix the two real bugs documented in StdioMCPClient's
docstring (an anyio cancel-scope task-identity rule, and a renamed SDK
attribute -- inputSchema/isError vs. input_schema/is_error).

Requires: pip install mcp

Run:

    python examples/mcp_live_client.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_live_server.py")

try:
    import mcp  # noqa: F401
except ImportError:
    print("Install the mcp package first: pip install mcp")
    sys.exit(1)

from airpy import ModelAgent, MockProvider, StdioMCPClient
from airpy.mcp_tools import tools_from_mcp
from airpy.providers import ModelResponse, ToolCallRequest


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


if __name__ == "__main__":
    section("Connecting to a real MCP server subprocess")

    with StdioMCPClient(command=sys.executable, args=[SERVER_SCRIPT]) as client:
        specs = client.list_tools()
        print("tools the server reports:", [(s.name, s.description) for s in specs])

        section("Calling a real MCP tool directly")
        result = client.call_tool("add", {"a": 3, "b": 4})
        print(f"add(3, 4) = {result!r}")

        section("Wrapped as aircore Tools, driven by ModelAgent's tool-calling loop")
        tools = tools_from_mcp(client)

        scripted_responses = [
            ModelResponse(content="", tool_calls=[
                ToolCallRequest(id="1", name="add", arguments={"a": 5, "b": 6}),
            ]),
            "the sum is 11",
        ]
        agent = ModelAgent("assistant", MockProvider(responses=scripted_responses),
                            prompt="add 5 and 6", tools=tools)
        answer = agent.execute()
        print(f"agent answer: {answer!r}")
        print(f"tool_call_log: {agent.tool_call_log}")

    print("\nclient closed cleanly -- no leftover subprocess or hung threads")
