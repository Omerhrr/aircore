"""A tiny real MCP server (two tools: add, greet), used by
examples/mcp_live_client.py to validate StdioMCPClient against an actual
MCP server subprocess -- not a mock. Requires the `mcp` package
(`pip install mcp`).

Not meant to be run directly except by mcp_live_client.py, which spawns
it as a subprocess (same relationship examples/live_deepseek.py has to a
real DeepSeek API key -- manual, live, not part of the automated test
suite).
"""

from mcp.server.mcpserver import MCPServer

server = MCPServer("aircore-example-server")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@server.tool()
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    server.run(transport="stdio")
