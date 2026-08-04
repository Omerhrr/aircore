"""MCP tool registry: letting ModelAgent's tools come from an external MCP
server, not just local aircore Tools.

The shape stays the same as everything else in this project: an MCP tool
becomes a plain aircore.Tool (via `tools_from_mcp`), so it's usable exactly
like any hand-written `@tool` function -- `ModelAgent(tools=[...])`,
`workflow.step()`, capability `requires=`, none of that code needs to know
a tool came from MCP instead of a local Python function.

Two layers, deliberately separated:

1. A minimal, duck-typed client contract -- `list_tools() -> List
   [MCPToolSpec]` and `call_tool(name, arguments) -> Any` -- that
   `tools_from_mcp()` builds Tools against. This is the only thing
   `tools_from_mcp` and `_invoke_tool` (model_agent.py) actually depend
   on, so it's fully testable with `MockMCPClient` (in-process, no
   subprocess, no network, no `mcp` package installed) the same way every
   other feature in this project has a Mock* counterpart proven by tests
   before anything real is layered on top.
2. `StdioMCPClient`, a real adapter implementing that contract over the
   official `mcp` Python SDK's stdio transport. Lazily imported (same
   pattern as litellm_provider.py) -- airpy has no hard dependency on
   `mcp` unless you actually construct one. Built and shape-checked
   against the real, installed `mcp` package's API (ClientSession,
   StdioServerParameters, stdio_client), but -- like LiteLLMProvider
   before its live DeepSeek run -- not yet validated against a real,
   running MCP server. Point it at one and it should work; if it doesn't,
   that's a real bug to report, not a "someday" gap.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

from aircore.tools import RequiresArg, Tool


@dataclass
class MCPToolSpec:
    """One tool as an MCP server describes it -- name, human-readable
    description, and its JSON-schema input shape (MCP's own `inputSchema`,
    already JSON-schema shaped, so no conversion is needed the way
    schema.py has to convert a Python signature for local Tools)."""
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPClient(Protocol):
    """The only contract `tools_from_mcp` needs. `MockMCPClient` and
    `StdioMCPClient` both satisfy it; so would a hand-rolled client for an
    HTTP-based MCP transport, with zero changes to `tools_from_mcp`
    itself."""

    def list_tools(self) -> Sequence[MCPToolSpec]: ...

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any: ...


class MCPToolCallError(Exception):
    """Raised by a client's call_tool() when the MCP server itself reports
    an error (or, for MockMCPClient, when the tool name doesn't exist).
    Never raised by tools_from_mcp -- it propagates up through the
    resulting Tool's __call__ exactly like any other tool exception, and
    ModelAgent's _invoke_tool (model_agent.py) already catches any
    Exception there and feeds it back to the model as text, not a crash."""


def tools_from_mcp(client: MCPClient, requires: RequiresArg = None) -> List[Tool]:
    """Lists every tool the given MCP client exposes and wraps each as an
    aircore.Tool, ready to hand to `ModelAgent(tools=tools_from_mcp(client)
    + [...other local tools...])`. `requires` (optional) is applied to
    every tool built this way -- the same capability gate a hand-written
    Tool would declare, since an MCP server is just as much an external
    capability as a local function with side effects, arguably more so."""
    return [_tool_from_spec(client, spec, requires) for spec in client.list_tools()]


def _tool_from_spec(client: MCPClient, spec: MCPToolSpec, requires: RequiresArg) -> Tool:
    def call(**kwargs: Any) -> Any:
        return client.call_tool(spec.name, kwargs)

    call.__name__ = spec.name
    return Tool(
        call,
        name=spec.name,
        description=spec.description,
        requires=requires,
        parameters_schema=spec.input_schema,
    )


class MockMCPClient:
    """An in-process MCPClient needing no subprocess, no network, and no
    `mcp` package -- the MCP equivalent of MockProvider. `tools` maps a
    tool name to the Python callable that "is" that tool for testing
    purposes; `descriptions`/`schemas` are optional per-name overrides,
    defaulting to a plain name-only description and an empty-object
    schema (fine for tests that don't care about exact schema shape)."""

    def __init__(self, tools: Dict[str, Any],
                 descriptions: Optional[Dict[str, str]] = None,
                 schemas: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._tools = tools
        self._descriptions = descriptions or {}
        self._schemas = schemas or {}

    def list_tools(self) -> List[MCPToolSpec]:
        return [
            MCPToolSpec(
                name=name,
                description=self._descriptions.get(name, name),
                input_schema=self._schemas.get(name, {"type": "object", "properties": {}, "required": []}),
            )
            for name in self._tools
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        fn = self._tools.get(name)
        if fn is None:
            raise MCPToolCallError(f"no such MCP tool '{name}'")
        return fn(**arguments)


class StdioMCPClient:
    """A real MCPClient over the official `mcp` Python SDK's stdio
    transport -- connects to an MCP server started as a subprocess
    (`command`/`args`/`env`, same shape as mcp.StdioServerParameters).

    `mcp`'s client API is async, and its `anyio`-based cancel scopes
    require the same asyncio Task to enter and exit a connection's `async
    with` blocks -- opening the connection in one scheduled coroutine and
    closing it in another (the first, simpler approach here) raises
    "Attempted to exit cancel scope in a different task than it was
    entered in" the moment you actually connect to a real server. This
    was caught by running a real toy MCP server locally and connecting to
    it end to end, not guessed at -- see the fix below: one single
    long-lived coroutine (`_session_task`) holds the connection open for
    the whole `with` block and services `list_tools()`/`call_tool()` calls
    off a queue, so every bit of `async with` nesting stays in that one
    task from open to close. Everything outside this file still sees
    plain, synchronous, blocking calls -- no `asyncio` anywhere else.

    Use as a context manager so the subprocess and session are cleanly
    torn down:

        with StdioMCPClient(command="npx", args=["-y", "some-mcp-server"]) as client:
            agent = ModelAgent("a", provider, prompt="...", tools=tools_from_mcp(client))
            agent.execute()

    One connection is held open for the whole `with` block -- list_tools()
    and call_tool() reuse it rather than reconnecting every call.
    """

    def __init__(self, command: str, args: Optional[List[str]] = None,
                 env: Optional[Dict[str, str]] = None, connect_timeout: float = 30.0) -> None:
        self.command = command
        self.args = args or []
        self.env = env
        self.connect_timeout = connect_timeout
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._commands: Any = None  # asyncio.Queue, created inside the loop thread
        self._session_future: Any = None  # concurrent.futures.Future for the whole session task

    def __enter__(self) -> "StdioMCPClient":
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise ImportError(
                "StdioMCPClient requires the 'mcp' package. Install it with: pip install mcp"
            ) from exc

        import concurrent.futures

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        ready: "concurrent.futures.Future[None]" = concurrent.futures.Future()

        async def _session_task() -> None:
            self._commands = asyncio.Queue()
            try:
                params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self._loop.call_soon_threadsafe(ready.set_result, None)
                        while True:
                            item = await self._commands.get()
                            if item is None:  # __exit__'s shutdown signal
                                break
                            coro_factory, result_future = item
                            try:
                                result = await coro_factory(session)
                                self._loop.call_soon_threadsafe(result_future.set_result, result)
                            except Exception as exc:  # noqa: BLE001 -- handed back to the
                                # synchronous caller as a normal exception, not lost here.
                                self._loop.call_soon_threadsafe(result_future.set_exception, exc)
            except Exception as exc:
                if not ready.done():
                    self._loop.call_soon_threadsafe(ready.set_exception, exc)

        self._session_future = asyncio.run_coroutine_threadsafe(_session_task(), self._loop)
        ready.result(timeout=self.connect_timeout)
        return self

    def _call(self, coro_factory: Any) -> Any:
        """Runs `coro_factory(session)` on the session's one long-lived
        task via the command queue, and blocks the calling (synchronous)
        thread for the result."""
        import concurrent.futures

        result_future: "concurrent.futures.Future[Any]" = concurrent.futures.Future()
        asyncio.run_coroutine_threadsafe(
            self._commands.put((coro_factory, result_future)), self._loop
        ).result()
        return result_future.result()

    def list_tools(self) -> List[MCPToolSpec]:
        result = self._call(lambda session: session.list_tools())
        return [
            MCPToolSpec(name=t.name, description=t.description or t.name, input_schema=t.input_schema)
            for t in result.tools
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        result = self._call(lambda session: session.call_tool(name, arguments))
        text = "\n".join(
            block.text for block in result.content if getattr(block, "type", None) == "text"
        )
        # `is_error`, not `isError` -- caught by running this against a
        # real toy MCP server (see the class docstring); the SDK's actual
        # attribute name doesn't match the camelCase wire field it's
        # populated from.
        if getattr(result, "is_error", False):
            raise MCPToolCallError(f"MCP tool '{name}' returned an error: {text}")
        return text

    def __exit__(self, *exc_info: Any) -> None:
        if self._loop is not None and self._commands is not None:
            # Signals _session_task to fall out of its while loop and then
            # unwind its own `async with` blocks -- in the same task that
            # opened them, which is exactly what anyio's cancel scopes
            # require.
            asyncio.run_coroutine_threadsafe(self._commands.put(None), self._loop).result()
        if self._session_future is not None:
            self._session_future.result(timeout=10)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
