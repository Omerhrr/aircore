"""airpy -- the provider-aware Python SDK layer on top of aircore.

This is where AI begins. aircore (the runtime) has no idea this package
exists and never imports anything from it -- airpy imports aircore, never
the other way around. That one-directional dependency is what keeps the
scheduler, capabilities, policy, journal, and consensus machinery honest
about being an execution runtime, not an AI framework with execution
features attached.

Contents:
- ModelProvider / ModelRequest / ModelResponse (providers.py): the
  interface any real adapter (OpenAI, Anthropic, etc.) implements.
- MockProvider (mock_provider.py): a ModelProvider needing no API key,
  used to prove the abstraction without real network calls or cost.
- ModelAgent (model_agent.py): an Executable backed by a ModelProvider --
  a first-class, interchangeable workflow step alongside Tool. See its
  docstring for why it's not named `Agent`.
- ask() (ask.py): one-line sugar for calling a model outside a Workflow,
  with none of the journal/policy/capability/retry guarantees a real
  workflow step gets.

M8: LiteLLMProvider (litellm_provider.py) is the first real adapter --
    lazily imports `litellm`, so airpy has no hard dependency on it unless
    you actually construct one. ModelAgent.usage() (see model_agent.py)
    reports whatever real usage/cost the response carries, which is how
    tokens/cost now flow into aircore's Metrics and Policy.max_cost without
    aircore ever knowing what "tokens" or a "model" are -- see
    executable.py's generic usage() hook.

OpenAIProvider (openai_provider.py): the first *native* (non-LiteLLM)
    adapter -- built directly against the `openai` package (lazily
    imported, same pattern) instead of proxying through litellm, so this
    project has one real adapter proving ModelProvider isn't secretly
    litellm-shaped. `base_url=` support means it also reaches any
    OpenAI-compatible endpoint, not just real OpenAI -- see its module
    docstring for how that's used to live-test it against DeepSeek's
    OpenAI-compatible API, reusing this project's existing
    DEEPSEEK_API_KEY, with no new credentials needed. One honest gap
    relative to LiteLLMProvider: no pricing lookup exists in the `openai`
    SDK, so `Usage.cost_usd` is always None here (never a guessed value).
Tool-calling loop: ModelAgent(tools=[...]) lets a model actually call
    aircore Tools -- schema.py builds a JSON schema from a Tool's Python
    signature, ModelRequest/ModelResponse carry tool schemas and parsed
    tool-call requests, and ModelAgent runs the ReAct-style loop. See
    model_agent.py's docstring for the two real limitations (bypasses the
    Scheduler entirely -- not journaled, not Policy-enforced; `identity=`
    is a partial capability-check mitigation, not a full fix).

JudgeConsensus (judge_consensus.py): a consensus strategy for aircore's
    ConsensusGroup (see aircore/consensus.py) that uses a model call to judge
    agreement instead of exact-string matching -- suited to free-text
    ModelAgent outputs, where majority()/unanimous() almost always fail
    even on substantively-agreeing answers. Lives entirely in airpy;
    aircore's Scheduler only ever sees it as a plain callable.

Structured output (structured_output.py): ModelAgent(output_schema=...)
    and JudgeConsensus(output_schema=..., confidence=...) both return
    validated, typed values -- a plain dict, or a Pydantic instance if
    output_schema was a Pydantic BaseModel subclass -- instead of raw
    text. Pydantic is optional and lazily used, same lazy-dependency
    pattern as litellm: a JSON-schema dict works with no extra
    dependency at all. Provider adapters may use ModelRequest.
    response_schema as a hint to enable a real JSON-mode API parameter
    (LiteLLMProvider does), but never validate anything themselves --
    parsing/validation is uniform across every provider, including
    MockProvider, in this one module.

MCP tool registry (mcp_tools.py): `tools_from_mcp(client)` turns any
    MCP server's tools into plain aircore Tools, usable in
    ModelAgent(tools=...) exactly like a hand-written @tool function --
    schema.py reads the MCP server's own JSON schema (via Tool.
    parameters_schema, see tools.py) instead of introspecting a Python
    signature. `MockMCPClient` proves this offline, no `mcp` package or
    subprocess required, same role as MockProvider; `StdioMCPClient` is
    the real adapter, lazily importing `mcp`, verified end to end against
    a live toy MCP server (see its docstring for the two real bugs that
    caught -- an anyio cancel-scope rule and a renamed SDK attribute).

Streaming (ModelAgent.stream(), in model_agent.py): yields text chunks as
    they arrive instead of blocking for the whole completion. Bypasses
    aircore's Scheduler entirely, the same way ask() does -- there's no
    single point in time a streamed answer "finished" that a Scheduler
    step's atomic StepStarted..StepFinished sequence could record. Not
    supported together with `tools=` (raises immediately, not lazily).

Session (session.py): a long-running, stateful conversation --
    `session.send(message)` instead of hand-constructing a new ModelAgent
    every turn. Unlike ask()/stream(), each turn runs through a real,
    one-step Workflow, so it's journaled and capability/Policy-enforced
    exactly like any other workflow step -- see session.py's docstring
    for what this adds on top of ModelAgent(memory=...) alone (session
    metadata, a per-turn journal audit trail, and max_history_turns to
    bound growth for a conversation meant to stay open a long time).

Provider catalog (provider_catalog.py): named constructors --
    `openai()`, `anthropic()`, `deepseek()`, `gemini()`, `qwen()`,
    `nvidia()`, `zai()`, `ollama()`, `lmstudio()`, `openrouter()` -- each a
    thin wrapper that builds a correctly-configured LiteLLMProvider (right
    model-string prefix, e.g. `anthropic/`, `dashscope/` for Qwen,
    `nvidia_nim/`, baked in) instead of a new adapter per provider. Chosen
    over ten native-SDK adapters because LiteLLM already reaches all of
    them through one call shape with zero extra dependencies -- see the
    module's docstring for exactly what was and wasn't validated (only
    DeepSeek has been exercised against a real API in this project; the
    other nine are covered by offline tests asserting the resulting
    model string, not a live call, since no API keys for them were
    available here).

PromptTemplate (prompt_template.py): `{variable}` substitution for
    building prompts from named pieces -- `PromptTemplate("Summarize
    {topic}").render(topic="deepseek")`. Built to close half of
    airlang-spec-v1.md section 5.4's gap (AirLang's `let`/agent `prompt` fields
    had nothing to plug values into). Deliberately minimal: named
    `{variable}` fields only (built on str.format()), no expression
    language, missing/extra variables both fail loudly at render() time
    rather than silently producing a bad prompt.

Cross-step data flow (model_agent.py's `prompt_bindings=`): the other
    half of the gap PromptTemplate alone didn't solve -- where a
    variable's *value* actually comes from at execution time -- is now
    closed too. `ModelAgent(prompt=PromptTemplate(...), prompt_bindings=
    workflow.bindings)` renders the template at execute() time, not
    construction time, reading whatever `aircore.Workflow.step(tool,
    as_="name")` has bound so far (see aircore/workflow.py's "Bindings"
    section and cross-step-data-flow.md). This closes it for hand-written
    airpy code; AirLang's `let` specifically still needs one more, smaller
    piece (artifact-to-producing-step linkage) before it can use this too
    -- see airlang-spec-v1.md section 5.4's status note.

Still not implemented: allowed_models-style policy (deliberately kept out
of aircore's Policy -- see policy.py), non-stdio MCP transports (HTTP/SSE),
streaming combined with tool-calling. Native provider SDKs: one now
exists (OpenAIProvider, above) -- the other nine provider_catalog.py
entries (anthropic, gemini, qwen, nvidia, zai, ollama, lmstudio,
openrouter) still go through LiteLLMProvider, unchanged, since nothing
has driven building a native adapter for any of them specifically.

Facade (Agent, Workflow, Tool, tool): airpy is meant to be the whole
    developer-facing surface -- `from airpy import Agent, Workflow`, never
    `from aircore import Workflow`. `Agent` is exactly `ModelAgent` under a
    friendlier name: model_agent.py's docstring proposed this exact alias
    from the start ("if you want a single name for this concept in your
    own code, alias it on import: `from airpy import ModelAgent as
    Agent`") -- this just makes that airpy's own default instead of
    something every caller has to do themselves. It's a plain alias, not a
    subclass, because there is nothing Agent-specific to add yet; inventing
    behavior here with no real use case driving it would be exactly the
    kind of speculative feature this project has consistently avoided.
    `Workflow`, `Tool`, and `tool` are re-exported unchanged from aircore --
    `Workflow("Audit").parallel(a, b, c).consensus(strategy=...)` already
    chains (see aircore/workflow.py's ParallelResults), so no new fluent
    wrapper was needed, just making aircore's existing chainable API
    reachable without an aircore import. aircore itself is unchanged and still
    doesn't know airpy exists.
"""

from aircore import Tool, Workflow, tool

from .ask import ask
from .judge_consensus import JudgeConsensus, JudgeConsensusFailed
from .litellm_provider import LiteLLMProvider
from .mcp_tools import MCPClient, MCPToolCallError, MCPToolSpec, MockMCPClient, StdioMCPClient, tools_from_mcp
from .mock_provider import MockProvider
from .model_agent import ModelAgent, ModelAgentToolLoopExceeded, ToolCallRecord
from .openai_provider import OpenAIProvider
from .provider_catalog import (
    anthropic, deepseek, gemini, lmstudio, nvidia, ollama, openai, openrouter, qwen, zai,
)
from .prompt_template import PromptTemplate, PromptTemplateError
from .providers import ModelProvider, ModelRequest, ModelResponse, Usage, ToolSchema, ToolCallRequest
from .session import Session, SessionClosed, SessionTurnFailed
from .structured_output import StructuredOutputError

# See "Facade" above -- Agent is deliberately just another name for
# ModelAgent, not a subclass.
Agent = ModelAgent

__all__ = [
    "Agent", "Workflow", "Tool", "tool",
    "ModelProvider", "ModelRequest", "ModelResponse", "Usage",
    "ToolSchema", "ToolCallRequest",
    "MockProvider", "LiteLLMProvider", "OpenAIProvider", "ModelAgent",
    "ModelAgentToolLoopExceeded", "ToolCallRecord", "ask",
    "JudgeConsensus", "JudgeConsensusFailed",
    "StructuredOutputError",
    "MCPClient", "MCPToolSpec", "MCPToolCallError", "MockMCPClient", "StdioMCPClient", "tools_from_mcp",
    "Session", "SessionClosed", "SessionTurnFailed",
    "openai", "anthropic", "deepseek", "gemini", "qwen", "nvidia", "zai", "ollama", "lmstudio", "openrouter",
    "PromptTemplate", "PromptTemplateError",
]

__version__ = "0.7.0"
