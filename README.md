# aircore / airpy / airlang / aircli

A Python library for building multi-step AI agent workflows, where each
step is either a plain Python function or a call to an LLM, run through a
scheduler that gives you a full audit log, permission checks, human
approval gates, crash-safe resume, and process sandboxing, as real,
built-in mechanisms rather than something you write yourself.

A `Workflow` is a named sequence of steps. Each step is a `Tool` (an
ordinary Python function) or a `ModelAgent` (a call to an LLM); the
scheduler treats them identically, so the same workflow can mix plain
Python logic and AI calls freely:

```python
from aircore import Workflow, tool

@tool
def fetch_data():
    return {"revenue": 1_200_000}

workflow = Workflow("Report").step(fetch_data)
journal = workflow.run()
```

Calling `.run()` doesn't just execute the steps. It produces a `Journal`,
a structured record of exactly what ran, in what order, with what input
and output, how long each step took, and whether it succeeded, that you
can inspect, log, or render as a trace afterward. On top of that base, the
runtime gives you concrete guarantees most hand-rolled agent code doesn't
have:

- A tool can declare `requires=Network` (or `Email`, `Payments`, any
  capability you define); an agent identity is granted a specific set of
  capabilities; a call that doesn't match is refused before the tool ever
  runs, not caught after something already happened.
- `Policy(approval_for={"send_payment"})` means that specific tool call
  blocks on a real approval decision (a terminal prompt, a Slack reply, a
  rule engine, whatever `approval_callback` you write) before it executes.
- `workflow.run(checkpoint_store=..., run_id=...)` means a process that
  crashes halfway through can be rerun and will skip every step, including
  an already-paid LLM call, that already succeeded.
- `Sandbox(max_runtime=30, max_memory_mb=512)` on a single step runs it in
  a real subprocess with a timeout and a best-effort resource limit.
- `.consensus()` runs several agents concurrently and reduces their
  answers to one, failing loudly (`ConsensusFailed`) on a genuine
  disagreement instead of silently picking a winner.
- `ModelAgent(tools=[...], use_mindgraph=True)` keeps a long tool-calling
  loop's token cost from growing every turn, by summarizing tool results
  into a compact memory instead of resending them in full on every
  subsequent call.

None of this is bolted onto a chat-completions wrapper after the fact.
It's what the scheduler actually does on every run, whether or not any
step involves an LLM at all.

---

## Table of contents

1. [Introduction](#introduction)
2. [The four packages](#the-four-packages)
3. [Install](#install)
4. [Quickstart](#quickstart)
5. [Core concepts](#core-concepts)
6. [What aircore gives you](#what-aircore-gives-you)
   - [Sequential and parallel steps](#sequential-and-parallel-steps)
   - [Capabilities](#capabilities)
   - [Policy](#policy)
   - [Retries](#retries)
   - [Consensus](#consensus)
   - [Memory](#memory)
   - [Approval](#approval)
   - [Checkpoint and resume](#checkpoint-and-resume)
   - [Sandboxed execution](#sandboxed-execution)
   - [Cross-step data flow](#cross-step-data-flow)
   - [Observability](#observability)
   - [MindGraph](#mindgraph)
7. [What airpy adds on top](#what-airpy-adds-on-top)
   - [The tool-calling loop](#the-tool-calling-loop)
   - [Token-efficient tool-calling](#token-efficient-tool-calling)
   - [Providers](#providers)
   - [Structured output](#structured-output)
   - [Consensus by LLM judge](#consensus-by-llm-judge)
   - [Streaming](#streaming)
   - [Long-running sessions](#long-running-sessions)
   - [PromptTemplate](#prompttemplate)
   - [MCP tools](#mcp-tools)
8. [AirLang: workflows as data](#airlang-workflows-as-data)
9. [The ai CLI](#the-ai-cli)
10. [What you can build with this](#what-you-can-build-with-this)
11. [Comparison with alternatives](#comparison-with-alternatives)
12. [Design principles](#design-principles)
13. [Examples and tests](#examples-and-tests)
14. [FAQ](#faq)

---

## Introduction

Most "AI agent" code people write by hand looks the same: send a prompt to
a model, get text back, loop again if the model asked for a tool. That
works for a demo. It stops working the moment a real system needs to
answer questions the demo never had to: what exactly did the agent do, and
in what order? Did it have permission to send that email? Who approved
the payment it just made? If the process dies halfway through a long job,
does it start over from zero, including paying again for LLM calls it
already made? What happens when two agents disagree with each other? Why
does a long-running agent's API bill keep climbing even when the work
it's doing hasn't changed?

`aircore` answers those by treating an AI call as one more kind of step in
an ordinary task scheduler, rather than treating the scheduler as
something you bolt onto an AI call. `aircore` itself has no idea what an
LLM is: it's a general-purpose scheduler, permission model, policy engine,
journal, retry mechanism, memory system, approval gate, checkpoint store,
and process sandbox, all as plain, dependency-free Python. `airpy` sits
one layer above it and adds the AI-specific part: `ModelAgent`, real
provider adapters, a tool-calling loop, structured output, multi-agent
consensus, and a token-efficient memory for long tool-calling loops. Since
a `ModelAgent` is scheduled, journaled, retried, and permission-checked by
the exact same mechanism a plain Python `Tool` is, every guarantee listed
above applies to an LLM call automatically, with no separate AI-specific
code path to keep in sync.

For concrete examples of what gets built on top of this, see [What you
can build with this](#what-you-can-build-with-this). Every code snippet in
this README is real and runs offline against `MockProvider` (no API key,
no network call) unless it says otherwise; most are trimmed directly from
a matching file in `examples/`.

---

## The four packages

The project is split into four independently installable packages with a
strict, one-directional dependency chain:

```
aircore   <-  airpy   <-  airlang
   ^           ^             ^
   └───────────┴─────────────┘
              aircli
```

| Package | What it is |
|---|---|
| **`aircore`** | The execution runtime. Scheduler, capabilities, policy, journal, consensus, memory, approval, checkpoint/resume, sandboxed execution, cross-step data flow, a token-efficient context memory (MindGraph). No model, prompt, or provider concept exists anywhere in this package, by design. Zero dependencies. |
| **`airpy`** | The provider-aware Python SDK on top of `aircore`. `ModelAgent`, real provider adapters (`LiteLLMProvider`, a native `OpenAIProvider`, a named catalog for ten providers), a ReAct-style tool-calling loop, consensus-by-LLM-judge, structured output, an MCP tool registry, streaming, long-running sessions, `PromptTemplate`. Imports `aircore`. `aircore` never imports `airpy`. |
| **`airlang`** | AirLang, a small declarative workflow language: a lexer, parser, IR, and an executor that compiles a parsed `.airlang` file into real `airpy`/`aircore` calls, for defining a workflow's shape as data instead of Python. Imports `airpy`. |
| **`aircli`** | The `ai` command. `ai run` and `ai trace` for both `.py` scripts and `.airlang` files, with `--json`/`--html` trace output. Depends on all three of the above. |

You don't have to use all four. Plenty of real uses only ever touch one or
two:

- **Just `aircore`.** Plain-Python task orchestration with real
  journaling, retries, and permission checks. No AI involved at all. Think
  of it as a small, embeddable alternative to a heavier workflow engine
  when you don't want a separate scheduler service, just a library.
- **`aircore` + `airpy`.** The common case. AI agents as first-class
  workflow steps, with everything `aircore` provides underneath them.
- **Add `airlang`** once you want a workflow's shape defined as a small,
  readable file instead of Python, for example so a non-engineer on your
  team can read (or even edit) what an agent pipeline actually does.
- **Add `aircli`** once you want to run and inspect those workflows from a
  terminal, with a visual trace instead of reading raw print statements.

---

## Install

`airpy`'s and `aircli`'s PyPI distribution names differ from their import
names. `airpy` and `aircli` were both too close to existing PyPI project
names and got rejected at registration, so they're published as `airpyy`
and `airclii`. You still `import airpy` and run the `ai` command exactly as
shown below; only the `pip install` name is different.

```bash
pip install aircore           # just the runtime
pip install airpyy            # runtime + SDK           (import as: airpy)
pip install airlang           # runtime + SDK + language frontend
pip install airclii           # everything, plus the ai command
```

Optional extras on `airpy`. Each one is a lazy import inside the single
module that needs it, so none of them are required just to install `airpy`
itself:

```bash
pip install "airpyy[litellm]"   # LiteLLMProvider, real access to 10+ model providers
pip install "airpyy[openai]"    # native OpenAIProvider, built directly on the openai package
pip install "airpyy[pydantic]"  # output_schema= as a Pydantic model, not just a JSON schema dict
pip install "airpyy[mcp]"       # StdioMCPClient, a real MCP server subprocess
```

For local development on this repo (all four packages, editable), install
from the root `pyproject.toml`:

```bash
git clone <this repo>
cd aircore
pip install -e ".[dev]"
pytest
```

---

## Quickstart

The smallest possible workflow: one plain Python function, run through the
scheduler, with a full journal to show for it.

```python
from aircore import Workflow, tool

@tool
def hello():
    return "Hello, World!"

workflow = Workflow("Hello")
workflow.step(hello)
journal = workflow.run()

print(journal.pretty())     # human-readable trace
print(journal.to_json())    # structured, for a dashboard or log
```

Add an LLM into the mix. `ModelAgent` (aliased as `Agent` in `airpy`'s
facade) is just another workflow step, nothing special about it:

```python
from airpy import Agent, MockProvider, Workflow

researcher = Agent("researcher", MockProvider(response="Q3 revenue grew 12%."),
                    "Summarize Q3 revenue performance.")

workflow = Workflow("Research").step(researcher)
journal = workflow.run()
print(journal.steps[0].output)
```

`MockProvider` needs no API key and makes no network call. Swap it for
`LiteLLMProvider(model="gpt-4o-mini")`, or any of the [named provider
constructors](#providers), to talk to a real model with zero other code
changes.

---

## Core concepts

Before the feature-by-feature walkthrough, four names you'll see
constantly:

- **`Workflow`.** A named sequence of steps. `.step()` runs one tool or
  agent. `.parallel()` runs several concurrently. `.consensus()` runs
  several concurrently and reduces their outputs to one agreed answer.
  `.run()` executes the whole thing and returns a `Journal`.
- **`Tool`** (the `@tool` decorator). Wraps a plain Python function as a
  named, schedulable, journaled unit of work. `ModelAgent` implements the
  exact same `Executable` interface, so an LLM call is a first-class,
  interchangeable step right alongside a `Tool`. `workflow.step()`,
  `.parallel()`, and `.consensus()` genuinely don't care which one they
  were handed.
- **`Journal`.** The complete, structured, replayable record of one run:
  every step's status, timing, output or error, retry history, usage and
  cost, and any approval decisions made along the way. `journal.pretty()`
  gives you a human-readable trace. `journal.to_json()` gives you a
  machine-readable one for a dashboard or log pipeline.
- **`Agent` (identity), `aircore.Agent`.** Not the same thing as
  `airpy.Agent` or `ModelAgent`. This one answers "who is executing," a
  name plus a set of granted `Capability` objects. `ModelAgent` answers
  "what gets executed." A tool declared `Tool(requires=Network)` demands
  that capability from whoever's calling it. Attach an identity via
  `workflow.step(tool, agent=some_identity)` and the scheduler checks the
  capability before the tool ever runs, not after something has already
  gone wrong.

---

## What aircore gives you

### Sequential and parallel steps

```python
workflow = Workflow("Research")
workflow.parallel(search_web, search_github)   # both run concurrently
workflow.step(merge)                            # then this, sequentially, once both finish
```

`.parallel()` fans out to worker threads and rejoins the sequential flow
only once every tool in the block has finished. A step after a parallel
block always sees the block as fully complete.

### Capabilities

A permission model checked before a tool runs, not after, and not by
convention. An agent identity carries a set of granted capabilities; a
tool declares what it requires; the scheduler refuses the call if the two
don't line up.

```python
from aircore import Workflow, tool, Agent, Network, Email

@tool(requires=Network)
def fetch_page():
    ...

@tool(requires=Email)
def send_email():
    ...

researcher = Agent("Researcher", capabilities=[Network])

workflow = Workflow("Demo")
workflow.step(fetch_page, agent=researcher)   # allowed, researcher has Network
workflow.step(send_email, agent=researcher)   # denied, researcher has no Email capability
```

Built-in capabilities are `Network`, `Filesystem`, `Email`, `Payments`, and
`Database`. Define your own with `Capability("YourName")` for anything
project-specific, for example a `PaymentRefund` capability that only a
specific identity is granted.

### Policy

Production-mode guardrails you set once on a `Workflow`, not something you
have to remember to check inside every tool.

```python
from aircore import Policy, PolicyViolation

# require every step to have an identified agent. an anonymous step is a
# pre-flight PolicyViolation, not a silently unrestricted call
prod = Workflow("Prod", policy=Policy(require_agent=True))

# cap concurrency and total wall-clock time
Policy(max_parallel=5, max_runtime=30.0)

# cap total spend, checked against real usage and cost as it's reported
Policy(max_cost=0.50)

# require a human decision before specific tools can run at all
Policy(approval_for={"send_payment", "delete_database"})
```

A violation raises `PolicyViolation` before the offending step (or the
whole run) ever executes. It's a pre-flight check, not a runtime surprise.

### Retries

Retries are opt-in and gated on the tool actually declaring itself safe to
repeat. There's no global "retry everything" switch to accidentally leave
on.

```python
@tool(idempotent=True, retries=3)
def flaky_read():
    ...   # retried up to 3 times on failure

@tool                            # idempotent=False, the default
def send_payment():
    ...   # never retried, even if it might have succeeded on a later try
```

Declaring `retries > 0` on a tool that isn't marked `idempotent=True` is a
`ValueError` at construction time. It's caught before you ever run
anything, not discovered in production when a payment gets sent twice.

### Consensus

Run several voters concurrently and reduce their outputs to one agreed
answer, or fail loudly if they don't agree.

```python
from aircore import Workflow, unanimous

workflow = Workflow("Vote")
workflow.consensus(model_a, model_b, model_c)                    # majority, the default
workflow.consensus(model_a, model_b, model_c, strategy=unanimous) # must all agree, or fail
```

A genuine tie, a non-unanimous vote under `unanimous`, or any single voter
raising an exception all fail with `ConsensusFailed` rather than silently
guessing a winner. That refusal to guess is deliberate: a consensus
mechanism that picks a winner on a tie isn't consensus, it's a coin flip
wearing a consensus mechanism's clothes. For free-text LLM answers, where
exact-string matching almost always fails even when the answers
substantively agree, use `airpy`'s `JudgeConsensus` instead, covered
[below](#consensus-by-llm-judge).

### Memory

Three scopes, three different lifetimes, so you pick the one that matches
what you actually need instead of reinventing scoping rules with a global
dict.

```python
from aircore import Memory

mem = Memory()
mem.temporary   # shared between steps within ONE run, wiped automatically after it finishes
mem.session     # persists across multiple run() calls made on this same Memory object
mem.project     # shared across any Memory instance constructed with the same project= name

# durable across process restarts, not just in-memory:
from aircore import FileMemoryScope
mem = Memory(session=FileMemoryScope("./session_state.json"))
```

`temporary` is for passing data between steps in a single run without
threading it through every function signature. `session` is for a
conversation or process that spans multiple separate `run()` calls.
`project` is for sharing state between entirely separate parts of a
system, for example a researcher agent logging a finding that a
completely separate auditor agent later reads, without either one knowing
about the other's `Memory` object directly.

### Approval

A human gate in front of the actions that actually matter, not a
suggestion buried in a comment somewhere.

```python
from aircore import Policy, auto_approve, cli_approval_callback

policy = Policy(approval_for={"send_payment"})
workflow = Workflow("Payments", policy=policy)
workflow.step(send_payment)

# a real, blocking y/n prompt in a terminal:
workflow.run(approval_callback=cli_approval_callback)

# or, for automated and CI runs where you've decided this is safe to auto-approve:
workflow.run(approval_callback=auto_approve)
```

You can also write your own `approval_callback`, any
`Callable[[ApprovalRequest], bool]` works, for example one that posts to
Slack and waits on a reply, or checks a request against a rule engine.

### Checkpoint and resume

Crash-safe, durable runs. If the process dies halfway through, rerunning
the same script picks up exactly where it left off instead of starting
over and, worse, re-paying for LLM calls that already succeeded.

```python
from aircore import FileCheckpointStore

store = FileCheckpointStore("./checkpoint.json")
journal = workflow.run(checkpoint_store=store, run_id="daily-report-2026-01-15")
```

Rerunning the same script with the same `run_id` skips every
already-succeeded step, including an already-answered, paid LLM call. This
is proven end to end against a real API call, not just asserted, in
`examples/production_readiness.py`, which checkpoints a workflow that
makes a real DeepSeek call and shows the second run genuinely skipping it.

### Sandboxed execution

Real process isolation for a single step, when you need one piece of a
workflow to run somewhere other than your main process.

```python
from aircore import Tool, Sandbox

def risky_computation():
    ...  # must be a module-level function, picklable, not a lambda or closure

workflow.step(Tool(risky_computation, name="risky",
                    sandbox=Sandbox(max_runtime=30, max_memory_mb=512)))
```

The step runs in a real subprocess with a wall-clock timeout, a
best-effort memory limit, and a best-effort network egress allowlist.
Being honest about scope here matters: this is not a container and not a
kernel namespace. See `aircore/sandbox.py` for exactly what is and isn't
guaranteed before you rely on it for anything security-sensitive.

### Cross-step data flow

Later steps reading earlier steps' real output, resolved at execute() time
rather than baked in up front.

```python
workflow = Workflow("Pipeline")
workflow.step(fetch_data, as_="raw")
workflow.step(Tool(lambda: workflow.bindings["raw"]["topic"], name="extract"), as_="topic")

# a PromptTemplate renders fresh at execute() time, reading whatever's
# actually in workflow.bindings by then, not a fixed string decided up front
from airpy import PromptTemplate, ModelAgent
template = PromptTemplate("Explain {topic} in two sentences.")
explainer = ModelAgent("explainer", provider, template, prompt_bindings=workflow.bindings)
workflow.step(explainer)
```

`as_="name"` binds whatever a step returns under that name into
`workflow.bindings`, a plain dict the scheduler fills in as each step
actually finishes. A `PromptTemplate` reading from it doesn't get
rendered until the moment it's actually sent to the model, so it always
reflects the real, current state of the run, not a snapshot from before
earlier steps had actually completed.

### Observability

Metrics and an execution graph, collected automatically. There's no
separate instrumentation step to remember.

```python
workflow.run()
print(workflow.metrics.summary())                  # step and tool counts, latency, usage and cost totals

from aircore import build_execution_graph, render_execution_graph
graph = build_execution_graph(workflow.journal)
print(render_execution_graph(graph))                # a tree view of what ran, in what order
```

### MindGraph

A ReAct-style tool-calling loop tends to accumulate context linearly:
every tool result gets resent verbatim on every subsequent turn, so a
2,000-token result paid for once becomes a 2,000-token tax on every
remaining turn of that loop. A long-running agent has the same problem
over wall-clock time instead of turns: state from hour one is still being
paid for on hour twelve.

`aircore.MindGraph` is a compact, graph-shaped memory built to fix both.
Each piece of information becomes a short, dense summary plus a pointer
back to the real value, and rendering a bounded neighborhood around any
node keeps prompt size roughly constant no matter how long the run has
been going. Old nodes can also be periodically folded into a single
aggregate summary instead of either being kept forever or silently
dropped.

You don't usually construct a `MindGraph` by hand. See [Token-efficient
tool-calling](#token-efficient-tool-calling) below for the `ModelAgent`
integration that actually uses it day to day.

---

## What airpy adds on top

### The tool-calling loop

```python
from aircore import tool, Network
from airpy import ModelAgent, MockProvider, ModelResponse, ToolCallRequest

@tool(requires=Network, description="Look up the current weather for a city")
def get_weather(city: str):
    return {"Lagos": "31C, sunny"}.get(city, "no data")

scripted = MockProvider(responses=[
    ModelResponse(content="", tool_calls=[ToolCallRequest(id="1", name="get_weather", arguments={"city": "Lagos"})]),
    "It's 31C and sunny in Lagos right now.",
])
weather_bot = ModelAgent("weather_bot", scripted, prompt="What's the weather in Lagos?", tools=[get_weather])

answer = weather_bot.execute()
print(weather_bot.tool_call_log)   # every call made, its arguments, and its result, inspectable after the fact
```

`tools=` turns any `ModelAgent` into a real, ReAct-style agent. It calls
`aircore.Tool` objects, gets results back, and keeps going until it has a
final answer, or until `max_turns` is exceeded, which raises
`ModelAgentToolLoopExceeded` rather than looping forever. Pass
`identity=some_agent_identity` to get the exact same capability checks as
a normal workflow step: a tool call the identity lacks the capability for
is denied and reported back to the model as plain text, so the model can
adapt or explain the limitation, rather than crashing the whole loop.

### Token-efficient tool-calling

```python
weather_bot = ModelAgent(
    "weather_bot", provider, prompt="...", tools=[get_weather],
    use_mindgraph=True,          # summarize tool results instead of resending them raw
)
```

With `use_mindgraph=True`, every tool result is summarized into a compact
`MindGraph` node before it re-enters the conversation, instead of being
sent back to the model in full and then resent, in full, on every
subsequent turn. Summarization is deterministic and free (no extra LLM
call, no extra tokens spent) for common structured data like numbers and
OHLC-style candle records, via a built-in summarizer, or you can register
your own per tool with `summarizers={tool_name: fn}`. An `expand_node`
tool is auto-injected into the loop so the model can still pull back a
specific result's exact, un-summarized value on demand, when the summary
genuinely isn't enough. `agent.mindgraph_savings` records the measured
raw-versus-summary token estimate for every call, so the reduction is
something you can actually see, not just something that's claimed.

Share one `MindGraph` across multiple agents, for example two consensus
specialists independently analyzing the same underlying input, and an
identical `(tool, arguments)` call made by the second agent is served
straight from the first agent's already-summarized node instead of being
invoked again:

```python
from aircore import MindGraph
shared = MindGraph()
specialist_a = ModelAgent("a", provider, prompt="...", tools=tools, use_mindgraph=True, mindgraph=shared)
specialist_b = ModelAgent("b", provider, prompt="...", tools=tools, use_mindgraph=True, mindgraph=shared)
```

For a long-running process, an always-on agent loop that might run for
days, periodically fold old nodes into one aggregate summary instead of
letting the graph grow without bound:

```python
if len(shared) > 200:
    shared.compact_oldest(100, summarize=lambda nodes: f"{len(nodes)} cycles summarized")
```

The node count settles into a steady state this way. A process that's been
running for a week doesn't cost more per cycle than one that just started.

### Providers

```python
from airpy import MockProvider, LiteLLMProvider, OpenAIProvider
from airpy import openai, anthropic, deepseek, gemini, qwen, nvidia, zai, ollama, lmstudio, openrouter

provider = deepseek()                              # LiteLLMProvider(model="deepseek/deepseek-chat")
provider = anthropic("claude-3-5-sonnet-20241022")  # LiteLLMProvider(model="anthropic/claude-3-5-sonnet-20241022")
provider = OpenAIProvider(model="gpt-4o-mini")      # a native adapter, not routed through litellm
```

`LiteLLMProvider` reaches over a hundred models through one unified call
shape. The ten named constructors above are ergonomic sugar over it: a
single function call instead of having to remember that Qwen's prefix on
LiteLLM is `dashscope/`, not `qwen/`, or that Nvidia NIM's is
`nvidia_nim/`. Real usage and cost from the provider's response flows
automatically into the journal, `workflow.metrics`, and `Policy.max_cost`,
with no extra wiring on your part.

### Structured output

```python
SCHEMA = {"type": "object", "properties": {
    "summary": {"type": "string"}, "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
}, "required": ["summary", "risk_level"]}

agent = ModelAgent("auditor", provider, "Audit this system.", output_schema=SCHEMA)
journal = Workflow("Audit").step(agent).run()
journal.steps[0].output   # a validated dict, not a string you have to parse yourself
```

`output_schema` also accepts a Pydantic `BaseModel` subclass, in which case
you get a real, validated instance back instead of a plain dict, if
`pydantic` is installed.

### Consensus by LLM judge

Exact-string `majority()` and `unanimous()` from `aircore` almost always
fail on free-text LLM answers that substantively agree but aren't
word-for-word identical. `JudgeConsensus` uses a model call to judge
agreement instead of a string comparison.

```python
from airpy import JudgeConsensus
judge = JudgeConsensus(provider, output_schema=SCHEMA, confidence=True)
Workflow("Audit").parallel(researcher, reviewer, professor).consensus(strategy=judge).run()
```

`confidence=True` adds a typed confidence score and a short reasoning
string to the journal, parsed structurally rather than scraped out of free
text.

### Streaming

```python
for chunk in agent.stream():
    print(chunk, end="", flush=True)
```

Streaming bypasses the scheduler on purpose. There's no single moment a
streamed answer "finished" that atomic per-step journaling could
meaningfully record, so this is a deliberate, direct-use convenience, not
something usable inside a workflow step. It isn't available once `tools=`
is set on the agent.

### Long-running sessions

```python
from airpy import Session

support = Session("support_agent", provider)
support.send("My order never arrived.")
support.send("It's order #9988.")
# every turn is still a real, journaled Workflow step underneath: support.journals[i]
```

A `Session` is what turns "a chat loop" into an agent whose every action
is audited and permission-checked the same way a normal workflow is. Every
`.send()` call is a real `Workflow.run()` under the hood, with a real
`Journal` you can inspect afterward, not a bare list of messages.

### PromptTemplate

```python
from airpy import PromptTemplate, PromptTemplateError

template = PromptTemplate("Investigate {topic} using {source}, and flag any {concern}.")
template.render(topic="bloom filters", source="the project docs", concern="false positives")
# a missing or an extra, unrecognized variable fails loudly at render time, not silently
```

### MCP tools

```python
from airpy.mcp_tools import StdioMCPClient, tools_from_mcp

client = StdioMCPClient(command=["python", "my_mcp_server.py"])
tools = tools_from_mcp(client)   # a list of ordinary aircore.Tool objects, usable as workflow steps or agent tools
```

Tools exposed by an external MCP server show up as plain `aircore.Tool`
objects. There's nothing MCP-specific downstream of `tools_from_mcp()`.
They work as a `ModelAgent`'s `tools=`, or as ordinary workflow steps,
identically to a tool you wrote yourself in Python.

---

## AirLang: workflows as data

For defining a workflow's shape once and running it without writing
Python, useful when a workflow's structure needs to be reviewed, versioned,
or edited by someone who isn't writing code.

```
# research.airlang
provider mock

agent Literature { provider mock }
agent Reddit     { provider mock }
agent GitHub     { provider mock }

workflow Research {
    parallel {
        Literature
        Reddit
        GitHub
    }
    consensus majority
    artifact Report
}
```

```bash
ai run research.airlang
ai trace research.airlang --html
```

Run `ail parse research.airlang --ir` to inspect exactly what a file
parses to, without running anything.

---

## The ai CLI

```bash
ai run script.py                        # run, print a one-line summary of any Workflow or Session it defined
ai run workflow.airlang                 # the same, for an AirLang file

ai trace script.py                      # run, then print the full execution graph
ai trace script.py --json               # the same, as structured JSON instead
ai trace script.py --html               # the same, and also write a self-contained, clickable HTML trace viewer
```

`ai trace` picks up any `Workflow` that actually ran, or any `Session`
that had at least one turn, sitting in the script's module-level namespace
once it finishes. Define one, call `.run()` or `.send()` on it, and
there's nothing special to import or configure to make it visible to the
CLI.

---

## What you can build with this

The pieces above are general enough to support a wide range of real
systems, not just demos. A few concrete shapes, roughly in order of how
much of the runtime they lean on:

**A research assistant that cites its work.** Multiple specialist agents
(a literature search, a code search, a web search) run in `.parallel()`,
get reduced to one answer with `JudgeConsensus`, and every step of the
process, including exactly which sources each specialist consulted, is
sitting in the `Journal` afterward for you to show the user or audit
later.

**A customer support agent with a real audit trail.** `Session` gives you
a multi-turn conversation where every single turn is independently
journaled, permission-checked against the agent's granted capabilities
(so a general support agent literally cannot approve a refund on its own,
only an identity holding a `PaymentRefund` capability can), and inspectable
after the fact. This is the difference between "a chat loop with some
prompt engineering" and something you'd actually be comfortable putting in
front of paying customers.

**An autonomous system that takes real-world actions, safely.** `Policy(
approval_for={"send_payment", "place_order"})` means the agent can reason
and plan freely, but anything with real consequences stops for a human
decision (or a programmatic gate you write, like a rule engine or a Slack
approval) before it actually happens. `Capability` checks mean an agent
that gets compromised or goes off the rails can't do more than whatever
capabilities its identity was actually granted, regardless of what it
decides to try.

**A long-running, always-on agent.** Think a monitoring agent, a trading
system, or a background job that reacts to events for days or weeks at a
time without a restart. MindGraph's compaction keeps its accumulated
context from growing without bound over that whole lifetime, so cycle
10,000 doesn't cost more than cycle one. This exact pattern is what
TradingOS, a full autonomous trading system built entirely on these four
packages, uses for its always-on decision loop: two specialist agents in
consensus, sharing a MindGraph so they don't duplicate analysis of the
same market data, checked against a hard risk gate before any trade is
placed, with every cycle's reasoning fully journaled.

**A durable batch pipeline over paid API calls.** Checkpoint and resume
means a nightly report generation job, or a large document-processing run,
that fails halfway through (a network blip, a rate limit, the process
getting killed) can be rerun and will skip every step, including every
already-answered LLM call, that already succeeded. You stop paying twice
for work a crash interrupted.

**A compliance-sensitive workflow.** Financial reporting, medical
documentation review, legal document analysis, anything where "what
exactly happened, and who approved it" has to be answerable after the
fact, not reconstructed from logs you hoped were verbose enough.
`Journal.to_json()` gives you that record as structured data from the
start.

**A multi-agent code review or content moderation system.** Several
agents independently review the same input in `.parallel()`, disagreement
surfaces as a real `ConsensusFailed` (or a `JudgeConsensus` verdict with a
confidence score) instead of one agent's opinion silently winning, and a
human approval gate sits in front of any action taken on the result.

**A small, embeddable task orchestrator with no AI at all.** Because
`aircore` has zero dependencies and no opinion about models, you can use
just the scheduler, retries, journal, and policy engine for plain Python
automation, and add `airpy` later if and when an LLM actually becomes part
of the picture.

---

## Comparison with alternatives

This isn't a claim of being strictly better than every framework below.
Some of them have a much larger ecosystem, a bigger community, or stronger
guarantees in one specific area (Temporal's durability, in particular, is
hard to match with anything that isn't also a distributed system). The
table below is about one specific thing: which of these guarantees are a
built-in, shipped primitive you call directly, versus something you'd have
to build yourself, get from a separate paid tier, or that isn't offered at
all. Researched in September 2026, current at time of writing, and worth
re-checking yourself since this space moves fast.

| | **aircore** | LangGraph | CrewAI | Microsoft Agent Framework | OpenAI Agents SDK | Temporal | LlamaIndex Workflows |
|---|---|---|---|---|---|---|---|
| Deployment | Python library, no server | Python/JS library, no server (LangSmith adds an optional hosted layer) | Python library, no server (AMP is an optional paid hosted layer) | Python/.NET library, no server | Python/JS library, no server for the core; Sandbox Agents needs OpenAI-hosted compute | Requires running a Temporal server/cluster plus worker processes | Python/TS library, no server (llama-agents adds an optional hosted API layer) |
| Built-in audit journal | Yes, `Journal` | Yes, via LangSmith tracing | Paid tier only (AMP); OSS core relies on external tools | Yes, telemetry/middleware pipeline | Yes, built-in Tracing | Yes, execution history is core to how it works | Not a dedicated product, relies on external tooling |
| Tool permission model | Yes, `Capability`/`requires=`, checked before the call | Not built in, DIY | Not built in, requested but unshipped | Partial, via middleware and a separate governance toolkit | Partial, sandbox `run_as` scoping only | Cluster-level access control, not per-tool | Not built in |
| Human approval gate | Yes, `Policy.approval_for` + `approval_callback` | Yes, `interrupt()` nodes | Yes, guardrails and AMP input points | Yes, first-class tool approval | Yes, human-in-the-loop plus guardrails | Not first-class, buildable on Signals/Updates | Yes, pause-and-resume on a human input event |
| Crash-safe checkpoint and resume | Yes, `FileCheckpointStore` | Yes, checkpointer-backed | Not confirmed as crash-resume grade | Part of the graph workflow engine | Not confirmed | Yes, this is Temporal's core guarantee | Partial, requires deliberate snapshotting |
| Sandboxed tool execution | Yes, `Sandbox`, a real subprocess | Optional separate package | Yes, via E2B/Daytona/Docker integrations | Yes, `ShellExecutor` (.NET) plus a governance toolkit | Yes, Sandbox Agents (OpenAI-hosted compute) | Not built in, left to your own Activities | Not built in, third-party sandbox integrations |
| Fail-loud multi-agent consensus | Yes, `.consensus()` raises `ConsensusFailed` on disagreement | Not found as a shipped primitive | Not found as a shipped primitive | Not found as a shipped primitive | Not found (has delegation via handoffs, not voting) | Not applicable, general-purpose engine | Not found as a shipped primitive |
| Long-loop context compaction | Yes, `MindGraph` | Not confirmed | Not confirmed | Yes, automatic context compaction | Not confirmed | Not LLM-context-specific (has large-payload storage) | Not confirmed, described as a DIY pattern |
| Core dependencies | Zero, for `aircore` itself | Several (`langchain-core` and friends) | Several | Several | Several | A running Temporal cluster | Several |

A few of these deserve more nuance than a table cell can hold:

- **Checkpoint and resume isn't the same guarantee everywhere.** Temporal's
  durable execution (mid-activity retries, exactly-once-style semantics,
  built on an operated cluster) is a stronger guarantee than any
  library-only checkpointer, including this one's. If you need that level
  of durability and are willing to run and operate a Temporal cluster,
  Temporal is the more mature choice for that specific guarantee.
- **Fail-loud consensus** turned up as a shipped, named primitive in none
  of the six alternatives researched for this table. Where similar ideas
  exist (LangGraph/CrewAI's multi-agent patterns, AutoGen's historical
  group-chat mode), they're a pattern you assemble yourself, not a
  function you call that raises a specific exception on disagreement.
- **LangGraph and CrewAI have far larger ecosystems** than this project:
  more integrations, more community examples, more tutorials, and (for
  CrewAI) a hosted platform with dashboards. If ecosystem size and
  community support matter more to you than the specific guarantees in
  this table, either is a reasonable choice.
- **Microsoft Agent Framework's context compaction** is the one other
  confirmed built-in answer to the same token-growth problem `MindGraph`
  solves here, monitoring token usage and compacting history automatically
  mid-loop. If you're already in the Microsoft/.NET or Azure ecosystem,
  it's worth a direct look.
- **If you don't need AI at all**, `aircore` by itself is a small,
  dependency-free task scheduler with retries, a journal, and a policy
  engine. Temporal is the heavier-weight, much more battle-tested choice
  for that same class of problem if you're willing to operate a cluster
  for it.

---

## Design principles

- **Provider-agnostic, always.** Nothing in `aircore` or in `airpy`'s core
  assumes a specific model or vendor. `MockProvider` and a real provider
  are fully interchangeable, with the same code either way.
- **No model or prompt concept inside aircore.** The runtime itself
  (scheduler, capabilities, policy, journal, consensus, memory,
  sandboxing) has zero dependencies and zero opinions about AI.
  Everything AI-specific lives one layer up, in `airpy`.
- **Fail loud, not silent.** A capability denial, a policy violation, a
  consensus tie, a missing prompt variable: all of these raise a specific,
  catchable exception. Nothing here guesses quietly on your behalf and
  hopes it guessed right.
- **Additive, not invasive.** Every feature added after the initial build,
  tool-calling, structured output, memory-backed conversations,
  MindGraph, and so on, is opt-in and backward compatible. Code written
  before a feature existed keeps working, unchanged, after it ships.

---

## Examples and tests

`examples/` has a runnable, offline demonstration of every feature covered
above. No API key needed, unless noted.

```bash
pytest                    # the full offline test suite
python examples/hello.py  # the smallest possible example
```

A handful of examples (`live_deepseek.py`, `production_readiness.py`,
`mcp_live_*.py`) need a real API key or a subprocess and are deliberately
excluded from the automated test suite. Everything else in `tests/` runs
offline, with no network access, and is exactly what CI exercises.

---

## FAQ

**Do I need `airlang` or `aircli` to use this?** No. Most real projects
only ever use `aircore` and `airpy`, writing workflows directly in Python.
`airlang` and `aircli` exist for the cases described in [The four
packages](#the-four-packages) above, not as a required part of the stack.

**Does this lock me into one model provider?** No, that's close to the
entire point. `LiteLLMProvider` alone reaches over a hundred models
through one interface, and `ModelProvider` is a small enough interface
that writing your own adapter for something not already covered is
realistic, not a multi-week project.

**What happens if two agents in a `.consensus()` block disagree?** It
depends on the strategy. `majority()` (the default) picks the majority
answer if there is one and fails with `ConsensusFailed` on a genuine tie.
`unanimous()` requires every voter to agree or fails the same way.
`JudgeConsensus` (in `airpy`) uses a model call to judge substantive
agreement on free text, rather than requiring an exact string match.
Nothing silently picks a winner when the voters don't actually agree.

**Is the sandbox a real security boundary?** It's real process isolation
with a wall-clock timeout, a best-effort memory limit, and a best-effort
network egress allowlist. It is explicitly not a container or a kernel
namespace. Read `aircore/sandbox.py`'s own docstring before relying on it
for anything where a determined, malicious payload is a real threat model,
not just untrusted-but-not-adversarial code.

**Can I use this without any AI at all?** Yes. `aircore` by itself is a
general-purpose task orchestrator: scheduler, retries, journal, policy,
capabilities, memory, and sandboxing, with zero dependencies and no
concept of a model anywhere in it.
