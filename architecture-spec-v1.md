# AI Execution Runtime — Architecture Spec (v1)

Status: **as-built**, updated to match the implementation after M0–M8 plus
post-M8 work (structured artifacts, MCP tool registry, streaming,
long-running sessions). This document was last synced with the pre-code
plan when M6 shipped; a lot has been added since (`ModelAgent`, the
tool-calling loop, `JudgeConsensus`, structured output, MCP, streaming,
`Session`) without this file being updated. Everything below reflects
what actually exists in the codebase today, not the original pre-code
plan — where the two differ, that's called out explicitly rather than
silently rewritten, because the divergences themselves are part of the
project's history and some were deliberate design decisions made along
the way, not just drift.

Scope: this document describes what's built. Anything under "Out of
scope" is explicitly deferred, not forgotten.

## 1. One-sentence answer

Why not just use PydanticAI + Temporal + LiteLLM + a homegrown policy layer?

**Because those concerns — capabilities, budgets, retries, consensus, replay — are currently assembled by every team from scratch, inconsistently, and none of the existing AI frameworks treat them as first-class execution guarantees rather than optional glue code.** This project provides a single runtime where those guarantees are built in, not bolted on.

To be precise about the claim: individual pieces of this already exist elsewhere — LiteLLM does budget/rate limits, LangGraph does human-in-the-loop, Guardrails does output validation. The gap isn't that no one has any of these. It's that no one unifies capability limits, budget caps, and resource limits into one declarative contract that the scheduler enforces around an entire multi-agent workflow, rather than each concern being a separate bolt-on per call. **Policy is the headline feature** — it's the thing a developer can point to and say "I can't assemble this from what exists, I'd have to build it myself."

**Non-goal:** `aircore` is not an agent framework. It doesn't compete with CrewAI or LangGraph on "how do I build an agent." It's an execution runtime that answers "what is allowed to happen while this runs" — `airpy` (built) is one possible thing that runs on top of it; a dedicated DSL (`airlang`, not built) would be another.

If the Policy claim stops being true — if another framework ships unified, scheduler-enforced budget+capability contracts — this project loses its reason to exist. That's the bar to keep checking against. `Policy` as actually implemented (§3) is narrower than the original plan (no `max_tokens`, `allowed_models`, or `approval_for` — see §3's Policy entry for why each was deliberately cut rather than half-built), so this claim should be read as "this is what's enforced today," not the original aspiration.

## 2. Project layout (as built)

```
project/
├── aircore/                  runtime -- provider-agnostic, zero knowledge of AI/prompts/models
│   ├── agent.py            Agent: a capability-holding identity (NOT a model-backed agent -- see §3)
│   ├── consensus.py        ConsensusGroup, majority(), unanimous(), Strategy type, ConsensusFailed
│   ├── effects.py          Capability (the effect system; module kept this name, see its own docstring)
│   ├── events.py           EventBus + every Event dataclass -- the single source of truth
│   ├── executable.py       Executable ABC: unifies Tool and (in airpy) ModelAgent
│   ├── graph.py            build_execution_graph()/render_execution_graph()
│   ├── journal.py          Journal, StepRecord, GroupRecord -- pure event-bus subscriber
│   ├── memory.py           Memory: session/project/temporary MemoryScope
│   ├── observability.py    Metrics, ToolStats -- another pure event-bus subscriber
│   ├── parallel.py         ParallelGroup (a marker; scheduler.py does the actual execution)
│   ├── policy.py           Policy, PolicyViolation
│   ├── scheduler.py        The only thing that executes anything
│   ├── tools.py            Tool, @tool decorator
│   └── workflow.py         Workflow: .step()/.parallel()/.consensus(), ParallelResults
├── airpy/                  provider-aware SDK layer -- imports aircore, never the reverse
│   ├── ask.py               ask(): one-line sugar, bypasses the Scheduler
│   ├── judge_consensus.py   JudgeConsensus: LLM-as-judge consensus strategy
│   ├── litellm_provider.py  LiteLLMProvider: first real ModelProvider adapter (lazy import)
│   ├── mcp_tools.py         tools_from_mcp(), MockMCPClient, StdioMCPClient
│   ├── mock_provider.py     MockProvider: no API key/network needed
│   ├── model_agent.py       ModelAgent: an Executable backed by a ModelProvider
│   ├── providers.py         ModelProvider/ModelRequest/ModelResponse/Usage/ToolSchema
│   ├── schema.py            tool_to_schema(): Tool -> ToolSchema (JSON schema)
│   ├── session.py           Session: long-running conversation, real per-turn Workflow
│   └── structured_output.py Shared JSON extract/validate/parse pipeline (optional Pydantic)
├── aircli/                  package: `ai` -- deliberately minimal, 2 subcommands
│   └── __main__.py          `ai run <script.py>`, `ai trace <script.py>`
├── examples/               20 runnable scripts, one (or a small family) per feature
├── tests/                  24 files, 212 tests, all offline/mocked (see §4a)
└── pyproject.toml          one package distributing aircore + airpy + aircli together
```

`aircore` is the product: it has zero knowledge of models, prompts, or providers, and a real test (`tests/test_airpy.py`) enforces that no `aircore` file imports `airpy`. `airpy` is how Python developers talk to it. A dedicated DSL (`airlang`) remains not started — see §7.

## 3. Core primitives (as built)

**Executable** (`aircore/executable.py`) — an ABC, not part of the original plan's primitive list, but the single most load-bearing abstraction added since. `Tool` and `airpy.ModelAgent` both implement it (`execute() -> Any`, plus an optional `usage() -> Optional[Dict[str, float]]` hook). The Scheduler dispatches against this interface only — it never needs to know whether a step is a plain function or a model call. This is what let every airpy feature (tool-calling, structured output, MCP tools, streaming, sessions) get built without a single change to `scheduler.py`'s core dispatch logic.

**Agent** (`aircore/agent.py`) — deliberately *not* what "Agent" means in most AI frameworks. This is a capability-holding identity: a name plus a set of granted `Capability` tokens, checked when a step declares `agent=`. It answers "who is executing," nothing about models or prompts. The model-backed concept (prompt + provider + tools) is `airpy.ModelAgent`, a distinct class on purpose — reusing the name `Agent` for both would have made `workflow.step(researcher, agent=identity_agent)` read like a bug even when correct. This split turned out to be one of the better calls made along the way: capability identity and model-backed intelligence are genuinely different concerns, and keeping them as separate classes cost nothing.

**Tool** (`aircore/tools.py`) — wraps a plain function: `idempotent`, `requires` (capabilities), `timeout`, `retries`, `description`, and `parameters_schema` (an optional override used when a tool's real JSON schema comes from somewhere other than its Python signature — e.g. an MCP server's own schema; see `airpy/mcp_tools.py`). `retries > 0` requires `idempotent=True`, enforced at construction, not at call time.

**ModelAgent** (`airpy/model_agent.py`) — an `Executable` backed by a `ModelProvider`. First-class and interchangeable with `Tool` in every workflow position (`step`/`parallel`/`consensus`) with zero `aircore` changes required. Also implements:
- A ReAct-style tool-calling loop (`tools=[...]`) — retry-safe (a mid-loop failure never replays an already-succeeded tool call; see the module docstring's "Retry semantics" section for the mechanism).
- Structured output (`output_schema=` — a JSON-schema dict or a Pydantic class) — `execute()` returns a validated, typed value instead of raw text, which the Journal stores automatically with no `aircore` awareness of what "structured output" even means.
- Memory-backed conversations (`memory=`, `conversation_id=`) — reuses `Memory`'s existing `get`/`set` contract, no new `aircore` API.
- `stream()` — yields text chunks instead of blocking; explicitly bypasses the Scheduler (see §4a) and does not support `tools=`.

**Workflow** (`aircore/workflow.py`) — `.step()`, `.parallel()`, `.consensus()`. `.parallel()` returns a `ParallelResults` handle (not `self`) so a following `.consensus()` can reduce its outputs *without re-executing the same steps* — `workflow.parallel(a, b, c).consensus(strategy=...)` costs 3 executions + 1 reduction, not 6 + 1. This (result reuse) was added after the naive form was shown to double-execute real, paid model calls.

**Capabilities** (`aircore/effects.py`, kept that module name as the technically-correct term for what's developer-facing as `Capability`) — `Network`, `Filesystem`, `Email`, `Payments`, `Database` built in, custom ones are just `Capability("Whatever")`. Enforced at call time by the Scheduler, opt-in per step (no `agent=` on a step means no enforcement — existing agent-less workflows are unaffected).

**Scheduler** (`aircore/scheduler.py`) — event-driven: everything it does is expressed as events on an `EventBus`; `Journal` and `Metrics` are pure subscribers, coupled to nothing but the bus. Executes sequential steps, `ParallelGroup`/`ConsensusGroup` blocks (thread pool, fan-in), idempotency-gated retries, and capability checks. `ConsensusGroup`'s exception handling is deliberately broad (`except Exception`, not just `ConsensusFailed`) because a real strategy (`JudgeConsensus`) can do I/O and raise anything — any exception fails that step gracefully, none crash the run.

**Memory** (`aircore/memory.py`) — `session` (lives with the `Memory` object), `project` (shared by name via a process-wide registry), `temporary` (cleared by `Workflow.run()` after every run, success or failure). In-process dict backend only — no SQLite or persistent backend; `project` scope doesn't survive a process restart or cross machines.

**Observability** (`aircore/observability.py`, `aircore/journal.py`, `aircore/graph.py`) — `Metrics` (per-tool stats, usage totals) and `Journal` (full step/group record, `.pretty()`/`.to_json()`) both attach automatically on every `Workflow.run()`, no manual instrumentation. `StrategyMetadataReported` is a generic, opaque-to-aircore event a consensus strategy can optionally emit (via a duck-typed `describe_last_call()` hook) to enrich its step's journal entry — `JudgeConsensus` uses it to record which model judged, select-vs-synthesize, and (if `confidence=True`) a typed confidence score and reasoning string. `build_execution_graph()` renders a `Journal` as a node/edge graph; `ai trace` prints it.

**Policy** (`aircore/policy.py`) — implemented: `require_agent` (pre-flight — a step with no `agent=` fails validation before anything runs), `max_parallel` (pre-flight, on `parallel`/`consensus` block size), `max_runtime`/`max_cost` (checked before each step starts, against wall-clock time and cumulative reported `cost_usd`). **Deliberately not implemented**, and not accepted as fields — this is a change from the original plan, not an oversight: `max_tokens` (never had a concrete enforcement design), `allowed_models` (genuinely LLM-specific vocabulary that doesn't belong in a Policy `aircore` — which has no concept of "model" — owns; would belong at the `airpy` layer if built), `approval_for` (no approval mechanism was ever designed — see §8, this was an open question in the original plan and is now explicitly closed as *not done*, not still pending).

**Consensus** (`aircore/consensus.py` + `airpy/judge_consensus.py`) — `ConsensusGroup` runs voters concurrently, then reduces with a `strategy: Callable[[Sequence[Any]], Any]`. `aircore` ships `majority()`/`unanimous()` (exact-match, suited to discrete/categorical outputs). `airpy.JudgeConsensus` — deliberately *not* baked into `aircore`, per an explicit design constraint — is an LLM-as-judge strategy for free-text or structured outputs, with `mode="synthesize"` (merge into one answer, the default) or `mode="select"` (choose one candidate verbatim), and `output_schema=`/`confidence=` for merging structured artifacts with a typed confidence score instead of scraped text markers. The Scheduler only ever sees any strategy as a plain callable — it has no idea `JudgeConsensus` makes a real model call.

**Artifact** — **not built as a distinct primitive**, unlike the original plan. What actually shipped covers the same need differently: `ModelAgent(output_schema=...)` and `JudgeConsensus(output_schema=...)` both return validated, typed values (a dict or a Pydantic instance) via `airpy/structured_output.py`'s shared parse/validate pipeline, and the Journal stores whatever an `Executable` returns with zero awareness of what "structured" even means — so a typed, schema-checked output already flows through every step and the journal without a separate `Artifact` class ever needing to exist. If a real need for artifact-specific concerns (versioning, lineage tracking beyond "which step produced this journal entry," non-JSON artifact types like embeddings or binary files) shows up, this is the gap to revisit.

**MCP tool registry** (`airpy/mcp_tools.py`, not in the original plan at all) — `tools_from_mcp(client)` turns any MCP server's tools into plain `aircore.Tool`s, usable in `ModelAgent(tools=...)` exactly like a hand-written `@tool` function. `MockMCPClient` proves this offline; `StdioMCPClient` is a real adapter over the official `mcp` SDK, validated end-to-end against a live toy MCP server (not just mocked — see its docstring for the two real bugs that live run caught).

**Session** (`airpy/session.py`, not in the original plan) — a long-running conversation. `session.send(message)` runs a real one-step `Workflow` per turn (journaled, capability/Policy-enforced, unlike `ask()`/`stream()`), tracks session metadata (`session_id`, `turn_count`, `created_at`/`last_active_at`/`ended_at`), and bounds history growth via `max_history_turns`.

## 4. Execution guarantees

What's actually promised today, stated precisely so it can be tested against:

- **Workflow structure is deterministic.** The same declared `parallel`/sequence/`consensus` structure executes the same way every run. Model *outputs* are not deterministic — not a guarantee this runtime makes or can make.
- **Retries only happen on steps declared `idempotent=True`.** Enforced at construction (`Tool`/`ModelAgent` both refuse `retries > 0` with `idempotent=False`), not just documented convention.
- **Capability violations are caught at call time, every time**, because the Scheduler is the only path through which a `Tool`/`ModelAgent` step gets invoked. Opt-in per step: a step with no `agent=` has no enforcement at all (this is intentional backward compatibility, not a gap — see `aircore/agent.py`'s docstring).
- **Policy limits (`require_agent`, `max_parallel`, `max_runtime`, `max_cost`) are enforced during execution**, not just logged after — a run that would exceed `max_cost` stops before spending more.
- **Every run produces a journal** sufficient to answer "what happened, in what order, with what inputs and outputs, and — for a consensus step — what strategy decided the outcome and why."
- **A consensus strategy's own exceptions never crash the workflow.** Any exception (not just the runtime's own `ConsensusFailed`) from `strategy(outputs)` fails that step gracefully.
- **Result reuse never re-executes.** `workflow.parallel(...).consensus(...)` (or `workflow.consensus(results, ...)`) is guaranteed to run each voter exactly once, even though the consensus step's outputs depend on them.

What this runtime does **not** guarantee: correctness of model output, absence of hallucination, that a workflow's *result* is the same across runs, or (new since the original plan) that a streamed (`ModelAgent.stream()`) call gets the same guarantees as a workflow step — see §4a.

### 4a. What deliberately bypasses the Scheduler

Not every `airpy` convenience goes through `Workflow`/journal/Policy. This is an explicit, documented boundary, not an inconsistency:

- **`ask()`** — one-line sugar for calling a model outside a `Workflow`. No journal, no policy, no capability check, no retry.
- **`ModelAgent.stream()`** — token-level streaming has no single point in time an atomic `StepStarted..StepFinished` journal entry could record, so it's explicitly workflow-adjacent. Also does not support `tools=` (raises immediately, not lazily, if set) — reconstructing tool_calls split across chunks is real additional complexity with no driving use case yet.
- **`ModelAgent`'s internal tool-calling loop** — tool calls made *inside* one `execute()` call (i.e. within the ReAct loop) never pass through the Scheduler; from the Scheduler's point of view one `ModelAgent` step is one atomic call. These internal calls are visible only via `ModelAgent.tool_call_log`, not the Journal. `identity=` is a partial capability-check mitigation for this gap, not a full fix.

`Session.send()` is the one long-running/interactive-feeling API that is *not* on this list — each turn runs through a real `Workflow`, deliberately, because the customer-care/audit use case this was built for specifically needs per-turn journaling and capability enforcement, not just a chat loop.

## 5. Execution model — one workflow, start to finish

```
1. Developer defines Workflow with steps (Tool and/or ModelAgent), Policy.
2. `ai run workflow.py` invokes the runtime (or the script calls .run() itself).
3. Workflow._validate() runs pre-flight Policy checks (require_agent, max_parallel)
   -- a violation here means no WorkflowStarted event, no journal, the run never begins.
4. Scheduler executes declared structure:
   - sequential steps run in order
   - ParallelGroup blocks run concurrently (thread pool, fan-in)
   - ConsensusGroup blocks run voters concurrently, then reduce via `strategy`
     (or, in reuse mode, skip re-running voters and reduce over a prior
     ParallelGroup's cached outputs)
   - each step's capability requirements are checked against its `agent=` before it fires
5. Each step emits StepStarted..ToolCalled..ToolSucceeded/ToolFailed..StepFinished
   on the EventBus; Journal and Metrics both subscribe and build their records
   from these events alone, never called directly.
6. If a step fails and its Executable is idempotent, retry per its own retries
   count; otherwise the step (and the workflow) fails.
7. Policy.max_runtime/max_cost are checked before each step starts; exceeding
   either stops the workflow before the next step, without interrupting a step
   already in flight.
8. On completion (or halt), the journal is finalized -- workflow.journal and
   workflow.metrics are both populated and readable.
```

No approval-gate step exists in this model — `approval_for`/human-in-the-loop was an open question in the original plan (§8) and was never built; see §3's Policy entry.

## 6. Execution journal and graph

Every workflow run produces a structured journal (`aircore/journal.py`): step sequence, tool calls, retries, group membership, usage, and — for a consensus step whose strategy supports it — strategy metadata (which model judged, select-vs-synthesize, confidence/reasoning), in order. `journal.pretty()` renders it human-readably; `journal.to_json()`/`to_dict()` for machine consumption.

The **execution graph** (`aircore/graph.py`) is a separate, derived rendering: `build_execution_graph(journal)` turns the journal into a node/edge structure; `render_execution_graph()` prints it; `ai trace <script.py>` runs both. The journal is the source of truth; the graph is a view of it.

What this delivers today: post-hoc debugging (which step failed and why, with the real error message) and audit trails (every tool call an agent made, whether it was permitted, and — for consensus steps — why the runtime decided what it decided). What it still does **not** attempt: full deterministic replay/resume of a failed workflow from mid-execution — unchanged from the original plan, still out of scope (§7).

## 7. Out of scope (explicit, unchanged unless noted)

- The DSL/parser/compiler (`airlang`) — still not started.
- A "planner" that rewrites or invents workflow structure — still excluded; every consensus strategy and tool-calling loop this project has built stays inside a structure the developer declared.
- Static (compile-time) proof of capability safety — still runtime interception only.
- Full durable replay/resume of failed workflows mid-execution.
- A general-purpose resource manager with fairness/queueing/deadlock avoidance.
- Package registry / marketplace for agents and tools.
- **New since the original plan:** `Policy.approval_for` / human-in-the-loop approval gates (open question in §8 of the original plan; now explicitly closed as not built — no approval mechanism exists anywhere in the runtime). `Policy.allowed_models` / `max_tokens`. Streaming combined with the tool-calling loop. Non-stdio MCP transports (HTTP/SSE) — only stdio is implemented. A distinct `Artifact` primitive (superseded by `output_schema=`, see §3).
- "AI OS," "AI Database," "AI Cloud," robotics — still cut, for the same reason as the original plan: none of them are buildable or validatable before the runtime itself exists, and it still does.

## 8. Validation workflow and what's actually been proven live

The original plan's Research workflow (`Question → parallel(Search Web, Search GitHub) → Summarize → Merge → Artifact`) was built (`examples/research.py`) and exercises Tool, Scheduler (parallel/fan-in), Policy, Capabilities, Memory, and Observability/Journal, same as planned.

Beyond that single planned example, this codebase now has 20 runnable examples, one per feature, and several things have been validated against **real, live external systems**, not just mocks — worth recording here since it's easy to lose track of which parts are "built and mocked" vs. "built and proven":

- **DeepSeek (real LLM API, via LiteLLMProvider)** — single-shot calls, the tool-calling loop, and `Policy.max_cost` were all run live (`examples/live_deepseek.py`). This live run caught a real bug (a wire-format mismatch in how tool_calls were serialized back to the provider) that no mock had caught.
- **A real MCP server subprocess (via StdioMCPClient)** — tool discovery, direct calls, and the full `ModelAgent` tool-calling loop were run against an actual `mcp`-SDK server (`examples/mcp_live_server.py` + `mcp_live_client.py`), not a mock. This live run caught two real bugs (an `anyio` cancel-scope task-identity requirement, and two SDK attributes that didn't match their constructor-kwarg names).

Everything else (JudgeConsensus, structured output, streaming's chunking, Session's journaling, result reuse) is validated by the automated test suite (212 tests, all offline/mocked, run on every change) plus manual example runs, but not yet against a live provider/server the way the two items above were. That's a reasonable place to be, not a gap to apologize for — but it's worth being precise about which claims rest on live validation and which rest on mocks, rather than letting the two blur together.

Domain-specific examples beyond Research (a fuller audit workflow, the customer-care sketch discussed but not built, a coding agent) remain not built — real applications on top of this runtime are the natural next step once a specific one is chosen, not before.

## 9. Milestones (completed)

- **M0 — Minimal execution + journal.** Done.
- **M1 — Scheduler: sequential + parallel + fan-in.** Done.
- **M2 — Capabilities.** Done.
- **M3 — Policy.** Done (see §3 for exact fields; narrower than originally planned).
- **M4 — Observability.** Done (Metrics + Journal + execution graph).
- **M5 — Memory.** Done (session/project/temporary).
- **M6 — Consensus.** Done (`majority`/`unanimous`; `JudgeConsensus` added later, see below).
- **M8 — Provider integration.** Done (`Executable` abstraction, `airpy` package, `MockProvider`, `LiteLLMProvider`; usage/cost flowing into Metrics/Policy). (There is no milestone numbered M7 in this project's history — provider integration was tracked as M8 from the start.)

**Post-M8, not originally numbered milestones** (chronological):
tool-calling loop → retry/replay safety fix → packaging + `ai` CLI → `JudgeConsensus` → result reuse (`ParallelResults`) → synthesize/select modes + richer consensus journal → structured outputs (`output_schema=`) → memory-backed conversations → MCP tool registry → streaming → long-running `Session`.

## 10. Open questions (revisited)

The original plan's three open questions, now answered by what was actually built:

- **What does an approval pause look like mechanically?** Never resolved — no approval mechanism was built (see §7). Still open if this becomes a real requirement.
- **Minimum viable Memory backend?** In-process dict, exactly as guessed. Still no SQLite or persistent backend; `project` scope doesn't survive a process restart.
- **Minimum viable Artifact implementation?** Superseded — no distinct `Artifact` class was built; `output_schema=` on `ModelAgent`/`JudgeConsensus` (a JSON-schema dict or Pydantic model, validated by `airpy/structured_output.py`) covers the "typed, structured step output" need instead. Revisit only if a concrete need shows up that this doesn't cover (binary/non-JSON artifacts, versioning/lineage beyond "which journal step produced this").

New open questions, raised by what's since been built:

- **Streaming + tool-calling.** `ModelAgent.stream()` currently refuses to run if `tools=` is set. Worth solving if an interactive, tool-using agent becomes a real use case.
- **MCP transports beyond stdio.** Only `StdioMCPClient` exists; HTTP/SSE-based MCP servers aren't supported.
- **First real application.** Every feature so far has been validated by its own example, not by a complete application built on top of the stack. A coding assistant, research assistant, or the customer-care system sketched in conversation (but not built) are the candidates on the table.
