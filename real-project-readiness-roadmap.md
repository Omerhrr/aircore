# Real-project readiness roadmap

Status: **planning document, nothing here is built yet**. Written in
response to "we will do all that" after an honest gap assessment (see the
end of this doc for the assessment that prompted it) — the user asked to
document the full scope first and pick items up later, rather than start
building against an unprioritized list. Nothing in this document should
be read as shipped; check each package's own `__init__.py` docstring and
`__version__` for what's actually true at any given time, the same way
every other status claim in this project is verified rather than assumed.

Seven workstreams, each scoped separately because they have genuinely
different risk profiles, dependencies, and who needs to make the call:

## 1. Provider live-testing (needs your API keys)

**Gap:** Of `airpy/provider_catalog.py`'s ten named constructors (openai,
anthropic, deepseek, gemini, qwen, nvidia, zai, ollama, lmstudio,
openrouter), only `deepseek()` has ever been exercised against a real
API — both through `LiteLLMProvider` (the original M8 validation, plus
the tool-calling wire-format bug it caught) and now through the native
`OpenAIProvider` (this session, via DeepSeek's OpenAI-compatible
endpoint). The other nine are covered only by offline tests asserting
the resulting model string is correct — never a live call.

**Why it matters:** the DeepSeek tool-calling bug (a wire-format
mismatch between what `ModelAgent` sent and what a real API expected)
was *only* caught by a live call — the offline tests didn't catch it
because they were testing against a fake that encoded the same wrong
assumption. There's no reason to believe the other nine providers are
bug-free just because they're unexercised; there's just no evidence
either way yet.

**What's needed:** API keys for whichever providers actually matter to
your project (not necessarily all nine) — `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DASHSCOPE_API_KEY` (Qwen),
`NVIDIA_NIM_API_KEY`, `ZAI_API_KEY`, `OPENROUTER_API_KEY` (Ollama/LM
Studio need no key, just a locally running server). I can't supply these
or make the call on which providers matter — that's a real product
decision only you can make.

**Scope once keys are available:** a small live example per provider
(mirroring `examples/live_deepseek.py` and `examples/
live_openai_provider.py`), run once to catch any wire-format surprises,
then folded into documentation as "live-verified" the same way DeepSeek
is now. Low code risk, mechanical work — the value is entirely in
actually making the calls.

## 2. AirLang: general branching (standalone `if`)

**Gap:** `if` only works in the one narrow shape AirLang-M3 built —
immediately after a `consensus judge { confidence true }` block, as a
confidence-gated fallback. A standalone `if` anywhere else, with any
comparator other than `<`, or with a `then` body that isn't exactly one
bare reference, still raises `AirLangNotYetSupportedError` (see `airlang/
executor.py`'s docstring).

**Why it matters:** any AirLang workflow that needs to branch on something
other than judge confidence — a tool's return value, a policy check, an
approval outcome — currently can't express that at all in the language.

**What's needed:** a real design decision, not just implementation. The
confidence-gated fallback works because `aircore.ConsensusGroup` already
has a narrow, purpose-built primitive (`fallback`/`fallback_below`) to
compile into. General branching needs either (a) a new `aircore` primitive
(a real conditional-step-selection mechanism in the scheduler, which is
a bigger change — it starts to look like DAG-with-conditional-edges, a
different execution model than the current flat step list) or (b)
scoping AirLang's `if` to a small number of additional narrow, purpose-built
shapes the way the confidence-gated case was scoped, rather than one
general mechanism. This needs a concrete `.airlang` use case to design
against — building it speculatively risks the same "guessed API,
probably wrong" trap this project has avoided everywhere else.

**Recommendation when picked up:** don't start here writing code — start
by writing down 2-3 real branching scenarios your project actually
needs, then design the minimal primitive that covers them, the same way
the confidence-gated fallback was scoped down from "general branching"
to exactly what one real spec example needed.

## 3. AirLang: body-level `approval { message }`

**Gap:** distinct from the now-working `policy { approval <tool> }`.
This is a workflow-body-level step that pauses unconditionally (not
tied to any one tool), showing a message to whoever's approving. Still
raises `AirLangNotYetSupportedError`.

**Why it matters:** `policy { approval <tool> }` gates a specific,
named, potentially-dangerous tool call. `approval { message "..." }` is
a different use case — a checkpoint for a human to review something
mid-workflow regardless of which tool comes next (e.g. "review the
research findings before I draft the final report").

**What's needed:** `aircore.approval.ApprovalRequest`/`Workflow.run(
approval_callback=...)` already exist and are tool-shaped (`tool_name`
is part of the request). A body-level approval step would need either a
synthetic no-op "step" that carries a message instead of a tool name, or
a small extension to `ApprovalRequest` to make `tool_name` optional and
add a `message` field. Smaller than item 2 — this is closer to "wire an
existing primitive to one more entry point" than "design a new one."
Reasonable to pick up independently of general branching.

## 4. AirLang: `artifact` schema enforcement

**Gap:** `artifact Name { schema AuditFinding }` records the schema
reference as metadata but never validates a step's actual output
against it. `type markdown`/`type json`/`type text` are the same —
documentation-only, per the original spec's §5.3 recommendation.

**Why it matters:** an artifact silently not matching its declared
schema currently produces no error — the whole point of declaring a
schema (catching a malformed report before it reaches whoever reads it)
doesn't happen.

**What's needed:** `airpy/structured_output.py` already does exactly
this kind of validation for `ModelAgent(output_schema=...)` — the
missing piece is purely in `airlang/executor.py`: wire a producer-linked
`artifact`'s `schema` field to call the same validation machinery
against that step's real output, and decide what happens on mismatch
(fail the step? warn and continue? — a real product decision, not an
implementation detail). Small, self-contained, and now that producer
linkage exists (this session's `let` work), there's finally a step to
validate the output *of* — this was blocked on that landing first.

## 5. Cross-process locking (FileCheckpointStore / FileMemoryScope)

**Gap:** both assume a single writer. Two processes writing to the same
`run_id`'s checkpoint file, or the same memory file, concurrently can
corrupt each other's writes — documented as a known limitation, not
silently ignored, but also not fixed.

**Why it matters:** any real deployment with more than one worker
process touching the same checkpoint/memory file needs this. A
single-script, single-process project (most of what's been built and
tested so far) never hits it.

**What's needed:** a real design decision on backend. Options, roughly
in order of how much they change: (a) OS-level file locking (`fcntl` on
Unix, no free equivalent on Windows — cross-platform is a real
complication given this project runs in both WSL and Windows contexts),
(b) a SQLite-backed store instead of raw JSON files (gets locking almost
for free, still zero extra infrastructure to run), (c) an actual
external store (Redis, Postgres) which is a much bigger dependency and
operational commitment than anything else in this project so far. My
default recommendation would be (b) — SQLite is stdlib, needs no server,
and solves the concurrency problem properly — but this is worth
confirming before building, since it changes the on-disk format anyone
depending on `FileCheckpointStore`'s JSON files today would see.

## 6. Sandboxing hardening

**Gap:** `Sandbox`'s network egress allowlist works by monkeypatching
`socket.socket.connect` — deny-by-default, but only catches code paths
that actually go through Python's `socket` module. Anything that
bypasses it (some C-extension HTTP clients, raw syscalls, subprocess
calling out to a system tool that makes its own connections) isn't
caught. Documented honestly as "best-effort," not a real security
boundary.

**Why it matters:** if sandboxed tools are ever expected to run
genuinely untrusted code (not just "someone's own tool that might have a
bug"), the current mechanism isn't strong enough to rely on.

**What's needed:** a real decision on threat model first. If the goal is
still "catch accidents, not stop a malicious actor," the current
mechanism may already be sufficient and this item can be closed as
"working as intended, scope documented." If the goal is a genuine
security boundary, that likely means OS-level isolation (a real
container, a network namespace, gVisor/Firecracker-style sandboxing) —
a materially bigger dependency than anything else in this project, and
worth being honest that it changes what "aircore" is (a pure-Python
library today; real OS-level sandboxing likely means shelling out to
external tooling).

## 7. Distributed execution, observability integrations, packaging

Grouped together because they're all "make this deployable/observable
beyond one developer's machine" rather than new runtime capability:

- **Distributed execution:** today, one `Workflow.run()` is one process,
  one machine. Nothing here supports running steps across multiple
  machines/workers. Not started, no design work done — this would be a
  substantial new primitive (task queue integration, most likely) and
  should only be scoped once there's a concrete need driving it, same
  standing rule as everything else in this project.
- **Observability integrations:** `Metrics`/`Journal` are internal,
  in-process data structures — nothing exports to OpenTelemetry,
  Prometheus, or any external observability system. Would be a new,
  additive module (an OTel exporter subscribing to the same `EventBus`
  everything else does) — lower risk than the items above, since it's
  purely additive and doesn't change existing behavior.
- **Packaging/distribution:** currently installed via `pip install -e .`
  from this local checkout only — nothing published to PyPI or a
  private index, no CI pipeline, no semantic-versioning policy beyond
  the ad hoc per-package bumps done so far (and `aircore` itself has
  stayed at 0.1.0 the entire project despite substantial growth — worth
  addressing as part of this, not just "add CI"). Mechanical work, low
  design risk, but real setup (PyPI account or private index, CI
  provider, a real versioning policy across all four packages).

## Suggested order, when we pick this back up

Not a commitment, just a starting recommendation based on effort vs.
value and what's blocked on what:

1. **Item 4 (artifact schema enforcement)** — smallest, self-contained,
   no external dependency, and was literally blocked on this session's
   `let` work landing first, so it's the most "ready to go" item here.
2. **Item 3 (body-level approval)** — small, wires an existing primitive
   to one more entry point, no new design needed.
3. **Item 1 (provider live-testing)** — as soon as you can supply keys
   for whichever providers matter; can happen in parallel with anything
   else since it's independent.
4. **Item 5 (cross-process locking)** — needs one design decision
   (SQLite vs. file locking vs. external store) before building, but
   otherwise self-contained.
5. **Item 7 (observability integration specifically)** — purely
   additive, low risk, whenever there's time.
6. **Item 2 (AirLang general branching)** and **item 6 (sandboxing
   hardening)** — both need real design conversations grounded in actual
   use cases/threat models before writing code; hold until there's a
   concrete driver.
7. **Item 7 (distributed execution, packaging)** — largest scope changes
   to what this project even is; only take on once everything above is
   solid and there's a real need.

---

## The assessment that prompted this document (for reference)

What's solid enough to build on today: the core runtime (scheduler,
policy, capabilities, retries, memory, consensus), real provider calls
with two independently-verified paths (LiteLLM and native
OpenAIProvider, both live-tested against DeepSeek), tool-calling,
structured output, human approval, checkpoint/resume, sandboxed
execution, and cross-step data flow — all tested both offline and live.
Good fit today for: single-machine, one-or-few-developer projects using
DeepSeek or another OpenAI-compatible endpoint, needing a workflow
engine with human-in-the-loop gates and crash recovery.

The seven gaps above are what stand between that and a broader "real
project" in the production/multi-provider/multi-process/distributed
sense — none of them block the narrower case today.
