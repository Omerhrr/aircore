# AirLang — Design Spec (v1)

Status: **as-built through AirLang-M3, plus policy-level `approval` and
`let`**. This document was written pre-implementation (per the
instruction that started it: design now, implement later); AirLang-M0
(lexer/parser/IR, `airlang/`), AirLang-M1 (the IR executor, `airlang/executor.py` +
`airlang/bindings.py`), AirLang-M2 (`ai run`/`ai trace` accepting `.airlang` files, in
`aircli`), AirLang-M3 (§5.1's confidence-gated fallback, closing the `if`
gap), a follow-on pass closing §5.2's `policy { approval <tool> }` gap
(`aircore/approval.py` + `Policy.approval_for`), and a further pass closing
§5.4's `let` gap (producer-linked `artifact` binding + agent prompts
compiling to `PromptTemplate`) have since all shipped — see §8 for the
phased breakdown and §5 for exactly what's still blocked (the
*body-level* `approval { message }` step specifically — not `policy {
approval <tool> }`, which now works — `if` used anywhere other than
immediately after a `consensus` block, and real schema enforcement for
`artifact`). Sections below are left as originally written except where a
footnote-style update marks what's since changed; where a construct's
semantics still can't run against `aircore`/`airpy` today, that remains
called out rather than glossed over — see examples/audit.airlang (parses;
still fails on an unbound tool reference and, if that were bound, its
`if` isn't in the supported shape) vs examples/research_with_fallback.airlang
(runs end to end, `if` included) vs examples/research_with_binding.airlang
(runs end to end, `let` included).

## 1. What AirLang is

AirLang is not a general-purpose programming language, and it must never grow
into one. It answers a fixed, narrow set of questions about one workflow:
which agents exist, what capabilities and tools they have, which provider
and model back them, how they collaborate (sequence / parallel / consensus),
what policy bounds the run, what artifact comes out, and (eventually) when
a human needs to approve something. Nothing else.

The analogy that matters: AirLang is to `aircore` what Terraform is to a cloud
API, what a Dockerfile is to a container runtime, what a GitHub Actions
YAML file is to CI infrastructure. It describes *what* should happen; the
runtime underneath decides scheduling, retries, thread pools, journaling,
metrics, and policy enforcement. AirLang never gets an opinion on any of that
— if it needs one, that's a sign the construct doesn't belong in AirLang.

**One file, one workflow.** `audit.airlang`, `research.airlang`,
`customer_support.airlang` — each file is a complete, independently runnable
unit (`ai run audit.airlang`), not a module imported by other files. This
keeps the language's job small: describe one execution, not a program
made of many.

## 2. What AirLang will never have

No classes, objects, inheritance, or interfaces. No arithmetic beyond the
one comparison `if` needs (§4.9). No generic functions, list
comprehensions, loops, or recursion. No exceptions (failures are the
runtime's business, reported in the Journal, not caught in AirLang). No
async/await, threads, file I/O, HTTP, SQL, JSON parsing, package
managers, or dependency injection. Every one of those belongs in the host
language (Python today, via `airpy`) or in the runtime (`aircore`) — AirLang's
job is to describe execution, not to perform it.

This list is a design commitment, not a first-draft limitation to relax
later. Every future "just add X to AirLang" proposal should be checked
against it first.

## 3. The stack

```
Applications
     │
     ▼
   AirLang (.airlang file)
     │  parse
     ▼
Execution Graph (IR)          -- language-agnostic, JSON-serializable
     │  execute
     ▼
IR executor (airlang/executor.py) -- imports airpy, builds real Agent/Workflow objects
     │
     ▼
   airpy
     │
     ▼
   aircore (Scheduler)
```

Two decisions worth being explicit about, because they weren't the
obvious default:

**AirLang does not compile to Python source.** There is no intermediate
`audit_generated.py` file. The parser produces an IR (§5) — a plain,
JSON-serializable data structure, not Python AST, not a Python
`Workflow` object — and a separate executor walks that IR and makes the
real `airpy` calls (`Agent(...)`, `Workflow(...).step(...)`, etc.)
directly, the same calls a human would type by hand. This is what keeps
the IR itself language-agnostic: it doesn't know Python exists. A future
`airjs`/`airgo` frontend could parse `.airlang` into the exact same IR shape
and hand it to its *own* executor, written against its own SDK. Nothing
about the IR format is Python-specific — only today's one executor is.
Be honest about what this claim is worth right now: there is exactly one
executor in this plan (Python, backed by `airpy`), so "language-agnostic
IR" is a design property being built in, not a capability that exists
yet. It costs nothing to get right now and would be expensive to retrofit
later, which is the whole justification for choosing it up front.

**The IR executor lives in a new top-level `airlang/` package, not inside
`airpy`.** Same one-directional dependency rule that already governs
`airpy` → `aircore`: `airlang` imports `airpy`, `airpy` never imports `airlang`.
`airlang/` contains the lexer, parser, IR dataclasses, *and* the executor
that turns IR into real `airpy` objects — all of it is "the AirLang
implementation," none of it belongs inside `airpy` itself, which stays
exactly what it is today: a pleasant Python SDK that has never heard of
AirLang and doesn't need to.

## 4. Syntax

Brace-delimited, not indentation-sensitive — AI workflows nest deeply
(agent definitions, parallel blocks, conditionals), and indentation rules
get worse, not better, as nesting grows.

### 4.1 `import`

```airlang
import github
import slack
```

Declares that this workflow depends on tools/capabilities registered
under these names by the host environment (§6 — AirLang cannot define what
`github` or `slack` actually *do*; it can only reference them by name).

### 4.2 `tool`

```airlang
tool clone_repo
tool run_tests
tool deploy
```

Declares a tool *reference* — a name the workflow body can use in a
`step` or an agent's `tools { }` block. AirLang cannot write a tool's body
(that would be arbitrary code); see §6 for how a name here gets bound to
a real `aircore.Tool` at execution time.

### 4.3 `capability`

```airlang
capability Git
capability Filesystem
capability Network
```

Declares which of `aircore.effects`' built-in capabilities (`Network`,
`Filesystem`, `Email`, `Payments`, `Database`) — or a custom one — this
workflow's agents may be granted. Maps directly to `aircore.Capability`.

### 4.4 `agent`

```airlang
agent Researcher {
    provider deepseek
    model deepseek-chat

    capabilities {
        Network
        Filesystem
    }

    tools {
        clone_repo
        search_docs
    }
}
```

One `agent` block compiles to one `airpy.Agent(name=..., provider=...,
model=..., requires=[...], tools=[...], prompt=...)`. `provider` names
resolve through `airpy.provider_catalog` (`deepseek` → `airpy.deepseek()`,
`anthropic` → `airpy.anthropic()`, etc. — see the provider catalog work);
`provider default` (or omitting `provider` entirely) uses whatever
default provider the executor is configured with (e.g. `MockProvider` in
tests, a configured `LiteLLMProvider` in real use). There is no `prompt`
field shown in the sketch above — see §5.4's open question, prompts need
to come from somewhere and "inline string in the agent block" is the
default assumption pending a decision.

### 4.5 `policy`

```airlang
policy {
    max_cost $2
    max_parallel 8
    timeout 5m
    approval deploy
}
```

Maps to `aircore.Policy`, field by field, including `approval` (post-M3 —
see §5.2 for the shipped mechanism and the one shape of "approval" that's
still unsupported). `timeout` maps to `Policy.max_runtime` (parsed from a
duration literal like `5m` into seconds). `$2` parses to `2.0` for
`max_cost`.

### 4.6 `workflow`

```airlang
workflow Audit {
    step clone_repo

    parallel {
        Researcher
        BusinessLogic
        InvariantChecker
    }

    consensus judge

    artifact Report
}
```

The body is a flat sequence of steps, matching `aircore.Workflow`'s own
model exactly (`.step()` / `.parallel()` / `.consensus()`, run in
declaration order) — this is not a coincidence, it's why AirLang can compile
to these calls almost one-to-one instead of needing a general execution
engine of its own.

- `step <tool-or-agent-name>` → `workflow.step(...)`
- `parallel { A B C }` → `workflow.parallel(A, B, C)` (member names refer
  to previously-declared `agent` blocks or `tool` references)
- `consensus <strategy>` or `consensus { strategy ... mode ... }` →
  `.consensus(strategy=majority|unanimous|JudgeConsensus(mode=...))`.
  Bare `consensus judge` means `JudgeConsensus()` with default settings;
  the block form sets `mode select|synthesize` and, per §5.1, is also
  where `confidence` would be requested once conditionals need it.
- `artifact Name { type markdown }` or `artifact Name { schema
  AuditFinding }` — see §4.8.

### 4.7 `provider`

```airlang
provider deepseek
provider openai
provider anthropic
```

A top-level default (used by any `agent` block that doesn't declare its
own `provider`), or `provider default` to explicitly defer to whatever
the executor is configured with.

### 4.8 `artifact`

```airlang
artifact AuditReport {
    type markdown
}

artifact Findings {
    schema AuditFinding
}
```

`schema X` maps to `ModelAgent(output_schema=X)` / structured output
(§3's existing `structured_output.py` pipeline) — `X` must be a schema
name resolvable the same way tools are (§6), since AirLang cannot define a
JSON schema or a Pydantic model body inline. `type markdown` (or `text`,
`json`) is a much weaker claim: today nothing in `aircore`/`airpy` enforces
an output *format*, only an output *schema* — `type` would need to be
either (a) sugar that does nothing but document intent, or (b) backed by
a real "coerce/validate as markdown" step that doesn't exist yet. Flagged
in §5.3.

### 4.9 `if` (conditional — narrow, not general branching)

```airlang
if confidence < 0.85 {
    HumanReviewer
}
```

This is the single construct in the whole language that looks like
"programming," and it's exactly the one flagged in §5.1 as not backed by
anything in the runtime today. See §5.1 for why this can't just be
"added" and what the two real options are.

### 4.10 `memory`

```airlang
memory session
memory project
memory temporary
```

Maps directly to `aircore.Memory.session` / `.project` / `.temporary` —
this one has no gap, `MemoryScope` already exists exactly as described.

### 4.11 `let` (minimal variable binding)

```airlang
let report = artifact Report
```

Flagged in §5.4 — binding a name to a step's output for later reference
implies passing one step's output into a later step's prompt or
arguments, which is not something `airpy` can do today (no template-
variable substitution exists yet; see the original roadmap's
`PromptTemplate` item, not yet built). `let` is included in the syntax
because it's clearly useful, not because it's ready to execute.

### 4.12 `approval`

```airlang
approval {
    message "Deploy?"
}
```

See §5.2. `aircore` now has a real approval primitive (`Policy.approval_for`
+ `Workflow.run(approval_callback=...)`, `aircore/approval.py`), but this
specific *body-level* construct — a pause tied to a message, not to any
one named tool — still has no runtime equivalent: `approval_for` gates a
tool by name, and this block names no tool at all. `policy { approval
<tool> }` (§4.5), which does name a tool, is what actually maps onto it
today; this block remains `AirLangNotYetSupportedError`.

## 5. Gaps: constructs with no runtime equivalent yet

This is the part of "designing AirLang" that matters most, and the part a
syntax sketch alone can't surface. Four of the constructs in §4 describe
something `aircore`/`airpy` cannot actually do yet. Each is a real decision,
not a detail — building the AirLang v1 executor without resolving these first
would mean AirLang either silently no-ops on part of its own syntax, or the
plan quietly grows a scope-creeping runtime feature in the middle of what
was supposed to be "just a parser."

### 5.1 Conditional execution (`if confidence < 0.85 { ... }`)

`aircore.Workflow` has no branching primitive at all — it's a fixed,
linear sequence of steps decided entirely at declaration time (§3's
`_steps: List[StepEntry]`, built once, run once, no runtime decision
point). This is by design (architecture-spec-v1.md notes no planner is
allowed to invent workflow logic at runtime), but it also means "run
`HumanReviewer` only if the preceding consensus's confidence was below
0.85" is not sugar over anything that exists — it's a genuinely new
runtime capability.

Two ways to actually deliver the compelling example from §4.9, in order
of how much new surface they need:

- **(a) Narrow, purpose-built primitive (recommended for AirLang v1's actual
  scope).** Not general `if`/branching — a single new `aircore` capability:
  a consensus step that can name a fallback `Executable` to run when the
  strategy's reported confidence is below a threshold (extending
  `ConsensusGroup`/the scheduler, using the same `describe_last_call()`
  metadata hook `JudgeConsensus(confidence=True)` already produces — see
  `StrategyMetadataReported`). This covers exactly the `if confidence <
  X { fallback }` case AirLang wants to express and nothing more general —
  no comparison operators beyond `<`, no arbitrary conditions, no
  branching on other fields. Small, real, addressable as its own aircore
  milestone before AirLang's executor needs it.
- **(b) General branching in `aircore.Workflow`.** A real, much bigger
  change — the scheduler would need a notion of runtime-decided next
  steps, which touches journaling, the graph renderer, and Policy's
  pre-flight validation (which currently assumes the full step list is
  known before anything runs). Not recommended for what AirLang v1 actually
  needs; this is the shape of thing AirLang's own "no general programming"
  rule argues against building at all, even in the runtime.

**Recommendation: (a).** Ship it as an `aircore` milestone (a fallback-on-
low-confidence extension to `ConsensusGroup`) before AirLang's executor tries
to compile `if`. Until then, AirLang's *parser* can still accept `if` syntax
(so `.airlang` files don't need rewriting later) while the *executor*
rejects it with a clear "not yet supported" error rather than silently
ignoring it.

**Shipped (AirLang-M3).** `ConsensusGroup` gained `fallback`/`fallback_below`/
`fallback_field` exactly as recommended (a): a strategy's
`describe_last_call()` metadata is checked against a threshold after the
consensus step succeeds, and the fallback runs as one more step nested
under the same consensus group if triggered — see the updated
`aircore/consensus.py`'s module docstring and `scheduler.py`'s
`_apply_consensus_strategy`. `airlang/executor.py` folds an `if <field> < X {
<single ref> }` node immediately following a `consensus` node into these
kwargs; `if` used anywhere else, with any comparator other than `<`, or
with a `then` body that isn't exactly one bare reference is still
`AirLangNotYetSupportedError` — this remains the narrow case (a), not general
branching (b). `if confidence < X` specifically also requires the
preceding consensus to be `judge` with `confidence true` (only
JudgeConsensus ever reports that field) — anything else is a loud
`AirLangBindingError` at build time, not a silently-inert fallback. See
`examples/research_with_fallback.airlang` for this running end to end via
`ai run`/`ai trace`, and `tests/test_consensus_fallback.py` /
`tests/test_ail_fallback.py` for the coverage.

### 5.2 Human approval (`approval { message "..." }`, `policy { approval X }`)

`Policy.approval_for` is explicitly absent from `aircore.policy.py` today,
for a stated reason: the mechanism itself isn't decided (blocking call?
webhook? CLI prompt requiring a human at a terminal? something async?).
This is a real open design question for `aircore`, not something AirLang can
resolve by having syntax for it. AirLang v1's parser can accept `approval`
blocks; the executor should reject them the same way as `if`, with the
same honesty — until `aircore` has an actual approval primitive, AirLang
can't compile to one.

**Shipped, partially (post-M3).** `aircore` chose the synchronous-callback
shape (not pause/resume — see `aircore/approval.py`'s module docstring for
why, tied to the same durable-state gap the CrewAI/LangGraph comparison
surfaced): `Policy.approval_for` names tools, `Workflow.run
(approval_callback=...)` supplies a `Callable[[ApprovalRequest], bool]`,
required pre-flight whenever `approval_for` is non-empty. `airlang/executor.py`
now maps `policy { approval <tool> }` straight onto `Policy.approval_for`
and forwards `approval_callback` through `execute_ir`/`execute_file`;
`aircli` passes `aircore.approval.cli_approval_callback` by default for `ai
run`/`ai trace` on `.airlang` files, so this works interactively out of the
box (see `tests/test_aicli_ail.py`). What's still unshipped: the
body-level `approval { message "..." }` step (§4.12) — a pause not tied
to any one named tool, which doesn't fit `approval_for`'s per-tool shape
and would need its own primitive — and anything resembling real
pause/resume across a process restart.

### 5.3 Artifact `type` (`type markdown`)

Nothing in `aircore`/`airpy` validates or coerces output into a stated
*format* today — only a *schema* (`output_schema=`, via
`structured_output.py`). `type markdown` (or `text`/`json`) has two
honest options: treat it as documentation only (parsed, stored, never
enforced — the executor doesn't fail a run whose "markdown" artifact
isn't actually markdown), or don't ship it in v1 at all and only support
`schema X`, which already maps to something real. **Recommendation:**
accept `type` as documentation-only metadata attached to the artifact
step's journal entry, not as a validated guarantee — cheap to support,
honestly labeled, no new runtime work required.

### 5.4 Variable binding (`let report = artifact Report`) and agent prompts

Two related gaps: nothing in `airpy` today lets one step's output feed
into a later step's prompt or arguments (no template-variable
substitution exists — the original project roadmap's `PromptTemplate`
item was never built), and the `agent` block sketch in §4.4 has no
`prompt` field at all, which every real `airpy.Agent`/`ModelAgent`
requires at construction. Both point at the same missing piece: AirLang
needs *some* story for "text with a variable in it" before `let` or a
real agent prompt can compile to anything. **Recommendation:** resolve
this by building `PromptTemplate` in `airpy` first (plain `{variable}`
substitution against named prior outputs, no expression language) — a
small, self-contained addition that both AirLang and hand-written `airpy`
code benefit from — then `let`/agent prompts in AirLang become sugar over
it, not a new problem AirLang has to solve by itself.

**Half shipped.** `PromptTemplate` exists (`airpy/prompt_template.py`,
`PromptTemplate("...{x}...").render(x=...)` — named fields only, fails
loudly on missing/extra variables) and is directly usable in hand-written
`airpy` code today (see `examples/prompt_template.py`). What it does
*not* solve, and what still blocks `let` in AirLang specifically: it has no
opinion on where a variable's value comes from — rendering is the
caller's job, and the caller must already have the value in hand.
`aircore.Workflow` builds its entire step list before `.run()` executes
anything (`workflow.py`'s `_steps`, populated once at declaration time),
so there is no point during `airlang.executor.build_workflow()` where a
*prior step's actual output* exists yet to hand to `PromptTemplate.
render()` for a *later* step's prompt.

**Fully shipped.** Deferred/lazy prompt construction landed in
`airpy`/`aircore`: `Workflow.step(tool, as_="name")` (and, since this pass,
`Workflow.consensus(..., as_="name")` too — see `aircore/workflow.py`'s
"Bindings" section) records a step's output in `workflow.bindings` once
it succeeds, and `ModelAgent(prompt=PromptTemplate(...),
prompt_bindings=workflow.bindings)` renders the template fresh at
execute() time from whatever's bound so far. `let` in AirLang is now wired to
this: `airlang/executor.py`'s `build_workflow()` treats an `artifact
<ArtifactName>` node immediately following a `step`/`ref`/`consensus`
node as that node's producer, and binds it under the `let` name if one
was declared for that artifact (a pre-scan pass over the body resolves
this before any step is built, so `let` can appear anywhere, not just
right after its artifact), or under the artifact's own name otherwise. An
`agent` prompt containing a literal `{` now compiles to a `PromptTemplate`
wired to `workflow.bindings` instead of a plain string. See
`examples/research_with_binding.airlang` for this running end to end
(offline, `provider mock`) and `airlang/executor.py`'s module docstring for
the exact scope: the artifact must *immediately* follow its producer (no
scanning back further than one node), and referencing an artifact name in
`let` that never appears as an `artifact` node anywhere in the body is an
`AirLangBindingError`, not a silent no-op. Still open, and explicitly a
separate gap from this one: `artifact`'s `schema`/`type` fields are
recorded but never validated against the step's actual output (§5.3).

## 6. Tool, schema, and provider binding (how a name in `.airlang` becomes a real Python object)

AirLang cannot author a tool's implementation, a schema's shape, or a custom
provider's config — by design (§2). So every `tool clone_repo`, `schema
AuditFinding`, or non-catalog `provider` name in a `.airlang` file is a
*reference* that must be resolved against real Python objects supplied by
the host environment at execution time. Proposed convention, to avoid
this being hand-waved:

- Running `audit.airlang` looks for a sibling `audit.airlang.py` (or an explicit
  `ai run audit.airlang --bindings bindings.py`) exposing a `TOOLS: dict[str,
  aircore.Tool]` and/or `SCHEMAS: dict[str, dict | type]` module-level
  dict. The executor resolves every `tool`/`schema` name against these
  before running; an unresolved name is a load-time error, not a runtime
  surprise mid-workflow.
- `provider` names that match `airpy.provider_catalog` functions
  (`openai`, `anthropic`, `deepseek`, `gemini`, `qwen`, `nvidia`, `zai`,
  `ollama`, `lmstudio`, `openrouter`) resolve with zero bindings file
  needed — this is most of the point of having named catalog
  constructors already. A provider name that doesn't match the catalog
  falls back to the same `bindings.py` (`PROVIDERS: dict[str,
  ModelProvider]`).
- `capability`/`import` names likewise resolve against `aircore.effects`'
  built-ins first, `bindings.py` (`CAPABILITIES: dict[str, Capability]`)
  otherwise.

This keeps AirLang itself capability-free (§2) while making the binding step
explicit and inspectable, rather than some kind of implicit registry
magic.

## 7. The IR

A plain, JSON-serializable dict — deliberately dataclass-shaped so the
same structure works as a Python object in the executor and as
`json.dumps`-able output for `ail parse --ir audit.airlang` (a debugging
tool, useful the same way `ai trace --json` already is).

```json
{
  "airlang_version": "0.1",
  "imports": ["github", "slack"],
  "tools": ["clone_repo", "run_tests", "deploy"],
  "capabilities": ["Git", "Filesystem", "Network"],
  "provider_default": "deepseek",
  "agents": [
    {
      "name": "Researcher",
      "provider": "deepseek",
      "model": "deepseek-chat",
      "capabilities": ["Network", "Filesystem"],
      "tools": ["clone_repo", "search_docs"],
      "prompt": null
    }
  ],
  "policy": {
    "max_cost": 2.0,
    "max_parallel": 8,
    "max_runtime": 300,
    "approval_for": ["deploy"]
  },
  "workflow": {
    "name": "Audit",
    "body": [
      {"kind": "step", "ref": "clone_repo"},
      {"kind": "parallel", "members": ["Researcher", "BusinessLogic", "InvariantChecker"]},
      {"kind": "consensus", "strategy": "judge", "mode": "synthesize", "confidence": false},
      {"kind": "if", "field": "confidence", "op": "<", "value": 0.85,
       "then": [{"kind": "ref", "name": "HumanReviewer"}]},
      {"kind": "artifact", "name": "AuditReport", "type": "markdown", "schema": null}
    ]
  }
}
```

Every node's `kind` maps to exactly one thing in §4's mapping table; a
node whose construct is still flagged in §5 (a standalone `if`, a
body-level `approval` step) is still valid IR — the parser never refuses
to produce it — but the executor raises a clear "not yet supported:
<construct>" error rather than silently dropping it (`policy.approval_for`
and `let` are no longer in this category — see §5.2 and §5.4). IR that
round-trips through `ail parse --ir` and back is how this gets tested
without needing the executor to exist yet.

## 8. Recommended v1 scope

Given §5's four gaps, shipping "faithfully compiles everything in §4"
as one milestone would silently mix a parser (low-risk, self-contained)
with at least one new `aircore` primitive (conditional fallback, §5.1) and
one new `airpy` primitive (`PromptTemplate`, §5.4). Splitting these
keeps each piece honestly scoped:

1. **AirLang-M0 — Lexer, parser, IR. Shipped.** Everything in §4 parses to
   §7's IR, including `if`/`approval`/`let` — the full grammar exists
   even though not all of it executes yet. `python -m ail parse file.airlang
   --ir` prints the IR as JSON. No `airpy`/`aircore` dependency at all in
   this milestone — pure syntax → data. (`airlang/lexer.py`, `airlang/parser.py`.)
2. **AirLang-M1 — Executor for the non-blocked subset. Shipped.** `import`,
   `tool` (ref), `capability`, `agent`, `provider`, `policy` (minus
   `approval_for`), `workflow`/`step`/`parallel`/`consensus`
   (majority/unanimous/judge, select/synthesize), `artifact` (schema=
   real, type= documentation-only per §5.3), top-level `memory`. Executor
   raised "not yet supported" for `if`/`approval`/`let` at ship time —
   `if` has since partially unblocked, see AirLang-M3 below.
   (`airlang/executor.py`, `airlang/bindings.py`.)
3. **AirLang-M2 — `ai run`/`ai trace` CLI integration. Shipped.** `.airlang`
   files run through `aircli`'s existing `run`/`trace` subcommands (detect
   the extension, execute via AirLang-M1's executor instead of `runpy`), get
   the same `Journal`/`--json`/`--html` trace viewer every Python-driven
   workflow already gets — AirLang workflows are not a second-class citizen
   in the CLI.
4. **AirLang-M3 — confidence-gated `if`. Shipped, partially.** §5.1(a)'s
   fallback landed in `aircore` (`ConsensusGroup.fallback`/`fallback_below`)
   and `airlang/executor.py` folds a qualifying `if` into it — see §5.1's
   "Shipped (AirLang-M3)" note for the exact shape and its limits. A
   standalone `if` (general branching) remains genuinely blocked.
5. **AirLang-M3.1 — policy-level `approval`. Shipped.** `aircore` decided on the
   mechanism §5.2 left open (a synchronous callback — `Policy.
   approval_for` + `Workflow.run(approval_callback=...)`, `aircore/
   approval.py`) once it became the highest-priority gap identified
   against CrewAI/LangGraph/OpenAI-Agents-SDK-style orchestrators. `airlang/
   executor.py` maps `policy { approval <tool> }` onto it directly;
   `aircli` wires in `cli_approval_callback` by default so `.airlang` files
   using it work interactively. The *body-level* `approval { message }`
   step (§4.12) — not tied to a named tool — is a different, still-
   unbuilt primitive; see §5.2's "Shipped, partially" note.
6. **AirLang-M3.2 — `let`. Shipped.** §5.4's second half (a value to plug
   into `PromptTemplate` at AirLang execution time) landed: producer-linked
   `artifact` binding in `airlang/executor.py`, plus a small, real `aircore`
   addition (`Workflow.consensus(..., as_=...)`, since a consensus
   group's one reduced value is exactly the shape `let` needed to bind
   from a `consensus`-produced artifact) — see §5.4's "Fully shipped"
   note for the exact scope.

## 9. Parser implementation note (for when M0 starts)

Hand-written recursive-descent, no parser-generator dependency — matches
how every other milestone in this project has been built (small,
explicit, no framework), and AirLang's grammar (§4) is small and fixed
enough that a generated parser would trade real control over error
messages for not much time saved. Revisit only if the grammar grows
substantially past what's specified here.

## 10. Open questions this document deliberately leaves open

- Exact syntax for an inline agent `prompt` (§4.4, §5.4) — resolved
  together with `PromptTemplate`, not before.
- Whether `airlang_version` needs to mean anything yet, or is just a
  forward-compatibility placeholder until there's a second version to
  distinguish it from.
- Whether `.airlang` files should be able to reference *other* `.airlang` files
  (e.g. reusing an `agent` block across workflows) — currently "one file,
  one workflow" (§1) says no; revisit only if real duplication across
  multiple `.airlang` files becomes a real, observed problem, not
  speculatively.
