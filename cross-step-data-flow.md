# Cross-step data flow

Status: **shipped at the aircore/airpy level** (option 1 below, deferred/
lazy construction). A sequential step's real output can now feed a later
step's prompt within one `Workflow.run()` — see `aircore/workflow.py`'s
"Bindings" section, `Workflow.step(tool, as_="name")`, and airpy's
`ModelAgent(prompt=PromptTemplate(...), prompt_bindings=workflow.
bindings)`. Tests: `tests/test_bindings.py`.

**AirLang's `let` is now closed too.** Rather than a new grammar addition
(`artifact Report { from ResearcherStep }`, floated below as the likely
shape), the fix that actually shipped needed no grammar change at all:
`airlang/executor.py`'s `build_workflow()` treats an `artifact <Name>` node
immediately following its producing `step`/`ref`/`consensus` node as
positional linkage — no explicit `from` needed, since AirLang's existing
convention (seen throughout airlang-spec-v1.md's examples) already places
`artifact` right after what produces it. A pre-scan pass over the body
resolves every `let name = artifact X` into an alias map before any step
is built, so the producer's output is bound under the `let` name (or the
artifact's own name, if no `let` names it). See `airlang/executor.py`'s
module docstring and `examples/research_with_binding.airlang` for this
running end to end. `Workflow.consensus(..., as_=...)` also had to be
added (a consensus group's one reduced value is exactly the shape `let`
needed when the producer is a `consensus`, not a plain `step`) — see
`aircore/workflow.py`'s Bindings section.

The rest of this document is left as originally written, describing the
problem and the two candidate solutions that were on the table before
option 1 was chosen and built.

---

Written down originally so this didn't get rediscovered from scratch
later. Surfaced while closing airlang-spec-v1.md section 5.4's `let`/agent-
prompt gap — `PromptTemplate` (`airpy/prompt_template.py`) shipped and
solved half of that gap; this document was the other half, the part that
was a genuine open design question rather than a small addition.

## The problem, precisely

`PromptTemplate.render(**values)` needs the actual values up front, as
plain Python objects the caller already has in hand. That's trivial in
hand-written `airpy` code — the values are just local variables. It is
not trivial for AirLang's `let report = artifact Report` (or, more generally,
"feed step N's output into step N+1's prompt/arguments"), because of how
`aircore.Workflow` is built:

- `Workflow.step()` / `.parallel()` / `.consensus()` only ever *record*
  steps into `self._steps` (a plain list) — see `aircore/workflow.py`.
  Nothing executes until `.run()` is called.
- `.run()` hands the whole, already-fully-built `self._steps` list to
  `Scheduler.run()`, which then executes it top to bottom (see
  `aircore/scheduler.py`).
- Every `Executable` in that list (a `Tool`, an `airpy.Agent`/
  `ModelAgent`) is fully constructed *before* `.run()` starts — a
  `ModelAgent`'s `prompt` is fixed at `__init__` time, baked into
  `self._request_prompt` (see `airpy/model_agent.py`).

So there is no point in the current design where "step N's real output"
exists yet at the moment step N+1's prompt needs to be decided — building
happens entirely before running. `airlang.executor.build_workflow()` hits
this literally: it constructs every `Agent` up front, in one pass over
the IR, before `workflow.run()` is ever called.

This is why `airlang/executor.py` raised `AirLangNotYetSupportedError` for `let`
until this pass (see this document's status note at the top for how it
was closed) — `PromptTemplate` was ready to be used the moment there was
a value to hand it; producing that value from a prior step's real output
was the unsolved part, until now.

## Why this wasn't a small fix, and which shape got built

Closing it meant changing something structural about how steps are
built or run, not adding a utility module. Two real shapes were on the
table, neither of which was "just wire it up":

1. **Deferred/lazy construction (chosen, shipped).** A `ModelAgent`'s
   prompt is resolved at *call time*, not construction time — `prompt=`
   now accepts a `PromptTemplate` in addition to a fixed string, resolved
   by `ModelAgent._resolve_prompt()` (airpy/model_agent.py) immediately
   before the request is sent, reading from `prompt_bindings=` (a mutable
   dict reference, normally `workflow.bindings`). "Prior step outputs"
   are exposed by an explicit name a workflow author assigns --
   `Workflow.step(tool, as_="name")` -- not by step id; the Scheduler
   writes into `workflow.bindings` in place as each named step succeeds
   (aircore/scheduler.py), and anything holding a reference to that same
   dict object sees updates as they happen. See aircore/workflow.py's
   "Bindings" section for the full picture, including the checkpoint.py-
   consistent scope decision (sequential steps only).
2. **A first-class binding/variable primitive in `aircore` itself.**
   (Not built.) A more general `Workflow`-scoped variable table with its
   own Journal/Policy/graph integration -- bigger than what a concrete
   need justified. Option 1 turned out sufficient: it doesn't touch the
   Journal (bindings are a plain dict, not a journaled primitive) or
   Policy at all, and required no changes to the execution graph
   renderer, which is exactly the kind of minimal footprint this project
   prefers when it's enough to solve the actual problem.

## What already worked without this (still true, still useful)

None of the following ever needed cross-step data flow, and all remain
fully supported:

- A `PromptTemplate` rendered from values known *before* a workflow is
  built (config, a prior *run's* output loaded from `Memory`, a
  constant) — works today, no gap.
- `Memory` (session/project/temporary scopes) for passing state between
  *separate runs* (e.g. `Session`'s per-turn conversation history) — this
  is a different kind of "data flow" (across `.run()` calls, via an
  explicit read/write API) and already exists; it does not help within
  a single run's step sequence, which is what this document was about.
- Structured output (`output_schema=`) plus reading `journal.steps[i].
  output` *after* `.run()` completes, by a human or by code that then
  kicks off a *second* `Workflow` — a manual, two-run version of the
  same idea, still possible, just no longer the only option within one
  run.

## AirLang's `let` (closed)

See the status note at the top of this document for what shipped. What's
still genuinely open in AirLang, and tracked in airlang-spec-v1.md/airlang/__init__.py
instead of here (it's not a cross-step-data-flow gap): the body-level
`approval { message }` step, a standalone `if` outside the consensus-
fallback shape, and real schema enforcement for `artifact`.
