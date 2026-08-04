"""aircli -- the `ai` command. Deliberately minimal.

`ai run <script.py>` executes a script and prints a one-line summary of
any aircore.Workflow it finds in the script's global scope after running
(i.e. any `workflow = Workflow(...)` your script defines at module level
and calls `.run()` on) -- and, since Session (airpy/session.py) shipped
after this CLI was first built, any airpy.Session that had at least one
`.send()` call, summarized as turn count and last-turn status. `ai trace
<script.py>` does the same but prints the full execution graph instead of
a summary -- one per Session turn, since each turn is its own Journal
(see Session's docstring for why). `ai trace --json` prints each journal
as JSON (Journal.to_json(), already existed, wasn't wired into the CLI
until now) instead of the rendered graph, for piping into something else.

`ai trace --html [--output path.html]` additionally writes a self-
contained, clickable HTML trace viewer (html_trace.py) -- one file, no
server, no CDN assets, opens in any browser. Click a step or group node
to see its status/latency/retries/usage(tokens, cost)/consensus metadata
(confidence, reasoning, judge decision, etc. -- whatever the strategy
reported)/output or error in a side panel. It's a static snapshot of the
Journal(s) produced by this run, not a live-updating dashboard -- see
html_trace.py's docstring for why that's a deliberate scope cut, and for
the one real gap it can't paper over (a step's *input* -- the prompt sent
to a ModelAgent, a Tool's arguments -- isn't recorded in the journal at
all, so there's nothing for the viewer to show there yet).

Both `ai run` and `ai trace` also accept a `.airlang` file (AirLang-M2) instead of
a `.py` script -- parsed and executed via the `airlang` package's IR executor
(AirLang-M1, see airlang/executor.py) instead of runpy, since a .airlang file has no
Python namespace to inspect afterward; the executor hands back the one
Workflow it built and ran directly. Most of AirLang's grammar runs, including
`if` (only immediately after a `consensus judge { confidence true }`
block, as a confidence-gated fallback -- general branching elsewhere is
still blocked) and `policy { approval <tool> }` (`_run_ail` passes
`aircore.approval.cli_approval_callback` by default, so this works
interactively). What still fails with one clear line on stderr
(AirLangNotYetSupportedError / AirLangBindingError), not a traceback: `let`, a
body-level `approval { message }` step, a body-level `memory` statement,
standalone `if` outside the consensus-fallback shape, or referencing a
tool/capability/provider with no binding. See examples/research.airlang (runs
end to end, provider mock, no bindings needed), examples/
research_with_fallback.airlang (runs end to end including the `if` fallback),
vs examples/audit.airlang (parses, but fails at its unbound `clone_repo` tool
reference) for these in practice.

None of these commands do anything a plain `python script.py` plus the
script's own `journal.pretty()`/`to_json()` calls couldn't already do --
the value is not having to add that boilerplate yourself. There is still
no `ai test` / `ai build` / `ai deploy`: those were speculative in the
original project vision, and shipping CLI surface for commands that don't
do anything real would be worse than not having them at all. "Finishing"
the CLI meant catching it up to features that already existed and had a
real journal to show (Session, and now .airlang), not inventing new surface
area beyond what those already-real features need.
"""

__version__ = "0.3.0"
