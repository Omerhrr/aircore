"""airlang -- the AirLang (AI language) frontend.

AirLang-M0 (lexer.py, parser.py): .airlang source -> IR, a plain JSON-serializable
dict (airlang-spec-v1.md section 7). No dependency on aircore or airpy.

AirLang-M1 (executor.py, bindings.py): IR -> a real, run airpy Workflow.
This is the first place in this package that imports airpy -- per
airlang-spec-v1.md section 3, `airlang` imports `airpy`, `airpy` never imports
`airlang`. Only the non-blocked subset of the grammar executes (section 8); a
body-level `approval { message }` step and a body-level `memory`
statement raise AirLangNotYetSupportedError rather than silently no-opping --
see executor.py's docstring for exactly what that covers and the real
scope decisions it had to make that the spec's IR sketch didn't fully
resolve (consensus always reduces the preceding parallel block; artifact
still doesn't validate a schema against real output).

`let <name> = artifact <ArtifactName>` IS now supported: an `artifact`
node immediately following a `step`/`ref`/`consensus` node is treated as
that node's producer and bound in `workflow.bindings` (the `let` name if
one was declared for it, the artifact's own name otherwise); an `agent`
prompt containing `{...}` compiles to a PromptTemplate reading those
bindings at execute() time instead of a plain string. See executor.py's
docstring for the exact scope (the artifact must immediately follow its
producer; referencing an artifact name in `let` that never appears as an
`artifact` node anywhere in the body is an AirLangBindingError) and
examples/research_with_binding.airlang for it running end to end.

Policy-level `approval <tool>` IS supported (closed alongside aircore/
approval.py): maps to Policy.approval_for, and running such a workflow
needs an `approval_callback` passed to execute_ir()/execute_file() -- see
executor.py's docstring.

AirLang-M2 (CLI integration, `ai run`/`ai trace` accepting .airlang files) lives
in aircli, not here -- see aircli/__main__.py. It passes
`aircore.approval.cli_approval_callback` by default, so a policy-level
`approval <tool>` line works interactively out of the box.

Still not built: any executor for a non-Python IR consumer (airjs/airgo
would each need their own, per airlang-spec-v1.md section 3's language-
agnostic-IR design); a body-level `approval { message }` step (different
from the now-working `policy { approval <tool> }`, see above); a
standalone `if` anywhere other than immediately after a `consensus judge
{ confidence true }` block (general branching was never built); real
schema enforcement for `artifact` (`schema`/`type` are recorded but never
validated against the step's actual output).
"""

from .bindings import Bindings, load_bindings_for
from .executor import AirLangBindingError, AirLangNotYetSupportedError, build_workflow, execute_file, execute_ir
from .lexer import AirLangSyntaxError, Token, tokenize
from .parser import IR_VERSION, parse, parse_file

__all__ = [
    "AirLangSyntaxError", "Token", "tokenize", "IR_VERSION", "parse", "parse_file",
    "Bindings", "load_bindings_for",
    "AirLangBindingError", "AirLangNotYetSupportedError", "build_workflow", "execute_ir", "execute_file",
]

__version__ = "0.4.0"
