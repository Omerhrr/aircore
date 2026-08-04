"""PromptTemplate: plain {variable} substitution for building prompts out
of named pieces, instead of hand-rolled f-strings/string concatenation.

This exists for one reason: closing half of airlang-spec-v1.md section 5.4's
gap. AirLang's `agent` block has no way to express a prompt built from
runtime values (only a static string, or none at all -- see
model_agent.py's own module docstring, which never had a templating
concept either), and `let name = artifact X` has nothing to actually
plug `name` into. PromptTemplate is the small, self-contained primitive
the spec recommended building first, so both AirLang and hand-written airpy
code get it once instead of AirLang inventing its own ad hoc version.

Deliberately minimal, matching structured_output.py's own restraint:
`{variable}` substitution only (built on str.format(), so `{{`/`}}`
escape literal braces the same way they always have in Python format
strings) -- no expression language, no filters, no conditionals, no
arbitrary code. Every variable the template mentions must be supplied to
render(), and render() rejects any extra variable it doesn't recognize --
both fail loudly at render time rather than silently leaving a literal
"{topic}" in a prompt or silently dropping a value nobody asked for.

What this does NOT do, and why that's a separate, larger problem: this
module has no idea where a variable's *value* comes from. Rendering a
template is the caller's job -- `PromptTemplate("Summarize: {report}").
render(report=some_string)` -- and the caller must already have
`some_string` in hand. For a hand-written airpy script that's trivial
(it's just a local variable). For AirLang's `let report = artifact Report`,
it is NOT trivial: aircore's Workflow builds its whole step list before
`.run()` executes anything (see workflow.py), so there is no point during
`airlang.executor.build_workflow()` where a prior step's actual output
exists yet to plug into a later agent's prompt. Closing that fully needs
either lazy/deferred prompt construction in ModelAgent (built at call
time, not construction time) or some other cross-step data-flow
primitive in the runtime -- a real design question of its own, not
solved by adding this module. AirLang's `let` therefore stays
AirLangNotYetSupportedError even after this ships; see airlang-spec-v1.md section
5.4's status note.
"""

from __future__ import annotations

import string
from typing import Any, Dict, FrozenSet


class PromptTemplateError(Exception):
    """Raised by render() for a missing or unexpected variable. Always
    fail-fast and specific -- a template with a typo'd variable name
    should never silently produce a prompt containing a literal
    "{typo}", and a caller passing a variable the template doesn't use
    is almost always a mistake worth surfacing, not silently ignoring."""


class PromptTemplate:
    def __init__(self, template: str) -> None:
        self.template = template
        self.variables: FrozenSet[str] = _extract_variables(template)

    def render(self, **values: Any) -> str:
        missing = self.variables - values.keys()
        if missing:
            raise PromptTemplateError(
                f"missing template variable(s): {sorted(missing)} -- template needs "
                f"{sorted(self.variables)}, got {sorted(values.keys())}"
            )
        extra = values.keys() - self.variables
        if extra:
            raise PromptTemplateError(
                f"unexpected variable(s) not used in this template: {sorted(extra)} -- "
                f"template only uses {sorted(self.variables)}"
            )
        return self.template.format(**values)

    def __repr__(self) -> str:
        return f"<PromptTemplate variables={sorted(self.variables)}>"


def _extract_variables(template: str) -> FrozenSet[str]:
    """Every `{name}` field in `template`, via the same parser str.format()
    itself uses -- so escaping (`{{`/`}}`) and field syntax stay
    identical to normal Python format strings, no separate parsing rules
    to learn. Positional (`{}`) or indexed (`{0}`) fields aren't
    supported -- PromptTemplate is for named variables only, since an
    unnamed `{}` would defeat the whole point of render(**values)
    validating what was actually supplied."""
    names = set()
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name is None:
            continue
        if field_name == "" or not field_name.isidentifier():
            raise PromptTemplateError(
                f"PromptTemplate only supports named {{variable}} fields, not "
                f"positional/indexed ones or attribute/index access -- got "
                f"{{{field_name}}} in {template!r}"
            )
        names.add(field_name)
    return frozenset(names)
