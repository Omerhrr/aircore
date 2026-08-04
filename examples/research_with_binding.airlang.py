"""Bindings for research_with_binding.airlang -- see airlang-spec-v1.md section 6
and airlang/bindings.py for the sibling-file convention this implements.

Scripts Researcher's response to a fixed, recognizable finding and
Critic's response to something that proves it actually received that
finding in its prompt (not the raw literal string "{report}", which is
what would show up here if the binding/template wiring were broken) --
the point of this file is to make airlang/executor.py's producer-linkage and
PromptTemplate compilation visible when run via `ai run`/`ai trace`
without needing a real provider or API key.
"""

from airpy import MockProvider

_FINDING = "Bloom filters trade a small false-positive rate for large space savings."


def _respond(request):
    if "Critique this finding" in request.prompt:
        # If {report} never got substituted, this branch is never reached
        # (the literal prompt sent would still contain "{report}", not
        # the finding text) -- so reaching this response at all is itself
        # proof the binding worked.
        assert _FINDING in request.prompt, (
            f"expected the Researcher's real finding in Critic's prompt, got: {request.prompt!r}"
        )
        return f"Critique: sound claim, but '{_FINDING[:20]}...' needs a citation."
    return _FINDING


PROVIDERS = {"mock": MockProvider(response=_respond)}
