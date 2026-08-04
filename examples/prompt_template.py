"""PromptTemplate acceptance example: {variable} substitution for
building a prompt from named pieces instead of hand-rolled f-strings --
see airpy/prompt_template.py's docstring for exactly what this does and
does not solve (it's a pure string-substitution utility; it has no idea
where a variable's value comes from, that's still the caller's job).

Run with: python examples/prompt_template.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airpy import Agent, MockProvider, PromptTemplate, PromptTemplateError, Workflow


if __name__ == "__main__":
    print("=== Building a prompt from named variables ===")
    template = PromptTemplate("Investigate {topic} using {source}, and flag any {concern}.")
    print(f"template.variables = {sorted(template.variables)}")

    prompt = template.render(topic="bloom filters", source="the project's own docs", concern="false positives")
    print(f"rendered: {prompt!r}")

    researcher = Agent("researcher", MockProvider(response=lambda req: f"Findings for: {req.prompt}"), prompt)
    workflow = Workflow("Research").step(researcher)
    journal = workflow.run()
    print(f"agent output: {journal.steps[0].output!r}")

    print("\n=== A missing variable fails loudly at render time, not silently ===")
    try:
        template.render(topic="bloom filters", source="docs")  # forgot `concern`
    except PromptTemplateError as exc:
        print(f"PromptTemplateError: {exc}")

    print("\n=== An unrecognized variable is rejected too (likely a typo) ===")
    try:
        template.render(topic="x", source="y", concern="z", extra="oops")
    except PromptTemplateError as exc:
        print(f"PromptTemplateError: {exc}")
