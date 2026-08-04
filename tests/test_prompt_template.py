"""PromptTemplate (airpy/prompt_template.py): plain {variable} prompt
substitution. Covers rendering, escaping, missing/extra-variable
failures, rejecting positional/attribute/index fields, and using a
rendered template as a real ModelAgent prompt end to end through a
Workflow -- proving this is a genuinely useful, self-contained airpy
primitive on its own, independent of anything AirLang-related.
"""

import pytest

from airpy import Agent, MockProvider, PromptTemplate, PromptTemplateError, Workflow


def test_renders_named_variables():
    template = PromptTemplate("Summarize {topic} using {source}.")
    assert template.render(topic="deepseek", source="docs") == "Summarize deepseek using docs."


def test_variables_property_reports_every_named_field():
    template = PromptTemplate("{a} and {b} and {a} again")
    assert template.variables == frozenset({"a", "b"})


def test_double_braces_are_literal_not_variables():
    template = PromptTemplate("literal {{braces}} and {real}")
    assert template.variables == frozenset({"real"})
    assert template.render(real="x") == "literal {braces} and x"


def test_template_with_no_variables_renders_unchanged():
    template = PromptTemplate("no variables here")
    assert template.render() == "no variables here"


def test_missing_variable_raises_with_a_clear_message():
    template = PromptTemplate("Summarize {topic} using {source}.")
    with pytest.raises(PromptTemplateError, match=r"missing template variable\(s\).*source"):
        template.render(topic="deepseek")


def test_extra_variable_raises_with_a_clear_message():
    template = PromptTemplate("Summarize {topic}.")
    with pytest.raises(PromptTemplateError, match=r"unexpected variable\(s\).*extra"):
        template.render(topic="deepseek", extra="oops")


def test_positional_field_is_rejected_at_construction():
    with pytest.raises(PromptTemplateError, match="named"):
        PromptTemplate("Summarize {}.")


def test_attribute_access_field_is_rejected_at_construction():
    with pytest.raises(PromptTemplateError, match="named"):
        PromptTemplate("Summarize {request.topic}.")


def test_index_access_field_is_rejected_at_construction():
    with pytest.raises(PromptTemplateError, match="named"):
        PromptTemplate("Summarize {items[0]}.")


def test_rendered_template_works_as_a_real_agent_prompt_end_to_end():
    template = PromptTemplate("Investigate {topic} using {source}.")
    prompt = template.render(topic="bloom filters", source="the docs")

    agent = Agent("researcher", MockProvider(response=lambda req: f"echo: {req.prompt}"), prompt)
    workflow = Workflow("Research").step(agent)
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.steps[0].output == "echo: Investigate bloom filters using the docs."


def test_repr_reports_sorted_variables():
    template = PromptTemplate("{b} then {a}")
    assert repr(template) == "<PromptTemplate variables=['a', 'b']>"
