"""Cross-step data flow (aircore/workflow.py's `bindings`, Workflow.step
(as_=...), airpy's ModelAgent(prompt=PromptTemplate(...),
prompt_bindings=...)) -- closes the gap cross-step-data-flow.md
documented: a sequential step's real output can now feed a later step's
prompt within a single Workflow.run(), which previously required two
separate runs (build the whole step list, including every prompt, before
anything executes).
"""

import pytest

from airpy import Agent, MockProvider, PromptTemplate, PromptTemplateError
from aircore import Tool, Workflow


def test_a_bound_step_output_is_recorded_in_workflow_bindings():
    workflow = Workflow("W")
    workflow.step(Tool(lambda: "the summary text", name="summarize"), as_="summary")
    journal = workflow.run()

    assert journal.status == "success"
    assert workflow.bindings == {"summary": "the summary text"}


def test_an_unbound_step_does_not_appear_in_bindings():
    workflow = Workflow("W")
    workflow.step(Tool(lambda: "output", name="plain"))  # no as_
    workflow.run()
    assert workflow.bindings == {}


def test_a_failed_step_binds_nothing():
    workflow = Workflow("W")

    def boom():
        raise RuntimeError("nope")

    workflow.step(Tool(boom, name="failing"), as_="never_bound")
    journal = workflow.run()
    assert journal.status == "failed"
    assert "never_bound" not in workflow.bindings


def test_bindings_are_cleared_at_the_start_of_every_run():
    workflow = Workflow("W")
    workflow.step(Tool(lambda: "first", name="a"), as_="value")
    workflow.run()
    assert workflow.bindings == {"value": "first"}

    # Re-running the same Workflow object (e.g. in a loop) must not let a
    # prior run's binding leak forward if this run's step never binds.
    workflow._steps = []
    workflow.step(Tool(lambda: "irrelevant", name="b"))  # no as_ this time
    workflow.run()
    assert workflow.bindings == {}


def test_a_later_agents_prompt_template_reads_an_earlier_steps_bound_output():
    workflow = Workflow("Pipeline")
    workflow.step(Tool(lambda: "bloom filters are space-efficient", name="research"), as_="findings")

    template = PromptTemplate("Critique this: {findings}")
    seen_prompts = []
    critic = Agent(
        "critic",
        MockProvider(response=lambda req: seen_prompts.append(req.prompt) or "looks fine"),
        template,
        prompt_bindings=workflow.bindings,
    )
    workflow.step(critic)
    journal = workflow.run()

    assert journal.status == "success"
    assert seen_prompts == ["Critique this: bloom filters are space-efficient"]
    assert journal.steps[-1].output == "looks fine"


def test_a_prompt_template_referencing_an_unbound_variable_fails_the_step_loudly():
    workflow = Workflow("Pipeline")
    # Nothing ever binds "findings" -- referencing it is a mistake (typo,
    # or the producing step hasn't run yet/never runs).
    template = PromptTemplate("Critique this: {findings}")
    critic = Agent("critic", MockProvider(response="unused"), template,
                    prompt_bindings=workflow.bindings)
    workflow.step(critic)
    journal = workflow.run()

    assert journal.status == "failed"
    assert "missing template variable" in journal.steps[0].error


def test_prompt_template_ignores_unrelated_bindings_already_present():
    workflow = Workflow("Pipeline")
    workflow.step(Tool(lambda: "unrelated value", name="other"), as_="other_name")
    workflow.step(Tool(lambda: "the real finding", name="research"), as_="findings")

    template = PromptTemplate("Critique this: {findings}")
    seen_prompts = []
    critic = Agent(
        "critic",
        MockProvider(response=lambda req: seen_prompts.append(req.prompt) or "ok"),
        template,
        prompt_bindings=workflow.bindings,
    )
    workflow.step(critic)
    journal = workflow.run()

    assert journal.status == "success"
    assert seen_prompts == ["Critique this: the real finding"]


def test_model_agent_with_prompt_template_requires_prompt_bindings():
    with pytest.raises(ValueError, match="prompt_bindings"):
        Agent("critic", MockProvider(response="x"), PromptTemplate("Critique {x}"))


def test_model_agent_rejects_a_non_string_non_template_prompt():
    with pytest.raises(TypeError):
        Agent("critic", MockProvider(response="x"), 12345)  # type: ignore[arg-type]


def test_plain_string_prompt_is_completely_unaffected_by_prompt_bindings_support():
    workflow = Workflow("W")
    agent = Agent("plain", MockProvider(response=lambda req: f"echo: {req.prompt}"), "hello world")
    workflow.step(agent)
    journal = workflow.run()
    assert journal.steps[0].output == "echo: hello world"


# --- Consensus group binding (Workflow.consensus(..., as_=...)) ---
# A consensus group reduces N voters to exactly one agreed value, the same
# shape as a single step's output -- so unlike a plain `parallel` block
# (no single value to bind), it can bind that one reduced result. See
# workflow.py's Bindings section.

def test_consensus_binds_its_agreed_value():
    workflow = Workflow("W")
    a = Tool(lambda: "yes", name="voter_a")
    b = Tool(lambda: "yes", name="voter_b")
    workflow.consensus(a, b, as_="verdict")
    journal = workflow.run()

    assert journal.status == "success"
    assert workflow.bindings == {"verdict": "yes"}


def test_consensus_reuse_mode_also_binds_via_parallel_results_consensus():
    workflow = Workflow("W")
    a = Tool(lambda: "agree", name="voter_a")
    b = Tool(lambda: "agree", name="voter_b")
    workflow.parallel(a, b).consensus(as_="verdict")
    journal = workflow.run()

    assert journal.status == "success"
    assert workflow.bindings == {"verdict": "agree"}


def test_a_failed_consensus_binds_nothing():
    workflow = Workflow("W")
    a = Tool(lambda: "yes", name="voter_a")
    b = Tool(lambda: "no", name="voter_b")  # disagreement -> majority() raises
    workflow.consensus(a, b, as_="never_bound")
    journal = workflow.run()

    assert journal.status == "failed"
    assert "never_bound" not in workflow.bindings


def test_consensus_without_as_does_not_bind():
    workflow = Workflow("W")
    a = Tool(lambda: "yes", name="voter_a")
    b = Tool(lambda: "yes", name="voter_b")
    workflow.consensus(a, b)  # no as_
    journal = workflow.run()

    assert journal.status == "success"
    assert workflow.bindings == {}


def test_a_later_agents_prompt_template_reads_a_consensus_agreed_value():
    workflow = Workflow("Pipeline")
    a = Tool(lambda: "space-efficient probabilistic set membership", name="voter_a")
    b = Tool(lambda: "space-efficient probabilistic set membership", name="voter_b")
    workflow.consensus(a, b, as_="verdict")

    template = PromptTemplate("Summarize: {verdict}")
    seen_prompts = []
    summarizer = Agent(
        "summarizer",
        MockProvider(response=lambda req: seen_prompts.append(req.prompt) or "summarized"),
        template,
        prompt_bindings=workflow.bindings,
    )
    workflow.step(summarizer)
    journal = workflow.run()

    assert journal.status == "success"
    assert seen_prompts == ["Summarize: space-efficient probabilistic set membership"]
