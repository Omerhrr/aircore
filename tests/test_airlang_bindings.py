"""End-to-end coverage of AirLang's `let`/producer-linked `artifact` binding
(airlang/executor.py's producer-linkage + aliasing passes, closing
airlang-spec-v1.md section 5.4). See test_ail_executor.py for the executor's
other, narrower unit tests -- this file specifically exercises the full
"a step's real output flows into a later agent's prompt via AirLang syntax
alone" path, the actual gap section 5.4 flagged.
"""

import pytest

from airlang.bindings import Bindings
from airlang.executor import AirLangBindingError, execute_ir
from airlang.parser import parse
from airpy import MockProvider


def test_artifact_immediately_after_a_step_binds_under_its_own_name():
    ir = parse("""
    agent Researcher { provider mock }
    workflow W {
        step Researcher
        artifact Findings
    }
    """)
    workflow = execute_ir(ir)
    assert workflow.journal.status == "success"
    assert workflow.bindings == {"Findings": "mock response"}


def test_let_aliases_the_artifact_binding_to_its_own_name():
    ir = parse("""
    agent Researcher { provider mock }
    workflow W {
        step Researcher
        artifact Findings
        let report = artifact Findings
    }
    """)
    workflow = execute_ir(ir)
    assert workflow.journal.status == "success"
    # The `let` name is what's actually usable in a later prompt --
    # binding under the artifact's own name too would be a second,
    # unrequested key nothing in AirLang ever asked for.
    assert workflow.bindings == {"report": "mock response"}
    assert workflow.airlang_lets == {"report": "Findings"}


def test_a_later_agents_prompt_reads_the_let_bound_value():
    seen_prompts = []

    def researcher_response(request):
        return "bloom filters are space-efficient"

    def critic_response(request):
        seen_prompts.append(request.prompt)
        return "solid summary"

    ir = parse("""
    agent Researcher { provider research_provider }
    agent Critic { provider critic_provider prompt "Critique this: {report}" }
    workflow W {
        step Researcher
        artifact Findings
        let report = artifact Findings
        step Critic
    }
    """)
    bindings = Bindings(providers={
        "research_provider": MockProvider(response=researcher_response),
        "critic_provider": MockProvider(response=critic_response),
    })
    workflow = execute_ir(ir, bindings)

    assert workflow.journal.status == "success"
    assert seen_prompts == ["Critique this: bloom filters are space-efficient"]
    assert workflow.journal.steps[-1].output == "solid summary"


def test_artifact_after_consensus_binds_the_agreed_value_via_let():
    def agreeing_response(request):
        return "agreed finding"

    ir = parse("""
    agent A { provider p }
    agent B { provider p }
    agent Critic { provider p prompt "Critique this: {verdict}" }
    workflow W {
        parallel { A B }
        consensus majority
        artifact Verdict
        let verdict = artifact Verdict
        step Critic
    }
    """)
    seen_prompts = []

    def critic_response(request):
        seen_prompts.append(request.prompt)
        return "reviewed"

    bindings = Bindings(providers={
        "p": MockProvider(response=lambda req: (
            critic_response(req) if req.prompt.startswith("Critique") else agreeing_response(req)
        )),
    })
    workflow = execute_ir(ir, bindings)

    assert workflow.journal.status == "success"
    assert seen_prompts == ["Critique this: agreed finding"]


def test_artifact_with_no_preceding_producer_stays_metadata_only():
    ir = parse("""
    workflow W {
        artifact Standalone
    }
    """)
    workflow = execute_ir(ir)
    assert workflow.journal.status == "success"
    assert workflow.bindings == {}
    assert workflow.airlang_artifacts == [{"name": "Standalone", "type": None, "schema": None}]


def test_let_referencing_an_artifact_thats_never_bound_fails_loudly_at_the_reading_step():
    # A real ordering mistake: `let` is declared, but the artifact it
    # names never follows a producer, so nothing ever binds it. The
    # later agent referencing {report} must fail loudly, not silently
    # send a literal "{report}" to the model.
    ir = parse("""
    agent Critic { provider mock prompt "Critique this: {report}" }
    workflow W {
        artifact Findings
        let report = artifact Findings
        step Critic
    }
    """)
    workflow = execute_ir(ir)
    assert workflow.journal.status == "failed"
    assert "missing template variable" in workflow.journal.steps[0].error


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
