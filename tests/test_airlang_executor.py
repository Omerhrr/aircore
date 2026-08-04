"""AirLang-M1: the IR executor (airlang/executor.py) -- IR -> a real, run airpy
Workflow. Covers the non-blocked subset (step/ref/parallel/consensus/
artifact/let/agent/policy/memory), the scope decisions the executor's
docstring calls out (consensus always reduces the preceding parallel
block; artifact still doesn't enforce a schema against real output),
binding resolution (builtin capabilities, MockProvider's zero-config
`mock` provider name, and AirLangBindingError for anything unresolved), that
a body-level `approval { message }` step / a body-level `memory` still
raise AirLangNotYetSupportedError rather than silently doing nothing, that
`policy { approval <tool> }` maps onto a real Policy.approval_for (see
approval.py and aircore's own test_approval.py for the underlying
mechanism), and that `let`/producer-linked `artifact` binding now closes
the loop end to end (see tests/test_ail_bindings.py for the fuller
end-to-end coverage of that).
"""

import pytest

from airpy import MockProvider
from aircore import Capability, Network, Tool

from airlang.bindings import Bindings
from airlang.executor import AirLangBindingError, AirLangNotYetSupportedError, build_workflow, execute_ir
from airlang.parser import parse


def test_single_agent_workflow_executes_end_to_end():
    ir = parse("""
    agent Researcher {
        provider mock
        model mock
    }
    workflow Research {
        step Researcher
    }
    """)
    workflow = execute_ir(ir)
    assert workflow.journal.status == "success"
    assert workflow.journal.steps[0].output == "mock response"


def test_bare_ref_is_sugar_for_step():
    ir = parse("""
    agent Researcher { provider mock }
    workflow W { Researcher }
    """)
    workflow = execute_ir(ir)
    assert workflow.journal.status == "success"


def test_parallel_then_consensus_reuses_results_not_double_execution():
    calls = {"n": 0}

    def counting_response(request):
        calls["n"] += 1
        return "agreed"

    ir = parse("""
    agent A { provider mock }
    agent B { provider mock }
    agent C { provider mock }
    workflow W {
        parallel { A B C }
        consensus majority
    }
    """)
    bindings = Bindings(providers={"mock": MockProvider(response=counting_response)})
    workflow = execute_ir(ir, bindings)

    assert workflow.journal.status == "success"
    # 3 voters + 0 extra calls for consensus (majority doesn't call the
    # provider at all) -- proves parallel+consensus didn't re-run voters.
    assert calls["n"] == 3
    consensus_step = [s for s in workflow.journal.steps if s.tool == "consensus"][0]
    assert consensus_step.output == "agreed"


def test_consensus_judge_uses_top_level_default_provider():
    ir = parse("""
    provider mock
    agent A { provider mock }
    agent B { provider mock }
    workflow W {
        parallel { A B }
        consensus judge
    }
    """)
    workflow = execute_ir(ir)
    assert workflow.journal.status == "success"


def test_consensus_without_preceding_parallel_is_a_binding_error():
    ir = parse("agent A { provider mock } \n workflow W { consensus majority }")
    with pytest.raises(AirLangBindingError, match="must immediately follow a parallel"):
        build_workflow(ir)


def test_consensus_judge_with_no_default_provider_is_a_binding_error():
    ir = parse("""
    agent A { provider mock }
    agent B { provider mock }
    workflow W {
        parallel { A B }
        consensus judge
    }
    """)
    with pytest.raises(AirLangBindingError, match="consensus judge needs a provider"):
        build_workflow(ir)


def test_artifact_is_recorded_as_metadata_not_a_journal_step():
    ir = parse("""
    agent A { provider mock }
    workflow W {
        step A
        artifact Report { type markdown }
    }
    """)
    workflow = build_workflow(ir)
    assert len(workflow._steps) == 1  # artifact added no step
    assert workflow.airlang_artifacts == [{"name": "Report", "type": "markdown", "schema": None}]


def test_unknown_tool_reference_is_a_binding_error():
    ir = parse("workflow W { step clone_repo }")
    with pytest.raises(AirLangBindingError, match="tool 'clone_repo' has no implementation"):
        build_workflow(ir)


def test_bound_tool_resolves_and_executes():
    @Tool
    def clone_repo():
        return "cloned"

    ir = parse("workflow W { step clone_repo }")
    workflow = execute_ir(ir, Bindings(tools={"clone_repo": clone_repo}))
    assert workflow.journal.status == "success"
    assert workflow.journal.steps[0].output == "cloned"


def test_unknown_capability_is_a_binding_error():
    ir = parse("""
    agent A {
        provider mock
        capabilities { Teleportation }
    }
    workflow W { step A }
    """)
    with pytest.raises(AirLangBindingError, match="unknown capability 'Teleportation'"):
        build_workflow(ir)


def test_builtin_capability_resolves_to_the_real_airun_capability():
    ir = parse("""
    agent A {
        provider mock
        capabilities { Network }
    }
    workflow W { step A }
    """)
    workflow = build_workflow(ir)
    agent = workflow._steps[0][0]
    assert Network in agent.requires


def test_custom_capability_resolves_via_bindings():
    custom = Capability("Teleportation")
    ir = parse("""
    agent A {
        provider mock
        capabilities { Teleportation }
    }
    workflow W { step A }
    """)
    workflow = build_workflow(ir, Bindings(capabilities={"Teleportation": custom}))
    agent = workflow._steps[0][0]
    assert custom in agent.requires


def test_unknown_provider_is_a_binding_error():
    ir = parse("agent A { provider vibes } \n workflow W { step A }")
    with pytest.raises(AirLangBindingError, match="unknown provider 'vibes'"):
        build_workflow(ir)


def test_agent_with_no_provider_at_all_is_a_binding_error():
    ir = parse("agent A { } \n workflow W { step A }")
    with pytest.raises(AirLangBindingError, match="has no provider"):
        build_workflow(ir)


def test_top_level_provider_default_applies_to_agents_without_their_own():
    ir = parse("""
    provider mock
    agent A { }
    workflow W { step A }
    """)
    workflow = execute_ir(ir)
    assert workflow.journal.status == "success"


def test_policy_fields_flow_into_the_real_airun_policy():
    ir = parse("""
    agent A { provider mock }
    agent B { provider mock }
    policy { max_parallel 1 }
    workflow W { parallel { A B } consensus majority }
    """)
    from aircore import PolicyViolation
    with pytest.raises(PolicyViolation):
        build_workflow(ir).run()


def test_policy_approval_maps_to_a_real_policy_approval_for():
    # AirLang-M3.1 (alongside aircore/approval.py): `policy { approval X }` now
    # builds -- it maps straight onto Policy.approval_for. Building never
    # needed an approval_callback; only running a workflow with a gated
    # tool does (see the next two tests).
    ir = parse("""
    agent A { provider mock }
    policy { approval A }
    workflow W { step A }
    """)
    workflow = build_workflow(ir)
    assert workflow.policy.approval_for == frozenset({"A"})


def test_running_a_policy_approval_workflow_without_a_callback_is_a_policy_violation():
    from aircore import PolicyViolation

    ir = parse("""
    agent A { provider mock }
    policy { approval A }
    workflow W { step A }
    """)
    with pytest.raises(PolicyViolation, match="approval_callback"):
        execute_ir(ir)


def test_running_a_policy_approval_workflow_with_auto_approve_succeeds():
    from aircore import auto_approve

    ir = parse("""
    agent A { provider mock }
    policy { approval A }
    workflow W { step A }
    """)
    workflow = execute_ir(ir, approval_callback=auto_approve)
    assert workflow.journal.status == "success"


def test_standalone_if_not_after_consensus_is_still_not_yet_supported():
    # AirLang-M3 folds `if` into a *preceding consensus's* fallback (see
    # test_ail_fallback.py) -- `if` anywhere else is still the general
    # branching case airlang-spec-v1.md section 5.1 didn't build a primitive
    # for.
    ir = parse("""
    agent A { provider mock }
    agent Reviewer { provider mock }
    workflow W {
        step A
        if confidence < 0.85 { Reviewer }
    }
    """)
    with pytest.raises(AirLangNotYetSupportedError, match="'if'"):
        build_workflow(ir)


def test_if_after_consensus_with_non_judge_strategy_is_a_binding_error():
    # This used to be the case that made `if` unconditionally "not yet
    # supported"; now that judge+confidence folding exists, a non-judge
    # strategy paired with `if confidence < X` is a real, specific
    # binding error instead (majority/unanimous can never report
    # confidence) -- see test_ail_fallback.py for the full behavior.
    ir = parse("""
    agent A { provider mock }
    agent B { provider mock }
    agent Reviewer { provider mock }
    workflow W {
        parallel { A B }
        consensus majority
        if confidence < 0.85 { Reviewer }
    }
    """)
    with pytest.raises(AirLangBindingError, match="needs a `consensus judge`"):
        build_workflow(ir)


def test_let_referencing_an_undeclared_artifact_is_a_binding_error():
    ir = parse("""
    workflow W {
        let report = artifact NeverDeclared
    }
    """)
    with pytest.raises(AirLangBindingError, match="NeverDeclared"):
        build_workflow(ir)


def test_body_level_memory_is_not_yet_supported():
    ir = parse("agent A { provider mock } \n workflow W { step A \n memory session }")
    with pytest.raises(AirLangNotYetSupportedError, match="memory"):
        build_workflow(ir)


def test_top_level_memory_sets_the_workflow_memory_scope():
    ir = parse("""
    memory session
    agent A { provider mock }
    workflow W { step A }
    """)
    workflow = build_workflow(ir)
    assert workflow.memory is not None


def test_prompt_defaults_when_omitted():
    ir = parse("agent A { provider mock }\nworkflow W { step A }")
    workflow = build_workflow(ir)
    agent = workflow._steps[0][0]
    assert agent.prompt == "You are A."


def test_explicit_prompt_is_used():
    ir = parse('agent A { provider mock \n prompt "Investigate." }\nworkflow W { step A }')
    workflow = build_workflow(ir)
    agent = workflow._steps[0][0]
    assert agent.prompt == "Investigate."
