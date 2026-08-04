"""Human-in-the-loop approval (aircore/approval.py, Policy.approval_for,
scheduler.py's gate in _run_tool).

Covers: pre-flight PolicyViolation when approval_for is set but no
approval_callback is given to run(); auto_approve lets a gated step run
normally; auto_deny fails just that step (ApprovalDenied), not the whole
process, and the workflow status reflects it; ApprovalRequested/
ApprovalDecided land in the Journal (StepRecord.approved, pretty()'s
"Approval: granted/denied" line); a tool not named in approval_for is
never gated even when other tools in the same workflow are; approval also
gates a member of a parallel/consensus group, not just sequential steps;
and Policy(approval_for=[...]) normalizes a plain list/tuple into a
frozenset the same way Tool(requires=...) does.
"""

import pytest

from aircore import Policy, PolicyViolation, Tool, Workflow, auto_approve, auto_deny


def _wf(approval_for, deploy_calls):
    workflow = Workflow("Deploy", policy=Policy(approval_for=approval_for))
    workflow.step(Tool(lambda: "prepared", name="prepare"))
    workflow.step(Tool(lambda: deploy_calls.append(1) or "deployed", name="deploy"))
    return workflow


def test_policy_approval_for_normalizes_any_iterable_to_a_frozenset():
    assert Policy(approval_for=["deploy", "delete"]).approval_for == frozenset({"deploy", "delete"})
    assert Policy(approval_for=("deploy",)).approval_for == frozenset({"deploy"})
    assert Policy(approval_for={"deploy"}).approval_for == frozenset({"deploy"})
    assert Policy().approval_for is None


def test_run_without_an_approval_callback_is_a_preflight_policy_violation():
    workflow = _wf({"deploy"}, deploy_calls=[])
    with pytest.raises(PolicyViolation, match="approval_callback"):
        workflow.run()
    # pre-flight means nothing ran at all -- no journal for a run that
    # never happened, same guarantee require_agent already gives.
    assert workflow.journal is None


def test_auto_approve_lets_the_gated_step_run_normally():
    calls = []
    workflow = _wf({"deploy"}, deploy_calls=calls)
    journal = workflow.run(approval_callback=auto_approve)

    assert journal.status == "success"
    assert calls == [1]
    deploy_step = next(s for s in journal.steps if s.tool == "deploy")
    assert deploy_step.approved is True
    assert deploy_step.status == "success"


def test_auto_deny_fails_only_the_gated_step():
    calls = []
    workflow = _wf({"deploy"}, deploy_calls=calls)
    journal = workflow.run(approval_callback=auto_deny)

    assert journal.status == "failed"
    assert calls == []  # deploy's fn never actually ran
    prepare_step = next(s for s in journal.steps if s.tool == "prepare")
    deploy_step = next(s for s in journal.steps if s.tool == "deploy")
    assert prepare_step.status == "success"  # the ungated step before it still ran
    assert deploy_step.status == "failed"
    assert deploy_step.approved is False
    assert "ApprovalDenied" in deploy_step.error


def test_a_tool_not_named_in_approval_for_is_never_gated():
    calls = []
    workflow = _wf({"deploy"}, deploy_calls=calls)
    journal = workflow.run(approval_callback=auto_deny)

    # "prepare" isn't in approval_for, so auto_deny is never even consulted
    # for it -- it succeeds regardless of what the callback would say.
    prepare_step = next(s for s in journal.steps if s.tool == "prepare")
    assert prepare_step.status == "success"
    assert prepare_step.approved is None


def test_workflow_with_no_approval_for_is_unaffected_by_a_callback():
    workflow = Workflow("Plain")
    workflow.step(Tool(lambda: "ok", name="a"))
    journal = workflow.run(approval_callback=auto_deny)  # never consulted
    assert journal.status == "success"
    assert journal.steps[0].approved is None


def test_pretty_renders_the_approval_decision():
    workflow = _wf({"deploy"}, deploy_calls=[])
    journal = workflow.run(approval_callback=auto_approve)
    assert "Approval: granted" in journal.pretty()

    workflow2 = _wf({"deploy"}, deploy_calls=[])
    journal2 = workflow2.run(approval_callback=auto_deny)
    assert "Approval: denied" in journal2.pretty()


def test_approval_gates_a_member_of_a_parallel_group():
    calls = []
    workflow = Workflow("Group", policy=Policy(approval_for={"risky"}))
    safe = Tool(lambda: "safe-output", name="safe")
    risky = Tool(lambda: calls.append(1) or "risky-output", name="risky")
    workflow.parallel(safe, risky)
    journal = workflow.run(approval_callback=auto_deny)

    assert journal.status == "failed"
    assert calls == []
    risky_step = next(s for s in journal.steps if s.tool == "risky")
    assert risky_step.approved is False


def test_approval_callback_receives_a_populated_approval_request():
    from aircore.approval import ApprovalRequest

    seen = []

    def recording_callback(request: ApprovalRequest) -> bool:
        seen.append(request)
        return True

    workflow = _wf({"deploy"}, deploy_calls=[])
    workflow.run(approval_callback=recording_callback)

    assert len(seen) == 1
    request = seen[0]
    assert request.tool_name == "deploy"
    assert request.workflow_name == "Deploy"
    assert isinstance(request.step_id, int)
