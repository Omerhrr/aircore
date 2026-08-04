"""Result reuse: workflow.parallel(...).consensus(strategy=...) (or the
equivalent workflow.consensus(results, strategy=...)) must not re-execute
the same voters a second time -- see workflow.py's ParallelResults,
consensus.py's ConsensusGroup(source_group=...), and scheduler.py's
_run_reused_consensus_group.

The whole point: for a workflow of expensive Tools/ModelAgents, the old
`workflow.parallel(a, b, c)` + `workflow.consensus(a, b, c, strategy=...)`
pattern paid for a, b, and c twice. These tests assert each voter's
underlying function is called exactly once regardless of how many
downstream consensus steps reduce over its output.
"""

from aircore import Workflow, Tool, ConsensusFailed
from airpy import MockProvider, JudgeConsensus, JudgeConsensusFailed


def _counting_tool(name, value):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return value

    return Tool(fn, name=name), calls


def test_parallel_then_consensus_via_chained_handle_executes_each_voter_once():
    a, a_calls = _counting_tool("a", "answer")
    b, b_calls = _counting_tool("b", "answer")
    c, c_calls = _counting_tool("c", "answer")

    workflow = Workflow("reuse-chained")
    workflow.parallel(a, b, c).consensus()
    journal = workflow.run()

    assert journal.status == "success"
    assert a_calls["n"] == 1
    assert b_calls["n"] == 1
    assert c_calls["n"] == 1

    consensus_steps = [s for s in journal.steps if s.tool == "consensus"]
    assert len(consensus_steps) == 1
    assert consensus_steps[0].output == "answer"


def test_parallel_then_consensus_via_explicit_results_argument_executes_each_voter_once():
    a, a_calls = _counting_tool("a", "answer")
    b, b_calls = _counting_tool("b", "answer")

    workflow = Workflow("reuse-explicit")
    results = workflow.parallel(a, b)
    workflow.consensus(results, strategy=lambda outputs: outputs[0])
    journal = workflow.run()

    assert journal.status == "success"
    assert a_calls["n"] == 1
    assert b_calls["n"] == 1


def test_reuse_journal_has_no_duplicate_voter_steps():
    # Old behavior: parallel group's 3 steps + consensus group's 3 more
    # steps + 1 synthetic = 7 steps. Reuse mode: 3 + 1 synthetic = 4.
    a, _ = _counting_tool("a", "x")
    b, _ = _counting_tool("b", "x")
    c, _ = _counting_tool("c", "x")

    workflow = Workflow("reuse-journal-shape")
    workflow.parallel(a, b, c).consensus()
    journal = workflow.run()

    assert len(journal.steps) == 4
    assert len(journal.groups) == 2
    voter_step_names = sorted(s.tool for s in journal.steps if s.tool != "consensus")
    assert voter_step_names == ["a", "b", "c"]


def test_reuse_mode_strategy_failure_fails_gracefully_without_crashing():
    a, _ = _counting_tool("a", "yes")
    b, _ = _counting_tool("b", "no")

    workflow = Workflow("reuse-strategy-fails")
    workflow.parallel(a, b).consensus()  # default strategy=majority, tie -> ConsensusFailed
    journal = workflow.run()  # must not raise

    assert journal.status == "failed"
    consensus_steps = [s for s in journal.steps if s.tool == "consensus"]
    assert len(consensus_steps) == 1
    assert consensus_steps[0].status == "failed"
    assert "ConsensusFailed" in consensus_steps[0].error


def test_reuse_mode_with_judge_consensus_still_costs_only_the_voters_plus_one_judge_call():
    call_counts = {"researcher": 0, "reviewer": 0, "professor": 0}

    def make(name, text):
        def fn():
            call_counts[name] += 1
            return text
        return Tool(fn, name=name)

    researcher = make("researcher", "A bloom filter is a probabilistic set membership structure.")
    reviewer = make("reviewer", "Bloom filters are space-efficient probabilistic structures.")
    professor = make("professor", "It's a probabilistic, space-saving structure.")

    judge_calls = {"n": 0}

    def judge_response(request):
        judge_calls["n"] += 1
        return "A bloom filter is a probabilistic set membership structure."

    judge_provider = MockProvider(response=judge_response)

    workflow = Workflow("reuse-judge")
    workflow.parallel(researcher, reviewer, professor).consensus(strategy=JudgeConsensus(judge_provider))
    journal = workflow.run()

    assert journal.status == "success"
    assert call_counts == {"researcher": 1, "reviewer": 1, "professor": 1}
    assert judge_calls["n"] == 1


def test_results_from_one_workflow_cannot_be_reused_on_another():
    a, _ = _counting_tool("a", "x")
    b, _ = _counting_tool("b", "x")

    workflow1 = Workflow("wf1")
    results = workflow1.parallel(a, b)

    workflow2 = Workflow("wf2")
    try:
        workflow2.consensus(results)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "different Workflow" in str(exc)


def test_normal_consensus_form_without_reuse_still_double_executes_as_before():
    # Explicit regression guard: passing the raw tools (not a
    # ParallelResults handle) to both parallel() and consensus() keeps the
    # old, non-reuse behavior -- reuse is opt-in, not a silent behavior
    # change for existing callers.
    a, a_calls = _counting_tool("a", "x")
    b, b_calls = _counting_tool("b", "x")

    workflow = Workflow("no-reuse")
    workflow.parallel(a, b)
    workflow.consensus(a, b)
    journal = workflow.run()

    assert journal.status == "success"
    assert a_calls["n"] == 2
    assert b_calls["n"] == 2
