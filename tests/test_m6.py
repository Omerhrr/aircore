import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool, Agent, Policy, PolicyViolation, majority, unanimous, ConsensusFailed


def test_majority_agreement_succeeds():
    @tool
    def a():
        return "yes"

    @tool
    def b():
        return "yes"

    @tool
    def c():
        return "no"

    workflow = Workflow("Vote")
    workflow.consensus(a, b, c)
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.steps[-1].tool == "consensus"
    assert journal.steps[-1].output == "yes"
    assert journal.groups[0].kind == "consensus"
    assert journal.groups[0].status == "success"


def test_majority_tie_fails():
    @tool
    def a():
        return "A"

    @tool
    def b():
        return "B"

    workflow = Workflow("Tie")
    workflow.consensus(a, b)
    journal = workflow.run()

    assert journal.status == "failed"
    assert journal.steps[-1].status == "failed"
    assert "ConsensusFailed" in journal.steps[-1].error
    assert "no majority" in journal.steps[-1].error


def test_unanimous_strategy_requires_exact_agreement():
    @tool
    def a():
        return "x"

    @tool
    def b():
        return "x"

    workflow = Workflow("Unanimous")
    workflow.consensus(a, b, strategy=unanimous)
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.steps[-1].output == "x"


def test_unanimous_strategy_fails_on_disagreement():
    @tool
    def a():
        return "x"

    @tool
    def b():
        return "y"

    workflow = Workflow("UnanimousDisagree")
    workflow.consensus(a, b, strategy=unanimous)
    journal = workflow.run()

    assert journal.status == "failed"
    assert "not unanimous" in journal.steps[-1].error


def test_voter_failure_fails_block_without_aggregation():
    @tool
    def ok():
        return "yes"

    @tool
    def boom():
        raise RuntimeError("model timeout")

    workflow = Workflow("VoterFails")
    workflow.consensus(ok, boom)
    journal = workflow.run()

    assert journal.status == "failed"
    assert journal.groups[0].status == "failed"
    # no synthetic 'consensus' step -- aggregation was never attempted
    assert "consensus" not in [s.tool for s in journal.steps]
    tool_names = [s.tool for s in journal.steps]
    assert tool_names == ["ok", "boom"]


def test_voters_run_concurrently():
    import threading
    import time

    active = []
    max_concurrent = []
    lock = threading.Lock()

    def make_voter(name, value):
        @tool(name=name)
        def _v():
            with lock:
                active.append(name)
                max_concurrent.append(len(active))
            time.sleep(0.05)
            with lock:
                active.remove(name)
            return value
        return _v

    workflow = Workflow("Concurrent")
    workflow.consensus(make_voter("v1", "yes"), make_voter("v2", "yes"), make_voter("v3", "no"))
    workflow.run()

    assert max(max_concurrent) > 1


def test_consensus_requires_at_least_two_voters():
    @tool
    def solo():
        return "x"

    workflow = Workflow("TooFew")
    try:
        workflow.consensus(solo)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_policy_require_agent_applies_to_consensus_voters():
    @tool
    def a():
        return "x"

    @tool
    def b():
        return "x"

    workflow = Workflow("ProdConsensus", policy=Policy(require_agent=True))
    workflow.consensus(a, b)  # no agent
    try:
        workflow.run()
        assert False, "expected PolicyViolation"
    except PolicyViolation:
        pass


def test_policy_max_parallel_applies_to_consensus_voters():
    @tool
    def a():
        return "x"

    @tool
    def b():
        return "x"

    @tool
    def c():
        return "x"

    workflow = Workflow("CappedConsensus", policy=Policy(max_parallel=2))
    workflow.consensus(a, b, c)
    try:
        workflow.run()
        assert False, "expected PolicyViolation"
    except PolicyViolation as exc:
        assert "consensus block" in str(exc)


def test_custom_strategy_function():
    def always_first(outputs):
        return outputs[0]

    @tool
    def a():
        return "first"

    @tool
    def b():
        return "second"

    workflow = Workflow("CustomStrategy")
    workflow.consensus(a, b, strategy=always_first)
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.steps[-1].output == "first"


if __name__ == "__main__":
    test_majority_agreement_succeeds()
    test_majority_tie_fails()
    test_unanimous_strategy_requires_exact_agreement()
    test_unanimous_strategy_fails_on_disagreement()
    test_voter_failure_fails_block_without_aggregation()
    test_voters_run_concurrently()
    test_consensus_requires_at_least_two_voters()
    test_policy_require_agent_applies_to_consensus_voters()
    test_policy_max_parallel_applies_to_consensus_voters()
    test_custom_strategy_function()
    print("All M6 tests passed.")
