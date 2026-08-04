"""Confidence-gated consensus fallback (aircore/consensus.py's `fallback`/
`fallback_below`/`fallback_field`) -- the narrow runtime primitive
airlang-spec-v1.md section 5.1 recommended in place of general branching, to
unblock AirLang's `if confidence < 0.85 { HumanReviewer }`.

Uses a small fake strategy (not airpy's JudgeConsensus) so this stays a
pure aircore-level test with no dependency on airpy -- consensus.py's
fallback mechanism only depends on a strategy optionally exposing
describe_last_call(), which is duck-typed, not airpy-specific.
"""

import pytest

from aircore import ConsensusFailed, Tool, Workflow, majority, unanimous
from aircore.consensus import ConsensusGroup


class _FakeStrategyWithConfidence:
    """A minimal describe_last_call()-exposing strategy, standing in for
    airpy's JudgeConsensus without pulling airpy into an aircore-level test."""

    def __init__(self, answer, confidence):
        self._answer = answer
        self._confidence = confidence
        self.calls = 0

    def __call__(self, outputs):
        self.calls += 1
        return self._answer

    def describe_last_call(self):
        return {"confidence": self._confidence}


def test_fallback_and_fallback_below_must_be_given_together():
    with pytest.raises(ValueError, match="together"):
        ConsensusGroup(tools=[Tool(lambda: "a", name="a"), Tool(lambda: "b", name="b")],
                        fallback=Tool(lambda: "x", name="x"))
    with pytest.raises(ValueError, match="together"):
        ConsensusGroup(tools=[Tool(lambda: "a", name="a"), Tool(lambda: "b", name="b")],
                        fallback_below=0.5)


def test_fallback_triggers_when_confidence_is_below_threshold():
    voter_a = Tool(lambda: "a", name="voter_a")
    voter_b = Tool(lambda: "b", name="voter_b")
    fallback = Tool(lambda: "human reviewed", name="human_review")
    strategy = _FakeStrategyWithConfidence("agreed", confidence=0.4)

    workflow = Workflow("W")
    workflow.consensus(voter_a, voter_b, strategy=strategy, fallback=fallback, fallback_below=0.85)
    journal = workflow.run()

    assert journal.status == "success"
    tool_names = [s.tool for s in journal.steps]
    assert tool_names == ["voter_a", "voter_b", "consensus", "human_review"]
    assert journal.steps[-1].output == "human reviewed"
    # the fallback step is nested under the same consensus group
    assert journal.steps[-1].group_id == journal.steps[-2].group_id


def test_fallback_does_not_trigger_when_confidence_is_above_threshold():
    voter_a = Tool(lambda: "a", name="voter_a")
    voter_b = Tool(lambda: "b", name="voter_b")
    fallback = Tool(lambda: "human reviewed", name="human_review")
    strategy = _FakeStrategyWithConfidence("agreed", confidence=0.95)

    workflow = Workflow("W")
    workflow.consensus(voter_a, voter_b, strategy=strategy, fallback=fallback, fallback_below=0.85)
    journal = workflow.run()

    assert journal.status == "success"
    tool_names = [s.tool for s in journal.steps]
    assert tool_names == ["voter_a", "voter_b", "consensus"]  # fallback never ran
    assert journal.steps[-1].output == "agreed"


def test_fallback_never_triggers_for_a_strategy_that_reports_no_metadata():
    voter_a = Tool(lambda: "same", name="voter_a")
    voter_b = Tool(lambda: "same", name="voter_b")
    fallback = Tool(lambda: "should never run", name="human_review")

    workflow = Workflow("W")
    workflow.consensus(voter_a, voter_b, strategy=majority, fallback=fallback, fallback_below=0.85)
    journal = workflow.run()

    assert journal.status == "success"
    assert [s.tool for s in journal.steps] == ["voter_a", "voter_b", "consensus"]


def test_fallback_failure_fails_the_whole_consensus_step():
    voter_a = Tool(lambda: "a", name="voter_a")
    voter_b = Tool(lambda: "b", name="voter_b")

    def blow_up():
        raise RuntimeError("reviewer unavailable")

    fallback = Tool(blow_up, name="human_review")
    strategy = _FakeStrategyWithConfidence("agreed", confidence=0.1)

    workflow = Workflow("W")
    workflow.consensus(voter_a, voter_b, strategy=strategy, fallback=fallback, fallback_below=0.85)
    journal = workflow.run()

    assert journal.status == "failed"
    assert journal.steps[-1].tool == "human_review"
    assert journal.steps[-1].status == "failed"
    assert "reviewer unavailable" in journal.steps[-1].error


def test_fallback_works_in_reuse_mode_after_parallel():
    voter_a = Tool(lambda: "a", name="voter_a")
    voter_b = Tool(lambda: "b", name="voter_b")
    fallback = Tool(lambda: "human reviewed", name="human_review")
    strategy = _FakeStrategyWithConfidence("agreed", confidence=0.2)

    workflow = Workflow("W")
    workflow.parallel(voter_a, voter_b).consensus(strategy=strategy, fallback=fallback, fallback_below=0.5)
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.steps[-1].tool == "human_review"
    assert journal.steps[-1].output == "human reviewed"


def test_fallback_field_can_be_a_custom_metadata_key():
    class _CustomFieldStrategy:
        def __call__(self, outputs):
            return "agreed"

        def describe_last_call(self):
            return {"agreement_score": 0.1}

    voter_a = Tool(lambda: "a", name="voter_a")
    voter_b = Tool(lambda: "b", name="voter_b")
    fallback = Tool(lambda: "escalated", name="escalate")

    workflow = Workflow("W")
    workflow.consensus(voter_a, voter_b, strategy=_CustomFieldStrategy(), fallback=fallback,
                        fallback_below=0.5, fallback_field="agreement_score")
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.steps[-1].output == "escalated"


def test_fallback_plain_function_is_wrapped_like_a_tool():
    voter_a = Tool(lambda: "a", name="voter_a")
    voter_b = Tool(lambda: "b", name="voter_b")

    def human_review():
        return "reviewed by a plain function"

    strategy = _FakeStrategyWithConfidence("agreed", confidence=0.1)

    workflow = Workflow("W")
    workflow.consensus(voter_a, voter_b, strategy=strategy, fallback=human_review, fallback_below=0.85)
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.steps[-1].output == "reviewed by a plain function"
