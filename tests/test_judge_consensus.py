"""JudgeConsensus: LLM-as-judge consensus strategy (airpy-only, not baked
into aircore -- see airpy/judge_consensus.py's docstring). Covers:

- direct unit behavior of the JudgeConsensus callable itself
- that it's a drop-in `strategy=` for aircore's ConsensusGroup, exercised
  through a real Workflow/Scheduler run
- that a judge reporting no consensus fails the step gracefully rather
  than crashing the workflow (this is exactly what scheduler.py's
  `_run_consensus_group` widening from `except ConsensusFailed` to
  `except Exception` was for -- a strategy can now raise any exception,
  not just ConsensusFailed, since it can do real I/O)
- that an arbitrary exception from the provider (e.g. a network/API
  error) is caught the same way, proving the scheduler fix actually
  matters and isn't just defensive dead code
"""

from aircore import Workflow, Tool
from airpy import MockProvider, JudgeConsensus, JudgeConsensusFailed


def test_judge_consensus_returns_the_judges_verdict():
    provider = MockProvider(response="Bloom filters are probabilistic set-membership structures.")
    judge = JudgeConsensus(provider)

    result = judge(["answer one", "answer two", "answer three"])

    assert result == "Bloom filters are probabilistic set-membership structures."


def test_judge_consensus_prompt_includes_every_output():
    seen_prompts = []

    def capture(request):
        seen_prompts.append(request.prompt)
        return "agreed answer"

    provider = MockProvider(response=capture)
    judge = JudgeConsensus(provider)

    judge(["first output", "second output", "third output"])

    assert len(seen_prompts) == 1
    assert "first output" in seen_prompts[0]
    assert "second output" in seen_prompts[0]
    assert "third output" in seen_prompts[0]


def test_judge_consensus_raises_on_no_consensus_marker():
    provider = MockProvider(response="NO CONSENSUS")
    judge = JudgeConsensus(provider)

    try:
        judge(["a", "b", "c"])
        assert False, "expected JudgeConsensusFailed"
    except JudgeConsensusFailed:
        pass


def test_judge_consensus_marker_check_is_case_insensitive_and_strips_whitespace():
    provider = MockProvider(response="  no consensus  \n")
    judge = JudgeConsensus(provider)

    try:
        judge(["a", "b"])
        assert False, "expected JudgeConsensusFailed"
    except JudgeConsensusFailed:
        pass


def test_judge_consensus_works_as_a_consensus_group_strategy_through_a_real_workflow():
    # Three "voters" with different exact wording but the same substance --
    # this is exactly the shape majority()/unanimous() cannot handle, and
    # exactly what JudgeConsensus exists for.
    voter_a = Tool(lambda: "A bloom filter is a space-efficient probabilistic structure.", name="voter_a")
    voter_b = Tool(lambda: "Bloom filters are probabilistic and space efficient.", name="voter_b")
    voter_c = Tool(lambda: "It's a probabilistic, space-saving data structure.", name="voter_c")

    judge_provider = MockProvider(response="A bloom filter is a space-efficient probabilistic structure.")

    workflow = Workflow("judge-test")
    workflow.consensus(voter_a, voter_b, voter_c, strategy=JudgeConsensus(judge_provider))
    journal = workflow.run()

    assert journal.status == "success"
    consensus_steps = [s for s in journal.steps if s.tool == "consensus"]
    assert len(consensus_steps) == 1
    assert consensus_steps[0].output == "A bloom filter is a space-efficient probabilistic structure."


def test_scheduler_fails_the_step_gracefully_when_judge_reports_no_consensus():
    voter_a = Tool(lambda: "answer one", name="voter_a")
    voter_b = Tool(lambda: "completely different answer", name="voter_b")

    judge_provider = MockProvider(response="NO CONSENSUS")

    workflow = Workflow("judge-no-consensus")
    workflow.consensus(voter_a, voter_b, strategy=JudgeConsensus(judge_provider))
    journal = workflow.run()

    # The whole workflow fails the step, not crashes -- a journal exists
    # and reports the failure, same as any other tool failure.
    assert journal.status == "failed"
    consensus_steps = [s for s in journal.steps if s.tool == "consensus"]
    assert len(consensus_steps) == 1
    assert consensus_steps[0].status == "failed"
    assert "JudgeConsensusFailed" in consensus_steps[0].error


def test_scheduler_fails_the_step_gracefully_when_the_provider_itself_raises():
    # This is the case the scheduler.py widening (except ConsensusFailed ->
    # except Exception) was specifically for: a strategy backed by a real
    # provider can raise something that has nothing to do with
    # ConsensusFailed/JudgeConsensusFailed at all (network error, API
    # error, etc.) -- the workflow must still fail gracefully, not crash.
    def blow_up(request):
        raise ConnectionError("simulated network failure calling the judge model")

    voter_a = Tool(lambda: "answer one", name="voter_a")
    voter_b = Tool(lambda: "answer two", name="voter_b")

    judge_provider = MockProvider(response=blow_up)

    workflow = Workflow("judge-provider-error")
    workflow.consensus(voter_a, voter_b, strategy=JudgeConsensus(judge_provider))
    journal = workflow.run()  # must not raise

    assert journal.status == "failed"
    consensus_steps = [s for s in journal.steps if s.tool == "consensus"]
    assert len(consensus_steps) == 1
    assert consensus_steps[0].status == "failed"
    assert "ConnectionError" in consensus_steps[0].error
    assert "simulated network failure" in consensus_steps[0].error


def test_judge_consensus_custom_prompt_template_and_marker():
    provider = MockProvider(response="DISAGREE")
    judge = JudgeConsensus(
        provider,
        prompt_template="Judge these {n} answers:\n{numbered_outputs}",
        no_consensus_marker="DISAGREE",
    )

    try:
        judge(["x", "y"])
        assert False, "expected JudgeConsensusFailed"
    except JudgeConsensusFailed:
        pass


def test_judge_consensus_is_not_importable_from_airun():
    # Structural check that this stays airpy-only, matching the user's
    # explicit constraint: "Do not bake it into aircore."
    import aircore
    assert not hasattr(aircore, "JudgeConsensus")
