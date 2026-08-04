"""JudgeConsensus modes (select/synthesize), the generic strategy-metadata
journal hook (StrategyMetadataReported), and confidence/reasoning capture.

Covers the follow-ups from examples/parallel_consensus.py's real DeepSeek
run: the default prompt was acting as a selector ("Answer 2: ...") instead
of a synthesizer. mode="synthesize" is now the default; mode="select" is
opt-in for when a caller genuinely wants one candidate chosen verbatim
(e.g. picking the best of several patches, where merging wouldn't even be
valid).
"""

import json

import pytest

from aircore import Workflow, Tool
from airpy import MockProvider, JudgeConsensus, JudgeConsensusFailed, StructuredOutputError


def test_default_mode_is_synthesize():
    judge = JudgeConsensus(MockProvider())
    assert judge.mode == "synthesize"


def test_synthesize_prompt_instructs_the_judge_not_to_reference_answer_numbers():
    seen_prompts = []

    def capture(request):
        seen_prompts.append(request.prompt)
        return "merged answer"

    judge = JudgeConsensus(MockProvider(response=capture), mode="synthesize")
    judge(["a", "b", "c"])

    prompt = seen_prompts[0]
    assert "do not mention" in prompt.lower()
    assert '"answer 1"' in prompt.lower() or "answer 1" in prompt.lower()


def test_select_mode_asks_for_verbatim_choice():
    seen_prompts = []

    def capture(request):
        seen_prompts.append(request.prompt)
        return "the chosen one"

    judge = JudgeConsensus(MockProvider(response=capture), mode="select")
    result = judge(["the chosen one", "something else"])

    assert result == "the chosen one"
    assert "verbatim" in seen_prompts[0].lower()


def test_invalid_mode_rejected_at_construction():
    try:
        JudgeConsensus(MockProvider(), mode="vote")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_custom_prompt_template_opts_out_of_mode_prompts():
    judge = JudgeConsensus(
        MockProvider(response="ok"),
        mode="select",
        prompt_template="Custom: {n} answers\n{numbered_outputs}",
    )
    assert judge.prompt_template == "Custom: {n} answers\n{numbered_outputs}"


def test_describe_last_call_is_none_before_any_call():
    judge = JudgeConsensus(MockProvider())
    assert judge.describe_last_call() is None


def test_describe_last_call_reports_strategy_mode_model_and_candidate_count():
    judge = JudgeConsensus(MockProvider(response="merged"), model="deepseek/deepseek-chat", mode="synthesize")
    judge(["a", "b", "c"])

    metadata = judge.describe_last_call()
    assert metadata["strategy"] == "JudgeConsensus"
    assert metadata["mode"] == "synthesize"
    assert metadata["judge_model"] == "deepseek/deepseek-chat"
    assert metadata["candidate_count"] == 3
    assert metadata["decision"] == "synthesized"


def test_describe_last_call_decision_is_selected_in_select_mode():
    judge = JudgeConsensus(MockProvider(response="a"), mode="select")
    judge(["a", "b"])
    assert judge.describe_last_call()["decision"] == "selected"


def test_describe_last_call_is_reset_to_none_after_a_failed_call():
    judge = JudgeConsensus(MockProvider(response="NO CONSENSUS"))
    try:
        judge(["a", "b"])
    except JudgeConsensusFailed:
        pass
    assert judge.describe_last_call() is None


def test_scheduler_records_strategy_metadata_on_the_consensus_step():
    voter_a = Tool(lambda: "answer a", name="voter_a")
    voter_b = Tool(lambda: "answer b", name="voter_b")

    judge = JudgeConsensus(MockProvider(response="merged answer"), model="deepseek/deepseek-chat")

    workflow = Workflow("metadata-test")
    workflow.consensus(voter_a, voter_b, strategy=judge)
    journal = workflow.run()

    consensus_steps = [s for s in journal.steps if s.tool == "consensus"]
    assert len(consensus_steps) == 1
    metadata = consensus_steps[0].metadata
    assert metadata is not None
    assert metadata["strategy"] == "JudgeConsensus"
    assert metadata["mode"] == "synthesize"
    assert metadata["judge_model"] == "deepseek/deepseek-chat"
    assert metadata["candidate_count"] == 2
    assert metadata["decision"] == "synthesized"


def test_journal_pretty_renders_strategy_metadata():
    voter_a = Tool(lambda: "answer a", name="voter_a")
    voter_b = Tool(lambda: "answer b", name="voter_b")

    judge = JudgeConsensus(MockProvider(response="merged answer"), model="deepseek/deepseek-chat")

    workflow = Workflow("metadata-pretty")
    workflow.consensus(voter_a, voter_b, strategy=judge)
    journal = workflow.run()

    rendered = journal.pretty()
    assert "Judge model: deepseek/deepseek-chat" in rendered
    assert "Decision: synthesized" in rendered


def test_plain_function_strategies_produce_no_metadata():
    # majority/unanimous don't have describe_last_call -- the hook must be
    # a true no-op for them, not an error.
    from aircore import majority

    a = Tool(lambda: "yes", name="a")
    b = Tool(lambda: "yes", name="b")

    workflow = Workflow("no-metadata")
    workflow.consensus(a, b, strategy=majority)
    journal = workflow.run()

    consensus_steps = [s for s in journal.steps if s.tool == "consensus"]
    assert consensus_steps[0].metadata is None


def test_confidence_true_captures_confidence_and_reasoning_in_metadata_not_in_the_return_value():
    # confidence=True switches JudgeConsensus into structured (JSON) mode --
    # see judge_consensus.py's module docstring on why: a real float and a
    # real string, parsed/validated the same way ModelAgent(output_schema=
    # ...) is, not scraped out of ad hoc text markers.
    response_json = json.dumps({
        "consensus": True,
        "answer": "A bloom filter is a probabilistic set-membership structure.",
        "confidence": 0.92,
        "reasoning": "All three answers agreed on the key properties.",
    })
    judge = JudgeConsensus(MockProvider(response=response_json), confidence=True)

    result = judge(["a", "b", "c"])

    # The return value (what downstream steps/consumers see) is just the
    # answer -- confidence/reasoning are journal-only detail, per the
    # explicit design goal of not forcing every caller to unpack a
    # structured result just to get the merged text.
    assert result == "A bloom filter is a probabilistic set-membership structure."

    metadata = judge.describe_last_call()
    assert metadata["confidence"] == 0.92
    assert metadata["reasoning"] == "All three answers agreed on the key properties."


def test_confidence_prompt_asks_for_the_structured_format():
    seen_prompts = []

    def capture(request):
        seen_prompts.append(request.prompt)
        return json.dumps({"consensus": True, "answer": "fine", "confidence": 0.5, "reasoning": "ok"})

    judge = JudgeConsensus(MockProvider(response=capture), confidence=True)
    judge(["a", "b"])

    assert '"confidence"' in seen_prompts[0]
    assert '"reasoning"' in seen_prompts[0]
    assert "JSON" in seen_prompts[0]


def test_confidence_true_raises_structured_output_error_when_judge_ignores_the_format():
    # confidence=True is a real contract now, not a best-effort text scrape
    # -- a judge that doesn't return valid JSON fails this step (gracefully,
    # same as any other strategy exception -- see scheduler.py), rather
    # than silently downgrading to an untyped answer.
    judge = JudgeConsensus(MockProvider(response="just a plain merged answer, not JSON"), confidence=True)

    with pytest.raises(StructuredOutputError):
        judge(["a", "b"])


def test_confidence_true_no_consensus_still_raises():
    no_consensus_response = json.dumps({
        "consensus": False, "answer": None, "confidence": 0.1, "reasoning": "they disagreed",
    })
    judge = JudgeConsensus(MockProvider(response=no_consensus_response), confidence=True)

    with pytest.raises(JudgeConsensusFailed):
        judge(["a", "b"])


def test_output_schema_merges_structured_voter_outputs_into_a_typed_artifact():
    widget_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["name", "count"],
    }
    merged = json.dumps({"consensus": True, "answer": {"name": "widget", "count": 5}})
    judge = JudgeConsensus(MockProvider(response=merged), output_schema=widget_schema)

    result = judge([{"name": "widget", "count": 4}, {"name": "widget", "count": 6}])

    assert result == {"name": "widget", "count": 5}


def test_output_schema_prompt_includes_json_representation_of_dict_voter_outputs():
    seen_prompts = []

    def capture(request):
        seen_prompts.append(request.prompt)
        return json.dumps({"consensus": True, "answer": {"name": "widget", "count": 5}})

    widget_schema = {"type": "object", "properties": {"name": {"type": "string"}, "count": {"type": "integer"}}}
    judge = JudgeConsensus(MockProvider(response=capture), output_schema=widget_schema)

    judge([{"name": "widget", "count": 4}, {"name": "widget", "count": 6}])

    assert '"count": 4' in seen_prompts[0]
    assert '"count": 6' in seen_prompts[0]
