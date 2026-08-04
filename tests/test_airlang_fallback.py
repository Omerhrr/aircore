"""AirLang-M3: `if confidence < X { Reviewer }` folded into a preceding
`consensus`'s confidence-gated fallback (airlang/executor.py's
_resolve_fallback, backed by aircore/consensus.py's `fallback`/
`fallback_below` -- see test_consensus_fallback.py for the aircore-level
primitive itself). Covers the happy path end to end through the real
airpy JudgeConsensus, and every validation _resolve_fallback enforces:
only `<`, only a single bare-ref then-body, only `consensus judge` with
`confidence true` for the `confidence` field specifically.
"""

import json

import pytest

from airpy import MockProvider

from airlang.bindings import Bindings
from airlang.executor import AirLangBindingError, AirLangNotYetSupportedError, build_workflow, execute_ir
from airlang.parser import parse


def test_low_confidence_triggers_the_fallback_end_to_end():
    # Only the judge call sets response_schema (JudgeConsensus's
    # confidence=True switches it into structured mode) -- voter agents'
    # plain single-shot calls never do, so this cleanly tells the two
    # apart without needing to inspect the prompt text.
    judge_response = json.dumps({"consensus": True, "answer": "agreed", "confidence": 0.4, "reasoning": "shaky"})

    def respond(request):
        return judge_response if request.response_schema else "voter answer"

    ir = parse("""
    provider mock
    agent A { provider mock }
    agent B { provider mock }
    agent Reviewer { provider mock }
    workflow W {
        parallel { A B }
        consensus { strategy judge mode select confidence true }
        if confidence < 0.85 { Reviewer }
    }
    """)
    workflow = execute_ir(ir, Bindings(providers={"mock": MockProvider(response=respond)}))

    assert workflow.journal.status == "success"
    assert workflow.journal.steps[-1].tool == "Reviewer"
    assert workflow.journal.steps[-1].output == "voter answer"  # Reviewer's own mock call


def test_high_confidence_does_not_trigger_the_fallback():
    judge_response = json.dumps({"consensus": True, "answer": "agreed", "confidence": 0.95, "reasoning": "solid"})

    def respond(request):
        return judge_response if request.response_schema else "voter answer"

    ir = parse("""
    provider mock
    agent A { provider mock }
    agent B { provider mock }
    agent Reviewer { provider mock }
    workflow W {
        parallel { A B }
        consensus { strategy judge mode select confidence true }
        if confidence < 0.85 { Reviewer }
    }
    """)
    workflow = execute_ir(ir, Bindings(providers={"mock": MockProvider(response=respond)}))

    assert workflow.journal.status == "success"
    assert workflow.journal.steps[-1].tool == "consensus"  # Reviewer never ran
    assert workflow.journal.steps[-1].output == "agreed"


def test_if_confidence_after_majority_is_a_binding_error():
    ir = parse("""
    provider mock
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


def test_if_confidence_after_judge_without_confidence_true_is_a_binding_error():
    ir = parse("""
    provider mock
    agent A { provider mock }
    agent B { provider mock }
    agent Reviewer { provider mock }
    workflow W {
        parallel { A B }
        consensus judge
        if confidence < 0.85 { Reviewer }
    }
    """)
    with pytest.raises(AirLangBindingError, match="never requested confidence"):
        build_workflow(ir)


def test_if_with_non_lt_comparator_is_not_yet_supported():
    ir = parse("""
    provider mock
    agent A { provider mock }
    agent B { provider mock }
    agent Reviewer { provider mock }
    workflow W {
        parallel { A B }
        consensus { strategy judge confidence true }
        if confidence >= 0.85 { Reviewer }
    }
    """)
    with pytest.raises(AirLangNotYetSupportedError, match="only supports '<'"):
        build_workflow(ir)


def test_if_with_multi_statement_then_body_is_not_yet_supported():
    ir = parse("""
    provider mock
    agent A { provider mock }
    agent B { provider mock }
    agent Reviewer { provider mock }
    agent SecondReviewer { provider mock }
    workflow W {
        parallel { A B }
        consensus { strategy judge confidence true }
        if confidence < 0.85 {
            Reviewer
            SecondReviewer
        }
    }
    """)
    with pytest.raises(AirLangNotYetSupportedError, match="exactly one bare"):
        build_workflow(ir)


def test_if_on_a_non_confidence_field_is_allowed_without_judge_validation():
    # The judge+confidence-true validation is specific to the "confidence"
    # field name -- a custom field (mapping to a strategy's own
    # describe_last_call() key) isn't second-guessed the same way, since
    # this executor has no way to know what a hypothetical custom
    # strategy does or doesn't report.
    ir = parse("""
    agent A { provider mock }
    agent B { provider mock }
    agent Reviewer { provider mock }
    workflow W {
        parallel { A B }
        consensus majority
        if agreement_score < 0.5 { Reviewer }
    }
    """)
    workflow = build_workflow(ir)  # no error -- majority never reports it, so it just never triggers
    workflow.run()
    assert workflow.journal.status == "success"
    assert workflow.journal.steps[-1].tool == "consensus"  # fallback never ran
