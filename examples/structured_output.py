"""Structured Artifacts: ModelAgent(output_schema=...) and
JudgeConsensus(output_schema=..., confidence=...) reduce three independent
audit reports into one typed, merged artifact -- not a string a caller has
to parse. Runs entirely offline against MockProvider, no API key needed;
swap in LiteLLMProvider (see examples/parallel_consensus.py) for a real
model.

Run with: python examples/structured_output.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow
from airpy import ModelAgent, MockProvider, JudgeConsensus


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["summary", "risk_level"],
}


if __name__ == "__main__":
    section("A single ModelAgent returning a validated artifact")

    # MockProvider stands in for a real model here -- it just returns this
    # fixed JSON string, but a real provider would be asked (via the
    # schema instruction ModelAgent adds to the prompt, see model_agent.py)
    # to produce exactly this shape.
    agent = ModelAgent(
        "researcher",
        MockProvider(response='{"summary": "no major issues found", "risk_level": "low"}'),
        prompt="Audit this system for security issues.",
        output_schema=REPORT_SCHEMA,
    )
    workflow = Workflow("SingleAgentArtifact")
    workflow.step(agent)
    journal = workflow.run()
    print(f"step output (a dict, not a string): {journal.steps[0].output!r}")

    section("Three agents, reduced into one merged artifact")

    researcher = ModelAgent(
        "researcher",
        MockProvider(response='{"summary": "no major issues found", "risk_level": "low"}'),
        prompt="Audit this system.", output_schema=REPORT_SCHEMA,
    )
    reviewer = ModelAgent(
        "reviewer",
        MockProvider(response='{"summary": "nothing concerning turned up", "risk_level": "low"}'),
        prompt="Audit this system.", output_schema=REPORT_SCHEMA,
    )
    professor = ModelAgent(
        "professor",
        MockProvider(response='{"summary": "system looks clean overall", "risk_level": "low"}'),
        prompt="Audit this system.", output_schema=REPORT_SCHEMA,
    )

    # The judge also returns a validated REPORT_SCHEMA artifact -- merging
    # three structured reports into one, not three strings into one string.
    # confidence=True adds typed confidence/reasoning fields to the
    # journal, parsed the same structured way, not scraped from text.
    merged_response = json.dumps({
        "consensus": True,
        "answer": {
            "summary": "All three auditors independently found no major issues.",
            "risk_level": "low",
        },
        "confidence": 0.95,
        "reasoning": "All three reports agreed on both the summary and the risk level.",
    })
    judge = JudgeConsensus(
        MockProvider(response=merged_response),
        output_schema=REPORT_SCHEMA,
        confidence=True,
    )

    workflow2 = Workflow("MergedArtifact")
    # .parallel(...).consensus(...) reuses the three agents' outputs
    # instead of re-running them -- see workflow.py's ParallelResults.
    workflow2.parallel(researcher, reviewer, professor).consensus(strategy=judge)
    journal2 = workflow2.run()

    print(journal2.pretty())

    consensus_step = next(s for s in journal2.steps if s.tool == "consensus")
    print(f"\nmerged artifact (a dict): {consensus_step.output!r}")
    print(f"typed confidence: {consensus_step.metadata['confidence']}")
