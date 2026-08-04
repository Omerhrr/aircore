"""End-to-end: structured ModelAgents feeding a structured JudgeConsensus,
through a real Workflow/Scheduler run with result reuse. This is the whole
"Structured Artifacts" milestone in one shape:

    ModelAgent(output_schema=Report)   \\
    ModelAgent(output_schema=Report)    >--> parallel --> consensus(JudgeConsensus(output_schema=Report))
    ModelAgent(output_schema=Report)   /

Each agent returns a validated dict (the journal stores it as that step's
output, automatically -- see model_agent.py); the judge reads all three as
JSON, merges them, and returns one validated, typed artifact -- which
*also* lands directly in the journal as the consensus step's output, with
no aircore changes required anywhere in this chain.
"""

import json

from aircore import Workflow
from airpy import MockProvider, ModelAgent, JudgeConsensus

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "risk_level": {"type": "string"},
    },
    "required": ["summary", "risk_level"],
}


def test_structured_agents_feed_a_structured_judge_end_to_end():
    researcher = ModelAgent(
        "researcher", MockProvider(response='{"summary": "no major issues found", "risk_level": "low"}'),
        prompt="audit this system", output_schema=REPORT_SCHEMA,
    )
    reviewer = ModelAgent(
        "reviewer", MockProvider(response='{"summary": "nothing concerning", "risk_level": "low"}'),
        prompt="audit this system", output_schema=REPORT_SCHEMA,
    )
    professor = ModelAgent(
        "professor", MockProvider(response='{"summary": "system looks clean", "risk_level": "low"}'),
        prompt="audit this system", output_schema=REPORT_SCHEMA,
    )

    merged_response = json.dumps({
        "consensus": True,
        "answer": {"summary": "All three auditors found no major issues.", "risk_level": "low"},
        "confidence": 0.95,
        "reasoning": "All three reports independently agreed the system is low risk.",
    })
    judge = JudgeConsensus(MockProvider(response=merged_response), output_schema=REPORT_SCHEMA, confidence=True)

    workflow = Workflow("structured-audit")
    workflow.parallel(researcher, reviewer, professor).consensus(strategy=judge)
    journal = workflow.run()

    assert journal.status == "success"

    voter_steps = {s.tool: s for s in journal.steps if s.tool != "consensus"}
    assert voter_steps["researcher"].output == {"summary": "no major issues found", "risk_level": "low"}
    assert voter_steps["reviewer"].output == {"summary": "nothing concerning", "risk_level": "low"}
    assert voter_steps["professor"].output == {"summary": "system looks clean", "risk_level": "low"}

    consensus_step = next(s for s in journal.steps if s.tool == "consensus")
    assert consensus_step.output == {
        "summary": "All three auditors found no major issues.",
        "risk_level": "low",
    }
    assert consensus_step.metadata["confidence"] == 0.95
    assert consensus_step.metadata["decision"] == "synthesized"
