"""
Real multi-agent workflow against a live provider.

Demonstrates:

- Multiple ModelAgents
- Parallel execution
- Consensus reduction
- Journal
- Metrics
- Real token/cost accounting

Run:

    export DEEPSEEK_API_KEY=sk-...
    python examples/parallel_consensus.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not os.environ.get("DEEPSEEK_API_KEY"):
    print("Set DEEPSEEK_API_KEY first.")
    sys.exit(1)

from aircore import Workflow, Policy
from airpy import ModelAgent, LiteLLMProvider, JudgeConsensus


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


provider = LiteLLMProvider(
    model="deepseek/deepseek-chat"
)

QUESTION = """
Explain what a Bloom Filter is.
Keep the answer under 40 words.
"""


if __name__ == "__main__":

    section("Three agents running in parallel")

    researcher = ModelAgent(
        "researcher",
        provider,
        prompt=QUESTION,
    )

    reviewer = ModelAgent(
        "reviewer",
        provider,
        prompt=QUESTION,
    )

    professor = ModelAgent(
        "professor",
        provider,
        prompt=QUESTION,
    )

    workflow = Workflow(
        "ParallelConsensus",
        policy=Policy(max_cost=0.10),
    )

    #
    # Run three agents concurrently, then reduce their outputs with an
    # LLM-as-judge instead of exact-string majority.
    #
    # majority() (aircore's default consensus strategy) does exact matching --
    # wrong for free-text model output, since three independent calls to
    # the same prompt almost never come back byte-identical even when they
    # agree in substance. JudgeConsensus asks a model to read all three
    # answers and merge them into one final answer (mode="synthesize", the
    # default) -- not pick one verbatim (mode="select"). It lives in
    # airpy, not aircore -- the runtime only ever sees it as a plain
    # callable.
    #
    # Chaining .consensus() straight off .parallel(...)'s return value
    # (instead of calling workflow.consensus(researcher, reviewer,
    # professor, ...) again) reuses these three agents' outputs instead of
    # re-running them -- 3 model calls + 1 judge call, not 6 + 1. See
    # ParallelResults in aircore/workflow.py.
    #
    # confidence=True asks the judge to also self-report how strongly the
    # three answers agreed and why -- not part of the returned answer, but
    # visible in journal.pretty() below, for post-run debugging/audit.
    #
    workflow.parallel(
        researcher,
        reviewer,
        professor,
    ).consensus(
        strategy=JudgeConsensus(
            provider,
            model="deepseek/deepseek-chat",
            confidence=True,
        ),
    )

    journal = workflow.run()

    section("Execution Journal")

    print(journal.pretty())

    section("Workflow Metrics")

    print(workflow.metrics.summary())

    section("Consensus Result")

    consensus_steps = [
        s for s in journal.steps
        if s.tool == "consensus"
    ]

    if consensus_steps:
        print(consensus_steps[-1].output)