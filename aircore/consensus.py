"""Consensus: a workflow step that runs several tools as voters and reduces
their outputs to one agreed value, using a stated strategy.

The runtime -- not the developer, and not a planner/LLM call -- decides
what "consensus" means and when it counts as failed. Same principle as
`parallel`: you declare what you want, the scheduler executes it
deterministically. See architecture-spec-v1.md section 3's note that no
planner is allowed to invent workflow logic; this is the strategy function
being the only thing that decides the outcome, and it's fixed at
declaration time, not chosen dynamically at runtime.

Voters run concurrently, reusing the same thread-pool mechanism as
ParallelGroup. If any voter tool fails, the whole consensus step fails --
there's no principled way to "agree" when one of the inputs never arrived.
If every voter succeeds, `strategy` is applied to their outputs in
step-id order (assignment order, not completion order -- same determinism
rule as parallel groups), and the result becomes a synthetic step's output,
so it shows up in the journal and execution graph exactly like a normal
tool call would.

Confidence-gated fallback (`fallback=`/`fallback_below=`): the narrow,
purpose-built alternative to general runtime branching that
airlang-spec-v1.md section 5.1 recommended over adding real `if`/branching to
`Workflow`. A strategy can optionally expose `describe_last_call() ->
dict` (see scheduler.py's `_apply_consensus_strategy`, already used to
populate the journal's per-consensus metadata -- airpy's JudgeConsensus
implements this when `confidence=True`). If `fallback` is set and that
dict's `fallback_field` entry (default `"confidence"`) is a number below
`fallback_below`, the scheduler runs `fallback` as one more step, nested
under this same consensus group, right after the consensus step itself.
This is deliberately not general branching: there's no expression
language, no arbitrary condition, just one numeric threshold against one
named metadata field a strategy chooses to report -- exactly the
`if confidence < 0.85 { HumanReviewer }` shape AirLang's design called for,
and nothing more general than that. A strategy that never reports the
named field (majority/unanimous never do) means the fallback simply never
triggers, not an error -- see Workflow.consensus()'s docstring for the
validation that catches the more useful failure mode (asking for a
fallback on a strategy that can't possibly report it).
"""

from __future__ import annotations

import uuid
from collections import Counter
from typing import Any, Callable, List, Optional, Sequence

from .executable import Executable

Strategy = Callable[[Sequence[Any]], Any]


class ConsensusFailed(Exception):
    """Raised by a strategy when the voters' outputs don't resolve to a
    single agreed value (e.g. no majority, or not unanimous). Caught by
    the scheduler and surfaced as a failed synthetic step, same as any
    other tool failure."""


def majority(outputs: Sequence[Any]) -> Any:
    """The most common output wins. Raises ConsensusFailed on a tie --
    silently picking one arbitrary winner would hide a real disagreement
    rather than report it."""
    counts = Counter(outputs)
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        raise ConsensusFailed(
            f"no majority: tie between {ranked[0][0]!r} and {ranked[1][0]!r} "
            f"({ranked[0][1]} votes each)"
        )
    return ranked[0][0]


def unanimous(outputs: Sequence[Any]) -> Any:
    """Every voter must produce the exact same output."""
    distinct = set(outputs)
    if len(distinct) > 1:
        raise ConsensusFailed(f"not unanimous: {len(distinct)} distinct outputs: {list(outputs)!r}")
    return outputs[0]


class ConsensusGroup:
    def __init__(self, tools: Optional[List[Executable]] = None, strategy: Strategy = majority,
                 name: str = "consensus", source_group: Optional[Any] = None,
                 fallback: Optional[Executable] = None, fallback_below: Optional[float] = None,
                 fallback_field: str = "confidence") -> None:
        """Two ways to build a ConsensusGroup:

        - `tools=[...]`: the normal case -- voters that the scheduler runs
          itself, exactly like a ParallelGroup, before reducing their
          outputs with `strategy`.
        - `source_group=<a ParallelGroup already on this workflow>`: reuse
          mode. No new voters are run -- the scheduler applies `strategy`
          directly to the outputs the referenced ParallelGroup already
          produced earlier in the same run. This is what
          `workflow.parallel(...).consensus(strategy=...)` (or
          `workflow.consensus(results, strategy=...)`) builds, so agreeing
          on an answer doesn't cost a second round of (possibly expensive)
          executions of the same voters. See scheduler.py's
          `_run_consensus_group` for the reuse path itself.

        `self.tools` is populated either way (from `source_group.tools` in
        reuse mode) so Policy checks like `max_parallel` in
        `Workflow._validate` don't need to know which mode this is."""
        if source_group is not None:
            if tools:
                raise ValueError("pass either tools= or source_group=, not both")
            self.tools = list(source_group.tools)
        else:
            if not tools or len(tools) < 2:
                raise ValueError("consensus() needs at least 2 voters to be meaningful")
            self.tools = tools
        if (fallback is None) != (fallback_below is None):
            raise ValueError(
                "fallback and fallback_below must be given together (a fallback with no "
                "threshold, or a threshold with nothing to fall back to, is a mistake)"
            )
        self.strategy = strategy
        self.name = name
        self.source_group = source_group
        self.fallback = fallback
        self.fallback_below = fallback_below
        self.fallback_field = fallback_field
        self.id = f"group-{uuid.uuid4().hex[:8]}"

    def __repr__(self) -> str:
        names = ", ".join(t.name for t in self.tools)
        strategy_name = getattr(self.strategy, "__name__", repr(self.strategy))
        mode = f" reuses={self.source_group.id}" if self.source_group is not None else ""
        return f"<ConsensusGroup {self.id} [{names}] strategy={strategy_name}{mode}>"
