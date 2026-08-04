"""Workflow: an ordered composition of steps.

A step is either a single Tool (sequential) or a ParallelGroup (a `parallel`
block). `delegate` (sub-agent delegation) is not part of M1 -- it needs
Capabilities (M2) to mean anything safe, so it's deferred.

Bindings (self.bindings, Workflow.step(..., as_=...)): closes the cross-
step data flow gap documented in cross-step-data-flow.md -- a sequential
step's output can now be recorded under a name, and read back by a later
step's prompt (see airpy's ModelAgent(prompt=PromptTemplate(...),
prompt_bindings=workflow.bindings)). `self.bindings` is a plain, mutable
dict, created once per Workflow and cleared at the start of every run()
(same lifecycle as `memory.temporary`) -- the Scheduler writes into it in
place as each named step succeeds (see scheduler.py's main loop), and
anything holding a reference to the same dict object sees updates as they
happen, with no new calling convention or Executable interface change
required. Scoped the same way checkpoint.py scoped itself: sequential
steps only, plus one deliberate extension -- `Workflow.consensus(...,
as_="name")` (and `ParallelResults.consensus(as_=...)`) also binds, since
a consensus group reduces its voters to exactly one agreed value, the
same shape as a single step's output. A `parallel` block (no consensus)
still never binds -- it has no single value to bind, only N outputs.
Group binding is NOT checkpoint-replayable the way a plain step's is
(checkpoint.py skips groups entirely and always re-runs them in full);
each fresh run rebinds it the normal way, so this only matters for a
resumed run reading a stale binding value, which cannot happen since the
whole group reruns."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from .agent import Agent
from .approval import ApprovalCallback
from .checkpoint import CheckpointStore
from .consensus import ConsensusGroup, Strategy, majority
from .executable import Executable
from .journal import Journal
from .memory import Memory
from .observability import Metrics
from .parallel import ParallelGroup
from .policy import Policy, PolicyViolation
from .scheduler import Scheduler

Step = Union[Executable, ParallelGroup, ConsensusGroup]
# Third element: the binding name this step's output should be recorded
# under in Workflow.bindings (see this module's docstring), or None for
# every step that doesn't bind anything -- which is every step built
# before this existed, so this is purely additive.
StepEntry = Tuple[Step, Optional[Agent], Optional[str]]
_GROUP_TYPES = (ParallelGroup, ConsensusGroup)


def _as_executable(x: Any) -> Executable:
    """Accepts anything that's already an Executable (a Tool, or a
    provider-backed executable from a layer like airpy) unchanged, and
    only wraps plain functions into a Tool. This is the seam that lets
    Tool and (e.g.) airpy's ModelAgent be interchangeable workflow steps
    without workflow.py needing to know airpy exists."""
    if isinstance(x, Executable):
        return x
    from .tools import tool as tool_decorator
    return tool_decorator(x)


class ParallelResults:
    """Handle returned by Workflow.parallel(), naming the ParallelGroup
    step that was just appended so a later consensus() can be told to
    reuse its outputs instead of re-running the same tools.

    Without this, `workflow.parallel(a, b, c)` followed by
    `workflow.consensus(a, b, c, strategy=...)` runs a, b, and c twice --
    harmless for cheap tools, real (in $ and latency) for ModelAgents
    backed by a paid provider. Passing the handle instead --
    `workflow.consensus(results, strategy=...)`, or the equivalent
    `workflow.parallel(a, b, c).consensus(strategy=...)` -- tells the
    scheduler there is nothing new to execute: just reduce the outputs
    the parallel step already produced. See consensus.py's
    `source_group` and scheduler.py's `_run_consensus_group` reuse path."""

    def __init__(self, workflow: "Workflow", group: ParallelGroup) -> None:
        self._workflow = workflow
        self.group = group

    def consensus(self, strategy: Strategy = majority, agent: Optional[Agent] = None,
                   fallback: Any = None, fallback_below: Optional[float] = None,
                   fallback_field: str = "confidence", as_: Optional[str] = None) -> "Workflow":
        """Sugar for `workflow.consensus(results, strategy=..., agent=...)`
        -- lets the whole thing read as one chain:
        `workflow.parallel(a, b, c).consensus(strategy=JudgeConsensus(p))`.
        `fallback`/`fallback_below`/`fallback_field`/`as_` are forwarded
        unchanged -- see Workflow.consensus()'s docstring and
        consensus.py's module docstring for what they do."""
        return self._workflow.consensus(self, strategy=strategy, agent=agent, fallback=fallback,
                                          fallback_below=fallback_below, fallback_field=fallback_field,
                                          as_=as_)

    def __repr__(self) -> str:
        return f"<ParallelResults of {self.group!r}>"


class Workflow:
    def __init__(self, name: str, policy: Optional[Policy] = None,
                 memory: Optional[Memory] = None) -> None:
        self.name = name
        self.id = str(uuid.uuid4())
        self.policy = policy or Policy()  # default: permissive, backward compatible
        # No memory by default -- workflows that never mention Memory are
        # completely unaffected, same backward-compatibility rule as every
        # other milestone.
        self.memory = memory
        self._steps: List[StepEntry] = []
        self.journal: Optional[Journal] = None  # populated by run()
        self.metrics: Optional[Metrics] = None  # populated by run()
        # See this module's docstring's "Bindings" section. Created once
        # here (not per-run) so a caller can wire the same dict object
        # into a ModelAgent's prompt_bindings= before the workflow has
        # ever been run -- cleared at the start of every run() instead,
        # same lifecycle as memory.temporary.
        self.bindings: Dict[str, Any] = {}

    def step(self, tool_or_fn: Any, agent: Optional[Agent] = None,
              as_: Optional[str] = None) -> "Workflow":
        """Append a sequential step. Accepts a Tool, or a plain function
        (auto-wrapped). `agent`, if given, is the acting agent for this
        call -- a tool that declares a required capability will be rejected
        if this agent wasn't granted it. No agent means unrestricted, so
        M0/M1-style workflows with no agents keep working unchanged.

        `as_`, if given, records this step's output in `self.bindings`
        under that name once it succeeds -- see this module's docstring's
        "Bindings" section. A step that fails records nothing (there's no
        output to bind); a name reused by two different steps is
        overwritten by whichever one runs (and therefore binds) last, not
        rejected as an error -- callers who need every name unique are
        expected to just pick unique names, the same way Tool names
        aren't enforced unique either."""
        self._steps.append((_as_executable(tool_or_fn), agent, as_))
        return self

    def parallel(self, *tools_or_fns: Any, agent: Optional[Agent] = None) -> ParallelResults:
        """Append a `parallel` block: all given tools run concurrently, and
        execution rejoins the sequential flow only once every one of them
        has finished (fan-in). `agent` applies to every tool in the block.

        Returns a `ParallelResults` handle (not `self`) naming this block,
        so it can be passed to `.consensus()` to reduce these same outputs
        instead of running the tools again -- see ParallelResults' and
        ConsensusGroup's docstrings. Nothing about `parallel()` executes
        eagerly: like every other step, this only records the block: the
        handle is just a reference to it, resolved at `.run()` time."""
        if len(tools_or_fns) < 2:
            raise ValueError("parallel() needs at least 2 tools to be meaningful")
        group = ParallelGroup([_as_executable(t) for t in tools_or_fns])
        self._steps.append((group, agent, None))
        return ParallelResults(self, group)

    def consensus(self, *tools_or_fns_or_results: Any, strategy: Strategy = majority,
                   agent: Optional[Agent] = None, fallback: Any = None,
                   fallback_below: Optional[float] = None, fallback_field: str = "confidence",
                   as_: Optional[str] = None) -> "Workflow":
        """Append a `consensus` block. Two forms:

        - `workflow.consensus(a, b, c, strategy=...)`: voters run
          concurrently (same mechanism as `parallel`), and once every voter
          has succeeded the scheduler reduces their outputs to one agreed
          value via `strategy` (majority by default -- see consensus.py).
          Any voter failing, or the strategy raising, fails the whole
          block.
        - `workflow.consensus(results, strategy=...)`, where `results` is
          the `ParallelResults` returned by an earlier `.parallel()` call
          on this same workflow: reuse mode. No voters are re-run --
          `strategy` is applied directly to the outputs that parallel
          block already produced. See ParallelResults' docstring for why
          this exists.

        `fallback=`/`fallback_below=`/`fallback_field=` (all optional, and
        only meaningful together with `fallback`/`fallback_below`) wire
        into ConsensusGroup's confidence-gated fallback -- see
        consensus.py's module docstring. `fallback` is wrapped via
        `_as_executable` the same way voters are, so a plain function
        works here too, not just a Tool/ModelAgent. Validated fail-fast
        at ConsensusGroup construction (below), not deferred to run time,
        same rule as every other constructor-time check in this
        project.

        `as_`, if given, records the group's one synthetic agreed-value
        step in `self.bindings` under that name once it succeeds -- the
        consensus counterpart to Step.step(..., as_=...) (see this
        module's docstring's "Bindings" section). Unlike a plain step's
        binding, this is the *only* way to bind a consensus/parallel
        group's output: individual voters/parallel members never get
        their own binding, by design (see the module docstring) -- there
        is exactly one value to name here, the strategy's reduced result."""
        wrapped_fallback = _as_executable(fallback) if fallback is not None else None

        if len(tools_or_fns_or_results) == 1 and isinstance(tools_or_fns_or_results[0], ParallelResults):
            results = tools_or_fns_or_results[0]
            if results._workflow is not self:
                raise ValueError(
                    "consensus(results, ...) was given a ParallelResults from a "
                    "different Workflow -- results can only be reused on the "
                    "workflow that produced them"
                )
            self._steps.append((ConsensusGroup(source_group=results.group, strategy=strategy,
                                                fallback=wrapped_fallback, fallback_below=fallback_below,
                                                fallback_field=fallback_field), agent, as_))
            return self

        if len(tools_or_fns_or_results) < 2:
            raise ValueError("consensus() needs at least 2 voters to be meaningful")
        tools = [_as_executable(t) for t in tools_or_fns_or_results]
        self._steps.append((ConsensusGroup(tools, strategy=strategy, fallback=wrapped_fallback,
                                            fallback_below=fallback_below, fallback_field=fallback_field),
                             agent, as_))
        return self

    def _validate(self, approval_callback: "ApprovalCallback | None",
                  checkpoint_store: "CheckpointStore | None", run_id: "str | None") -> None:
        """Pre-flight policy checks. Raises PolicyViolation and never
        schedules anything if the workflow isn't valid to run at all."""
        if (checkpoint_store is None) != (run_id is None):
            raise ValueError(
                "checkpoint_store and run_id must be given together, or neither -- a "
                "CheckpointStore with no run_id has nothing to key its records under, "
                "and a run_id with no CheckpointStore has nowhere to persist them "
                "(see aircore.checkpoint's module docstring)"
            )

        if self.policy.approval_for and approval_callback is None:
            raise PolicyViolation(
                f"Policy.approval_for={sorted(self.policy.approval_for)!r} requires an "
                f"approval_callback to be passed to run() -- see aircore.approval for a "
                f"ready-made cli_approval_callback, or write your own "
                f"Callable[[ApprovalRequest], bool]"
            )

        for step, agent, _as_name in self._steps:
            tools = step.tools if isinstance(step, _GROUP_TYPES) else [step]

            if self.policy.require_agent and agent is None:
                names = ", ".join(t.name for t in tools)
                raise PolicyViolation(
                    f"step '{names}' requires an Agent because Policy(require_agent=True)"
                )

            if self.policy.max_parallel is not None and isinstance(step, _GROUP_TYPES):
                if len(step.tools) > self.policy.max_parallel:
                    kind = "consensus block" if isinstance(step, ConsensusGroup) else "parallel block"
                    raise PolicyViolation(
                        f"{kind} [{', '.join(t.name for t in step.tools)}] has "
                        f"{len(step.tools)} tools, exceeding Policy(max_parallel="
                        f"{self.policy.max_parallel})"
                    )

    def run(self, approval_callback: "ApprovalCallback | None" = None,
            checkpoint_store: "CheckpointStore | None" = None,
            run_id: "str | None" = None) -> Journal:
        """Runs the workflow and returns the Journal (unchanged return type
        for backward compatibility). Metrics are collected automatically on
        every run, with no extra wiring required -- available afterward as
        `workflow.metrics`, same way the journal is available as
        `workflow.journal`.

        `approval_callback`, if this workflow's Policy.approval_for names
        any tools, is required (see _validate() below) -- a
        Callable[[ApprovalRequest], bool] the scheduler calls, blocking,
        before running any gated tool. See approval.py. Workflows with no
        approval_for are completely unaffected whether or not this is
        passed.

        `checkpoint_store`/`run_id` (must be given together, or neither):
        enables durable resume-after-crash for this run -- see
        checkpoint.py's module docstring for exactly what that covers
        (sequential steps only, JSON-serializable outputs only,
        position-indexed against this same Workflow's declared step
        order). Rerunning the identical script with the same
        checkpoint_store/run_id after a crash skips every already-
        succeeded sequential step and continues from the first one that
        didn't finish."""
        self._validate(approval_callback, checkpoint_store, run_id)
        # Cleared at the start of every run, not just constructed once --
        # a Workflow object run twice (e.g. in a loop, or a retried
        # process) never lets a prior run's bound values silently leak
        # into a new one. See this module's docstring's "Bindings"
        # section for why this is a plain dict, not something recreated.
        self.bindings.clear()
        scheduler = Scheduler()
        journal = Journal()
        metrics = Metrics()
        journal.attach(scheduler.bus)
        metrics.attach(scheduler.bus)
        try:
            scheduler.run(self, self._steps, max_runtime=self.policy.max_runtime,
                          max_cost=self.policy.max_cost, approval_callback=approval_callback,
                          checkpoint_store=checkpoint_store, run_id=run_id)
        finally:
            # Runtime-enforced guarantee for the `temporary` scope: it never
            # survives past the run it was written in, whether that run
            # succeeded or failed.
            if self.memory is not None:
                self.memory.temporary.clear()
        self.journal = journal
        self.metrics = metrics
        return journal
