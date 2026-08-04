"""The scheduler: executes workflow structure and emits events.

Executes a flat list of steps, where each step is either a single Tool
(sequential) or a ParallelGroup (concurrent, fan-in on completion). The
scheduler owns nothing about journaling or observability -- it only emits
events onto its EventBus. Everything else in the runtime finds out what
happened by subscribing, not by being called directly.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING, List

from .events import (
    EventBus,
    WorkflowStarted,
    WorkflowFinished,
    StepStarted,
    ToolCalled,
    ToolSucceeded,
    ToolFailed,
    StepFinished,
    RetryAttempted,
    UsageReported,
    StrategyMetadataReported,
    ApprovalRequested,
    ApprovalDecided,
    GroupStarted,
    GroupFinished,
)
from .agent import Agent
from .approval import ApprovalCallback, ApprovalDenied, ApprovalRequest
from .checkpoint import CheckpointError, CheckpointRecord, CheckpointStore
from .consensus import ConsensusGroup
from .effects import CapabilityDenied
from .executable import Executable, ToolTimeout
from .parallel import ParallelGroup

if TYPE_CHECKING:
    from .workflow import Workflow, StepEntry


class Scheduler:
    def __init__(self) -> None:
        self.bus = EventBus()
        self._cumulative_cost_usd = 0.0
        # parallel/consensus voters call _run_tool from worker threads, and
        # each can report usage/cost concurrently -- this makes the
        # increment below atomic instead of a lost-update race.
        self._cost_lock = threading.Lock()
        # group.id -> outputs, in group.tools order, for every ParallelGroup
        # that finished successfully this run. This is what lets a later
        # reuse-mode ConsensusGroup (built from `source_group=`, see
        # consensus.py and workflow.py's ParallelResults) reduce over
        # outputs that already exist instead of re-executing the same
        # tools -- see `_run_consensus_group`'s reuse branch below.
        self._group_outputs: dict = {}
        self._approval_callback: "ApprovalCallback | None" = None

    def run(self, workflow: "Workflow", steps: List["StepEntry"],
            max_runtime: "float | None" = None, max_cost: "float | None" = None,
            approval_callback: "ApprovalCallback | None" = None,
            checkpoint_store: "CheckpointStore | None" = None,
            run_id: "str | None" = None) -> None:
        # Stashed on self, not threaded through every method's signature --
        # Scheduler is single-use per Workflow.run() call (see workflow.py),
        # so this is equivalent to a parameter without touching every
        # _run_tool/_run_group/_run_consensus_group call site. Workflow._
        # validate() already guarantees this is not None whenever
        # workflow.policy.approval_for is non-empty, before this method is
        # ever reached -- see workflow.py.
        self._approval_callback = approval_callback
        workflow_start = time.perf_counter()
        self.bus.emit(WorkflowStarted(workflow_id=workflow.id, workflow_name=workflow.name))

        overall_status = "success"
        next_step_id = 1

        # See checkpoint.py's module docstring for the full scope decision
        # -- position-indexed (Workflow._steps order), sequential steps
        # only, JSON-serializable outputs only. An empty dict (never seen
        # this run_id before, or checkpointing not in use at all) behaves
        # identically to a normal run -- nothing here special-cases "first
        # run" vs. "resumed run with nothing to replay yet".
        checkpoint: "dict[int, CheckpointRecord]" = (
            checkpoint_store.load(run_id) if checkpoint_store is not None and run_id is not None else {}
        )

        for entry_index, (step, agent, as_name) in enumerate(steps):
            # Policy.max_runtime, checked before every step. A step already
            # running when the limit is hit is not interrupted -- this only
            # stops the *next* step from starting. See policy.py.
            if max_runtime is not None and (time.perf_counter() - workflow_start) > max_runtime:
                overall_status = "failed"
                break

            # Policy.max_cost, checked the same way -- against cumulative
            # cost_usd reported via UsageReported so far. Same limitation as
            # max_runtime: a step already in flight is not interrupted, and
            # a step whose provider doesn't report cost_usd (e.g.
            # MockProvider) contributes 0 regardless of what it "actually"
            # cost.
            if max_cost is not None and self._cumulative_cost_usd > max_cost:
                overall_status = "failed"
                break

            is_group = isinstance(step, (ConsensusGroup, ParallelGroup))
            if entry_index in checkpoint and not is_group:
                # Groups are never checkpointed (see checkpoint.py, point
                # 2) -- this branch can only be taken for a plain
                # sequential step, so `is_group` above is what keeps a
                # stale/mismatched checkpoint entry from ever being
                # consulted for one.
                cached = checkpoint[entry_index]
                if cached.tool_name != step.name:
                    raise CheckpointError(
                        f"checkpoint for run_id={run_id!r}, step position {entry_index}, was "
                        f"recorded for tool '{cached.tool_name}', but this run's step at that "
                        f"position is '{step.name}' -- either the workflow definition changed "
                        f"between the checkpointed run and this one (steps must stay in the "
                        f"same order to resume), or run_id was reused for a different workflow"
                    )
                next_step_id = self._replay_step(workflow, cached, next_step_id)
                if as_name is not None:
                    # A binding must resolve the same way on a resumed run
                    # as it did the first time -- otherwise a later step
                    # reading it would see nothing (or something stale)
                    # after a resume, silently breaking cross-step data
                    # flow. See workflow.py's "Bindings" section.
                    workflow.bindings[as_name] = cached.output
                continue

            if isinstance(step, ConsensusGroup):
                consensus_output_sink: dict = {}
                status, next_step_id = self._run_consensus_group(workflow, step, next_step_id, agent,
                                                                   output_sink=consensus_output_sink)
                if status == "success" and as_name is not None:
                    # Exactly one synthetic step is ever added per
                    # consensus group (the reduced/agreed value) -- see
                    # workflow.py's Bindings section for why this is the
                    # one group kind that can bind at all.
                    workflow.bindings[as_name] = next(iter(consensus_output_sink.values()))
            elif isinstance(step, ParallelGroup):
                status, next_step_id = self._run_group(workflow, step, next_step_id, agent)
            else:
                this_step_id = next_step_id
                output_sink: dict = {}
                status, next_step_id = self._run_tool(workflow, step, next_step_id, group_id=None,
                                                        agent=agent, output_sink=output_sink)
                if status == "success":
                    if as_name is not None:
                        workflow.bindings[as_name] = output_sink[this_step_id]
                    if checkpoint_store is not None and run_id is not None:
                        # Raises CheckpointError (uncaught, on purpose --
                        # see checkpoint.py, point 3) if this step's
                        # output isn't JSON-serializable. The step itself
                        # already succeeded and is already in the
                        # Journal; what fails here is only "this run can
                        # no longer be durably resumed past this point",
                        # which is exactly what should be loud.
                        checkpoint_store.record(run_id, entry_index, step.name, output_sink[this_step_id])

            if status == "failed":
                overall_status = "failed"
                break  # no retry policy yet -- first failure stops the workflow

        workflow_duration_ms = (time.perf_counter() - workflow_start) * 1000
        self.bus.emit(WorkflowFinished(workflow_id=workflow.id, workflow_name=workflow.name,
                                        status=overall_status, duration_ms=workflow_duration_ms))

    def _replay_step(self, workflow: "Workflow", cached: "CheckpointRecord", step_id: int) -> int:
        """Skips actually calling the step's execute() -- it already
        succeeded in an earlier attempt at this run_id. Emits the same
        StepStarted..StepFinished sequence a real execution would
        (StepStarted tagged replayed=True), using the cached output, so a
        replayed step reads the same in the Journal/trace viewer as a
        fresh one except for that one flag. No ToolCalled/UsageReported --
        nothing was actually called this run, and reporting fresh
        usage/cost for a call that didn't happen would double-count it
        against Policy.max_cost. Returns the next step_id, same +1
        contract every plain sequential step follows."""
        self.bus.emit(StepStarted(workflow_id=workflow.id, step_id=step_id,
                                   tool_name=cached.tool_name, group_id=None, replayed=True))
        self.bus.emit(ToolSucceeded(workflow_id=workflow.id, step_id=step_id,
                                     tool_name=cached.tool_name, output=cached.output, group_id=None))
        self.bus.emit(StepFinished(workflow_id=workflow.id, step_id=step_id,
                                    tool_name=cached.tool_name, status="success",
                                    duration_ms=0.0, group_id=None))
        return step_id + 1

    def _run_tool(self, workflow: "Workflow", tool: Executable, step_id: int,
                   group_id: str | None, agent: "Agent | None" = None,
                   output_sink: "dict | None" = None):
        """Run one tool as a step, emitting the full StepStarted..StepFinished
        sequence. Shared by sequential steps and each member of a parallel
        or consensus group. Returns (status, next_step_id).

        `output_sink`, if given, gets `output_sink[step_id] = output` on
        success -- this is how _run_consensus_group collects voter outputs
        for the strategy function without changing this method's return
        contract (still (status, next_step_id), same as every other caller
        expects).

        If `tool.requires` names capabilities `agent` wasn't granted, the
        tool is never called -- CapabilityDenied is raised and handled
        exactly like any other tool failure, so it shows up in the journal
        as a failed step with that error.

        If `tool.name` is in `workflow.policy.approval_for`, the tool is
        also never called until `self._approval_callback` returns True --
        ApprovalDenied is raised (and handled the same way) otherwise. See
        approval.py."""
        self.bus.emit(StepStarted(workflow_id=workflow.id, step_id=step_id,
                                   tool_name=tool.name, group_id=group_id))
        self.bus.emit(ToolCalled(workflow_id=workflow.id, step_id=step_id,
                                  tool_name=tool.name, group_id=group_id))

        start = time.perf_counter()
        try:
            # No agent attached to a step means capability checks are not
            # enforced for it at all -- enforcement is opt-in per step, so
            # M0/M1-style workflows that never use agents are unaffected.
            # Policy.require_agent (M3) is what makes an agent mandatory.
            # This check runs once, outside the retry loop below -- a
            # capability denial is a permission error, not a transient
            # failure, so it is never retried regardless of tool.retries.
            if agent is not None and tool.requires:
                missing = agent.missing(tool.requires)
                if missing:
                    names = ", ".join(c.name for c in missing)
                    raise CapabilityDenied(
                        f"tool '{tool.name}' requires capability/capabilities "
                        f"[{names}], but {agent.name} was not granted "
                        f"{'it' if len(missing) == 1 else 'them'}"
                    )

            if workflow.policy.approval_for and tool.name in workflow.policy.approval_for:
                self.bus.emit(ApprovalRequested(workflow_id=workflow.id, step_id=step_id,
                                                 tool_name=tool.name, group_id=group_id))
                approved = self._approval_callback(ApprovalRequest(
                    workflow_id=workflow.id, workflow_name=workflow.name,
                    step_id=step_id, tool_name=tool.name, group_id=group_id,
                ))
                self.bus.emit(ApprovalDecided(workflow_id=workflow.id, step_id=step_id,
                                               tool_name=tool.name, approved=approved,
                                               group_id=group_id))
                if not approved:
                    raise ApprovalDenied(
                        f"tool '{tool.name}' requires approval (Policy.approval_for), and "
                        f"the approval_callback denied it"
                    )

            output = self._call_with_retries(workflow, tool, step_id, group_id)
            if output_sink is not None:
                output_sink[step_id] = output
            duration_ms = (time.perf_counter() - start) * 1000
            self.bus.emit(ToolSucceeded(workflow_id=workflow.id, step_id=step_id,
                                         tool_name=tool.name, output=output, group_id=group_id))
            usage = tool.usage()
            if usage:
                self.bus.emit(UsageReported(workflow_id=workflow.id, step_id=step_id,
                                             tool_name=tool.name, usage=usage, group_id=group_id))
                with self._cost_lock:
                    self._cumulative_cost_usd += usage.get("cost_usd", 0.0) or 0.0
            self.bus.emit(StepFinished(workflow_id=workflow.id, step_id=step_id,
                                        tool_name=tool.name, status="success",
                                        duration_ms=duration_ms, group_id=group_id))
            return "success", step_id + 1
        except Exception as exc:  # noqa: BLE001 -- typed runtime errors beyond CapabilityDenied are M3+
            duration_ms = (time.perf_counter() - start) * 1000
            error_message = (f"{type(exc).__name__}: {exc}"
                              if isinstance(exc, (CapabilityDenied, ApprovalDenied, ToolTimeout)) else str(exc))
            self.bus.emit(ToolFailed(workflow_id=workflow.id, step_id=step_id,
                                      tool_name=tool.name, error=error_message, group_id=group_id))
            self.bus.emit(StepFinished(workflow_id=workflow.id, step_id=step_id,
                                        tool_name=tool.name, status="failed",
                                        duration_ms=duration_ms, group_id=group_id))
            return "failed", step_id + 1

    def _call_with_retries(self, workflow: "Workflow", tool: Executable, step_id: int, group_id: str | None):
        """Calls tool.execute() once (through _execute_with_timeout, see
        below), then retries on failure only if the executable is
        declared idempotent (Tool, and anything else implementing
        Executable, refuses retries>0 on a non-idempotent instance at
        construction time, so that invariant is enforced before this ever
        runs). Raises the last exception if every attempt fails --
        including ToolTimeout, so a tool that times out on every attempt
        is retried exactly like any other failure, up to tool.retries."""
        attempt = 0
        while True:
            try:
                return self._execute_with_timeout(tool)
            except Exception as exc:
                if tool.idempotent and attempt < tool.retries:
                    attempt += 1
                    self.bus.emit(RetryAttempted(workflow_id=workflow.id, step_id=step_id,
                                                  tool_name=tool.name, attempt=attempt,
                                                  error=str(exc), group_id=group_id))
                    continue
                raise

    def _execute_with_timeout(self, tool: Executable):
        """Runs tool.execute(), bounding how long the scheduler waits for
        it by `tool.timeout` (seconds) if the Executable sets one --
        checked via getattr, since `timeout` is a Tool-specific attribute,
        not part of the Executable contract itself (see executable.py).
        No timeout attribute, or timeout=None (Tool's default): behaves
        exactly as before this existed -- a plain, unbounded call.

        Real, previously-missing enforcement: Tool.timeout was stored on
        every Tool since M0/M2 but the scheduler never actually read it
        anywhere -- this closes that gap. See ToolTimeout's docstring
        (executable.py) for the one honest limitation this still has: the
        wait stops, the call itself does not (Python cannot force-kill a
        running thread) -- Tool(sandbox=Sandbox(max_runtime=...)) (see
        sandbox.py) is what gives an actual forced kill, via a real OS
        process instead of a thread."""
        timeout = getattr(tool, "timeout", None)
        if timeout is None:
            return tool.execute()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(tool.execute)
            try:
                return future.result(timeout=timeout)
            except FutureTimeoutError as exc:
                raise ToolTimeout(
                    f"tool '{tool.name}' did not finish within timeout={timeout}s"
                ) from exc

    def _run_group(self, workflow: "Workflow", group: ParallelGroup, next_step_id: int,
                    agent: "Agent | None" = None):
        """Run every tool in the group concurrently via a thread pool, then
        fan in: execution doesn't continue to the next step until every
        member of the group has finished. Returns (status, next_step_id)."""
        group_start = time.perf_counter()
        self.bus.emit(GroupStarted(workflow_id=workflow.id, group_id=group.id,
                                    tool_names=tuple(t.name for t in group.tools)))

        # assign step ids up front so ordering in the journal is stable
        # regardless of which thread finishes first
        assignments = [(tool, next_step_id + i) for i, tool in enumerate(group.tools)]
        next_step_id += len(group.tools)
        output_sink: dict = {}

        statuses = []
        with ThreadPoolExecutor(max_workers=len(assignments)) as pool:
            futures = {
                pool.submit(self._run_tool, workflow, tool, step_id, group.id, agent, output_sink): step_id
                for tool, step_id in assignments
            }
            for future in as_completed(futures):
                status, _ = future.result()
                statuses.append(status)

        group_status = "failed" if "failed" in statuses else "success"
        # Cache outputs (in group.tools/assignment order, not completion
        # order) so a later reuse-mode ConsensusGroup can reduce over them
        # without re-running these same tools. Only cached on success --
        # a reuse consensus step referencing a failed parallel group fails
        # with a clear error instead of reducing over partial outputs (see
        # _run_consensus_group).
        if group_status == "success":
            self._group_outputs[group.id] = [output_sink[step_id] for _, step_id in assignments]
        group_duration_ms = (time.perf_counter() - group_start) * 1000
        self.bus.emit(GroupFinished(workflow_id=workflow.id, group_id=group.id,
                                     status=group_status, duration_ms=group_duration_ms))
        return group_status, next_step_id

    def _run_consensus_group(self, workflow: "Workflow", group: ConsensusGroup, next_step_id: int,
                              agent: "Agent | None" = None, output_sink: "dict | None" = None):
        """Run every voter concurrently (same mechanism as _run_group), then
        -- only if every voter succeeded -- apply group.strategy to their
        outputs and record the agreed value as one more synthetic step in
        this same group, so it appears in the journal/graph exactly like a
        normal tool's output. Returns (status, next_step_id).

        `output_sink`, if given, gets the agreed value recorded into it
        (keyed by the synthetic consensus step's id) on success -- same
        convention as `_run_tool`'s `output_sink`, and how the main run()
        loop above resolves a `Workflow.consensus(..., as_="name")`
        binding without this method's return contract changing.

        If `group.source_group` is set (reuse mode -- see consensus.py and
        workflow.py's ParallelResults), no voters are run here at all: the
        outputs an earlier ParallelGroup already produced this run
        (cached in `self._group_outputs`, see `_run_group`) are reused
        directly. This is what makes `workflow.parallel(a, b, c).consensus
        (strategy=...)` cost 3 executions + 1 judge call instead of 6 + 1."""
        group_start = time.perf_counter()

        if group.source_group is not None:
            group_status, next_step_id = self._run_reused_consensus_group(
                workflow, group, next_step_id, group_start, agent, output_sink=output_sink
            )
            return group_status, next_step_id

        self.bus.emit(GroupStarted(workflow_id=workflow.id, group_id=group.id,
                                    tool_names=tuple(t.name for t in group.tools),
                                    kind="consensus"))

        assignments = [(tool, next_step_id + i) for i, tool in enumerate(group.tools)]
        next_step_id += len(group.tools)
        # Deliberately a different dict than the `output_sink` parameter
        # above -- that one collects the *final agreed value* (one entry,
        # written by _apply_consensus_strategy below); this one collects
        # each voter's individual output first, same convention as
        # _run_group's output_sink. Naming them the same would shadow the
        # parameter and silently drop the agreed-value binding.
        voter_output_sink: dict = {}

        statuses = []
        with ThreadPoolExecutor(max_workers=len(assignments)) as pool:
            futures = {
                pool.submit(self._run_tool, workflow, tool, step_id, group.id, agent, voter_output_sink): step_id
                for tool, step_id in assignments
            }
            for future in as_completed(futures):
                status, _ = future.result()
                statuses.append(status)

        if "failed" in statuses:
            # A voter failing means there's nothing principled to agree on
            # -- the strategy is never invoked, and no synthetic step is
            # added (the failed voter's own step already explains why).
            group_status = "failed"
        else:
            # Outputs in step-id (assignment) order, not completion order --
            # same determinism rule as parallel groups, and it matters more
            # here since the strategy result can depend on ordering.
            voter_outputs = [voter_output_sink[step_id] for _, step_id in assignments]
            group_status, next_step_id = self._apply_consensus_strategy(workflow, group, voter_outputs,
                                                                          next_step_id, agent,
                                                                          output_sink=output_sink)

        group_duration_ms = (time.perf_counter() - group_start) * 1000
        self.bus.emit(GroupFinished(workflow_id=workflow.id, group_id=group.id,
                                     status=group_status, duration_ms=group_duration_ms))
        return group_status, next_step_id

    def _run_reused_consensus_group(self, workflow: "Workflow", group: ConsensusGroup, next_step_id: int,
                                     group_start: float, agent: "Agent | None" = None,
                                     output_sink: "dict | None" = None):
        """Reuse-mode path for `_run_consensus_group`: apply `group.strategy`
        to a ParallelGroup's already-cached outputs, running no voters of
        its own. `GroupStarted` still reports the reused tools' names, so
        the journal reads the same way it would if this group had run them
        itself -- the difference is there are no per-voter steps nested
        under it this time, just the one synthetic consensus step (see
        journal.py's `pretty()`, which nests steps under a group purely by
        matching `group_id`)."""
        self.bus.emit(GroupStarted(workflow_id=workflow.id, group_id=group.id,
                                    tool_names=tuple(t.name for t in group.tools),
                                    kind="consensus"))

        outputs = self._group_outputs.get(group.source_group.id)
        if outputs is None:
            # Either the source parallel group failed (nothing was cached)
            # or -- defensively -- this consensus step somehow ran before
            # its source group did. Either way there is nothing to reduce,
            # and this fails exactly like a voter failing would.
            group_status = "failed"
            next_step_id = self._emit_failed_consensus_step(
                workflow, group, next_step_id,
                error=f"source parallel group '{group.source_group.id}' produced no "
                      f"outputs to reuse (it may have failed, or hasn't run yet)",
            )
        else:
            group_status, next_step_id = self._apply_consensus_strategy(workflow, group, outputs,
                                                                          next_step_id, agent,
                                                                          output_sink=output_sink)

        group_duration_ms = (time.perf_counter() - group_start) * 1000
        self.bus.emit(GroupFinished(workflow_id=workflow.id, group_id=group.id,
                                     status=group_status, duration_ms=group_duration_ms))
        return group_status, next_step_id

    def _apply_consensus_strategy(self, workflow: "Workflow", group: ConsensusGroup,
                                   outputs: list, next_step_id: int, agent: "Agent | None" = None,
                                   output_sink: "dict | None" = None):
        """Runs `group.strategy(outputs)` as one synthetic step (StepStarted
        .. StepFinished, same event sequence as any tool), and returns
        (status, next_step_id). Shared by the normal (voters-run-here) and
        reuse (`source_group`) paths through `_run_consensus_group` -- the
        only difference between them is where `outputs` came from."""
        consensus_step_id = next_step_id
        next_step_id += 1
        self.bus.emit(StepStarted(workflow_id=workflow.id, step_id=consensus_step_id,
                                   tool_name=group.name, group_id=group.id))
        self.bus.emit(ToolCalled(workflow_id=workflow.id, step_id=consensus_step_id,
                                  tool_name=group.name, group_id=group.id))
        step_start = time.perf_counter()
        try:
            agreed = group.strategy(outputs)
            if output_sink is not None:
                output_sink[consensus_step_id] = agreed
            step_duration_ms = (time.perf_counter() - step_start) * 1000
            # Optional, duck-typed hook: a strategy object (e.g. airpy's
            # JudgeConsensus) can expose describe_last_call() -> dict to
            # surface details about how it reached `agreed` (which model
            # judged, select vs. synthesize, confidence/reasoning, ...).
            # Plain function strategies (majority/unanimous) don't have
            # this attribute, so this is a no-op for them -- aircore's
            # Scheduler never has to know what any of the dict means.
            describe = getattr(group.strategy, "describe_last_call", None)
            metadata = describe() if callable(describe) else None
            if metadata:
                self.bus.emit(StrategyMetadataReported(workflow_id=workflow.id, step_id=consensus_step_id,
                                                         tool_name=group.name, metadata=dict(metadata),
                                                         group_id=group.id))
            self.bus.emit(ToolSucceeded(workflow_id=workflow.id, step_id=consensus_step_id,
                                         tool_name=group.name, output=agreed, group_id=group.id))
            self.bus.emit(StepFinished(workflow_id=workflow.id, step_id=consensus_step_id,
                                        tool_name=group.name, status="success",
                                        duration_ms=step_duration_ms, group_id=group.id))

            # Confidence-gated fallback (see consensus.py's module
            # docstring): triggers only when the strategy actually
            # reported a *numeric* value for `fallback_field` in its
            # metadata -- a strategy that never reports it (majority/
            # unanimous) means this silently never fires, not an error;
            # ConsensusGroup's own construction-time validation (see
            # Workflow.consensus()) is where the more useful "you asked
            # for a fallback on a strategy that can't report this" case
            # gets caught, not here.
            if group.fallback is not None and metadata is not None:
                trigger_value = metadata.get(group.fallback_field)
                if isinstance(trigger_value, (int, float)) and trigger_value < group.fallback_below:
                    fallback_status, next_step_id = self._run_tool(
                        workflow, group.fallback, next_step_id, group_id=group.id, agent=agent
                    )
                    if fallback_status == "failed":
                        return "failed", next_step_id

            return "success", next_step_id
        except Exception as exc:  # noqa: BLE001 -- a strategy can be arbitrary code (e.g.
            # airpy's JudgeConsensus makes a real provider call), not just the pure,
            # I/O-free functions (majority/unanimous) this was originally written
            # against. Any exception a strategy raises must fail this step gracefully,
            # the same way a Tool or ModelAgent failure does, not crash the whole run.
            step_duration_ms = (time.perf_counter() - step_start) * 1000
            error_message = f"{type(exc).__name__}: {exc}"
            self.bus.emit(ToolFailed(workflow_id=workflow.id, step_id=consensus_step_id,
                                      tool_name=group.name, error=error_message,
                                      group_id=group.id))
            self.bus.emit(StepFinished(workflow_id=workflow.id, step_id=consensus_step_id,
                                        tool_name=group.name, status="failed",
                                        duration_ms=step_duration_ms, group_id=group.id))
            return "failed", next_step_id

    def _emit_failed_consensus_step(self, workflow: "Workflow", group: ConsensusGroup,
                                     next_step_id: int, error: str) -> int:
        """Records the synthetic consensus step as failed without ever
        calling `group.strategy` -- used when there's nothing valid to
        reduce (reuse mode with no cached outputs). Returns next_step_id."""
        consensus_step_id = next_step_id
        next_step_id += 1
        now_ms = 0.0
        self.bus.emit(StepStarted(workflow_id=workflow.id, step_id=consensus_step_id,
                                   tool_name=group.name, group_id=group.id))
        self.bus.emit(ToolCalled(workflow_id=workflow.id, step_id=consensus_step_id,
                                  tool_name=group.name, group_id=group.id))
        self.bus.emit(ToolFailed(workflow_id=workflow.id, step_id=consensus_step_id,
                                  tool_name=group.name, error=error, group_id=group.id))
        self.bus.emit(StepFinished(workflow_id=workflow.id, step_id=consensus_step_id,
                                    tool_name=group.name, status="failed",
                                    duration_ms=now_ms, group_id=group.id))
        return next_step_id
