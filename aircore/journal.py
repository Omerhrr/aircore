"""The journal: a structured, replayable record of a workflow run.

The journal is a pure event-bus subscriber. It never talks to the scheduler
directly -- it only ever reacts to events. This is what lets observability,
a future execution-graph renderer, or a web dashboard subscribe the same way
without the scheduler knowing or caring who's listening.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .events import (
    Event,
    EventBus,
    WorkflowStarted,
    WorkflowFinished,
    StepStarted,
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


@dataclass
class StepRecord:
    id: int
    tool: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: Optional[float] = None
    status: Optional[str] = None
    output: Any = None
    error: Optional[str] = None
    group_id: Optional[str] = None
    retries: int = 0
    retry_errors: List[str] = field(default_factory=list)
    usage: Optional[Dict[str, float]] = None
    # Populated only when the step's strategy (consensus) or executable
    # exposes optional describe_last_call()-style metadata -- see
    # StrategyMetadataReported in events.py. Opaque to the journal itself,
    # rendered generically in pretty()/to_dict().
    metadata: Optional[Dict[str, Any]] = None
    # None: approval was never requested for this step (the common case --
    # most tools aren't in Policy.approval_for). True/False: an
    # approval_callback was actually called and this is what it returned
    # -- see approval.py and events.py's ApprovalRequested/ApprovalDecided.
    approved: Optional[bool] = None
    # True only if this step was skipped and replayed from a
    # checkpoint.py CheckpointStore rather than actually executed this
    # run -- see events.py's StepStarted.replayed and scheduler.py's
    # _replay_step.
    replayed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroupRecord:
    id: str
    tool_names: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: Optional[float] = None
    status: Optional[str] = None
    step_ids: List[int] = field(default_factory=list)
    kind: str = "parallel"  # "parallel" | "consensus"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Journal:
    """Accumulates a complete record of one workflow run by listening to events."""

    workflow_name: str = ""
    workflow_id: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: Optional[float] = None
    status: Optional[str] = None
    steps: List[StepRecord] = field(default_factory=list)
    groups: List[GroupRecord] = field(default_factory=list)

    _steps_by_id: Dict[int, StepRecord] = field(default_factory=dict, repr=False)
    _groups_by_id: Dict[str, GroupRecord] = field(default_factory=dict, repr=False)

    def attach(self, bus: EventBus) -> None:
        """Subscribe this journal to every event on the given bus."""
        bus.subscribe(WorkflowStarted, self._on_workflow_started)
        bus.subscribe(WorkflowFinished, self._on_workflow_finished)
        bus.subscribe(StepStarted, self._on_step_started)
        bus.subscribe(ToolSucceeded, self._on_tool_succeeded)
        bus.subscribe(ToolFailed, self._on_tool_failed)
        bus.subscribe(StepFinished, self._on_step_finished)
        bus.subscribe(RetryAttempted, self._on_retry_attempted)
        bus.subscribe(UsageReported, self._on_usage_reported)
        bus.subscribe(StrategyMetadataReported, self._on_strategy_metadata_reported)
        bus.subscribe(ApprovalDecided, self._on_approval_decided)
        bus.subscribe(GroupStarted, self._on_group_started)
        bus.subscribe(GroupFinished, self._on_group_finished)

    def _on_workflow_started(self, e: WorkflowStarted) -> None:
        self.workflow_name = e.workflow_name
        self.workflow_id = e.workflow_id
        self.started_at = e.timestamp

    def _on_workflow_finished(self, e: WorkflowFinished) -> None:
        self.finished_at = e.timestamp
        self.duration_ms = e.duration_ms
        self.status = e.status

    def _on_step_started(self, e: StepStarted) -> None:
        record = StepRecord(id=e.step_id, tool=e.tool_name, started_at=e.timestamp,
                             group_id=e.group_id, replayed=e.replayed)
        self._steps_by_id[e.step_id] = record
        self.steps.append(record)
        if e.group_id is not None:
            self._groups_by_id[e.group_id].step_ids.append(e.step_id)

    def _on_group_started(self, e: GroupStarted) -> None:
        record = GroupRecord(id=e.group_id, tool_names=list(e.tool_names),
                              started_at=e.timestamp, kind=e.kind)
        self._groups_by_id[e.group_id] = record
        self.groups.append(record)

    def _on_group_finished(self, e: GroupFinished) -> None:
        record = self._groups_by_id[e.group_id]
        record.finished_at = e.timestamp
        record.duration_ms = e.duration_ms
        record.status = e.status

    def _on_tool_succeeded(self, e: ToolSucceeded) -> None:
        self._steps_by_id[e.step_id].output = e.output

    def _on_tool_failed(self, e: ToolFailed) -> None:
        self._steps_by_id[e.step_id].error = e.error

    def _on_step_finished(self, e: StepFinished) -> None:
        record = self._steps_by_id[e.step_id]
        record.finished_at = e.timestamp
        record.duration_ms = e.duration_ms
        record.status = e.status

    def _on_retry_attempted(self, e: RetryAttempted) -> None:
        record = self._steps_by_id[e.step_id]
        record.retries = e.attempt
        record.retry_errors.append(f"attempt {e.attempt}: {e.error}")

    def _on_usage_reported(self, e: UsageReported) -> None:
        self._steps_by_id[e.step_id].usage = dict(e.usage)

    def _on_strategy_metadata_reported(self, e: StrategyMetadataReported) -> None:
        self._steps_by_id[e.step_id].metadata = dict(e.metadata)

    def _on_approval_decided(self, e: ApprovalDecided) -> None:
        self._steps_by_id[e.step_id].approved = e.approved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow": self.workflow_name,
            "workflow_id": self.workflow_id,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "steps": [s.to_dict() for s in self.steps],
            "groups": [g.to_dict() for g in self.groups],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def _render_step(self, step: "StepRecord", indent: str = "") -> List[str]:
        lines = [f"{indent}Step {step.id}", f"{indent}" + "-" * 10]
        lines.append(f"{indent}Tool: {step.tool}")
        lines.append(f"{indent}Started: {step.started_at}")
        lines.append(f"{indent}Finished: {step.finished_at}")
        lines.append(f"{indent}Latency: {step.duration_ms:.2f}ms" if step.duration_ms is not None else f"{indent}Latency: -")
        lines.append(f"{indent}Status: {step.status}")
        if step.replayed:
            lines.append(f"{indent}(replayed from checkpoint)")
        if step.retries:
            lines.append(f"{indent}Retries: {step.retries}")
        if step.usage:
            usage_str = ", ".join(f"{k}={v}" for k, v in step.usage.items())
            lines.append(f"{indent}Usage: {usage_str}")
        if step.approved is not None:
            lines.append(f"{indent}Approval: {'granted' if step.approved else 'denied'}")
        if step.metadata:
            for key, value in step.metadata.items():
                label = key.replace("_", " ").capitalize()
                if isinstance(value, str) and "\n" in value:
                    lines.append(f"{indent}{label}:\n{indent}{value}")
                else:
                    lines.append(f"{indent}{label}: {value}")
        if step.status == "success":
            lines.append(f"{indent}Output:\n{indent}{step.output!r}")
        else:
            lines.append(f"{indent}Error:\n{indent}{step.error}")
        lines.append("")
        return lines

    def pretty(self) -> str:
        """Human-readable rendering of the run. Steps that belong to a
        parallel group are nested and shown ordered by step id (assignment
        order), not by which one happened to finish first."""
        lines = [f"Workflow: {self.workflow_name}", ""]
        rendered_step_ids = set()

        # walk groups in the order they started, but steps within Workflow's
        # declared sequence -- since M1 keeps steps flat on the journal, we
        # render in step-id order and expand a group the first time one of
        # its members is encountered.
        steps_by_group: Dict[str, List["StepRecord"]] = {}
        for step in self.steps:
            if step.group_id is not None:
                steps_by_group.setdefault(step.group_id, []).append(step)

        seen_groups = set()
        for step in sorted(self.steps, key=lambda s: s.id):
            if step.id in rendered_step_ids:
                continue
            if step.group_id is not None:
                if step.group_id in seen_groups:
                    continue
                seen_groups.add(step.group_id)
                group = self._groups_by_id[step.group_id]
                header = "Consensus Group" if group.kind == "consensus" else "Parallel Group"
                lines.append(f"{header} {group.id} [{', '.join(group.tool_names)}]")
                lines.append(f"Status: {group.status}  Latency: {group.duration_ms:.2f}ms" if group.duration_ms is not None else "Status: -")
                lines.append("")
                for member in sorted(steps_by_group[step.group_id], key=lambda s: s.id):
                    lines.extend(self._render_step(member, indent="  "))
                    rendered_step_ids.add(member.id)
            else:
                lines.extend(self._render_step(step))
                rendered_step_ids.add(step.id)

        lines.append(f"Workflow status: {self.status} ({self.duration_ms:.2f}ms)" if self.duration_ms is not None else f"Workflow status: {self.status}")
        return "\n".join(lines)
