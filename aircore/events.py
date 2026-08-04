"""Event definitions and the event bus.

Every execution primitive emits events instead of calling other subsystems
directly. The journal, observability, CLI, and (later) execution graph all
subscribe to the bus. Nothing couples directly to the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, DefaultDict, List
from collections import defaultdict
import threading


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Event:
    """Base class for all runtime events."""
    timestamp: str = field(default_factory=_now)


@dataclass(frozen=True)
class WorkflowStarted(Event):
    workflow_id: str = ""
    workflow_name: str = ""


@dataclass(frozen=True)
class WorkflowFinished(Event):
    workflow_id: str = ""
    workflow_name: str = ""
    status: str = ""  # "success" | "failed"
    duration_ms: float = 0.0


@dataclass(frozen=True)
class StepStarted(Event):
    workflow_id: str = ""
    step_id: int = 0
    tool_name: str = ""
    group_id: str | None = None
    # True only for a step the scheduler skipped re-executing because
    # checkpoint.py already had a recorded success for it at this run_id --
    # see scheduler.py's _replay_step. False (the default) for every step
    # that actually ran this time, whether or not checkpointing is in use
    # at all.
    replayed: bool = False


@dataclass(frozen=True)
class ToolCalled(Event):
    workflow_id: str = ""
    step_id: int = 0
    tool_name: str = ""
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    group_id: str | None = None


@dataclass(frozen=True)
class ToolSucceeded(Event):
    workflow_id: str = ""
    step_id: int = 0
    tool_name: str = ""
    output: Any = None
    group_id: str | None = None


@dataclass(frozen=True)
class ToolFailed(Event):
    workflow_id: str = ""
    step_id: int = 0
    tool_name: str = ""
    error: str = ""
    group_id: str | None = None


@dataclass(frozen=True)
class StepFinished(Event):
    workflow_id: str = ""
    step_id: int = 0
    tool_name: str = ""
    status: str = ""  # "success" | "failed"
    duration_ms: float = 0.0
    group_id: str | None = None


@dataclass(frozen=True)
class UsageReported(Event):
    """Emitted right after a successful ToolSucceeded, only when
    Executable.usage() returned something other than None. `usage` is
    whatever dict the executable reported -- aircore doesn't interpret the
    keys, it just carries them to the Journal/Metrics."""
    workflow_id: str = ""
    step_id: int = 0
    tool_name: str = ""
    usage: dict = field(default_factory=dict)
    group_id: str | None = None


@dataclass(frozen=True)
class StrategyMetadataReported(Event):
    """Emitted right after a consensus strategy succeeds, only when the
    strategy exposes an optional `describe_last_call() -> dict | None`
    method (duck-typed -- plain callables like majority/unanimous don't
    have one, and are unaffected). `metadata` is whatever dict the
    strategy reported -- aircore doesn't interpret the keys, it just carries
    them to the Journal, same principle as UsageReported for Executable.
    This is how a strategy like airpy's JudgeConsensus can surface things
    like which model judged, whether it selected or synthesized, and its
    confidence/reasoning, without aircore's Scheduler knowing what any of
    that means."""
    workflow_id: str = ""
    step_id: int = 0
    tool_name: str = ""
    metadata: dict = field(default_factory=dict)
    group_id: str | None = None


@dataclass(frozen=True)
class RetryAttempted(Event):
    """A tool call failed on a tool declared idempotent=True with retries>0,
    and the scheduler is about to call it again. `attempt` is 1-indexed
    (the first retry, i.e. the second overall call, is attempt=1)."""
    workflow_id: str = ""
    step_id: int = 0
    tool_name: str = ""
    attempt: int = 0
    error: str = ""
    group_id: str | None = None


@dataclass(frozen=True)
class ApprovalRequested(Event):
    """Emitted right before the scheduler calls an approval_callback for a
    tool named in Policy.approval_for -- see approval.py. Recorded in the
    Journal so "a human/system was asked" is part of the audit trail even
    before the decision is known."""
    workflow_id: str = ""
    step_id: int = 0
    tool_name: str = ""
    group_id: str | None = None


@dataclass(frozen=True)
class ApprovalDecided(Event):
    """Emitted right after an approval_callback returns, whatever it
    returned. `approved=False` is what causes the step to fail with
    ApprovalDenied -- but the decision itself is recorded here regardless,
    so a denied step's journal entry reads as "approval was requested and
    refused," not as an opaque failure."""
    workflow_id: str = ""
    step_id: int = 0
    tool_name: str = ""
    approved: bool = False
    group_id: str | None = None


@dataclass(frozen=True)
class GroupStarted(Event):
    """A `parallel` or `consensus` block began executing. Its child steps
    each emit their own StepStarted/ToolCalled/... events tagged with this
    group_id. `kind` distinguishes the two in the journal/graph."""
    workflow_id: str = ""
    group_id: str = ""
    tool_names: tuple = ()
    kind: str = "parallel"  # "parallel" | "consensus"


@dataclass(frozen=True)
class GroupFinished(Event):
    workflow_id: str = ""
    group_id: str = ""
    status: str = ""  # "success" | "failed"
    duration_ms: float = 0.0


class EventBus:
    """Simple synchronous pub/sub bus.

    Subscribers register for a specific event class (or `Event` itself to
    receive everything). Handlers run synchronously, in subscription order.
    """

    def __init__(self) -> None:
        self._listeners: DefaultDict[type, List[Callable[[Event], None]]] = defaultdict(list)
        # `parallel` blocks emit events from multiple worker threads at once;
        # this serializes dispatch so listeners (e.g. the journal) never see
        # interleaved, half-applied updates.
        self._lock = threading.Lock()

    def subscribe(self, event_type: type, handler: Callable[[Event], None]) -> None:
        self._listeners[event_type].append(handler)

    def emit(self, event: Event) -> None:
        with self._lock:
            # exact-type subscribers
            for handler in self._listeners.get(type(event), []):
                handler(event)
            # subscribers to the base Event type get everything
            if type(event) is not Event:
                for handler in self._listeners.get(Event, []):
                    handler(event)
