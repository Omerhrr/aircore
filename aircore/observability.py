"""Observability: aggregate metrics, built the same way the Journal is.

Metrics is an independent EventBus subscriber, not something derived from
the Journal after the fact -- keeping the two separate means neither has to
know the other exists, matching the "nobody couples directly to the
scheduler" rule from the M0 design.

What this deliberately does NOT track yet, and why:
- tokens_total / cost_total_usd: there is no model provider integration in
  the runtime yet (tools are plain Python functions). Adding these fields
  now would mean they're always zero/None, which is worse than not having
  them -- a metric that's always empty looks like a bug, not a scope
  decision. Add these once a real model call exists to produce them.
- retries: closed. Tools can declare `idempotent=True, retries=N`; the
  scheduler retries on failure up to N times and emits RetryAttempted for
  each one. Metrics.retries_total now actually counts these.
- tokens/cost: closed, generically. Any Executable can implement
  usage() -> Optional[dict] (see executable.py); the scheduler emits
  UsageReported when it does, and Metrics.usage_totals sums whatever
  numeric keys show up across the whole run. aircore still doesn't know
  these keys mean "tokens" or "dollars" -- that convention lives in
  airpy's ModelAgent, not here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

from .events import (
    EventBus,
    WorkflowStarted,
    WorkflowFinished,
    GroupStarted,
    StepFinished,
    RetryAttempted,
    UsageReported,
)


@dataclass
class ToolStats:
    calls: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    total_duration_ms: float = 0.0
    usage_totals: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    @property
    def avg_duration_ms(self) -> Optional[float]:
        return self.total_duration_ms / self.calls if self.calls else None


@dataclass
class Metrics:
    workflow_name: str = ""
    status: Optional[str] = None
    total_duration_ms: Optional[float] = None

    steps_total: int = 0
    steps_succeeded: int = 0
    steps_failed: int = 0
    groups_total: int = 0
    retries_total: int = 0
    usage_totals: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    by_tool: Dict[str, ToolStats] = field(default_factory=lambda: defaultdict(ToolStats))

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(WorkflowStarted, self._on_workflow_started)
        bus.subscribe(WorkflowFinished, self._on_workflow_finished)
        bus.subscribe(GroupStarted, self._on_group_started)
        bus.subscribe(StepFinished, self._on_step_finished)
        bus.subscribe(RetryAttempted, self._on_retry_attempted)
        bus.subscribe(UsageReported, self._on_usage_reported)

    def _on_workflow_started(self, e: WorkflowStarted) -> None:
        self.workflow_name = e.workflow_name

    def _on_workflow_finished(self, e: WorkflowFinished) -> None:
        self.status = e.status
        self.total_duration_ms = e.duration_ms

    def _on_group_started(self, e: GroupStarted) -> None:
        self.groups_total += 1

    def _on_step_finished(self, e: StepFinished) -> None:
        self.steps_total += 1
        stats = self.by_tool[e.tool_name]
        stats.calls += 1
        stats.total_duration_ms += e.duration_ms
        if e.status == "success":
            self.steps_succeeded += 1
            stats.successes += 1
        else:
            self.steps_failed += 1
            stats.failures += 1

    def _on_retry_attempted(self, e: RetryAttempted) -> None:
        self.retries_total += 1
        self.by_tool[e.tool_name].retries += 1

    def _on_usage_reported(self, e: UsageReported) -> None:
        stats = self.by_tool[e.tool_name]
        for key, value in e.usage.items():
            self.usage_totals[key] += value
            stats.usage_totals[key] += value

    def summary(self) -> str:
        lines = [
            f"Workflow: {self.workflow_name}",
            f"Status: {self.status}",
            f"Total duration: {self.total_duration_ms:.2f}ms" if self.total_duration_ms is not None else "Total duration: -",
            f"Steps: {self.steps_total} ({self.steps_succeeded} succeeded, {self.steps_failed} failed)",
            f"Parallel groups: {self.groups_total}",
            f"Retries: {self.retries_total}",
        ]
        if self.usage_totals:
            usage_str = ", ".join(f"{k}={v}" for k, v in self.usage_totals.items())
            lines.append(f"Usage totals: {usage_str}")
        lines.append("")
        lines.append("By tool:")
        for name, stats in self.by_tool.items():
            avg = f"{stats.avg_duration_ms:.2f}ms" if stats.avg_duration_ms is not None else "-"
            line = (
                f"  {name}: {stats.calls} calls, {stats.successes} ok, "
                f"{stats.failures} failed, {stats.retries} retries, avg {avg}"
            )
            if stats.usage_totals:
                usage_str = ", ".join(f"{k}={v}" for k, v in stats.usage_totals.items())
                line += f", usage: {usage_str}"
            lines.append(line)
        return "\n".join(lines)
