"""Execution graph: the user-facing representation of a journal.

Per architecture-spec-v1.md section 6: the journal is the serialized data
model (source of truth), the graph is a rendering of it -- node per agent
call/tool call/parallel block, edges show execution order and grouping.
Since M0-M3 execution is strictly a sequence of steps and parallel blocks
(no `delegate`/branching yet), the shape is a tree, not a general graph;
this module builds a tree today and can grow into a real graph once
delegation or branching exist.

This is a text renderer, not a web dashboard. `aircli`'s `ai trace` command
(added later) builds on exactly this -- render_execution_graph() for the
default text output, plus `--json`/`--html` (aircli/html_trace.py) for the
other two formats -- so this module's tree-building logic is still the
one source of truth for "what does this run's structure look like,"
whichever output format a caller wants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .journal import Journal, StepRecord


@dataclass
class GraphNode:
    kind: str  # "workflow" | "group" | "step"
    label: str
    status: Optional[str] = None
    duration_ms: Optional[float] = None
    children: List["GraphNode"] = field(default_factory=list)


def build_execution_graph(journal: Journal) -> GraphNode:
    root = GraphNode(kind="workflow", label=journal.workflow_name,
                      status=journal.status, duration_ms=journal.duration_ms)

    steps_by_group: Dict[str, List[StepRecord]] = {}
    for step in journal.steps:
        if step.group_id is not None:
            steps_by_group.setdefault(step.group_id, []).append(step)

    seen_groups = set()
    for step in sorted(journal.steps, key=lambda s: s.id):
        if step.group_id is not None:
            if step.group_id in seen_groups:
                continue
            seen_groups.add(step.group_id)
            group = journal._groups_by_id[step.group_id]
            group_node = GraphNode(kind="group", label=f"{group.kind}:{group.id}",
                                    status=group.status, duration_ms=group.duration_ms)
            for member in sorted(steps_by_group[step.group_id], key=lambda s: s.id):
                group_node.children.append(GraphNode(
                    kind="step", label=member.tool, status=member.status,
                    duration_ms=member.duration_ms,
                ))
            root.children.append(group_node)
        else:
            root.children.append(GraphNode(
                kind="step", label=step.tool, status=step.status,
                duration_ms=step.duration_ms,
            ))

    return root


def render_execution_graph(node: GraphNode, depth: int = 0) -> str:
    prefix = "  " * depth + ("- " if depth > 0 else "")
    status_part = f" [{node.status}]" if node.status else ""
    duration_part = f" ({node.duration_ms:.2f}ms)" if node.duration_ms is not None else ""
    line = f"{prefix}{node.kind}: {node.label}{status_part}{duration_part}"
    lines = [line]
    for child in node.children:
        lines.append(render_execution_graph(child, depth + 1))
    return "\n".join(lines)
