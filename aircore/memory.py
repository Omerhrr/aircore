"""Memory: scoped key-value stores tools can read and write.

Three scopes, each with different persistence semantics:

- session: lives as long as the Memory object you construct. Reuse the same
  Memory instance across multiple workflow.run() calls and session data
  persists between them; construct a new Memory and it starts empty. No
  runtime-enforced lifecycle beyond "it's a plain dict tied to this object."
- project: shared across every Memory instance constructed with the same
  `project=` name, via a process-wide registry. This is what lets two
  different Workflows (e.g. two different agents in the same project) see
  each other's project-scoped data. A Memory() with no project name behaves
  like an isolated scope, shared with nobody.
- temporary: scratch space for one workflow run. This is the one scope the
  runtime actively manages: Workflow.run() clears it after every run,
  success or failure, so temporary data can never leak into the next run.
  It is NOT cleared per-step -- steps within a single run share it.

Tools access memory the same way they access anything else in this
runtime: as a plain Python closure over a Memory instance, not through a
special calling convention or scheduler-injected argument. That keeps the
Tool/Scheduler contract unchanged (tools are still called with no
arguments), but it also means memory reads/writes are invisible to the
event bus -- the Journal and Metrics have no way to know a tool touched
memory unless the tool's own return value says so. That's a real
limitation, not an oversight: making memory access observable would mean
giving tools a different calling convention (e.g. scheduler-injected
memory argument), which is a bigger API change than M5 is scoped for.

Minimum viable backend: in-process dict, per architecture-spec-v1.md
section 10's open question. No SQLite or other persistent backend yet --
every scope vanishes when the process exits, and `project` scope is only
shared within a single process, not across machines or restarts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class MemoryScope:
    """A single namespaced key-value store."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def snapshot(self) -> Dict[str, Any]:
        """A shallow copy -- useful for inspection/debugging without handing
        out the live dict for uncontrolled mutation."""
        return dict(self._data)

    def __repr__(self) -> str:
        return f"MemoryScope({self._data!r})"


# Process-wide registry backing the `project` scope. Keyed by project name
# so any Memory("some-project") anywhere in the process shares the same
# underlying MemoryScope.
_project_scopes: Dict[str, MemoryScope] = {}


class Memory:
    def __init__(self, project: Optional[str] = None) -> None:
        self.session = MemoryScope()
        self.temporary = MemoryScope()
        self.project: MemoryScope = (
            _project_scopes.setdefault(project, MemoryScope())
            if project is not None
            else MemoryScope()
        )
