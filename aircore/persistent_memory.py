"""FileMemoryScope: a JSON-file-backed, MemoryScope-compatible store --
the persistent-across-restarts option memory.py's module docstring names
as a real gap ("every scope vanishes when the process exits").

Drop-in wherever a MemoryScope is accepted -- ModelAgent(memory=...,
conversation_id=...), Session(memory=...), or in place of Memory().
session/.project -- since it implements the exact same duck-typed
contract every one of those already checks for (get/set, per model_
agent.py's own constructor check), not a MemoryScope subclass. Nothing
downstream needs to know it isn't the plain in-process MemoryScope.

Two honest constraints, the same two FileCheckpointStore already
documents for the same underlying reasons:

- JSON-serializable values only. set() raises TypeError immediately (not
  silently) if a value doesn't round-trip through json.dumps -- a
  conversation history (a list of {"role", "content"} dicts, exactly what
  ModelAgent's memory-backed conversations already store) is always fine;
  a value holding a live object, a Pydantic model instance, or anything
  else non-JSON is not.
- Reads and rewrites the whole file on every set()/delete()/clear() call
  -- fine for a conversation's worth of history or a workflow's worth of
  bound state, not built for high-frequency writes or large values. A
  real high-throughput need would want a proper database, not this.

What this does NOT give you: cross-process *concurrent* access safety.
Two processes writing to the same path at the same time can race (last
write wins, no locking) -- fine for "one long-running process, restarted
occasionally," not fine for "many workers sharing one memory file at
once." That's a different, bigger problem (the same class of thing a
real database solves), not something this module reaches for.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict


class FileMemoryScope:
    """See this module's docstring. `path` is a single JSON file holding
    every key this scope has ever been given -- there is no separate
    namespacing beyond "one FileMemoryScope per file"; use a different
    path (or a key prefix of your own) for genuinely separate scopes."""

    def __init__(self, path: str) -> None:
        self.path = path

    def _read_all(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        with open(self.path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return json.loads(content) if content else {}

    def _write_all(self, data: Dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        return self._read_all().get(key, default)

    def set(self, key: str, value: Any) -> None:
        try:
            json.dumps(value)
        except TypeError as exc:
            raise TypeError(
                f"FileMemoryScope.set({key!r}, ...) -- value isn't JSON-serializable "
                f"({exc}); this store only persists plain JSON-compatible data "
                f"(str/int/float/bool/None/list/dict of those)"
            ) from exc
        data = self._read_all()
        data[key] = value
        self._write_all(data)

    def delete(self, key: str) -> None:
        data = self._read_all()
        if key in data:
            del data[key]
            self._write_all(data)

    def clear(self) -> None:
        self._write_all({})

    def __contains__(self, key: str) -> bool:
        return key in self._read_all()

    def snapshot(self) -> Dict[str, Any]:
        """A shallow copy of everything currently stored -- same
        inspection-only contract as MemoryScope.snapshot()."""
        return self._read_all()

    def __repr__(self) -> str:
        return f"FileMemoryScope({self.path!r})"
