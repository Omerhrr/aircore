"""Durable checkpointing / resume-after-crash for sequential steps.

The second gap the CrewAI/LangGraph/OpenAI-Agents-SDK comparison
surfaced, after approval.py: `Workflow.run()` is one synchronous,
in-process call start to finish. If the process crashes halfway through,
there is no way to pick back up -- you rerun the whole workflow from
scratch, which is wasteful (paid LLM calls repeated) and dangerous for
any non-idempotent tool (an email gets sent twice).

What got built, and the real scope decisions behind it:

1. **Position-indexed replay, not a general durable-execution engine.**
   A `Workflow` is built by re-running the same Python code that
   constructed it the first time -- the same Tool/Agent objects, in the
   same order, are reconstructed deterministically. So "resume" doesn't
   need to serialize and reconstruct live Executable objects (which is
   the genuinely hard part of durable execution -- a Tool closure or a
   ModelAgent holding a real provider client isn't picklable in any
   meaningful way). It only needs to remember, per top-level step
   *position* in `Workflow._steps`, whether that step already succeeded
   and what it returned -- then skip re-running it and replay the cached
   output instead, letting execution continue from the first
   not-yet-recorded position. This is a much narrower promise than
   Temporal/Restate-style event-sourced durable execution, and is
   exactly why it's achievable without a rewrite.

2. **Sequential steps only -- `parallel`/`consensus` groups are never
   checkpointed, and always re-run in full on resume.** A group that
   crashed mid-flight (2 of 3 voters done) has no natural single
   "position" to record partial progress against, and building per-member
   group checkpointing is real additional complexity with no concrete
   need driving it yet. This is a genuine, stated limitation, not an
   oversight: a workflow whose expensive work lives inside `parallel`/
   `consensus` blocks gets no resume benefit from this at all today.

3. **JSON-serializable outputs only.** A checkpoint has to actually be
   written somewhere durable (a file, eventually maybe a database) --
   this module requires every checkpointed step's output to round-trip
   through `json.dumps`, and raises `CheckpointError` immediately if it
   doesn't, rather than silently lossy-stringifying it (which would hand
   back the wrong type/value on replay) or silently skipping the
   checkpoint (which would look durable while quietly not being). A tool
   returning a Pydantic model, a Pandas DataFrame, or anything else
   non-JSON simply can't be checkpointed today -- return a plain dict
   instead if you need durability for that step.

4. **A cheap, not a complete, determinism guard.** Real durable-execution
   systems (Temporal included) have to solve "what if the code changed
   between the crashed run and the resumed one" -- this module does the
   minimum useful version of that: each checkpoint entry also records the
   tool's `name`, and resume refuses (raises `CheckpointError`) if the
   step at that position now has a different name than what was recorded.
   This catches the common case (steps reordered, added, or removed) but
   not a subtler one (same name, semantically different tool) -- a real,
   stated gap, not a promise this module doesn't keep.

5. **CheckpointStore is a small protocol, not a specific backend.**
   `FileCheckpointStore` (a JSON file on disk) is the actual durable
   implementation -- no server or database dependency, same "no
   infrastructure until something needs it" rule as html_trace.py's
   self-contained viewer. `InMemoryCheckpointStore` exists only to test
   the replay mechanism itself; an in-process dict does not survive the
   crash it's meant to protect against, and using it for anything real
   defeats the entire point.

Usage: `workflow.run(checkpoint_store=FileCheckpointStore("run.json"),
run_id="my-run-1")`. Rerunning the identical script with the same
`checkpoint_store`/`run_id` after a crash skips every already-succeeded
sequential step and continues from the first one that didn't finish.
Both `checkpoint_store` and `run_id` must be given together, or neither
-- see workflow.py's `run()`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Protocol


class CheckpointError(Exception):
    """Raised when a succeeded step's output can't be checkpointed (not
    JSON-serializable -- see this module's docstring, point 3), or when a
    checkpoint entry's recorded tool name doesn't match the step at that
    position in this run (point 4). Either way, a loud, immediate failure
    -- never a silently incomplete or silently wrong checkpoint."""


@dataclass(frozen=True)
class CheckpointRecord:
    """One durably-recorded step: the tool's name (for the mismatch guard)
    and its JSON-serializable output, as it will be replayed on resume."""
    tool_name: str
    output: Any


class CheckpointStore(Protocol):
    """The interface `Workflow.run(checkpoint_store=...)` needs. Anything
    implementing these two methods works -- `FileCheckpointStore` and
    `InMemoryCheckpointStore` below are the two provided, but a caller
    with a real durable-storage need (S3, a database row, ...) can write
    their own against this same shape."""

    def load(self, run_id: str) -> Dict[int, CheckpointRecord]:
        """Every previously-recorded step for this run_id, keyed by its
        position in Workflow._steps. Empty dict for a run_id never seen
        before -- that's what makes a *first* run and a *resumed* run
        with an empty/stale checkpoint indistinguishable to the scheduler,
        by design: nothing behaves differently, everything just runs."""
        ...

    def record(self, run_id: str, entry_index: int, tool_name: str, output: Any) -> None:
        """Durably records one succeeded sequential step. Called by the
        scheduler immediately after that step's StepFinished(status=
        "success") -- see scheduler.py. Must raise CheckpointError (not
        silently drop the write) if `output` isn't JSON-serializable."""
        ...


def _require_json_serializable(tool_name: str, entry_index: int, output: Any) -> None:
    try:
        json.dumps(output)
    except TypeError as exc:
        raise CheckpointError(
            f"step '{tool_name}' (position {entry_index}) succeeded, but its output "
            f"isn't JSON-serializable ({exc}) -- checkpointing requires a plain "
            f"JSON-compatible output (str/int/float/bool/None/list/dict of those). "
            f"This workflow can't be safely resumed past this step with the "
            f"checkpoint_store in use."
        ) from exc


class InMemoryCheckpointStore:
    """A dict-backed CheckpointStore -- exists to test the replay
    mechanism itself (see tests/test_checkpoint.py), not to provide real
    durability. Nothing here survives the process crash this whole module
    exists to recover from; use FileCheckpointStore (or your own,
    real-storage-backed CheckpointStore) for anything that has to
    actually work across a restart."""

    def __init__(self) -> None:
        self._runs: Dict[str, Dict[int, CheckpointRecord]] = {}

    def load(self, run_id: str) -> Dict[int, CheckpointRecord]:
        return dict(self._runs.get(run_id, {}))

    def record(self, run_id: str, entry_index: int, tool_name: str, output: Any) -> None:
        _require_json_serializable(tool_name, entry_index, output)
        self._runs.setdefault(run_id, {})[entry_index] = CheckpointRecord(tool_name, output)


class FileCheckpointStore:
    """A JSON file on disk -- the real, durable implementation: survives
    a process crash and restart, needs no server or database. `record()`
    reads and rewrites the whole file on every call (fine for the
    tens-to-low-hundreds of steps a real workflow has; not built for
    anything beyond that -- a real high-throughput need would want a
    proper append-only log or a database, not this).

    One file can hold checkpoints for multiple run_ids (keyed inside the
    JSON), so a single FileCheckpointStore can be reused across every
    workflow in an application rather than needing one file per run."""

    def __init__(self, path: str) -> None:
        self.path = path

    def _read_all(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        if not os.path.exists(self.path):
            return {}
        with open(self.path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return json.loads(content) if content else {}

    def load(self, run_id: str) -> Dict[int, CheckpointRecord]:
        run = self._read_all().get(run_id, {})
        return {
            int(index_str): CheckpointRecord(entry["tool_name"], entry["output"])
            for index_str, entry in run.items()
        }

    def record(self, run_id: str, entry_index: int, tool_name: str, output: Any) -> None:
        _require_json_serializable(tool_name, entry_index, output)
        data = self._read_all()
        run = data.setdefault(run_id, {})
        run[str(entry_index)] = {"tool_name": tool_name, "output": output}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
