"""ParallelGroup: a workflow step that runs several tools concurrently.

This is deliberately just a marker the Scheduler recognizes -- it carries no
execution logic itself. Concurrency (thread pool), event emission, and
fan-in back to sequential flow all live in scheduler.py, matching M0's rule
that the Scheduler is the only thing that executes anything.
"""

from __future__ import annotations

import uuid
from typing import List

from .executable import Executable


class ParallelGroup:
    def __init__(self, tools: List[Executable]) -> None:
        self.tools = tools
        self.id = f"group-{uuid.uuid4().hex[:8]}"

    def __repr__(self) -> str:
        names = ", ".join(t.name for t in self.tools)
        return f"<ParallelGroup {self.id} [{names}]>"
