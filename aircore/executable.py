"""Executable: the one interface the Scheduler actually needs.

Before this, the Scheduler's `_run_tool` was typed against `Tool`
specifically. That was fine while Tool was the only thing that could be a
step, but it meant "add a new kind of step" required touching scheduler
internals. This makes the contract explicit and minimal: anything with a
name, the retry/capability metadata every step already carries, and an
execute() method can be scheduled -- Tool implements it, and so will
anything a provider-aware layer (like airpy) adds on top, without aircore
needing to know that layer exists.

This file, and everything else in aircore, has zero knowledge of models,
prompts, or providers. That's deliberate: aircore is an execution runtime
that happens to be able to execute AI, not an AI runtime with execution
features bolted on. See architecture-spec-v1.md's addendum on this.

Kept synchronous by deliberate choice, not oversight: every concurrency
primitive already in this runtime (parallel, consensus) uses a thread
pool, not asyncio, and a provider adapter can make a blocking HTTP call
inside execute() exactly like any other tool call does today. Revisit if a
real integration genuinely needs async -- don't switch the runtime over
speculatively before that's demonstrated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from .effects import Capability


class ToolTimeout(Exception):
    """Raised when a step's execute() doesn't return within its `timeout`
    attribute (seconds) -- checked generically via getattr(tool,
    'timeout', None) in scheduler.py's _execute_with_timeout, so this
    applies to any Executable that happens to set a `timeout` attribute
    (Tool always has one; airpy's ModelAgent currently doesn't, so it's
    unaffected until/unless it grows one).

    A real, but honestly limited, guarantee: this stops the scheduler
    from *waiting* any longer for the call, it does not forcibly stop the
    call itself -- Python cannot safely kill a running thread. A
    non-cooperative call keeps running in the background even after this
    is raised and the step is marked failed in the Journal. For a
    genuine forced termination, use Tool(sandbox=Sandbox(max_runtime=...))
    (see sandbox.py) -- that runs in a real OS process, which can
    actually be killed."""


class Executable(ABC):
    name: str
    idempotent: bool = False
    retries: int = 0
    requires: Tuple[Capability, ...] = ()

    @abstractmethod
    def execute(self) -> Any:
        """Run this step and return its output. Called with no arguments --
        anything an execution needs (prompt, provider, closed-over memory)
        must already be bound on the instance by the time this runs."""
        raise NotImplementedError

    def usage(self) -> Optional[Dict[str, float]]:
        """Optional: numeric usage/cost data about the most recent
        execute() call -- e.g. {"tokens_in": 120, "cost_usd": 0.002}.
        Returns None by default. aircore has no opinion on what the keys
        mean; the Scheduler just reports whatever's here (see
        UsageReported in events.py) and Metrics/Journal sum and record
        whatever numeric keys show up. This is deliberately generic --
        "cost" and "tokens" aren't hardcoded AI vocabulary here, they're
        just conventional key names a caller (like airpy's ModelAgent)
        happens to use. Tool never overrides this."""
        return None
