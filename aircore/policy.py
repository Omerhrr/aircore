"""Policy: the execution contract a workflow must satisfy.

Where Capabilities (M2) answer "can this identified agent invoke this
tool," Policy answers a different question: "must every step in this
workflow have an identified agent at all, and what other constraints bound
the whole run." Kept as a separate primitive on purpose -- conflating the
two makes the runtime harder to reason about.

Fields implemented:

- require_agent: pre-flight. If True, any step (or parallel-group member)
  with no `agent=` attached fails validation before the workflow runs at
  all -- no WorkflowStarted event, no journal, just a PolicyViolation. This
  is a stricter guarantee than a capability check: a capability check can
  only run once there's an agent to check against, so an anonymous step
  under `require_agent=True` never even reaches that point.
- max_parallel: pre-flight. Rejects any `parallel()` or `consensus()` block
  whose size exceeds the limit, before execution starts.
- max_runtime: enforced during execution. Checked before starting each
  step; once exceeded, no further steps start and the workflow finishes
  with status "failed". A step already in flight when the limit is hit is
  not interrupted -- true cancellation needs cooperative or forced
  interruption of running tool code, which is out of scope.
- max_cost: enforced during execution, checked the same way as
  max_runtime -- against cumulative `cost_usd` reported via any
  Executable's usage() (see executable.py, observability.py). This only
  means something once something actually reports cost_usd; a workflow
  built entirely of plain Tools or a mock-backed ModelAgent will never
  trip it, because nothing is reporting a cost. That's not a bug -- it's
  the same "we don't invent numbers" rule that kept this field unimplemented
  until a real provider integration (airpy's LiteLLMProvider) existed to
  produce a real cost figure.
- approval_for: a set of tool names that must be approved by a human (or
  whatever a caller wires in) before the scheduler will actually call
  them. Pre-flight: if this is non-empty, `Workflow.run()` requires an
  `approval_callback` argument (a `Callable[[ApprovalRequest], bool]`,
  see approval.py) or raises PolicyViolation before the run starts --
  same "never silently promise something it doesn't enforce" rule as
  require_agent. Enforced during execution: the scheduler calls
  `approval_callback` the moment it's about to run a gated tool and
  raises ApprovalDenied (failing just that step, journaled, not a crash)
  if it returns False. See approval.py's module docstring for the real
  scope decision here -- a synchronous, blocking callback, not a durable
  pause/resume mechanism (aircore has no durable workflow state to resume
  from yet).

Deliberately NOT implemented, and not accepted as fields, so Policy never
silently promises something it doesn't enforce:

- allowed_models: this is genuinely LLM-specific vocabulary ("model") that
  doesn't belong in aircore's Policy, which otherwise has no concept of what
  a "model" is -- aircore only knows Executable, not ModelAgent. Enforcing
  this belongs at the airpy layer (e.g. ModelAgent validating against a
  list it's told about), not here. Revisit only if a provider-agnostic way
  to express it emerges.
- default_agent: a real migration path from `require_agent=True` back to
  convenience, but deferred until require_agent itself has been used for
  real and its rough edges are known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Optional, Union


@dataclass
class Policy:
    require_agent: bool = False
    max_parallel: Optional[int] = None
    max_runtime: Optional[float] = None  # seconds
    max_cost: Optional[float] = None  # USD, against cumulative reported cost_usd
    approval_for: Optional[Union[FrozenSet[str], Iterable[str]]] = None

    def __post_init__(self) -> None:
        # Normalizes any iterable of names (a set, list, tuple, ...) into
        # a frozenset -- same convenience `Tool(requires=...)` gives
        # (tools.py's _normalize_requires), so `Policy(approval_for=
        # ["deploy"])` and `Policy(approval_for={"deploy"})` both work.
        if self.approval_for is not None and not isinstance(self.approval_for, frozenset):
            self.approval_for = frozenset(self.approval_for)


class PolicyViolation(Exception):
    """Raised for pre-flight policy failures (require_agent, max_parallel).

    Distinct from a failed step: a PolicyViolation means the workflow was
    never valid to run in the first place, so it never starts -- there is
    no journal for a run that never happened.
    """
