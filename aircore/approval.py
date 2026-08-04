"""Human-in-the-loop approval: a synchronous gate in front of specific tools.

This closes the gap flagged in policy.py since M3 and called out again in
the CrewAI/LangGraph/OpenAI-Agents-SDK comparison: every one of those
frameworks already has some notion of "pause and ask a human before doing
this," and aircore didn't.

Two real shapes were on the table for this, and it's worth being explicit
about which one got built and why:

1. A synchronous callback (this module): `Policy(approval_for={"deploy"})`
   marks a tool name as approval-gated; `Workflow.run(approval_callback=...)`
   supplies a `Callable[[ApprovalRequest], bool]` the scheduler calls,
   blocking, the moment it's about to run a gated tool. Nothing about this
   needs durable state -- it fits aircore's existing execution model exactly
   as it already works today (Scheduler.run() is one synchronous,
   in-process call from start to finish), so it costs nothing beyond this
   one module plus the scheduler/policy wiring.
2. Pause-and-resume: the workflow actually halts mid-run, the Journal
   records a "pending approval" step, and a *separate* call resumes
   execution later -- potentially after the process restarted. This is
   what LangGraph's interrupt/checkpoint model does, and it's the shape a
   real production approval queue eventually wants. It was deliberately
   NOT built here: it needs durable, resumable workflow state, which aircore
   does not have (see the durable-checkpointing gap noted alongside this
   one) -- building a fake version of pause/resume that doesn't actually
   survive a restart would be worse than not having it, because it would
   look like the real thing without being it.

So: approval here means "someone or something answers synchronously while
this process is running," never "pause for three days and resume on a
different machine." That's a real, honest limit on what this primitive
promises -- not the full LangGraph-equivalent, but a genuine, enforced gate
(a gated tool truly cannot run without the callback returning True, checked
by the scheduler itself, not by convention), and a real foundation: a
future durable-resume mechanism could replace `cli_approval_callback` with
one that persists the pending request and blocks a worker thread on it,
without changing Policy.approval_for's shape at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class ApprovalRequest:
    """What an approval_callback is handed. Deliberately thin -- just
    enough to identify which step, in which workflow, is asking, and to
    show a human (or route to whatever's answering) something meaningful.
    Not the step's full input (that's the same gap ModelAgent/Tool inputs
    aren't in the Journal at all yet -- see journal.py), so a callback
    that needs to know *what arguments* a tool is about to be called with
    can't get that from this object today."""
    workflow_id: str
    workflow_name: str
    step_id: int
    tool_name: str
    group_id: Optional[str] = None


ApprovalCallback = Callable[[ApprovalRequest], bool]


class ApprovalDenied(Exception):
    """Raised internally when an approval_callback returns False for a
    tool named in Policy.approval_for. Caught by the scheduler exactly
    like CapabilityDenied -- fails that step gracefully (recorded in the
    Journal via ApprovalDecided, see events.py), never an unhandled
    crash."""


def cli_approval_callback(request: ApprovalRequest) -> bool:
    """A ready-made approval_callback for interactive use -- this is what
    `ai run`/`ai trace` wire in by default for .airlang files (see aircli).
    Blocks on a real y/n prompt via input(). Not meant for anything
    unattended: write your own callback (poll a database, wait on a
    webhook, check a Slack approval, ...) for anything beyond a human at a
    terminal -- ApprovalCallback is a plain callable, nothing here is
    special-cased to this implementation."""
    prompt = (
        f"\nApproval required -- workflow '{request.workflow_name}', "
        f"step {request.step_id}: run tool '{request.tool_name}'? [y/N] "
    )
    answer = input(prompt).strip().lower()
    return answer in ("y", "yes")


def auto_approve(request: ApprovalRequest) -> bool:
    """Approves everything unconditionally. A deliberate escape hatch for
    tests and automation that has already decided every approval-gated
    tool in this run is fine -- using this in place of a real approval
    mechanism defeats the entire point of Policy.approval_for, so it is
    named loudly rather than being the default anywhere."""
    return True


def auto_deny(request: ApprovalRequest) -> bool:
    """Denies everything unconditionally -- exists to test that a
    workflow behaves correctly (fails that one step, not the whole
    process) when approval is refused."""
    return False
