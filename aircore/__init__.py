"""aircore -- the AI execution runtime.

M0: sequential execution + journal + events.
M1: `parallel` blocks with fan-in.
M2: Capabilities -- agents declare granted capabilities, tools declare
    required ones (`requires=`), the scheduler enforces the match at call
    time. Enforcement is opt-in per step: a step with no agent attached is
    unrestricted, regardless of what the tool requires.
M3: Policy -- require_agent makes anonymous steps a pre-flight
    PolicyViolation instead of silently unrestricted; max_parallel and
    max_runtime bound concurrency and wall-clock time.
M4: Observability -- Metrics (step/tool counts, latency) and the execution
    graph (tree rendering of a finished journal), both built the same way
    as the Journal: independent EventBus subscribers / a pure function of
    the journal, nothing coupled directly to the scheduler.
Retries: Tool(idempotent=True, retries=N) -- the scheduler retries a
    failed call up to N times, only for tools explicitly declared
    idempotent (declaring retries>0 on a non-idempotent tool is a
    construction-time error). Metrics.retries_total and the journal's
    per-step retry history are both populated from this.
M5: Memory -- session/project/temporary scopes (see memory.py). Tools
    access it via closure, not a scheduler-injected argument, which means
    memory reads/writes are NOT visible to the Journal or Metrics -- a real
    gap relative to every other primitive, documented in memory.py rather
    than silently accepted.
M6: Consensus -- N tools run as voters (same concurrency mechanism as
    `parallel`), then a strategy (majority/unanimous, both raising
    ConsensusFailed on disagreement) reduces their outputs to one agreed
    value, recorded as a synthetic step in the journal/graph. This was the
    last milestone on the original list (architecture-spec-v1.md section 9).
Approval (approval.py): Policy.approval_for names tools that need a human
    (or whatever's wired in) to say yes before the scheduler will call them
    -- Workflow.run(approval_callback=...) supplies a
    Callable[[ApprovalRequest], bool], required pre-flight (PolicyViolation
    otherwise) whenever approval_for is non-empty. A synchronous, blocking
    gate, not a durable pause/resume mechanism -- see approval.py's module
    docstring for exactly why that's the scope, not an oversight.
Checkpointing (checkpoint.py): Workflow.run(checkpoint_store=...,
    run_id=...) makes a run resumable after a crash -- rerunning the same
    script with the same checkpoint_store/run_id skips every already-
    succeeded sequential step and continues from the first one that didn't
    finish. Position-indexed replay of Workflow's own deterministic step
    construction, not general durable execution: sequential steps only
    (parallel/consensus groups always re-run in full), JSON-serializable
    outputs only, and a cheap (not complete) determinism guard comparing
    each replayed step's recorded tool name against the step actually at
    that position this run. See checkpoint.py's module docstring for the
    full scope decision.
Sandboxed execution (sandbox.py): Tool(sandbox=Sandbox(max_runtime=...,
    max_memory_mb=..., allowed_hosts=...)) runs that tool's function in a
    real OS subprocess instead of in-process -- a genuine forced kill on
    timeout (unlike Tool.timeout, see below), a best-effort network
    egress allowlist (a socket-module-level check, not a kernel firewall),
    and a Unix-only best-effort memory cap. Requires `fn` to be a
    picklable, module-level callable, not a lambda/closure -- see
    sandbox.py's module docstring for exactly what is and isn't covered.
Tool.timeout is now actually enforced (it was stored but silently ignored
    before this) -- scheduler.py's _execute_with_timeout bounds how long
    the scheduler *waits* for a call via a thread-based timeout, raising
    ToolTimeout (executable.py). This does not forcibly stop a
    non-cooperative call (Python cannot safely kill a thread); Tool(
    sandbox=Sandbox(max_runtime=...)) is what gives an actual forced
    termination, via a real process instead.
Cross-step data flow (workflow.py's `bindings`): Workflow.step(tool,
    as_="name") records that step's output in `workflow.bindings` (a
    plain dict, mutated in place as the run proceeds) once it succeeds --
    closing the gap cross-step-data-flow.md documented ("no point in the
    design where step N's real output exists yet when step N+1 is being
    built"). airpy's ModelAgent(prompt=PromptTemplate(...),
    prompt_bindings=workflow.bindings) is the consumer side: the prompt is
    rendered fresh at execute() time (not construction time) from
    whatever's bound so far. Sequential steps only, same scope decision as
    checkpoint.py -- a parallel/consensus group's individual members don't
    get their own bindings.
Persistent Memory (persistent_memory.py): FileMemoryScope -- a JSON-file-
    backed, MemoryScope-compatible store (get/set/delete/clear/
    snapshot/__contains__, the exact same duck-typed contract every
    memory=-accepting call site already checks for), so a conversation's
    history (or anything else stored via Memory) survives a process
    restart. Drop-in wherever a MemoryScope is accepted -- not a special
    case. JSON-serializable values only, and no cross-process write
    locking -- see its own module docstring for exactly what that does
    and doesn't cover.

Executable: the Scheduler no longer knows about Tool specifically -- it
    only requires a `name`, `idempotent`/`retries`/`requires` metadata, and
    an `execute()` method. Tool implements this. So does anything a
    provider-aware layer adds on top (see the airpy package, a sibling to
    aircore, NOT imported by anything in here). aircore stays 100%
    provider-agnostic: no model, prompt, or provider concept exists
    anywhere in this package, by design.

Still not implemented: Policy.default_agent (see policy.py), checkpointing/
resume for parallel/consensus groups (checkpoint.py covers sequential
steps only), real filesystem/OS-level sandboxing (sandbox.py gives
process isolation + a best-effort egress allowlist, not a container or
namespace), bindings for parallel/consensus group members (workflow.py's
bindings cover sequential steps only), and cross-process write locking
for FileCheckpointStore/FileMemoryScope (both assume one writer at a
time).

(Token/cost metrics WERE listed here as unimplemented in an earlier
version of this docstring -- that was stale even before this round of
changes. See observability.py's own docstring: any Executable can
implement usage() -> Optional[dict], the scheduler emits UsageReported
when it does, and Metrics.usage_totals sums whatever numeric keys show
up. Closed since M8, proven again live in examples/
production_readiness.py's real DeepSeek run.)
"""

from .agent import Agent
from .approval import ApprovalCallback, ApprovalDenied, ApprovalRequest, auto_approve, auto_deny, cli_approval_callback
from .checkpoint import (
    CheckpointError, CheckpointRecord, CheckpointStore,
    FileCheckpointStore, InMemoryCheckpointStore,
)
from .consensus import ConsensusFailed, ConsensusGroup, majority, unanimous
from .effects import Capability, CapabilityDenied, Network, Filesystem, Email, Payments, Database
from .events import EventBus, Event
from .executable import Executable, ToolTimeout
from .graph import GraphNode, build_execution_graph, render_execution_graph
from .journal import Journal
from .memory import Memory, MemoryScope
from .observability import Metrics, ToolStats
from .persistent_memory import FileMemoryScope
from .policy import Policy, PolicyViolation
from .sandbox import (
    EgressDenied, Sandbox, SandboxedToolError, SandboxTimeout,
    SandboxViolation, run_sandboxed,
)
from .tools import tool, Tool
from .workflow import Workflow

__all__ = [
    "Workflow", "tool", "Tool", "Executable", "Journal", "EventBus", "Event",
    "Agent", "Capability", "CapabilityDenied",
    "Network", "Filesystem", "Email", "Payments", "Database",
    "Policy", "PolicyViolation",
    "Metrics", "ToolStats", "GraphNode", "build_execution_graph", "render_execution_graph",
    "Memory", "MemoryScope", "FileMemoryScope",
    "ConsensusGroup", "ConsensusFailed", "majority", "unanimous",
    "ApprovalRequest", "ApprovalCallback", "ApprovalDenied",
    "cli_approval_callback", "auto_approve", "auto_deny",
    "CheckpointError", "CheckpointRecord", "CheckpointStore",
    "FileCheckpointStore", "InMemoryCheckpointStore",
    "ToolTimeout",
    "Sandbox", "SandboxViolation", "SandboxTimeout", "EgressDenied",
    "SandboxedToolError", "run_sandboxed",
]

__version__ = "0.1.0"
