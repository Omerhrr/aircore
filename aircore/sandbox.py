"""Sandboxed tool execution: process isolation, a best-effort network
egress allowlist, and a genuine forced-kill timeout for a Tool's function.

The third gap the CrewAI/LangGraph/OpenAI-Agents-SDK comparison surfaced,
after approval.py and checkpoint.py: Capability grants (effects.py) are a
*logical* check -- "is this agent allowed to call this tool" -- not an
OS-level one. Nothing stops a Tool's own code, once called, from doing
anything the host process can do: opening arbitrary sockets, reading
arbitrary files, running forever. This module is what closes that,
scoped honestly rather than pretending to more than it delivers:

1. **Process isolation, not a container.** `run_sandboxed(fn, sandbox)`
   runs `fn()` in a real OS subprocess (via `multiprocessing`, "spawn"
   start method for portability), not the calling process's own thread.
   This is what makes `Sandbox.max_runtime` a *genuine* forced
   termination -- unlike `Tool.timeout` (scheduler.py's
   `_execute_with_timeout`, a thread-based wait that cannot actually stop
   a running call), a subprocess can be killed outright. It is still not
   a container: no filesystem isolation, no separate user/namespace, no
   seccomp profile -- the subprocess can read/write anything the host
   process's user could. Real container/VM isolation is a much larger,
   platform-specific undertaking with no proven need driving it yet;
   this is the achievable, portable slice.

2. **The egress allowlist is application-level, not a firewall.**
   `Sandbox.allowed_hosts` is enforced by monkeypatching
   `socket.socket.connect` inside the subprocess before `fn()` runs --
   real enforcement against anything that goes through Python's socket
   module under the hood (covers `requests`, `urllib`, `http.client`,
   most HTTP libraries), but NOT a kernel-level guarantee: a C extension
   that opens a raw socket, a subprocess the tool shells out to, or a
   creative bypass of the patched method could get around it. Calling
   this "sandboxed networking" would overstate what it does; it's a
   deliberately-labeled best-effort check.

3. **Memory limits are Unix-only and best-effort.** `Sandbox.max_memory_mb`
   uses `resource.setrlimit(RLIMIT_AS, ...)` inside the subprocess --
   silently a no-op on platforms without the `resource` module (Windows).
   Exceeding it typically kills the subprocess (OOM or a hard OS
   response) rather than raising a clean, distinguishable exception --
   getting a uniform "out of memory" signal across platforms isn't
   something Python's stdlib gives for free, and this doesn't invent one.

4. **`fn` and its return value must be picklable.** The "spawn" start
   method (chosen over "fork" for portability, and to avoid the known
   fork-from-a-multithreaded-process hazards relevant here since
   `run_sandboxed` can be called from inside the Scheduler's own thread
   pool for a parallel/consensus voter) needs to pickle `fn` by import
   path to hand it to the child process. A lambda or a closure over
   unpicklable state (a live network client, an open file handle, ...)
   will fail immediately in `process.start()`, in the parent, with
   Python's own clear pickling error -- not a mysterious hang. A
   module-level function is what this needs. This is a real, structural
   constraint on what `Tool(sandbox=...)` can wrap, not a bug.

Usage: `Tool(my_fn, name=..., sandbox=Sandbox(max_runtime=30,
max_memory_mb=256, allowed_hosts={"api.example.com"}))`. Every field on
Sandbox is optional and independent -- set only the limits you need.
"""

from __future__ import annotations

import multiprocessing
import queue
import socket
import traceback
from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, Iterable, Optional, Union

try:
    import resource  # Unix only
except ImportError:  # pragma: no cover -- exercised only on Windows
    resource = None  # type: ignore[assignment]


class SandboxViolation(Exception):
    """Base class for a failure caused by the sandbox mechanism itself
    (a forced timeout, a denied egress attempt) -- as opposed to
    SandboxedToolError, which wraps a failure the tool's own code
    raised."""


class SandboxTimeout(SandboxViolation):
    """The sandboxed subprocess didn't finish within Sandbox.max_runtime
    and was terminated. A genuine forced kill -- see this module's
    docstring, point 1, for how this differs from Tool.timeout."""


class EgressDenied(SandboxViolation):
    """Raised inside the sandboxed subprocess when tool code attempts a
    network connection to a host not in Sandbox.allowed_hosts, and
    re-raised with the same message in the parent process. See this
    module's docstring, point 2, for exactly what this does and doesn't
    cover."""


class SandboxedToolError(Exception):
    """Wraps an exception the tool's own code raised inside the sandbox
    (or a subprocess crash with no result at all -- e.g. an OOM kill).
    The original exception object doesn't survive the process boundary
    intact -- its class may not exist/import cleanly in the parent, and
    pickling arbitrary exception instances isn't reliable -- so this
    carries the original type name, message, and traceback text instead
    of pretending to perfectly preserve exception identity across
    processes."""

    def __init__(self, original_type: str, message: str, traceback_text: str = "") -> None:
        super().__init__(f"{original_type}: {message}")
        self.original_type = original_type
        self.original_message = message
        self.traceback_text = traceback_text


@dataclass(frozen=True)
class Sandbox:
    """Describes how isolated a Tool's execution should be. Every field
    is optional and independent -- `Sandbox()` (all None) still runs
    `fn()` in a subprocess (real process isolation, no forced limits),
    which is already a meaningfully stronger guarantee than running it
    in-process. See this module's docstring for what each field does and
    doesn't guarantee."""
    max_runtime: Optional[float] = None
    max_memory_mb: Optional[int] = None
    allowed_hosts: Optional[Union[FrozenSet[str], Iterable[str]]] = None

    def __post_init__(self) -> None:
        if self.allowed_hosts is not None and not isinstance(self.allowed_hosts, frozenset):
            object.__setattr__(self, "allowed_hosts", frozenset(self.allowed_hosts))


def _install_egress_allowlist(allowed_hosts: FrozenSet[str]) -> None:
    """Monkeypatches socket.socket.connect for the remaining lifetime of
    this process. Safe to do unconditionally -- this only ever runs
    inside a fresh, single-purpose sandbox subprocess (see _child_main),
    never in the parent process, so there's no other code in this
    process whose behavior this could unexpectedly change. Deny by
    default: nothing is implicitly allowed (not even localhost) -- an
    allowed_hosts set that should include "127.0.0.1" must say so
    explicitly."""
    original_connect = socket.socket.connect

    def guarded_connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host not in allowed_hosts:
            raise EgressDenied(
                f"connection to '{host}' denied -- not in this sandbox's "
                f"allowed_hosts ({sorted(allowed_hosts)})"
            )
        return original_connect(self, address, *args, **kwargs)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]


def _child_main(fn: Callable[[], Any], sandbox: "Sandbox", result_queue) -> None:
    """Runs inside the subprocess. Never lets an exception propagate past
    this function -- every outcome (success, the tool's own exception, an
    egress denial) is put on `result_queue` as a plain, picklable tuple;
    run_sandboxed (the parent side, below) is what turns that back into a
    real return value or a raised exception."""
    if sandbox.max_memory_mb is not None and resource is not None:
        limit_bytes = sandbox.max_memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except (ValueError, OSError):
            pass  # best-effort -- some platforms/limits refuse this; not fatal

    if sandbox.allowed_hosts is not None:
        _install_egress_allowlist(sandbox.allowed_hosts)

    try:
        result = fn()
        result_queue.put(("ok", result))
    except EgressDenied as exc:
        result_queue.put(("egress_denied", str(exc)))
    except Exception as exc:  # noqa: BLE001 -- the tool's own code can raise anything
        result_queue.put(("error", type(exc).__name__, str(exc), traceback.format_exc()))


def run_sandboxed(fn: Callable[[], Any], sandbox: "Sandbox") -> Any:
    """Runs `fn()` in an isolated subprocess honoring `sandbox`'s limits.
    Returns `fn()`'s return value, which must itself be picklable to
    cross back over the process boundary. See this module's docstring
    (point 4) for the picklability constraint on `fn` itself.

    Raises SandboxTimeout, EgressDenied, or SandboxedToolError (wrapping
    whatever `fn()` itself raised, or a subprocess crash with no result)
    -- never a bare, unclassified exception from the multiprocessing
    machinery itself."""
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_child_main, args=(fn, sandbox, result_queue))
    process.start()
    process.join(timeout=sandbox.max_runtime)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
        raise SandboxTimeout(
            f"sandboxed call exceeded max_runtime={sandbox.max_runtime}s and was terminated"
        )

    try:
        # A short grace period, not get_nowait(): the child's write to
        # result_queue can lag a hair behind process.join() returning
        # (multiprocessing.Queue uses a background feeder thread) -- this
        # avoids a rare, spurious "no result" for a subprocess that did
        # in fact finish cleanly.
        outcome = result_queue.get(timeout=2.0)
    except queue.Empty:
        raise SandboxedToolError(
            "SubprocessCrashed",
            f"the sandboxed subprocess exited (code={process.exitcode}) without "
            f"producing a result -- likely killed by the OS (e.g. an OOM kill if "
            f"max_memory_mb={sandbox.max_memory_mb} was exceeded), or it crashed natively",
        )

    kind = outcome[0]
    if kind == "ok":
        return outcome[1]
    if kind == "egress_denied":
        raise EgressDenied(outcome[1])
    # kind == "error"
    _, original_type, message, traceback_text = outcome
    raise SandboxedToolError(original_type, message, traceback_text)
