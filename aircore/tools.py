"""The @tool decorator.

M0 keeps this deliberately minimal: a tool is just a named, callable unit of
work. Capability requirements, idempotency flags, timeouts, and cost
estimates are metadata that get added in M2/M3 -- adding the fields now, as
plain attributes with safe defaults, so later milestones don't require
changing the decorator's shape.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from .effects import Capability
from .executable import Executable
from .sandbox import Sandbox, run_sandboxed

RequiresArg = Union[Capability, Iterable[Capability], None]


def _normalize_requires(requires: RequiresArg) -> Tuple[Capability, ...]:
    if requires is None:
        return ()
    if isinstance(requires, Capability):
        return (requires,)
    return tuple(requires)


class Tool(Executable):
    """Wraps a plain function as a named, schedulable unit of work.

    `requires` names the capability (or capabilities) the *caller* must
    hold to invoke this tool -- the tool itself doesn't have a capability,
    it demands one from whoever's calling it. Renamed from the earlier
    `capability=` kwarg for exactly that reason."""

    def __init__(self, fn: Callable[..., Any], *, name: str | None = None,
                 idempotent: bool = False, requires: RequiresArg = None,
                 timeout: float | None = None, retries: int = 0,
                 description: str | None = None,
                 parameters_schema: Optional[Dict[str, Any]] = None,
                 sandbox: Optional[Sandbox] = None) -> None:
        if retries > 0 and not idempotent:
            raise ValueError(
                f"tool '{name or fn.__name__}' declares retries={retries} but "
                f"idempotent=False -- retrying a non-idempotent tool could repeat "
                f"a side effect (e.g. double-send an email). Set idempotent=True "
                f"only if calling the tool twice is actually safe, or leave retries=0."
            )
        self.fn = fn
        self.name = name or fn.__name__
        self.idempotent = idempotent
        self.requires: Tuple[Capability, ...] = _normalize_requires(requires)
        self.timeout = timeout
        self.retries = retries
        # Not used by the Scheduler at all -- purely informational metadata
        # for anything that wants to describe this tool to a human or a
        # model (e.g. airpy's tool-calling loop, which needs a description
        # to hand a model). Falls back to the function's docstring so most
        # tools don't need to set this explicitly.
        self.description = description or (fn.__doc__ or "").strip().split("\n")[0] or self.name
        # Same category as `description`: purely descriptive, unread by the
        # Scheduler. airpy's schema.py normally derives a JSON schema by
        # introspecting `fn`'s Python signature -- that only works when
        # `fn` genuinely has one (a real function with typed parameters).
        # A tool whose schema comes from somewhere else entirely (e.g. an
        # MCP server's own tool listing -- see airpy/mcp_tools.py, where
        # `fn` is a generic `**kwargs` wrapper with no useful signature of
        # its own) sets this instead, and schema.py uses it as-is rather
        # than introspecting. None means "derive it from fn as before" --
        # every Tool built before this existed behaves identically.
        self.parameters_schema = parameters_schema
        # None (the default): execute() calls self.fn() directly,
        # in-process, exactly as before this existed -- every Tool ever
        # built without this argument is completely unaffected. See
        # sandbox.py's module docstring for what Sandbox actually
        # guarantees (real process isolation, a best-effort egress
        # allowlist, Unix-only best-effort memory limits) and its real
        # constraint (fn must be a picklable, module-level callable --
        # not a lambda or closure -- since it crosses a process boundary).
        self.sandbox = sandbox
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def execute(self) -> Any:
        """Satisfies the Executable interface. The Scheduler calls this,
        not __call__ -- __call__ stays around for anyone calling a Tool
        directly outside a workflow (tests do this occasionally).

        If `sandbox` was given, `self.fn` is never called directly here
        at all -- it's handed to run_sandboxed() (sandbox.py), which runs
        it in a subprocess and enforces whatever limits `sandbox`
        declares."""
        if self.sandbox is not None:
            return run_sandboxed(self.fn, self.sandbox)
        return self.fn()

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"


def tool(_fn: Callable[..., Any] | None = None, *, name: str | None = None,
         idempotent: bool = False, requires: RequiresArg = None,
         timeout: float | None = None, retries: int = 0,
         description: str | None = None,
         parameters_schema: Optional[Dict[str, Any]] = None,
         sandbox: Optional[Sandbox] = None) -> Any:
    """Decorator turning a plain function into a Tool.

    Usable bare (`@tool`) or with metadata (`@tool(idempotent=True, retries=3)`).
    `@tool(sandbox=Sandbox(...))` works too -- but remember sandbox.py's
    picklability constraint: a function *decorated* with `@tool` at
    module level is still a plain module-level function underneath (the
    decorator returns a Tool wrapping it), so this is fine; a tool
    defined as a nested/local function is not.
    """

    def wrap(fn: Callable[..., Any]) -> Tool:
        return Tool(fn, name=name, idempotent=idempotent, requires=requires,
                     timeout=timeout, retries=retries, description=description,
                     parameters_schema=parameters_schema, sandbox=sandbox)

    if _fn is not None:
        return wrap(_fn)
    return wrap
