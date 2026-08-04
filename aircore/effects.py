"""Capabilities (the effect system).

Module is named effects.py per architecture-spec-v1.md section 3: it is
technically an effect system, but the developer-facing name is
`Capability` because that reads more plainly. An agent declares which
capabilities it holds; a tool declares which capability it requires. The
scheduler is the only path through which tools get invoked, so it's also
the only place capability checks need to happen -- see scheduler.py.

Enforcement here is runtime interception, not static analysis. Tool
implementations are arbitrary Python, so there is no way to prove from the
DSL/SDK layer what a tool actually does internally; the guarantee is only
"this call was checked before it happened," not "this tool is safe."
"""

from __future__ import annotations


class Capability:
    """A named permission a tool can require and an agent can hold.

    Equality/hash are by name, so `Capability("Network") == Capability("Network")`
    -- two capability tokens with the same name are the same capability, no
    need to share a single instance.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Capability) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("Capability", self.name))

    def __repr__(self) -> str:
        return f"Capability({self.name!r})"


# Common built-in capabilities. Custom ones are just `Capability("Whatever")`.
Network = Capability("Network")
Filesystem = Capability("Filesystem")
Email = Capability("Email")
Payments = Capability("Payments")
Database = Capability("Database")


class CapabilityDenied(Exception):
    """Raised when a tool call requires a capability the acting agent
    wasn't granted. Caught by the scheduler like any other tool failure --
    it surfaces in the journal as a failed step with this as the error."""
