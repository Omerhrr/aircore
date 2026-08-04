"""Agent: the minimal M2 slice.

Full spec (model + prompt + tools + capabilities, see architecture-spec-v1.md
section 3) needs model/provider integration that isn't part of any milestone
yet. For M2, an Agent is only what capability enforcement needs: a name and
the set of capabilities it's been granted. Model/prompt binding is added
when a milestone actually needs it -- not speculatively now.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Set

from .effects import Capability


class Agent:
    def __init__(self, name: str, capabilities: Optional[Iterable[Capability]] = None) -> None:
        self.name = name
        self.capabilities: Set[Capability] = set(capabilities or [])

    def grants(self, capability: Optional[Capability]) -> bool:
        """A tool with no declared capability requirement is always allowed.
        Otherwise the agent must hold that exact capability."""
        if capability is None:
            return True
        return capability in self.capabilities

    def missing(self, requires: Iterable[Capability]) -> List[Capability]:
        """Every capability in `requires` this agent was not granted."""
        return [c for c in requires if c not in self.capabilities]

    def __repr__(self) -> str:
        names = ", ".join(c.name for c in self.capabilities)
        return f"<Agent {self.name} capabilities=[{names}]>"
