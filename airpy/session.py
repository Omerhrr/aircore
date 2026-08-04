"""Session: a long-running, stateful conversation, built entirely out of
existing primitives -- ModelAgent, Memory, and a real Workflow per turn.
No aircore changes were needed for this, same as JudgeConsensus, MCP tools,
and structured output before it: airpy composes aircore's primitives, it
doesn't extend the runtime itself.

What this adds on top of ModelAgent(memory=..., conversation_id=...)
(model_agent.py) -- which already gives multi-turn history:

1. Ergonomics: `session.send(message)` instead of hand-constructing a new
   ModelAgent every turn with the same provider/tools/schema/memory/
   conversation_id (see examples/memory_conversations.py, which does
   exactly that by hand -- Session is that pattern, wrapped).
2. Real per-turn journaling: each `send()` runs its ModelAgent through an
   actual aircore Workflow (`workflow.step(agent, agent=...)` +
   `workflow.run()`), not a bypass like ask()/ModelAgent.stream() --
   capability requirements, Policy, retries, and a full journal all apply
   to every turn, exactly as if you'd built that Workflow yourself. This
   is the difference between "a chat loop" and "a long-running employee
   whose every action is capability-checked and audited" -- see
   `session.journals` for the full per-turn audit trail.
3. Session-level state a single ModelAgent doesn't track on its own:
   `session_id`, `created_at`, `last_active_at`, `turn_count`, and
   `close()`/`ended_at` for an explicit end-of-session marker.
4. `max_history_turns`: an actual "long-running" concern -- unbounded
   conversation history (what memory-backed ModelAgent alone gives you)
   is fine for a short exchange, but a session meant to stay open for a
   long time (a customer support call, say) needs some bound on how much
   history keeps getting resent every turn. A simple sliding window,
   trimmed from the oldest turns, applied before each send().
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

from aircore.agent import Agent as Identity
from aircore.journal import Journal
from aircore.memory import Memory, MemoryScope
from aircore.policy import Policy
from aircore.workflow import Workflow

from .model_agent import ModelAgent, RequiresArg
from .providers import ModelProvider


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionClosed(Exception):
    """Raised by send() on a Session that's already had close() called."""


class SessionTurnFailed(Exception):
    """Raised by send() when the turn's Workflow run failed -- the
    underlying journal step (see the exception's `journal` attribute for
    the full record) already explains why in `.error`; this just turns
    that into a normal Python exception, since send() promises to return
    an answer, unlike a Workflow step, which fails gracefully by design."""

    def __init__(self, message: str, journal: Journal) -> None:
        super().__init__(message)
        self.journal = journal


class Session:
    """A long-running conversation. Constructed once with everything a
    turn needs (provider, model, tools, output_schema, the capabilities
    this agent requires); `send()` is called once per message.

    `memory`/`conversation_id` are optional -- if omitted, the Session
    creates its own private `Memory().session` scope and uses its own
    `session_id` as the conversation key, so two Sessions never
    accidentally share history. Pass your own `memory=`/`conversation_id=`
    (e.g. `Memory.project`) if you want this session to share or persist
    history the way `examples/memory_conversations.py` demonstrates for
    plain ModelAgents."""

    def __init__(
        self,
        name: str,
        provider: ModelProvider,
        model: str = "mock",
        tools: Optional[list] = None,
        output_schema: Optional[Any] = None,
        requires: RequiresArg = None,
        memory: Optional[MemoryScope] = None,
        conversation_id: Optional[str] = None,
        policy: Optional[Policy] = None,
        max_history_turns: Optional[int] = None,
        max_turns: int = 5,
    ) -> None:
        if max_history_turns is not None and max_history_turns < 1:
            raise ValueError("max_history_turns must be at least 1 (each turn is one exchange)")

        self.name = name
        self.provider = provider
        self.model = model
        self.tools = tools
        self.output_schema = output_schema
        self.requires = requires
        self.policy = policy
        self.max_history_turns = max_history_turns
        self.max_turns = max_turns

        self.session_id = f"session-{uuid.uuid4().hex[:8]}"
        self.memory: MemoryScope = memory if memory is not None else Memory().session
        self.conversation_id = conversation_id or self.session_id

        self.created_at = _now()
        self.last_active_at: Optional[str] = None
        self.ended_at: Optional[str] = None
        self.turn_count = 0
        # One Journal per successful or failed send() -- a session-level
        # audit trail, exactly the kind of "why did the AI do this"
        # record the runtime's journal already exists for.
        self.journals: List[Journal] = []

    def send(self, message: str, agent: Optional[Identity] = None) -> Any:
        """Sends one message, runs it through a real one-step Workflow
        (so it's journaled, capability-checked against `agent` if
        `requires=` was set, and Policy-enforced exactly like any other
        workflow step), and returns the agent's answer -- a string, or a
        validated structured value if `output_schema=` was given.

        `agent`, if given, is the identity attempting this turn -- same
        meaning as `workflow.step(tool, agent=identity)`'s `agent=`.
        Raises SessionClosed if close() was already called, or
        SessionTurnFailed if the turn's step failed (capability denial,
        provider error, a bad structured response, etc. -- see the
        raised exception's `.journal` for the full record)."""
        if self.ended_at is not None:
            raise SessionClosed(f"session '{self.session_id}' was closed at {self.ended_at}")

        turn_agent = ModelAgent(
            self.name, self.provider, prompt=message, model=self.model,
            tools=self.tools, output_schema=self.output_schema, requires=self.requires,
            memory=self.memory, conversation_id=self.conversation_id, max_turns=self.max_turns,
        )
        workflow = Workflow(f"{self.session_id}-turn-{self.turn_count + 1}", policy=self.policy)
        workflow.step(turn_agent, agent=agent)
        journal = workflow.run()

        self.turn_count += 1
        self.last_active_at = _now()
        self.journals.append(journal)
        # Trimmed after the turn, not before -- so the bound applies to
        # the history as it actually stands after this exchange, not as
        # of the start of it (which would let it grow one turn past the
        # limit every time before ever trimming).
        self._trim_history_if_needed()

        if journal.status != "success":
            error = journal.steps[-1].error if journal.steps else "unknown error"
            raise SessionTurnFailed(
                f"session '{self.session_id}' turn {self.turn_count} failed: {error}", journal
            )

        return journal.steps[0].output

    def _trim_history_if_needed(self) -> None:
        if self.max_history_turns is None:
            return
        history = list(self.memory.get(self.conversation_id, []))
        max_messages = self.max_history_turns * 2  # each turn is one user + one assistant message
        if len(history) > max_messages:
            self.memory.set(self.conversation_id, history[-max_messages:])

    def close(self) -> None:
        """Marks the session ended -- purely a metadata marker (`ended_at`,
        checked by send()); does not clear history, so `.history` still
        reads back a closed session's conversation."""
        self.ended_at = _now()

    @property
    def history(self) -> List[dict]:
        """The conversation so far, in the same {"role", "content"} shape
        ModelAgent.conversation_history() returns."""
        return list(self.memory.get(self.conversation_id, []))

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self.ended_at else "open"
        return f"<Session {self.session_id} ({status}) turns={self.turn_count}>"
