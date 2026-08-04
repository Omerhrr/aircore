"""ModelAgent: a ModelProvider bound to a prompt, wearing an Executable.

Named ModelAgent rather than reusing `Agent` on purpose: `aircore.Agent` is
already a real thing in this codebase -- the capability-holding identity
checked in `workflow.step(tool, agent=identity)`. That Agent answers "who
is executing." This class answers a completely different question -- "what
gets executed" -- it's a provider-backed sibling of Tool, not an identity.
Reusing the name `Agent` for both would make `workflow.step(researcher,
agent=identity_agent)` read like a mistake even when it's correct. If you
want a single name for this concept in your own code, alias it on import:
`from airpy import ModelAgent as Agent`.

Because ModelAgent implements aircore.Executable, it's a first-class,
interchangeable workflow step -- it can be passed to workflow.step(),
workflow.parallel(), or workflow.consensus() exactly like a Tool, and it
gets the exact same capability/policy/retry enforcement aircore already
built for Tool, with zero changes to aircore required. That's the entire
point of the Executable interface: aircore never needs to know this class
exists.

Tool-calling loop (ModelAgent(tools=[...])):

When `tools` is given, execute() runs a ReAct-style loop instead of a
single call: send the prompt (and tool schemas) to the provider; if the
response carries tool_calls, run each named aircore.Tool with the model-
supplied arguments, feed the results back as a new message, and ask
again; stop once a response has no more tool_calls, or raise
ModelAgentToolLoopExceeded after `max_turns`. With no tools given,
execute() is the exact single-shot call it always was -- this is why
every ModelAgent constructed before this existed still behaves
identically.

Two real limitations, not oversights:

1. Tool calls made inside this loop never pass through aircore's Scheduler
   -- from the Scheduler's point of view, one ModelAgent step is one
   execute() call, full stop. That means these internal tool calls are
   NOT journaled (see ToolCallRecord / tool_call_log below for the only
   place they're visible) and NOT capability-checked by aircore's Policy
   the way a declared workflow step would be. `identity=` (below) is a
   partial mitigation for the capability gap specifically, reusing
   aircore.Agent directly rather than reinventing permission logic -- but
   it's still bypassing the Scheduler's enforcement path entirely, which
   a future milestone might need to close properly.
2. Conversation history here is a plain list of dicts (OpenAI-style
   role/content messages), not a real aircore.Memory-integrated concept.
   Multi-turn *conversations* (as opposed to multi-turn *tool-calling
   within one execute() call*) still need the caller to manage that
   themselves, same as before this existed.

Memory-backed conversations (ModelAgent(memory=..., conversation_id=...)):

By default (no `memory`) every execute() call starts from a blank slate --
just `prompt` (or `prompt` + this call's own tool-calling scratch), same
as always. Passing a `memory` (any aircore MemoryScope -- Memory.session,
Memory.project, or Memory.temporary, or any other object with the same
get/set contract) and a `conversation_id` makes execute() load that
scope's prior history for this id, send it ahead of the new prompt, and
append this turn (the user prompt and the final assistant answer) back to
it afterward -- so a second execute() call on the same agent (or a
different ModelAgent sharing the same memory/conversation_id) continues
the conversation instead of repeating it from scratch.

Deliberately NOT persisted: the tool-calling loop's internal scratch
messages (intermediate tool_calls/tool results within a single execute()
call) -- only the outward-facing user prompt and final answer become part
of the conversation history. This keeps history compact and readable, at
the cost of a future turn not seeing exactly which tools were called to
produce a past answer, only what was asked and what was answered. `Memory.
temporary` technically works as a memory= target but defeats the purpose
for cross-run continuity, since Workflow.run() clears it after every
run -- `session` (same Memory object, kept alive across runs) or
`project` (shared by name) are the scopes this is meant for.

Structured output (ModelAgent(output_schema=...)):

When `output_schema` is given (a JSON-schema dict, or a Pydantic
BaseModel subclass), execute() returns a validated, typed value -- a dict
or a Pydantic instance -- instead of raw text, and the journal stores that
value as this step's output automatically. See structured_output.py for
the shared parse/validate pipeline (also used by JudgeConsensus). Works
with or without `tools`: in the tool-calling loop, the schema instruction
is added to every turn's request, and only the *final* answer (the turn
with no more tool_calls) is parsed against it.

Cross-step data flow (ModelAgent(prompt=PromptTemplate(...),
prompt_bindings=workflow.bindings)):

Closes the gap documented in cross-step-data-flow.md and aircore/
workflow.py's "Bindings" section -- a `prompt` no longer has to be a
fixed string decided at construction time. Pass a `PromptTemplate`
(prompt_template.py) instead of a plain string, plus `prompt_bindings=`
(a mutable dict reference -- normally the *same* dict object as
`workflow.bindings`, obtained via an earlier `workflow.step(other_tool,
as_="some_name")`), and the prompt is rendered fresh at execute() time
(not __init__ time), using whatever's actually in `prompt_bindings` at
that exact moment -- which, since the Scheduler writes into that same
dict in place as earlier steps complete, is however many prior steps
have actually finished by the time this one runs. Only the template's own
declared `{variable}` names are pulled from `prompt_bindings` (extra
unrelated bindings already in the dict are ignored, not rejected) --
render()'s existing strict "no missing, no extra" contract (prompt_
template.py) still applies to that filtered subset, so a template
referencing a variable that hasn't been bound yet (e.g. it names a step
that runs *later*, or never ran, or the name was misspelled) fails loudly
with PromptTemplateError at execute() time, not silently with a broken
prompt. A plain string `prompt` (the original, and still the default) is
completely unaffected by any of this -- prompt_bindings is simply unused
for a string prompt, whether or not it's passed.

Retry semantics change once `tools` is set (fixed after initial ship --
see the gap this closes):

With no tools, `idempotent`/`retries` mean exactly what they mean for a
Tool: the Scheduler may retry the whole execute() call, which is safe
because a single-shot generation call has no side effects to replay.

With tools, blindly letting the Scheduler retry the whole execute() call
would be unsafe -- a failure on turn 3 of a loop would cause the Scheduler
to call execute() again from turn 1, re-invoking whatever tools already
succeeded on turns 1-2, including side-effecting ones. So when `tools` is
non-empty, the Executable-visible `self.idempotent`/`self.retries` are
forced to False/0 (the Scheduler never retries the loop as a whole,
regardless of what was passed to __init__), and the developer's actual
`idempotent`/`retries` values are repurposed to control retrying just the
*model call* within a single turn, before any tool from that turn has
run -- which is safe to retry, the same way a single-shot call is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Union

from aircore.agent import Agent as Identity
from aircore.effects import Capability
from aircore.executable import Executable
from aircore.tools import Tool

from .prompt_template import PromptTemplate
from .providers import ModelProvider, ModelRequest, ModelResponse, ToolCallRequest
from .schema import tool_to_schema
from .structured_output import is_pydantic_model, parse_structured_output, schema_from


def _tool_call_to_wire_format(call: ToolCallRequest) -> dict:
    """Converts a ToolCallRequest back into the OpenAI-style wire format a
    real provider expects to see when it's echoed back as part of the
    assistant's turn in message history -- {id, type, function: {name,
    arguments-as-a-JSON-string}}. This is exactly the bug a live DeepSeek
    test caught: putting the raw ToolCallRequest object into `messages`
    instead of this shape produced a 400 from the real API ("tool_calls:
    empty array") that no mock ever exercised, because neither
    MockProvider nor the fake litellm module used in tests validated the
    wire format strictly enough to catch it."""
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
    }

RequiresArg = Union[Capability, Iterable[Capability], None]


class ModelAgentToolLoopExceeded(Exception):
    """Raised when a ModelAgent's tool-calling loop hits max_turns without
    the model returning a final answer (no more tool_calls in its
    response). Caught by the Scheduler like any other execute() failure --
    it becomes a normal failed step, retried if the agent is idempotent."""


@dataclass
class ToolCallRecord:
    """One entry in ModelAgent.tool_call_log -- the inspectable record of
    what happened inside the loop. NOT written to the aircore journal: these
    calls never pass through the Scheduler (see the module docstring's
    note on why), so this is the only place to see them after the fact."""
    name: str
    arguments: Dict[str, Any]
    result: Any
    error: Optional[str] = None


def _normalize_requires(requires: RequiresArg):
    if requires is None:
        return ()
    if isinstance(requires, Capability):
        return (requires,)
    return tuple(requires)


_SCHEMA_INSTRUCTION = """

Respond ONLY with valid JSON matching this schema, and nothing else \
(no explanation, no markdown code fences):

{schema_json}
"""


class ModelAgent(Executable):
    def __init__(self, name: str, provider: ModelProvider, prompt: Union[str, PromptTemplate],
                 model: str = "mock", idempotent: bool = True,
                 retries: int = 0, requires: RequiresArg = None,
                 tools: Optional[List[Tool]] = None, max_turns: int = 5,
                 identity: Optional[Identity] = None,
                 output_schema: Optional[Any] = None,
                 memory: Optional[Any] = None,
                 conversation_id: Optional[str] = None,
                 prompt_bindings: Optional[Dict[str, Any]] = None) -> None:
        # Model calls default to idempotent=True (unlike Tool's
        # idempotent=False default) because a read-only generation call is
        # normally safe to retry on a transient failure -- but this is
        # still just a default, not a guarantee; override it for any
        # provider/prompt combination that has real side effects.
        if retries > 0 and not idempotent:
            raise ValueError(
                f"ModelAgent '{name}' declares retries={retries} but idempotent=False -- "
                f"same rule as Tool: only retry calls that are actually safe to repeat."
            )
        if isinstance(prompt, PromptTemplate) and prompt_bindings is None:
            raise ValueError(
                f"ModelAgent '{name}' was given a PromptTemplate for `prompt` but no "
                f"`prompt_bindings` -- a template needs somewhere to pull its variables "
                f"from at execute() time (normally workflow.bindings; see this module's "
                f"docstring's 'Cross-step data flow' section)."
            )
        if not isinstance(prompt, (str, PromptTemplate)):
            raise TypeError(
                f"ModelAgent '{name}': prompt must be a str or a PromptTemplate, got {prompt!r}"
            )
        self.name = name
        self.provider = provider
        # See "Cross-step data flow" above. A plain string: rendered
        # exactly once, right here (unchanged behavior for every
        # ModelAgent built before this existed). A PromptTemplate: NOT
        # rendered here -- self._resolve_prompt() renders it fresh from
        # self._prompt_bindings at execute() time instead.
        self.prompt = prompt
        self._prompt_bindings = prompt_bindings
        self.model = model
        self.requires = _normalize_requires(requires)
        self.tools = tools or []
        self.max_turns = max_turns

        # output_schema (a JSON-schema dict, or a Pydantic BaseModel
        # subclass) is normalized once here, at construction -- a bad
        # schema fails fast (TypeError from schema_from), the same
        # fail-fast-at-construction rule Tool already applies to
        # retries/idempotent. When set, execute() returns a validated,
        # structured value (a dict, or a Pydantic instance) instead of raw
        # text -- see structured_output.py -- and that's what lands in the
        # journal as this step's output, automatically: aircore's Journal
        # already stores whatever an Executable returns, with no
        # awareness of what "structured output" even means.
        self.output_schema = output_schema
        self._output_json_schema = schema_from(output_schema) if output_schema is not None else None
        self._output_model_cls = output_schema if is_pydantic_model(output_schema) else None

        # See the module docstring's "Memory-backed conversations" section.
        # Validated at construction, fail-fast, same rule as everything
        # else in this constructor: a memory= without a conversation_id=
        # has no key to store history under, so it's a mistake, not a
        # valid "no memory" configuration (that's just memory=None).
        if memory is not None and conversation_id is None:
            raise ValueError(
                f"ModelAgent '{name}' was given memory= but no conversation_id= -- "
                f"a conversation needs a key to store its history under."
            )
        if memory is not None and not (hasattr(memory, "get") and hasattr(memory, "set")):
            raise TypeError(
                f"memory must be a MemoryScope (or any object with get(key, default) "
                f"and set(key, value)), got {memory!r}"
            )
        self.memory = memory
        self.conversation_id = conversation_id
        # Only used to gate which of `self.tools` this agent may actually
        # invoke during its own loop -- unrelated to (and not a substitute
        # for) the `agent=` identity a Workflow step checks when this
        # ModelAgent itself is used as a step.
        self.identity = identity
        self.last_response: Optional[ModelResponse] = None
        self.tool_call_log: List[ToolCallRecord] = []

        # See the module docstring's "Retry semantics change once `tools`
        # is set" section. `_loop_call_*` are always the developer's actual
        # values; `self.idempotent`/`self.retries` are what the Scheduler
        # sees, and get overridden below when tools are present.
        self._loop_call_idempotent = idempotent
        self._loop_call_retries = retries
        if self.tools:
            self.idempotent = False
            self.retries = 0
        else:
            self.idempotent = idempotent
            self.retries = retries

    def execute(self) -> Any:
        if not self.tools:
            # Unchanged single-shot behavior for a plain string prompt --
            # every ModelAgent built before this existed still does
            # exactly this (_resolve_prompt() returns self.prompt
            # unchanged, plus a schema instruction suffix when
            # output_schema is set, exactly like the old _request_prompt
            # did). With memory=, prior conversation history is sent as
            # `messages` ahead of the new prompt. With a PromptTemplate
            # prompt, this is where it actually gets rendered -- see the
            # module docstring's "Cross-step data flow" section.
            request = self._build_single_shot_request()
            response = self.provider.generate(request)
            self.last_response = response
            if self.memory is not None:
                self._remember_turn(response.content)
            return self._finalize_output(response.content)
        return self._run_tool_calling_loop()

    def _resolve_prompt(self) -> str:
        """Renders `self.prompt` into the actual text sent to the
        provider, plus the schema instruction suffix if output_schema was
        given. A plain string: returned unchanged (schema suffix aside).
        A PromptTemplate: rendered fresh, right now, against whatever's
        currently in `self._prompt_bindings` -- filtered to just the
        template's own declared variables, so unrelated bindings already
        in that dict (e.g. from steps this template doesn't care about)
        are ignored, not rejected; a variable the template DOES declare
        but that isn't bound yet raises PromptTemplateError, same as
        calling PromptTemplate.render() directly with a missing kwarg.
        Called freely within one execute()/stream() call -- nothing here
        is cached, but nothing mutates prompt_bindings mid-call either, so
        resolving more than once in the same call is safe, just slightly
        redundant."""
        if isinstance(self.prompt, PromptTemplate):
            available = {k: v for k, v in self._prompt_bindings.items() if k in self.prompt.variables}
            base = self.prompt.render(**available)
        else:
            base = self.prompt
        if self._output_json_schema is not None:
            schema_json = json.dumps(self._output_json_schema, indent=2)
            return base + _SCHEMA_INSTRUCTION.format(schema_json=schema_json)
        return base

    def _build_single_shot_request(self) -> ModelRequest:
        """Shared by execute() and stream() for the no-tools case: the
        resolved prompt (see _resolve_prompt()), plus prior conversation
        history as `messages` if memory= is set."""
        resolved_prompt = self._resolve_prompt()
        if self.memory is not None:
            messages = self._load_history() + [{"role": "user", "content": resolved_prompt}]
            return ModelRequest(prompt=resolved_prompt, messages=messages, model=self.model,
                                 response_schema=self._output_json_schema)
        return ModelRequest(prompt=resolved_prompt, model=self.model,
                             response_schema=self._output_json_schema)

    def stream(self, on_token: Optional[Callable[[str], None]] = None) -> Iterator[str]:
        """Yields text chunks as they arrive from the provider, instead of
        blocking for the whole completion the way execute() does. Two
        real limitations, not oversights:

        1. Bypasses aircore's Scheduler entirely, the same way ask()
           (ask.py) already does -- there is no single point in time a
           streamed answer "finished" that a Scheduler step's atomic
           StepStarted..StepFinished sequence could record, so this is
           explicitly a workflow-adjacent convenience, not something
           `workflow.step()` can run. Use execute() (via a normal
           workflow step) when you need journaling/Policy/capability
           enforcement; use stream() for direct, interactive use, the same
           tradeoff ask() already makes.
        2. Does not work with `tools` set -- raises NotImplementedError.
           Reconstructing tool_calls from a token stream (they can arrive
           split across chunks) is real additional complexity with no
           current use case driving it; a ModelAgent with tools should use
           execute() (or ask()) instead.

        `on_token`, if given, is called with each chunk as it arrives (for
        callers who'd rather register a callback than iterate). The
        underlying provider's `stream()` is always fully consumed here
        either way, so `self.last_response.content` is the complete,
        accumulated text once the generator is exhausted -- the same
        after-the-fact inspection point execute() already provides via
        last_response. Structured output (output_schema=) is still
        parsed/validated against the full accumulated text at the end,
        not incrementally; memory (memory=/conversation_id=), if set, is
        updated the same way execute() updates it, once the stream is
        fully consumed. Usage/cost are not tracked for a streamed call --
        see LiteLLMProvider.stream()'s docstring for why.

        The `tools` check below runs immediately, not lazily -- a plain
        `def` wrapping a `yield`-based generator would otherwise defer
        that check until the caller starts iterating (a common generator
        gotcha), which would make `agent.stream()` silently accept a bad
        call and only fail once consumed."""
        if self.tools:
            raise NotImplementedError(
                f"ModelAgent '{self.name}' has tools set -- streaming with tool-calling "
                f"is not supported. Use execute() instead."
            )
        return self._stream_impl(on_token)

    def _stream_impl(self, on_token: Optional[Callable[[str], None]]) -> Iterator[str]:
        request = self._build_single_shot_request()
        chunks: List[str] = []
        for chunk in self.provider.stream(request):
            chunks.append(chunk)
            if on_token is not None:
                on_token(chunk)
            yield chunk

        full_content = "".join(chunks)
        self.last_response = ModelResponse(content=full_content, model=self.model)
        if self.memory is not None:
            self._remember_turn(full_content)
        # Not returned (generators can't easily return a value callers
        # would naturally consume via `for chunk in agent.stream()`) --
        # available afterward via self.last_response.content, or, for
        # output_schema=, by calling self._finalize_output(full_content)
        # yourself. Raising here on a bad structured response would
        # surface after the loop has already been fully consumed by the
        # caller, same as any generator's deferred-execution semantics.
        if self._output_json_schema is not None:
            self._finalize_output(full_content)  # validates; raises StructuredOutputError if malformed

    def _load_history(self) -> List[dict]:
        """A shallow copy of this conversation's prior turns -- see the
        module docstring's "Memory-backed conversations" section. Only
        ever called when self.memory is not None."""
        return list(self.memory.get(self.conversation_id, []))

    def _remember_turn(self, assistant_content: str) -> None:
        """Appends this turn's user prompt and final assistant answer to
        the conversation's history and writes it back to memory. Re-reads
        history fresh (rather than reusing the `messages` list built for
        the API call) so that a tool-calling loop's internal scratch
        messages never get persisted -- only the outward-facing prompt and
        final answer do, per the module docstring."""
        history = self._load_history()
        history.append({"role": "user", "content": self._resolve_prompt()})
        history.append({"role": "assistant", "content": assistant_content or ""})
        self.memory.set(self.conversation_id, history)

    def conversation_history(self) -> List[dict]:
        """Public read-only view of this agent's conversation so far (empty
        if memory= wasn't set, or nothing has been said yet)."""
        return self._load_history() if self.memory is not None else []

    def _finalize_output(self, content: str) -> Any:
        """Applied to whatever text the provider returned as the final
        answer (single-shot, or the loop's last turn with no more
        tool_calls). Returns the raw text unchanged when no output_schema
        was requested -- the exact, unchanged behavior of every ModelAgent
        built before this existed. Raises StructuredOutputError (via
        parse_structured_output) when output_schema is set and the model's
        response doesn't parse/validate -- the Scheduler already handles
        this exactly like any other execute() failure."""
        if self._output_json_schema is None:
            return content
        return parse_structured_output(content, self._output_json_schema, self._output_model_cls)

    def _generate_with_retry(self, request: ModelRequest) -> ModelResponse:
        """Retries only the model call itself, using the developer's
        original idempotent/retries values -- safe because this runs
        before any tool from the current turn has been invoked, so a
        retry here never replays a tool call."""
        attempt = 0
        while True:
            try:
                return self.provider.generate(request)
            except Exception:
                if self._loop_call_idempotent and attempt < self._loop_call_retries:
                    attempt += 1
                    continue
                raise

    def _run_tool_calling_loop(self) -> Any:
        tools_by_name = {t.name: t for t in self.tools}
        schemas = [tool_to_schema(t) for t in self.tools]
        # Resolved once, up front -- every turn of this loop reuses the
        # same resolved prompt text (a PromptTemplate is rendered exactly
        # once per execute() call, not once per turn; nothing about
        # prompt_bindings changes mid-loop).
        resolved_prompt = self._resolve_prompt()
        # Seeded with prior conversation turns when memory= is set (never
        # the loop's own tool-calling scratch from a *previous* call --
        # that was deliberately never persisted, see _remember_turn).
        history = self._load_history() if self.memory is not None else []
        messages: List[dict] = list(history) + [{"role": "user", "content": resolved_prompt}]
        self.tool_call_log = []

        for _turn in range(self.max_turns):
            request = ModelRequest(prompt=resolved_prompt, messages=messages,
                                    model=self.model, tools=schemas,
                                    response_schema=self._output_json_schema)
            response = self._generate_with_retry(request)
            self.last_response = response

            if not response.tool_calls:
                if self.memory is not None:
                    self._remember_turn(response.content)
                return self._finalize_output(response.content)

            messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [_tool_call_to_wire_format(c) for c in response.tool_calls],
            })

            for call in response.tool_calls:
                result_text, error = self._invoke_tool(tools_by_name, call.name, call.arguments)
                self.tool_call_log.append(ToolCallRecord(
                    name=call.name, arguments=call.arguments, result=result_text, error=error,
                ))
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result_text,
                })

        raise ModelAgentToolLoopExceeded(
            f"'{self.name}' exceeded max_turns={self.max_turns} without a final answer "
            f"(the model kept requesting tool calls)"
        )

    def _invoke_tool(self, tools_by_name: Dict[str, Tool], name: str, arguments: Dict[str, Any]):
        """Returns (result_text, error). Never raises -- a tool failure or
        a capability denial becomes text fed back to the model (so it can
        adapt, e.g. try different arguments or give up), not a crash of
        the whole loop."""
        tool = tools_by_name.get(name)
        if tool is None:
            error = f"no such tool '{name}'"
            return f"Error: {error}", error

        if self.identity is not None and tool.requires:
            missing = self.identity.missing(tool.requires)
            if missing:
                names = ", ".join(c.name for c in missing)
                error = f"identity '{self.identity.name}' lacks capability/capabilities [{names}]"
                return f"Error: {error}", error

        try:
            result = tool(**arguments)
            return str(result), None
        except Exception as exc:  # noqa: BLE001 -- fed back to the model as text, not re-raised
            error = f"{type(exc).__name__}: {exc}"
            return f"Error calling {name}: {error}", error

    def usage(self) -> Optional[Dict[str, Any]]:
        """Implements Executable's generic usage() hook (see executable.py).
        Reports whatever numeric usage the last response actually carried --
        MockProvider leaves these as None, so a mock-backed ModelAgent
        reports nothing (the scheduler skips emitting UsageReported
        entirely when every value is None), while a real provider like
        LiteLLMProvider populates them from the actual API response.

        Deliberately excludes latency: duration_ms is already recorded for
        every step in the journal regardless of what kind of Executable it
        is, so repeating it here would be redundant, and worse, would make
        a mock-backed ModelAgent always report "usage" even when nothing
        meaningful was actually measured."""
        if self.last_response is None:
            return None
        usage = self.last_response.usage
        values = {
            "tokens_in": usage.prompt_tokens,
            "tokens_out": usage.completion_tokens,
            "cost_usd": usage.cost_usd,
        }
        reported = {k: v for k, v in values.items() if v is not None}
        return reported or None

    def __repr__(self) -> str:
        return f"<ModelAgent {self.name} model={self.model!r}>"
