"""ModelProvider: the one interface airpy needs from any model backend.

The runtime (aircore) never sees a ChatCompletion, a Responses API payload,
a Claude Messages object, or a Gemini GenerateContent response -- every
provider adapter normalizes to ModelRequest in, ModelResponse out. Adding
a new provider means implementing this one method; nothing else in airpy
or aircore needs to change.

ModelResponse's shape (content/usage/finish_reason/tool_calls/
structured_output/raw) was deliberately NOT speculatively designed ahead
of a real provider -- it's shaped by what LiteLLMProvider (litellm.py)
actually gets back from a real completion call. `text` was renamed to
`content` and `tokens_in`/`tokens_out` were consolidated into `usage` as
part of that -- see architecture-spec-v1.md's addendum on why fields
weren't added until they had a real source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class ToolSchema:
    """What a provider needs to offer a tool to a model. Built from an
    aircore.Tool's Python signature by airpy/schema.py -- providers.py itself
    has no idea what an aircore.Tool is, keeping this module's only
    dependency the shape of a request/response, not aircore."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema for the tool's argument object


@dataclass
class ToolCallRequest:
    """A model's request to call one tool, already parsed -- provider
    adapters are responsible for turning whatever raw shape their SDK
    returns (e.g. a JSON string of arguments) into this."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ModelRequest:
    prompt: str
    model: str = "mock"
    # If given, takes precedence over `prompt` for providers that support
    # multi-turn history (needed for the tool-calling loop, which has to
    # send back what the model said and what a tool returned). `prompt`
    # stays the primary field for simple, single-shot calls -- nothing
    # built before the tool-calling loop existed constructs `messages`.
    messages: Optional[List[dict]] = None
    tools: Optional[List[ToolSchema]] = None
    # Set by ModelAgent(output_schema=...) / JudgeConsensus(output_schema=
    # ..., confidence=...) -- see structured_output.py. A provider adapter
    # MAY use this as a hint to enable a real JSON-mode API parameter
    # (LiteLLMProvider does); it is never required to, and never expected
    # to validate the schema itself -- parsing/validation happens
    # uniformly in airpy (structured_output.py), the same way for every
    # provider, so a provider adapter that ignores this field entirely
    # still works, just without the extra hint.
    response_schema: Optional[Dict[str, Any]] = None


@dataclass
class Usage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


@dataclass
class ModelResponse:
    content: str
    model: str = "mock"
    latency_ms: float = 0.0
    usage: Usage = field(default_factory=Usage)
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[ToolCallRequest]] = None
    # Not populated by any provider adapter yet -- no adapter attempts
    # structured output parsing in this pass. Present now because the
    # shape is what a real provider exposes; wiring it up is separate,
    # later work.
    structured_output: Any = None
    raw: Any = None  # the untouched provider payload, for debugging only


class ModelProvider(ABC):
    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError

    def stream(self, request: ModelRequest) -> Iterator[str]:
        """Optional. Yields text chunks as they arrive instead of blocking
        for the whole completion -- used by ModelAgent.stream() (see
        model_agent.py), which lives entirely in airpy and bypasses the
        Scheduler the same way ask() already does, since true token-level
        streaming doesn't fit atomic per-step journaling (one step has one
        recorded output, produced at one point in time). Deliberately not
        `@abstractmethod`: a provider adapter that never implements this
        (the base default just raises) still satisfies ModelProvider for
        every non-streaming use -- generate() is the only real contract.
        MockProvider and LiteLLMProvider both implement it."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")
