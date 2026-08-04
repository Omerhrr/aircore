"""LiteLLMProvider: the first real ModelProvider adapter.

LiteLLM was chosen over a single native SDK (per the M8 decision) because
it gives access to many providers (OpenAI, Anthropic, Gemini, Ollama,
etc.) through one call shape, while native adapters can still be added
later for provider-specific features LiteLLM doesn't expose.

`litellm` is imported lazily, inside __init__, not at module level --
airpy itself has no hard dependency on it. Only constructing a
LiteLLMProvider requires it to be installed (`pip install litellm`);
importing airpy, or using MockProvider, never does.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator, List, Optional

from .providers import ModelProvider, ModelRequest, ModelResponse, ToolCallRequest, Usage


class LiteLLMProvider(ModelProvider):
    def __init__(self, model: str = "gpt-4o-mini", **litellm_kwargs: Any) -> None:
        try:
            import litellm
        except ImportError as exc:
            raise ImportError(
                "LiteLLMProvider requires the 'litellm' package. Install it with: "
                "pip install litellm"
            ) from exc
        self._litellm = litellm
        self.model = model
        self.litellm_kwargs = litellm_kwargs

    def generate(self, request: ModelRequest) -> ModelResponse:
        model = request.model if request.model != "mock" else self.model
        messages = request.messages if request.messages is not None else [
            {"role": "user", "content": request.prompt}
        ]

        completion_kwargs = dict(model=model, messages=messages, **self.litellm_kwargs)
        if request.tools:
            completion_kwargs["tools"] = [
                {"type": "function", "function": {
                    "name": t.name, "description": t.description, "parameters": t.parameters,
                }}
                for t in request.tools
            ]
        if request.response_schema is not None:
            # Best-effort hint to the real API that JSON is expected --
            # not every provider/model litellm proxies to honors this, and
            # none of them are asked to validate the *shape* of the JSON
            # against request.response_schema specifically. That
            # validation happens uniformly in airpy regardless of
            # provider -- see structured_output.py, used by ModelAgent and
            # JudgeConsensus -- so this is purely an accuracy nudge, never
            # a correctness dependency.
            completion_kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        response = self._litellm.completion(**completion_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        content = choice.message.content
        finish_reason = getattr(choice, "finish_reason", None)
        tool_calls = self._parse_tool_calls(choice.message)

        raw_usage = getattr(response, "usage", None)
        usage = Usage(
            prompt_tokens=getattr(raw_usage, "prompt_tokens", None) if raw_usage else None,
            completion_tokens=getattr(raw_usage, "completion_tokens", None) if raw_usage else None,
            total_tokens=getattr(raw_usage, "total_tokens", None) if raw_usage else None,
            cost_usd=self._extract_cost(response),
        )

        return ModelResponse(
            content=content,
            model=getattr(response, "model", model),
            latency_ms=latency_ms,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw=response,
        )

    def stream(self, request: ModelRequest) -> Iterator[str]:
        """Real token streaming via litellm's `stream=True`. Scoped to
        plain text content only for this first pass -- tool_calls arrive
        split across chunks too and reconstructing them mid-stream is real
        additional complexity (see ModelAgent.stream()'s docstring for why
        the tool-calling loop is explicitly out of scope here); a request
        with `tools=` set still works with generate(), just not stream().
        Usage/cost are not available from a streamed response in the same
        way generate() gets them -- ModelAgent.stream() does not call
        usage() after streaming for this reason."""
        model = request.model if request.model != "mock" else self.model
        messages = request.messages if request.messages is not None else [
            {"role": "user", "content": request.prompt}
        ]
        completion_kwargs = dict(model=model, messages=messages, stream=True, **self.litellm_kwargs)

        for chunk in self._litellm.completion(**completion_kwargs):
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content

    def _parse_tool_calls(self, message: Any) -> Optional[List[ToolCallRequest]]:
        raw_calls = getattr(message, "tool_calls", None)
        if not raw_calls:
            return None
        parsed = []
        for call in raw_calls:
            try:
                arguments = json.loads(call.function.arguments)
            except (json.JSONDecodeError, AttributeError, TypeError):
                # A malformed arguments payload shouldn't crash the whole
                # response -- surface it as an empty-args call; ModelAgent's
                # tool invocation will then fail loudly against the tool's
                # real signature instead of silently here.
                arguments = {}
            parsed.append(ToolCallRequest(
                id=getattr(call, "id", ""),
                name=call.function.name,
                arguments=arguments,
            ))
        return parsed

    def _extract_cost(self, response: Any) -> Optional[float]:
        """litellm.completion_cost() needs per-model pricing data it
        doesn't have for every model/provider combination -- this is best
        effort, and returning None (rather than 0.0) when it fails matters:
        Policy.max_cost and Metrics.usage_totals only sum values that are
        actually reported, so a None here means "unknown," not "free.\""""
        try:
            return self._litellm.completion_cost(completion_response=response)
        except Exception:
            return None
