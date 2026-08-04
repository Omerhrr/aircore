"""OpenAIProvider: the first native (non-LiteLLM) ModelProvider adapter.

Why this exists: LiteLLMProvider is currently the *only* real adapter --
nothing in this project proves ModelProvider is actually adapter-agnostic
as opposed to secretly shaped around however litellm happens to normalize
things. Building a second, independent adapter against the `openai`
package directly (no litellm involved at all) is what actually tests
that. `openai`'s official Python SDK was chosen over hand-rolled HTTP
calls because it's the standard client most OpenAI-compatible providers
(DeepSeek included -- see below) already document against.

Lazy import, same pattern as LiteLLMProvider: `openai` is imported inside
__init__, not at module level, so airpy has no hard dependency on it.
Only constructing an OpenAIProvider requires it installed (`pip install
openai`); importing airpy, or using MockProvider/LiteLLMProvider, never
does.

`base_url=` is the whole reason this can be live-tested without a new API
key: DeepSeek's API is OpenAI-compatible (same request/response shape),
so `OpenAIProvider(model="deepseek-chat", api_key=os.environ[
"DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")` talks to the
same DeepSeek account this project already has credentials for, through
`openai`'s client instead of litellm's. This is also, generically, how
any other OpenAI-compatible endpoint (Ollama's OpenAI-compatible route,
vLLM, etc.) would be reached with this adapter -- not DeepSeek-specific
code, just DeepSeek-shaped test coverage.

Honest gap relative to LiteLLMProvider: `openai`'s SDK has no equivalent
of `litellm.completion_cost()` -- no built-in per-model pricing lookup --
so `Usage.cost_usd` is always None here, never a guess. Policy.max_cost
and Metrics.usage_totals already treat None as "unknown," not "free" (see
providers.py's Usage docstring and litellm_provider.py's
`_extract_cost`), so this is a real, silently-correct degradation, not a
bug: a workflow relying on OpenAIProvider for cost enforcement gets no
enforcement, and that's visible (usage_totals simply won't include a
cost_usd figure for these calls) rather than wrong.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator, List, Optional

from .providers import ModelProvider, ModelRequest, ModelResponse, ToolCallRequest, Usage


class OpenAIProvider(ModelProvider):
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, **client_kwargs: Any) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "OpenAIProvider requires the 'openai' package. Install it with: "
                "pip install openai"
            ) from exc
        # api_key=None/base_url=None fall through to openai's own defaults
        # (OPENAI_API_KEY env var, OpenAI's real endpoint) -- passing None
        # explicitly is the same as not passing the kwarg at all, per the
        # SDK's own constructor.
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, **client_kwargs)
        self.model = model

    def generate(self, request: ModelRequest) -> ModelResponse:
        model = request.model if request.model != "mock" else self.model
        messages = request.messages if request.messages is not None else [
            {"role": "user", "content": request.prompt}
        ]

        completion_kwargs = dict(model=model, messages=messages)
        if request.tools:
            completion_kwargs["tools"] = [
                {"type": "function", "function": {
                    "name": t.name, "description": t.description, "parameters": t.parameters,
                }}
                for t in request.tools
            ]
        if request.response_schema is not None:
            # Same best-effort accuracy nudge as LiteLLMProvider -- never a
            # correctness dependency, see providers.py's ModelRequest
            # docstring. Not every OpenAI-compatible endpoint honors this
            # (DeepSeek's does).
            completion_kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        response = self._client.chat.completions.create(**completion_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        content = choice.message.content
        finish_reason = choice.finish_reason
        tool_calls = self._parse_tool_calls(choice.message)

        raw_usage = response.usage
        usage = Usage(
            prompt_tokens=getattr(raw_usage, "prompt_tokens", None) if raw_usage else None,
            completion_tokens=getattr(raw_usage, "completion_tokens", None) if raw_usage else None,
            total_tokens=getattr(raw_usage, "total_tokens", None) if raw_usage else None,
            # No pricing lookup in the openai SDK -- see this module's
            # docstring. None means unknown, not free.
            cost_usd=None,
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
        """Real token streaming via the SDK's `stream=True`. Same scope cut
        as LiteLLMProvider.stream(): plain text content only, not usable
        together with `tools=` (see ModelAgent.stream()'s docstring for
        why the tool-calling loop is out of scope for streaming
        entirely), and no usage/cost afterward (a streamed response
        doesn't carry it the way generate() does)."""
        model = request.model if request.model != "mock" else self.model
        messages = request.messages if request.messages is not None else [
            {"role": "user", "content": request.prompt}
        ]
        stream = self._client.chat.completions.create(model=model, messages=messages, stream=True)
        for chunk in stream:
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
                # Same defensive fallback as LiteLLMProvider._parse_tool_calls
                # -- a malformed arguments payload shouldn't crash the whole
                # response; ModelAgent's tool invocation fails loudly
                # against the tool's real signature instead of silently
                # here.
                arguments = {}
            parsed.append(ToolCallRequest(
                id=getattr(call, "id", ""),
                name=call.function.name,
                arguments=arguments,
            ))
        return parsed
