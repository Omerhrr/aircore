"""MockProvider: a ModelProvider that needs no API key and no network.

This exists to prove the ModelProvider/ModelAgent abstraction end to end
without requiring credentials or making real (costly, non-deterministic)
API calls. A real adapter (OpenAI, Anthropic, etc.) implements the exact
same interface -- ModelAgent, Workflow, the Scheduler, none of them change
when a real provider is swapped in.
"""

from __future__ import annotations

import time
from typing import Callable, Iterator, List, Optional, Union

from .providers import ModelProvider, ModelRequest, ModelResponse

ResponseSource = Union[str, Callable[[ModelRequest], str]]
ScriptedItem = Union[str, ModelResponse, Callable[[ModelRequest], "str | ModelResponse"]]
_UNSET = object()


class MockProvider(ModelProvider):
    def __init__(self, response: ResponseSource = _UNSET, latency_ms: float = 0.0,
                 responses: Optional[List[ScriptedItem]] = None) -> None:
        """Two modes, mutually exclusive:

        - `response=`: every call returns the same thing (a fixed string,
          or a function of the request). This is the normal case.
        - `responses=`: a script consumed one item per call, in order --
          needed to test the tool-calling loop, which calls generate()
          multiple times per execute() (once to get a tool-call request,
          again after the tool result is fed back). Each item is a plain
          string (a final answer), a full ModelResponse (e.g. one with
          tool_calls set, to simulate the model asking to call a tool), or
          a function of the request returning either. The last item
          repeats if generate() is called more times than the script has
          entries."""
        if response is not _UNSET and responses is not None:
            raise ValueError("pass either response= or responses=, not both")
        self._response = "mock response" if response is _UNSET else response
        self._responses = responses
        self._call_index = 0
        self._latency_ms = latency_ms

    def generate(self, request: ModelRequest) -> ModelResponse:
        if self._latency_ms:
            time.sleep(self._latency_ms / 1000)

        if self._responses is not None:
            index = min(self._call_index, len(self._responses) - 1)
            self._call_index += 1
            item = self._responses[index]
            item = item(request) if callable(item) else item
            if isinstance(item, ModelResponse):
                return item
            return ModelResponse(content=item, model=request.model, latency_ms=self._latency_ms)

        content = self._response(request) if callable(self._response) else self._response
        return ModelResponse(content=content, model=request.model, latency_ms=self._latency_ms)

    def stream(self, request: ModelRequest) -> Iterator[str]:
        """Simulates token streaming by chunking the same content generate()
        would have returned into whitespace-separated pieces -- good
        enough to exercise ModelAgent.stream()'s accumulation logic in
        tests without needing a real provider's real streaming API."""
        content = self._resolve_content(request)
        chunks = content.split(" ")
        for i, chunk in enumerate(chunks):
            if self._latency_ms:
                time.sleep(self._latency_ms / 1000 / max(len(chunks), 1))
            yield chunk if i == 0 else " " + chunk

    def _resolve_content(self, request: ModelRequest) -> str:
        """Shared by generate() and stream() -- the same content a
        non-streaming call would have returned, still consuming one item
        from `responses=`'s script if that mode is in use, so a test
        mixing generate() and stream() calls on one MockProvider still
        advances through the script in call order."""
        if self._responses is not None:
            index = min(self._call_index, len(self._responses) - 1)
            self._call_index += 1
            item = self._responses[index]
            item = item(request) if callable(item) else item
            return item.content if isinstance(item, ModelResponse) else item

        content = self._response(request) if callable(self._response) else self._response
        return content
