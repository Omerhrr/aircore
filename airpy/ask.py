"""ask(): the one-line convenience for calling a model outside a Workflow.

This intentionally does NOT go through the Scheduler -- there's no
journal, no policy check, no capability check, no retry, when you call
ask() directly. It's sugar for "I just want a quick answer right now,"
not a replacement for putting a ModelAgent in a Workflow when you want
those guarantees. If you need any of that, use `workflow.step(ModelAgent(...))`
instead.
"""

from __future__ import annotations

from typing import Union

from .model_agent import ModelAgent
from .providers import ModelProvider, ModelRequest


def ask(target: Union[ModelAgent, ModelProvider], prompt: str | None = None,
        model: str = "mock") -> str:
    if isinstance(target, ModelAgent):
        if prompt is not None:
            raise TypeError(
                "ask(agent) already has its prompt bound -- pass prompt= only "
                "when calling ask() with a raw ModelProvider."
            )
        return target.execute()

    if isinstance(target, ModelProvider):
        if prompt is None:
            raise ValueError("ask(provider, prompt=...) requires a prompt.")
        response = target.generate(ModelRequest(prompt=prompt, model=model))
        return response.content

    raise TypeError(f"ask() expects a ModelAgent or ModelProvider, got {type(target).__name__}")
