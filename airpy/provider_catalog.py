"""Named provider constructors: openai(), anthropic(), deepseek(), gemini(),
qwen(), nvidia(), zai(), ollama(), lmstudio(), openrouter().

Every one of these is a thin constructor over LiteLLMProvider -- not a new
adapter, not a new dependency, not new request/response handling. LiteLLM
already reaches all ten of these providers through one call shape (see
litellm_provider.py); the only real gap was developer experience --
remembering that Anthropic needs `model="anthropic/claude-..."`, that
Qwen's provider prefix is `dashscope/` (not `qwen/`), that Nvidia NIM's is
`nvidia_nim/`, and so on, is exactly the kind of detail FastAPI-over-
Starlette hides that this module hides here. `airpy.providers.anthropic
("claude-3-5-sonnet-20241022")` reads the way you'd want to write it;
`LiteLLMProvider(model="anthropic/claude-3-5-sonnet-20241022")` is what
actually runs underneath, unchanged.

Building ten separate native-SDK adapter classes instead (own auth, own
streaming/tool-call parsing per provider) was explicitly considered and
rejected -- see the roadmap discussion this came out of. That would only
be justified by a provider-specific capability LiteLLM's unified interface
doesn't expose (e.g. Anthropic prompt caching, OpenAI's Responses API);
none of that has a real use case driving it yet, so it isn't built.

Model-string prefixes below were verified against LiteLLM's own current
provider docs (docs.litellm.ai/docs/providers/<name>), not guessed:
openai (no prefix), anthropic/, deepseek/, gemini/, dashscope/ (Qwen),
nvidia_nim/, zai/, ollama/, lm_studio/, openrouter/. Being honest about
what that verification does and doesn't cover: no API keys for any of
these ten providers were available in this environment, so -- unlike the
DeepSeek tool-calling path, which was validated against the real API
earlier in this project -- none of these ten constructors have been
exercised against a real network call here. Each one is covered by an
offline test asserting the resulting LiteLLMProvider.model string is
correct (the thing this module is actually responsible for getting
right); actually reaching each provider's API also depends on
LiteLLMProvider/litellm's request handling, already covered by M8's
tests, and on the right API key being set in the environment
(OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY,
DASHSCOPE_API_KEY, NVIDIA_NIM_API_KEY, ZAI_API_KEY, OPENROUTER_API_KEY --
Ollama and LM Studio run locally and need no key, just a running server).

Every function forwards **litellm_kwargs straight to LiteLLMProvider
(api_key=, api_base=, temperature=, etc.) -- nothing here is hidden or
hardcoded beyond the provider prefix and a reasonable default model name.

Note: `openai()` here still returns a LiteLLMProvider, unchanged --
openai_provider.py's OpenAIProvider (a separate, native adapter added
later, using the `openai` package directly instead of litellm) is not
wired into this catalog. It's constructed directly (`OpenAIProvider(
model=..., api_key=..., base_url=...)`), not through a catalog function,
since its main point is being a second *independent* adapter proving
ModelProvider isn't secretly litellm-shaped -- routing it through the
same catalog pattern as the other nine LiteLLM-backed constructors would
undercut that. See openai_provider.py's module docstring.
"""

from __future__ import annotations

from typing import Any

from .litellm_provider import LiteLLMProvider


def openai(model: str = "gpt-4o-mini", **litellm_kwargs: Any) -> LiteLLMProvider:
    """OpenAI needs no provider prefix -- litellm treats a bare model name
    as OpenAI by default. Requires OPENAI_API_KEY."""
    return LiteLLMProvider(model=model, **litellm_kwargs)


def anthropic(model: str = "claude-3-5-sonnet-20241022", **litellm_kwargs: Any) -> LiteLLMProvider:
    """Requires ANTHROPIC_API_KEY."""
    return LiteLLMProvider(model=f"anthropic/{model}", **litellm_kwargs)


def deepseek(model: str = "deepseek-chat", **litellm_kwargs: Any) -> LiteLLMProvider:
    """Requires DEEPSEEK_API_KEY. The one provider in this list whose
    tool-calling wire format was actually live-tested earlier in this
    project (see model_agent.py's _tool_call_to_wire_format docstring) --
    that validation predates this constructor but exercises the same
    LiteLLMProvider path underneath."""
    return LiteLLMProvider(model=f"deepseek/{model}", **litellm_kwargs)


def gemini(model: str = "gemini-1.5-pro", **litellm_kwargs: Any) -> LiteLLMProvider:
    """Google AI Studio, via litellm's `gemini/` prefix. Requires
    GEMINI_API_KEY."""
    return LiteLLMProvider(model=f"gemini/{model}", **litellm_kwargs)


def qwen(model: str = "qwen-plus", **litellm_kwargs: Any) -> LiteLLMProvider:
    """Alibaba Cloud's Qwen models, via litellm's Dashscope integration --
    the provider prefix is `dashscope/`, not `qwen/` (Dashscope is the
    name of Alibaba's API product; Qwen is the model family it serves).
    Requires DASHSCOPE_API_KEY."""
    return LiteLLMProvider(model=f"dashscope/{model}", **litellm_kwargs)


def nvidia(model: str = "meta/llama3-70b-instruct", **litellm_kwargs: Any) -> LiteLLMProvider:
    """Nvidia NIM. Requires NVIDIA_NIM_API_KEY. Default model is a NIM
    catalog example, not a claim about what you should actually run --
    pass whatever model= you've deployed/subscribed to."""
    return LiteLLMProvider(model=f"nvidia_nim/{model}", **litellm_kwargs)


def zai(model: str = "glm-4.7", **litellm_kwargs: Any) -> LiteLLMProvider:
    """Z.AI (Zhipu AI)'s GLM models. Requires ZAI_API_KEY."""
    return LiteLLMProvider(model=f"zai/{model}", **litellm_kwargs)


def ollama(model: str = "llama3", **litellm_kwargs: Any) -> LiteLLMProvider:
    """Local models served by Ollama. No API key -- needs a running
    `ollama serve` (default http://localhost:11434); pass api_base= to
    point elsewhere."""
    return LiteLLMProvider(model=f"ollama/{model}", **litellm_kwargs)


def lmstudio(model: str = "local-model", **litellm_kwargs: Any) -> LiteLLMProvider:
    """Local models served by LM Studio's built-in server. No API key --
    needs LM Studio's server running (default http://localhost:1234/v1);
    pass api_base= to point elsewhere. `model` should match whatever LM
    Studio reports the loaded model as."""
    return LiteLLMProvider(model=f"lm_studio/{model}", **litellm_kwargs)


def openrouter(model: str, **litellm_kwargs: Any) -> LiteLLMProvider:
    """OpenRouter proxies many providers/models behind one API -- unlike
    the others here, there's no sane default model (OpenRouter's whole
    point is choosing among many), so `model` is required, in OpenRouter's
    own `vendor/model` form, e.g. `openrouter("anthropic/claude-3.5-
    sonnet")`. Requires OPENROUTER_API_KEY."""
    return LiteLLMProvider(model=f"openrouter/{model}", **litellm_kwargs)
