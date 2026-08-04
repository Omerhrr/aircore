"""airpy's provider catalog (provider_catalog.py): named constructors --
openai/anthropic/deepseek/gemini/qwen/nvidia/zai/ollama/lmstudio/
openrouter -- each a thin wrapper over LiteLLMProvider. What these tests
actually prove, and what they don't:

- each constructor builds a LiteLLMProvider whose .model carries the
  correct litellm provider prefix (the one thing this module is
  responsible for getting right), and forwards arbitrary litellm_kwargs
  (api_key=, api_base=, etc.) through unchanged
- no real network call is made or needed here -- LiteLLMProvider.__init__
  only imports and configures litellm, it doesn't call out; the real
  `litellm` package IS installed in this environment (unlike some earlier
  tests in this suite that fake it out), so constructing these directly
  exercises the same import path a real deployment would

What this deliberately does NOT prove: that any of these ten providers'
APIs actually accept these model strings and respond correctly. No API
keys for openai/anthropic/deepseek/gemini/qwen/nvidia/zai/openrouter were
available in this environment, and ollama/lmstudio need a locally running
server that also isn't available here -- see provider_catalog.py's
docstring for the honest breakdown of what has (DeepSeek, in an earlier
live test) and hasn't (these ten constructors specifically) been
exercised against a real API.
"""

import pytest

litellm = pytest.importorskip("litellm", reason="requires the real litellm package")

from airpy import (
    LiteLLMProvider,
    anthropic, deepseek, gemini, lmstudio, nvidia, ollama, openai, openrouter, qwen, zai,
)


def test_openai_needs_no_prefix():
    provider = openai()
    assert isinstance(provider, LiteLLMProvider)
    assert provider.model == "gpt-4o-mini"


def test_openai_custom_model():
    assert openai("gpt-4o").model == "gpt-4o"


def test_anthropic_prefix():
    assert anthropic().model == "anthropic/claude-3-5-sonnet-20241022"
    assert anthropic("claude-3-opus-20240229").model == "anthropic/claude-3-opus-20240229"


def test_deepseek_prefix():
    assert deepseek().model == "deepseek/deepseek-chat"
    assert deepseek("deepseek-reasoner").model == "deepseek/deepseek-reasoner"


def test_gemini_prefix():
    assert gemini().model == "gemini/gemini-1.5-pro"


def test_qwen_uses_dashscope_prefix_not_qwen():
    # The one prefix in this catalog that doesn't match its own name --
    # litellm routes Qwen models through its Dashscope integration.
    assert qwen().model == "dashscope/qwen-plus"
    assert qwen("qwen-max").model == "dashscope/qwen-max"


def test_nvidia_uses_nvidia_nim_prefix():
    assert nvidia().model.startswith("nvidia_nim/")


def test_zai_prefix():
    assert zai().model == "zai/glm-4.7"


def test_ollama_prefix():
    assert ollama().model == "ollama/llama3"
    assert ollama("mistral").model == "ollama/mistral"


def test_lmstudio_prefix():
    assert lmstudio().model == "lm_studio/local-model"


def test_openrouter_requires_an_explicit_model():
    with pytest.raises(TypeError):
        openrouter()  # no sane default -- OpenRouter's whole point is choosing among many
    assert openrouter("anthropic/claude-3.5-sonnet").model == "openrouter/anthropic/claude-3.5-sonnet"


def test_kwargs_forward_through_to_litellm_provider():
    provider = anthropic("claude-3-5-sonnet-20241022", api_key="sk-test-123", temperature=0.2)
    assert provider.litellm_kwargs["api_key"] == "sk-test-123"
    assert provider.litellm_kwargs["temperature"] == 0.2
