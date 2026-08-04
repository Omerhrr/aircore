# aircore / airpy / airlang / aircli

A provider-agnostic AI execution runtime, split into four independently
installable packages with a strict, one-directional dependency chain:

```
aircore   <-  airpy   <-  airlang
   ^           ^             ^
   └───────────┴─────────────┘
              aircli
```

- **`aircore`** -- the execution runtime: scheduler, capabilities, policy,
  journal, consensus, memory, approval, checkpoint/resume, sandboxed
  execution, cross-step data flow. No model, prompt, or provider concept
  exists anywhere in this package, by design. Zero dependencies.
- **`airpy`** -- the provider-aware Python SDK on top of `aircore`:
  `ModelAgent`, real provider adapters (`LiteLLMProvider`, native
  `OpenAIProvider`), consensus strategies, structured output, an MCP tool
  registry, streaming, long-running sessions, `PromptTemplate`. Imports
  `aircore`; `aircore` never imports `airpy`.
- **`airlang`** -- AirLang, a declarative workflow language: lexer,
  parser, IR, and an executor that compiles a parsed `.airlang` file to
  real `airpy`/`aircore` calls. Imports `airpy`; `airpy` never imports
  `airlang`. See `airlang-spec-v1.md`.
- **`aircli`** -- the `ai` command: `ai run`/`ai trace` for both `.py`
  scripts and `.airlang` files, with `--json`/`--html` trace output.
  Depends on all three of the above.

## Install

`airpy`'s and `aircli`'s PyPI distribution names differ from their
import names -- `airpy`/`aircli` were both too close to existing PyPI
project names and got rejected, so they're published as `airpyy` /
`airclii`. You still `import airpy` and run the `ai` command exactly as
before; only the `pip install` name changes.

```
pip install aircore           # just the runtime
pip install airpyy            # runtime + SDK           (import as: airpy)
pip install airlang           # runtime + SDK + language frontend
pip install airclii           # everything, plus the `ai` command
```

Optional extras on `airpy`: `litellm`, `pydantic`, `mcp`, `openai` --
each is a lazy import inside the one module that needs it, so none of
them are required just to install `airpy` itself. (`pip install
"airpyy[litellm]"` etc.)

For local development on this repo (all four packages, editable,
`pip install -e .` style), see the root `pyproject.toml` instead of the
per-package ones under `packaging/` -- those are used only by the
release workflow (`.github/workflows/publish.yml`) to build the four
separate PyPI distributions.

## Docs

- `architecture-spec-v1.md` -- the runtime's original design spec.
- `airlang-spec-v1.md` -- AirLang's design spec.
- `cross-step-data-flow.md` -- the bindings/`let` design writeup.
- `real-project-readiness-roadmap.md` -- known gaps and what's planned.
