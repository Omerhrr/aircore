"""Bindings: how a name in a .airlang file (a `tool`, a `schema`, a non-catalog
`provider`, a custom `capability`) becomes a real Python object at
execution time -- see airlang-spec-v1.md section 6. AirLang cannot author a
tool's implementation, a schema's shape, or a capability class (section 2
-- no arbitrary code), so every such name is a *reference* that must be
resolved against objects the host environment supplies.

Two ways to supply a Bindings:
- Build one directly in Python (`Bindings(tools={...})`) -- what the
  executor's own tests do, and what any code calling execute_ir()/
  build_workflow() directly (not through the CLI) will normally do.
- The file convention (`load_bindings_for`): running `audit.airlang` looks
  for a sibling `audit.airlang.py` exposing module-level `TOOLS` / `SCHEMAS`
  / `PROVIDERS` / `CAPABILITIES` dicts. This is what `ai run audit.airlang`
  (AirLang-M2) uses so a .airlang file's tool implementations live in a normal,
  inspectable Python file right next to it, not some hidden registry.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Bindings:
    tools: Dict[str, Any] = field(default_factory=dict)
    schemas: Dict[str, Any] = field(default_factory=dict)
    providers: Dict[str, Any] = field(default_factory=dict)
    capabilities: Dict[str, Any] = field(default_factory=dict)


def load_bindings_for(airlang_path: str) -> Bindings:
    """Looks for `<airlang_path>.py` (e.g. `audit.airlang` -> `audit.airlang.py`).
    Returns an empty Bindings (not an error) if it doesn't exist -- a
    .airlang file that only uses builtin capabilities and catalog providers,
    with no `tool`/`schema` references, needs no bindings file at all."""
    bindings_path = airlang_path + ".py"
    if not os.path.exists(bindings_path):
        return Bindings()

    spec = importlib.util.spec_from_file_location("_ail_bindings", bindings_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    return Bindings(
        tools=dict(getattr(module, "TOOLS", {})),
        schemas=dict(getattr(module, "SCHEMAS", {})),
        providers=dict(getattr(module, "PROVIDERS", {})),
        capabilities=dict(getattr(module, "CAPABILITIES", {})),
    )
