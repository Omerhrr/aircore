"""Converts an aircore.Tool into a ToolSchema a model provider can use.

For most tools this is the one place in airpy that reads a Tool's Python
signature to build a JSON schema -- everything else just calls the tool.
Simple type mapping only (str/int/float/bool/list/dict); anything else
falls back to "string" rather than guessing at a nested schema. A
parameter counts as required if it has no default value.

If `tool.parameters_schema` is set (see aircore/tools.py), that's used
as-is instead of introspecting `tool.fn` -- this is the case for tools
whose real schema comes from somewhere else entirely, like an MCP
server's own tool listing (see airpy/mcp_tools.py), where `fn` is a
generic `**kwargs` wrapper with no signature worth introspecting.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

from aircore.tools import Tool

from .providers import ToolSchema

_TYPE_MAP: Dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool_to_schema(tool: Tool) -> ToolSchema:
    if tool.parameters_schema is not None:
        return ToolSchema(name=tool.name, description=tool.description, parameters=tool.parameters_schema)

    signature = inspect.signature(tool.fn)
    properties: Dict[str, Any] = {}
    required = []

    for param_name, param in signature.parameters.items():
        json_type = _TYPE_MAP.get(param.annotation, "string")
        properties[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return ToolSchema(
        name=tool.name,
        description=tool.description,
        parameters={"type": "object", "properties": properties, "required": required},
    )
