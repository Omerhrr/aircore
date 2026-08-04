"""Structured output: turning a model's free text into a validated,
typed value instead of a string the caller has to parse themselves.

Used by two callers:
- ModelAgent(output_schema=...) (model_agent.py) -- an agent's own answer.
- JudgeConsensus(output_schema=..., confidence=...) (judge_consensus.py)
  -- a consensus strategy's merged/selected answer, and its typed
  confidence/reasoning fields.

Deliberately minimal. `schema_from()` accepts either a plain JSON-schema
dict (no extra dependency) or a Pydantic BaseModel subclass (lazily
imported, same lazy-dependency pattern as litellm_provider.py -- airpy has
no hard dependency on pydantic unless you actually pass one in).
`validate_against_schema()` is NOT a full JSON Schema implementation: it
checks `required` keys are present and does a best-effort top-level `type`
check per property. No $ref, no nested schema validation, no format
validators. That's enough to catch the common failure mode (the model
returns prose, or omits a field) without building a JSON Schema validator
from scratch -- if a real need for deeper validation shows up, a caller
can always pass a Pydantic model instead, which validates as deeply as its
own field types say to.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Type


class StructuredOutputError(Exception):
    """Raised when a model's response can't be turned into the requested
    structured output -- it wasn't valid JSON, didn't match the schema, or
    (with a Pydantic model_cls) failed that model's own validation. Caught
    by the scheduler exactly like any other Executable/strategy failure
    (see scheduler.py's broad exception handling in _run_tool and
    _apply_consensus_strategy) -- it fails the step gracefully, it doesn't
    crash the run."""


def is_pydantic_model(obj: Any) -> bool:
    """Duck-types a Pydantic v1 or v2 BaseModel subclass without importing
    pydantic -- so this check (and therefore schema_from) works whether or
    not pydantic is installed, and costs nothing when it isn't."""
    return isinstance(obj, type) and (hasattr(obj, "model_json_schema") or hasattr(obj, "schema"))


def schema_from(output_schema: Any) -> Dict[str, Any]:
    """Normalizes `output_schema` (a plain JSON-schema dict, or a Pydantic
    BaseModel subclass) into a JSON-schema dict. Raises TypeError for
    anything else -- this runs at construction time (ModelAgent.__init__,
    JudgeConsensus.__init__), so a bad schema fails fast, before any model
    call is made, the same way Tool(retries>0, idempotent=False) fails
    fast instead of at execution time."""
    if isinstance(output_schema, dict):
        return output_schema
    if is_pydantic_model(output_schema):
        if hasattr(output_schema, "model_json_schema"):  # pydantic v2
            return output_schema.model_json_schema()
        return output_schema.schema()  # pydantic v1
    raise TypeError(
        f"output_schema must be a JSON-schema dict or a Pydantic BaseModel "
        f"subclass, got {output_schema!r}"
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_JSON_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def extract_json(text: str) -> str:
    """Best-effort extraction of a JSON payload out of a model's raw text
    response. Models frequently wrap JSON in a ```json ... ``` fence, or
    add a sentence before/after it despite being told not to -- this
    handles the fenced case explicitly, and otherwise falls back to the
    first '{' through the last '}' (or '[' through last ']')."""
    text = text.strip()
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]

    return text


def validate_against_schema(data: Any, schema: Dict[str, Any]) -> None:
    """Minimal, top-level-only JSON Schema check -- see module docstring
    for what this deliberately doesn't do. Raises StructuredOutputError
    with a specific reason on the first problem found."""
    schema_type = schema.get("type", "object")
    check = _JSON_TYPE_CHECKS.get(schema_type)
    if check is not None and not check(data):
        raise StructuredOutputError(
            f"expected top-level type {schema_type!r}, got {type(data).__name__}: {data!r}"
        )

    if schema_type != "object" or not isinstance(data, dict):
        return

    for required_key in schema.get("required", []):
        if required_key not in data:
            raise StructuredOutputError(
                f"missing required field {required_key!r} in {data!r}"
            )

    properties = schema.get("properties", {})
    for key, value in data.items():
        prop_schema = properties.get(key)
        if not prop_schema:
            continue
        prop_type = prop_schema.get("type")
        prop_check = _JSON_TYPE_CHECKS.get(prop_type)
        if prop_check is not None and value is not None and not prop_check(value):
            raise StructuredOutputError(
                f"field {key!r} expected type {prop_type!r}, got "
                f"{type(value).__name__}: {value!r}"
            )


def parse_structured_output(text: str, schema: Dict[str, Any],
                             model_cls: Optional[Type] = None) -> Any:
    """The full pipeline: extract JSON out of raw text, parse it, validate
    it against `schema`, and -- if `model_cls` (a Pydantic class) is given
    -- construct and return an instance of it (which applies that model's
    own, deeper validation); otherwise returns the plain parsed dict/list/
    scalar. Raises StructuredOutputError (never a bare JSONDecodeError or
    pydantic ValidationError) at every failure point, with the offending
    text included, so a caller catching StructuredOutputError doesn't also
    need to know what parser or validator produced it."""
    candidate = extract_json(text)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            f"response was not valid JSON ({exc}): {text!r}"
        ) from exc

    validate_against_schema(data, schema)

    if model_cls is not None:
        try:
            return model_cls(**data) if isinstance(data, dict) else model_cls(data)
        except Exception as exc:  # noqa: BLE001 -- pydantic raises its own
            # ValidationError (v1/v2 differ), and a non-pydantic model_cls
            # could raise anything from its constructor. Either way this
            # is a structured-output failure, not an unhandled crash.
            raise StructuredOutputError(f"{model_cls.__name__} validation failed: {exc}") from exc

    return data
