"""airpy/structured_output.py in isolation -- the shared JSON-extraction,
schema-normalization, and validation pipeline used by both
ModelAgent(output_schema=...) and JudgeConsensus(output_schema=...,
confidence=...). No provider/workflow involved here; see
test_structured_output_agent.py and test_judge_consensus_modes.py for the
end-to-end wiring.
"""

import pytest

from airpy.structured_output import (
    StructuredOutputError,
    extract_json,
    is_pydantic_model,
    parse_structured_output,
    schema_from,
    validate_against_schema,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer"},
    },
    "required": ["name", "count"],
}


def test_schema_from_passes_through_a_plain_dict():
    assert schema_from(SCHEMA) is SCHEMA


def test_schema_from_rejects_non_schema_non_pydantic_values():
    with pytest.raises(TypeError):
        schema_from("not a schema")
    with pytest.raises(TypeError):
        schema_from(42)


def test_extract_json_handles_plain_json():
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_json_strips_a_json_code_fence():
    text = 'Here you go:\n```json\n{"a": 1}\n```\nHope that helps.'
    assert extract_json(text) == '{"a": 1}'


def test_extract_json_strips_a_plain_code_fence():
    text = '```\n{"a": 1}\n```'
    assert extract_json(text) == '{"a": 1}'


def test_extract_json_falls_back_to_first_brace_to_last_brace():
    text = 'Sure, the answer is {"a": 1} -- let me know if you need more.'
    assert extract_json(text) == '{"a": 1}'


def test_validate_against_schema_passes_for_matching_data():
    validate_against_schema({"name": "bloom filter", "count": 3}, SCHEMA)  # no raise


def test_validate_against_schema_raises_on_missing_required_field():
    with pytest.raises(StructuredOutputError, match="count"):
        validate_against_schema({"name": "x"}, SCHEMA)


def test_validate_against_schema_raises_on_wrong_top_level_type():
    with pytest.raises(StructuredOutputError, match="object"):
        validate_against_schema(["not", "an", "object"], SCHEMA)


def test_validate_against_schema_raises_on_wrong_property_type():
    with pytest.raises(StructuredOutputError, match="count"):
        validate_against_schema({"name": "x", "count": "not a number"}, SCHEMA)


def test_validate_against_schema_ignores_unknown_extra_properties():
    validate_against_schema({"name": "x", "count": 1, "extra": "ignored"}, SCHEMA)  # no raise


def test_parse_structured_output_end_to_end_returns_plain_dict():
    result = parse_structured_output('{"name": "bloom filter", "count": 3}', SCHEMA)
    assert result == {"name": "bloom filter", "count": 3}


def test_parse_structured_output_handles_a_fenced_and_chatty_response():
    text = 'Sure! Here is the JSON:\n```json\n{"name": "x", "count": 2}\n```\nLet me know!'
    result = parse_structured_output(text, SCHEMA)
    assert result == {"name": "x", "count": 2}


def test_parse_structured_output_raises_structured_output_error_on_invalid_json():
    with pytest.raises(StructuredOutputError):
        parse_structured_output("this is not json at all", SCHEMA)


def test_parse_structured_output_raises_structured_output_error_on_schema_mismatch():
    with pytest.raises(StructuredOutputError):
        parse_structured_output('{"name": "x"}', SCHEMA)  # missing "count"


def test_is_pydantic_model_false_for_plain_classes_and_instances():
    class NotPydantic:
        pass

    assert is_pydantic_model(NotPydantic) is False
    assert is_pydantic_model(NotPydantic()) is False
    assert is_pydantic_model("a string") is False


# --- Pydantic-specific: skipped entirely if pydantic isn't installed, so
# the rest of this file (and airpy generally) never requires it. ---

pydantic = pytest.importorskip("pydantic")


class Widget(pydantic.BaseModel):
    name: str
    count: int


def test_is_pydantic_model_true_for_a_basemodel_subclass():
    assert is_pydantic_model(Widget) is True


def test_schema_from_a_pydantic_model_produces_a_json_schema_dict():
    schema = schema_from(Widget)
    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert "count" in schema["properties"]


def test_parse_structured_output_with_model_cls_constructs_a_typed_instance():
    schema = schema_from(Widget)
    result = parse_structured_output('{"name": "bloom filter", "count": 3}', schema, model_cls=Widget)
    assert isinstance(result, Widget)
    assert result.name == "bloom filter"
    assert result.count == 3


def test_parse_structured_output_with_model_cls_raises_on_pydantic_validation_failure():
    schema = schema_from(Widget)
    with pytest.raises(StructuredOutputError):
        parse_structured_output('{"name": "x", "count": "not an int"}', schema, model_cls=Widget)
