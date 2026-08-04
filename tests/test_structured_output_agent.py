"""ModelAgent(output_schema=...) end to end: single-shot, inside the
tool-calling loop, and through a real Workflow/Scheduler run (so the
parsed structured value shows up as the journal step's output). Uses
MockProvider throughout -- structured_output.py's parsing/validation is
provider-agnostic by design (see litellm_provider.py's comment on why it
only forwards a JSON-mode hint and doesn't validate anything itself).
"""

import pytest

from aircore import Workflow, Tool
from airpy import MockProvider, ModelAgent
from airpy.structured_output import StructuredOutputError

WIDGET_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer"},
    },
    "required": ["name", "count"],
}


def test_without_output_schema_behavior_is_unchanged():
    provider = MockProvider(response="plain text answer")
    agent = ModelAgent("a", provider, prompt="hi")
    assert agent.execute() == "plain text answer"


def test_output_schema_returns_a_parsed_dict():
    provider = MockProvider(response='{"name": "widget", "count": 3}')
    agent = ModelAgent("a", provider, prompt="describe the widget", output_schema=WIDGET_SCHEMA)
    result = agent.execute()
    assert result == {"name": "widget", "count": 3}


def test_output_schema_handles_a_fenced_response():
    provider = MockProvider(response='```json\n{"name": "widget", "count": 3}\n```')
    agent = ModelAgent("a", provider, prompt="describe the widget", output_schema=WIDGET_SCHEMA)
    assert agent.execute() == {"name": "widget", "count": 3}


def test_output_schema_prompt_includes_schema_instructions():
    seen_prompts = []

    def capture(request):
        seen_prompts.append(request.prompt)
        return '{"name": "widget", "count": 1}'

    provider = MockProvider(response=capture)
    agent = ModelAgent("a", provider, prompt="describe the widget", output_schema=WIDGET_SCHEMA)
    agent.execute()

    assert "describe the widget" in seen_prompts[0]
    assert "JSON" in seen_prompts[0]
    assert '"count"' in seen_prompts[0]


def test_output_schema_request_carries_response_schema_hint():
    seen_requests = []

    def capture(request):
        seen_requests.append(request)
        return '{"name": "widget", "count": 1}'

    provider = MockProvider(response=capture)
    agent = ModelAgent("a", provider, prompt="describe the widget", output_schema=WIDGET_SCHEMA)
    agent.execute()

    assert seen_requests[0].response_schema == WIDGET_SCHEMA


def test_invalid_json_raises_structured_output_error():
    provider = MockProvider(response="not json at all")
    agent = ModelAgent("a", provider, prompt="describe the widget", output_schema=WIDGET_SCHEMA)
    with pytest.raises(StructuredOutputError):
        agent.execute()


def test_schema_mismatch_raises_structured_output_error():
    provider = MockProvider(response='{"name": "widget"}')  # missing "count"
    agent = ModelAgent("a", provider, prompt="describe the widget", output_schema=WIDGET_SCHEMA)
    with pytest.raises(StructuredOutputError):
        agent.execute()


def test_bad_schema_type_rejected_at_construction():
    provider = MockProvider()
    with pytest.raises(TypeError):
        ModelAgent("a", provider, prompt="x", output_schema="not a schema")


def test_structured_output_works_inside_the_tool_calling_loop():
    from airpy.providers import ModelResponse, ToolCallRequest

    lookup = Tool(lambda item: {"widget": 3}.get(item, 0), name="lookup")

    responses = [
        ModelResponse(content="", tool_calls=[ToolCallRequest(id="1", name="lookup", arguments={"item": "widget"})]),
        '{"name": "widget", "count": 3}',
    ]
    provider = MockProvider(responses=responses)
    agent = ModelAgent("a", provider, prompt="how many widgets?", tools=[lookup], output_schema=WIDGET_SCHEMA)

    result = agent.execute()
    assert result == {"name": "widget", "count": 3}


def test_structured_output_lands_in_the_journal_as_the_step_output():
    provider = MockProvider(response='{"name": "widget", "count": 3}')
    agent = ModelAgent("researcher", provider, prompt="describe the widget", output_schema=WIDGET_SCHEMA)

    workflow = Workflow("structured-output-journal")
    workflow.step(agent)
    journal = workflow.run()

    assert journal.status == "success"
    step = journal.steps[0]
    assert step.output == {"name": "widget", "count": 3}


def test_scheduler_fails_the_step_gracefully_on_bad_structured_output():
    provider = MockProvider(response="not json")
    agent = ModelAgent("researcher", provider, prompt="describe the widget", output_schema=WIDGET_SCHEMA)

    workflow = Workflow("structured-output-failure")
    workflow.step(agent)
    journal = workflow.run()  # must not raise

    assert journal.status == "failed"
    assert "not valid JSON" in journal.steps[0].error


pydantic = pytest.importorskip("pydantic")


class Widget(pydantic.BaseModel):
    name: str
    count: int


def test_output_schema_accepts_a_pydantic_model_and_returns_a_typed_instance():
    provider = MockProvider(response='{"name": "widget", "count": 3}')
    agent = ModelAgent("a", provider, prompt="describe the widget", output_schema=Widget)

    result = agent.execute()

    assert isinstance(result, Widget)
    assert result.name == "widget"
    assert result.count == 3
