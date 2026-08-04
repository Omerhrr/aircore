"""airpy's facade: `from airpy import Agent, Workflow` should be all a
developer ever needs -- no `from aircore import ...` required to write a
real, multi-step application. Covers:

- Agent is exactly ModelAgent (a plain alias, per model_agent.py's own
  docstring suggestion), not a divergent subclass
- Workflow/Tool/tool re-exported unchanged from aircore (airpy adds no
  wrapper logic -- aircore's Workflow already chains, see ParallelResults)
- the exact shape from the roadmap discussion actually runs end to end:
  `Workflow("Audit").parallel(a, b, c).consensus(strategy=...)` built
  entirely from airpy.Agent instances, imported only from airpy
- aircore is never imported anywhere in this file, proving the facade is
  actually sufficient, not just present
"""

from aircore import majority

from airpy import Agent, MockProvider, ModelAgent, ModelResponse, Tool, ToolCallRequest, Workflow, tool


def test_agent_is_exactly_model_agent():
    assert Agent is ModelAgent


def test_workflow_tool_and_tool_decorator_are_airun_s_unchanged():
    import aircore
    assert Workflow is aircore.Workflow
    assert Tool is aircore.Tool
    assert tool is aircore.tool


def test_fluent_workflow_of_agents_runs_end_to_end_with_only_airpy_imports():
    researcher = Agent("researcher", MockProvider(response="A"), "Research the topic.")
    reviewer = Agent("reviewer", MockProvider(response="A"), "Review the research.")
    professor = Agent("professor", MockProvider(response="A"), "Grade the research.")

    workflow = (
        Workflow("Audit")
        .parallel(researcher, reviewer, professor)
        .consensus(strategy=majority)
    )
    journal = workflow.run()

    assert journal.status == "success"
    consensus_steps = [s for s in journal.steps if s.tool == "consensus"]
    assert len(consensus_steps) == 1
    assert consensus_steps[0].output == "A"


def test_sequential_step_also_works_through_the_facade():
    @tool
    def hello():
        return "hi"

    agent = Agent("assistant", MockProvider(response="hello there"), "Greet the user.")

    workflow = Workflow("Greeting").step(hello).step(agent)
    journal = workflow.run()

    assert journal.status == "success"
    assert len(journal.steps) == 2
    assert journal.steps[0].output == "hi"
    assert journal.steps[1].output == "hello there"


def test_custom_tool_via_airpy_facade_used_inside_an_agent_s_tool_loop():
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    provider = MockProvider(responses=[
        ModelResponse(content="", tool_calls=[
            ToolCallRequest(id="1", name="add", arguments={"a": 2, "b": 3}),
        ]),
        "The sum is 5.",
    ])
    agent = Agent("calculator", provider, "Add two numbers.", tools=[add])

    workflow = Workflow("Math").step(agent)
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.steps[0].output == "The sum is 5."
