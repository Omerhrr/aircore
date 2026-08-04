import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool, Agent, Policy, PolicyViolation, Network, Executable
from airpy import ModelAgent, MockProvider, ask, ModelProvider, ModelRequest, ModelResponse


def test_model_agent_is_an_executable():
    agent = ModelAgent("m", MockProvider(response="x"), prompt="p")
    assert isinstance(agent, Executable)


def test_model_agent_works_as_sequential_workflow_step():
    agent = ModelAgent("summarizer", MockProvider(response="a summary"), prompt="summarize this")
    workflow = Workflow("W")
    workflow.step(agent)
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.steps[0].tool == "summarizer"
    assert journal.steps[0].output == "a summary"


def test_tool_and_model_agent_coexist_in_same_workflow():
    @tool
    def fetch():
        return "raw data"

    agent = ModelAgent("analyzer", MockProvider(response="analysis"), prompt="analyze")

    workflow = Workflow("Mixed")
    workflow.step(fetch)
    workflow.step(agent)
    journal = workflow.run()

    assert journal.status == "success"
    assert [s.tool for s in journal.steps] == ["fetch", "analyzer"]
    assert [s.output for s in journal.steps] == ["raw data", "analysis"]


def test_model_agent_in_parallel_block():
    a1 = ModelAgent("a1", MockProvider(response="r1"), prompt="p")
    a2 = ModelAgent("a2", MockProvider(response="r2"), prompt="p")

    workflow = Workflow("Parallel")
    workflow.parallel(a1, a2)
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.groups[0].kind == "parallel"
    outputs = {s.tool: s.output for s in journal.steps}
    assert outputs == {"a1": "r1", "a2": "r2"}


def test_model_agents_reach_consensus():
    yes1 = ModelAgent("gpt", MockProvider(response="yes"), prompt="ship?")
    yes2 = ModelAgent("claude", MockProvider(response="yes"), prompt="ship?")
    no = ModelAgent("gemini", MockProvider(response="no"), prompt="ship?")

    workflow = Workflow("Vote")
    workflow.consensus(yes1, yes2, no)
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.steps[-1].output == "yes"


def test_capability_enforcement_applies_to_model_agent():
    agent = ModelAgent("web", MockProvider(response="fetched"), prompt="search", requires=Network)
    sandboxed = Agent("Sandboxed", capabilities=[])

    workflow = Workflow("Restricted")
    workflow.step(agent, agent=sandboxed)
    journal = workflow.run()

    assert journal.status == "failed"
    assert "CapabilityDenied" in journal.steps[0].error


def test_capability_enforcement_allows_when_granted():
    agent = ModelAgent("web", MockProvider(response="fetched"), prompt="search", requires=Network)
    networked = Agent("Networked", capabilities=[Network])

    workflow = Workflow("Allowed")
    workflow.step(agent, agent=networked)
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.steps[0].output == "fetched"


def test_policy_require_agent_applies_to_model_agent():
    agent = ModelAgent("m", MockProvider(response="x"), prompt="p")
    workflow = Workflow("Prod", policy=Policy(require_agent=True))
    workflow.step(agent)  # no identity agent attached
    try:
        workflow.run()
        assert False, "expected PolicyViolation"
    except PolicyViolation:
        pass


def test_model_agent_retries_declared_non_idempotent_is_rejected():
    try:
        ModelAgent("m", MockProvider(response="x"), prompt="p", idempotent=False, retries=2)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "idempotent" in str(exc)


def test_model_agent_default_idempotent_true_differs_from_tool_default():
    agent = ModelAgent("m", MockProvider(response="x"), prompt="p")
    assert agent.idempotent is True  # different default than Tool's idempotent=False


def test_model_agent_retries_on_transient_failure():
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("timeout")
        return "recovered"

    class FlakyProvider(ModelProvider):
        def generate(self, request: ModelRequest) -> ModelResponse:
            text = flaky(request)
            return ModelResponse(content=text, model=request.model)

    agent = ModelAgent("m", FlakyProvider(), prompt="p", idempotent=True, retries=3)
    workflow = Workflow("Retries")
    workflow.step(agent)
    journal = workflow.run()

    assert journal.status == "success"
    assert journal.steps[0].output == "recovered"
    assert journal.steps[0].retries == 1


def test_ask_with_model_agent():
    agent = ModelAgent("m", MockProvider(response="42"), prompt="the answer?")
    assert ask(agent) == "42"


def test_ask_with_raw_provider_and_prompt():
    provider = MockProvider(response=lambda req: f"echo:{req.prompt}")
    assert ask(provider, prompt="hi") == "echo:hi"


def test_ask_rejects_prompt_with_agent():
    agent = ModelAgent("m", MockProvider(response="x"), prompt="p")
    try:
        ask(agent, prompt="not allowed")
        assert False, "expected TypeError"
    except TypeError:
        pass


def test_ask_rejects_provider_without_prompt():
    provider = MockProvider(response="x")
    try:
        ask(provider)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_airun_does_not_import_airpy():
    """aircore must stay 100% provider-agnostic -- nothing in it should have
    an actual import statement pulling in airpy. Docstrings are allowed to
    *mention* airpy by name for documentation purposes (several already
    do, explaining the Executable seam) -- this checks real import lines,
    not prose."""
    import aircore
    import inspect
    airun_dir = os.path.dirname(inspect.getfile(aircore))
    for fname in os.listdir(airun_dir):
        if not fname.endswith(".py"):
            continue
        with open(os.path.join(airun_dir, fname)) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("import airpy") or stripped.startswith("from airpy"):
                    assert False, f"{fname} has a real import of airpy: {stripped!r}"


if __name__ == "__main__":
    test_model_agent_is_an_executable()
    test_model_agent_works_as_sequential_workflow_step()
    test_tool_and_model_agent_coexist_in_same_workflow()
    test_model_agent_in_parallel_block()
    test_model_agents_reach_consensus()
    test_capability_enforcement_applies_to_model_agent()
    test_capability_enforcement_allows_when_granted()
    test_policy_require_agent_applies_to_model_agent()
    test_model_agent_retries_declared_non_idempotent_is_rejected()
    test_model_agent_default_idempotent_true_differs_from_tool_default()
    test_model_agent_retries_on_transient_failure()
    test_ask_with_model_agent()
    test_ask_with_raw_provider_and_prompt()
    test_ask_rejects_prompt_with_agent()
    test_ask_rejects_provider_without_prompt()
    test_airun_does_not_import_airpy()
    print("All airpy tests passed.")
