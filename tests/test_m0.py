import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool


def test_single_tool_runs_and_journals():
    @tool
    def hello():
        return "Hello, World!"

    workflow = Workflow("Hello")
    workflow.step(hello)
    journal = workflow.run()

    assert journal.workflow_name == "Hello"
    assert journal.status == "success"
    assert len(journal.steps) == 1

    step = journal.steps[0]
    assert step.tool == "hello"
    assert step.status == "success"
    assert step.output == "Hello, World!"
    assert step.duration_ms is not None
    assert step.started_at is not None
    assert step.finished_at is not None


def test_journal_serializes_to_json():
    @tool
    def hello():
        return "Hello, World!"

    workflow = Workflow("Hello")
    workflow.step(hello)
    journal = workflow.run()

    data = journal.to_dict()
    assert data["workflow"] == "Hello"
    assert data["status"] == "success"
    assert data["steps"][0]["output"] == "Hello, World!"

    # must actually serialize without error
    journal.to_json()


def test_failed_tool_marks_workflow_failed_and_stops():
    @tool
    def boom():
        raise ValueError("kaboom")

    @tool
    def never_runs():
        return "should not appear"

    workflow = Workflow("Failing")
    workflow.step(boom)
    workflow.step(never_runs)
    journal = workflow.run()

    assert journal.status == "failed"
    assert len(journal.steps) == 1  # second step never started
    assert journal.steps[0].status == "failed"
    assert "kaboom" in journal.steps[0].error


def test_multiple_sequential_steps_run_in_order():
    calls = []

    @tool
    def step_a():
        calls.append("a")
        return "a-done"

    @tool
    def step_b():
        calls.append("b")
        return "b-done"

    workflow = Workflow("Sequential")
    workflow.step(step_a)
    workflow.step(step_b)
    journal = workflow.run()

    assert calls == ["a", "b"]
    assert journal.status == "success"
    assert [s.id for s in journal.steps] == [1, 2]


if __name__ == "__main__":
    test_single_tool_runs_and_journals()
    test_journal_serializes_to_json()
    test_failed_tool_marks_workflow_failed_and_stops()
    test_multiple_sequential_steps_run_in_order()
    print("All M0 tests passed.")
