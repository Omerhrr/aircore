import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool, build_execution_graph, render_execution_graph


def test_metrics_populated_automatically_after_run():
    @tool
    def a():
        return "a"

    @tool
    def b():
        return "b"

    workflow = Workflow("Sequential")
    workflow.step(a)
    workflow.step(b)
    assert workflow.metrics is None  # not populated before run()
    workflow.run()

    m = workflow.metrics
    assert m is not None
    assert m.workflow_name == "Sequential"
    assert m.status == "success"
    assert m.steps_total == 2
    assert m.steps_succeeded == 2
    assert m.steps_failed == 0
    assert m.groups_total == 0
    assert m.retries_total == 0  # neither tool here declares retries
    assert set(m.by_tool.keys()) == {"a", "b"}
    assert m.by_tool["a"].calls == 1
    assert m.by_tool["a"].avg_duration_ms is not None


def test_metrics_counts_failures_and_groups():
    @tool
    def ok():
        return "ok"

    @tool
    def boom():
        raise ValueError("bad")

    workflow = Workflow("WithFailure")
    workflow.parallel(ok, boom)
    workflow.run()

    m = workflow.metrics
    assert m.status == "failed"
    assert m.groups_total == 1
    assert m.steps_succeeded == 1
    assert m.steps_failed == 1
    assert m.by_tool["boom"].failures == 1


def test_metrics_summary_does_not_error():
    @tool
    def a():
        return "a"

    workflow = Workflow("Summary")
    workflow.step(a)
    workflow.run()
    text = workflow.metrics.summary()
    assert "Summary" in text
    assert "a:" in text


def test_execution_graph_shape_sequential():
    @tool
    def a():
        return "a"

    @tool
    def b():
        return "b"

    workflow = Workflow("Seq")
    workflow.step(a)
    workflow.step(b)
    workflow.run()

    graph = build_execution_graph(workflow.journal)
    assert graph.kind == "workflow"
    assert graph.status == "success"
    assert len(graph.children) == 2
    assert [c.kind for c in graph.children] == ["step", "step"]
    assert [c.label for c in graph.children] == ["a", "b"]


def test_execution_graph_nests_parallel_group():
    @tool
    def a():
        return "a"

    @tool
    def b():
        return "b"

    @tool
    def after():
        return "after"

    workflow = Workflow("ParallelGraph")
    workflow.parallel(a, b)
    workflow.step(after)
    workflow.run()

    graph = build_execution_graph(workflow.journal)
    assert len(graph.children) == 2  # one group node, one step node
    group_node, step_node = graph.children
    assert group_node.kind == "group"
    assert len(group_node.children) == 2
    assert {c.label for c in group_node.children} == {"a", "b"}
    assert step_node.kind == "step"
    assert step_node.label == "after"


def test_execution_graph_renders_without_error():
    @tool
    def a():
        return "a"

    workflow = Workflow("Render")
    workflow.step(a)
    workflow.run()

    graph = build_execution_graph(workflow.journal)
    text = render_execution_graph(graph)
    assert "workflow: Render" in text
    assert "step: a" in text


if __name__ == "__main__":
    test_metrics_populated_automatically_after_run()
    test_metrics_counts_failures_and_groups()
    test_metrics_summary_does_not_error()
    test_execution_graph_shape_sequential()
    test_execution_graph_nests_parallel_group()
    test_execution_graph_renders_without_error()
    print("All M4 tests passed.")
