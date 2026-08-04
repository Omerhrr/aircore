import sys
import os
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool


def test_parallel_group_runs_concurrently():
    active = []
    max_concurrent = []
    lock = threading.Lock()

    def make_tool(name):
        @tool(name=name)
        def _t():
            with lock:
                active.append(name)
                max_concurrent.append(len(active))
            time.sleep(0.05)
            with lock:
                active.remove(name)
            return f"{name}-done"
        return _t

    workflow = Workflow("Parallel")
    workflow.parallel(make_tool("a"), make_tool("b"), make_tool("c"))
    journal = workflow.run()

    assert journal.status == "success"
    assert max(max_concurrent) > 1, "tools did not overlap -- not actually concurrent"
    assert len(journal.groups) == 1
    assert journal.groups[0].status == "success"
    assert len(journal.steps) == 3


def test_fan_in_continues_after_group_completes():
    order = []

    @tool
    def a():
        order.append("a")
        return "a"

    @tool
    def b():
        order.append("b")
        return "b"

    @tool
    def after():
        order.append("after")
        return "after"

    workflow = Workflow("FanIn")
    workflow.parallel(a, b)
    workflow.step(after)
    journal = workflow.run()

    assert journal.status == "success"
    assert order[-1] == "after", "sequential step after parallel block did not wait for fan-in"
    assert len(journal.steps) == 3


def test_group_failure_marks_workflow_failed():
    @tool
    def ok():
        return "fine"

    @tool
    def bad():
        raise RuntimeError("group member failed")

    @tool
    def never_runs():
        return "should not appear"

    workflow = Workflow("GroupFailure")
    workflow.parallel(ok, bad)
    workflow.step(never_runs)
    journal = workflow.run()

    assert journal.status == "failed"
    assert journal.groups[0].status == "failed"
    # the sequential step after the failed group never ran
    assert not any(s.tool == "never_runs" for s in journal.steps)


def test_parallel_requires_at_least_two_tools():
    @tool
    def solo():
        return "x"

    workflow = Workflow("TooFew")
    try:
        workflow.parallel(solo)
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_parallel_group_runs_concurrently()
    test_fan_in_continues_after_group_completes()
    test_group_failure_marks_workflow_failed()
    test_parallel_requires_at_least_two_tools()
    print("All M1 tests passed.")
