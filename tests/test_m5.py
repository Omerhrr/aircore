import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool, Memory


def test_temporary_shared_between_steps_in_one_run():
    mem = Memory()

    @tool
    def write():
        mem.temporary.set("x", 42)
        return "written"

    @tool
    def read():
        return mem.temporary.get("x")

    workflow = Workflow("W", memory=mem)
    workflow.step(write)
    workflow.step(read)
    journal = workflow.run()

    assert journal.steps[1].output == 42


def test_temporary_cleared_after_successful_run():
    mem = Memory()

    @tool
    def write():
        mem.temporary.set("x", 1)
        return "ok"

    workflow = Workflow("W", memory=mem)
    workflow.step(write)
    workflow.run()

    assert mem.temporary.snapshot() == {}


def test_temporary_cleared_even_after_failed_run():
    mem = Memory()

    @tool
    def write_then_fail():
        mem.temporary.set("x", 1)
        raise RuntimeError("boom")

    workflow = Workflow("W", memory=mem)
    workflow.step(write_then_fail)
    journal = workflow.run()

    assert journal.status == "failed"
    assert mem.temporary.snapshot() == {}


def test_temporary_does_not_leak_into_next_run():
    mem = Memory()

    @tool
    def write():
        mem.temporary.set("x", "run1-data")
        return "ok"

    @tool
    def read():
        return mem.temporary.get("x", "nothing")

    wf1 = Workflow("Run1", memory=mem)
    wf1.step(write)
    wf1.run()

    wf2 = Workflow("Run2", memory=mem)
    wf2.step(read)
    j2 = wf2.run()

    assert j2.steps[0].output == "nothing"


def test_session_persists_across_runs_on_same_memory():
    mem = Memory()

    @tool
    def write():
        mem.session.set("name", "Ada")
        return "ok"

    @tool
    def read():
        return mem.session.get("name")

    wf1 = Workflow("Turn1", memory=mem)
    wf1.step(write)
    wf1.run()

    wf2 = Workflow("Turn2", memory=mem)
    wf2.step(read)
    j2 = wf2.run()

    assert j2.steps[0].output == "Ada"


def test_session_is_isolated_between_separate_memory_instances():
    mem_a = Memory()
    mem_b = Memory()
    mem_a.session.set("shared_key", "a-value")
    assert mem_b.session.get("shared_key") is None


def test_project_scope_shared_across_instances_with_same_name():
    mem1 = Memory(project="acme")
    mem2 = Memory(project="acme")
    mem1.project.set("key", "value")
    assert mem2.project.get("key") == "value"


def test_project_scope_isolated_across_different_names():
    mem1 = Memory(project="acme")
    mem2 = Memory(project="other")
    mem1.project.set("key", "acme-value")
    assert mem2.project.get("key") is None


def test_project_scope_isolated_when_no_project_name_given():
    mem1 = Memory()  # project=None
    mem2 = Memory()  # project=None
    mem1.project.set("key", "value")
    assert mem2.project.get("key") is None


def test_workflow_without_memory_is_unaffected():
    @tool
    def noop():
        return "ok"

    workflow = Workflow("NoMemory")  # memory=None, default
    workflow.step(noop)
    journal = workflow.run()
    assert journal.status == "success"
    assert workflow.memory is None


if __name__ == "__main__":
    test_temporary_shared_between_steps_in_one_run()
    test_temporary_cleared_after_successful_run()
    test_temporary_cleared_even_after_failed_run()
    test_temporary_does_not_leak_into_next_run()
    test_session_persists_across_runs_on_same_memory()
    test_session_is_isolated_between_separate_memory_instances()
    test_project_scope_shared_across_instances_with_same_name()
    test_project_scope_isolated_across_different_names()
    test_project_scope_isolated_when_no_project_name_given()
    test_workflow_without_memory_is_unaffected()
    print("All M5 tests passed.")
