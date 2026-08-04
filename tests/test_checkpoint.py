"""Durable checkpointing / resume-after-crash (aircore/checkpoint.py,
Scheduler.run()'s replay branch, Workflow.run(checkpoint_store=...,
run_id=...)).

Simulates "the process died and you rerun the same script" the only way
that's meaningful inside one test process: build a Workflow whose step 2
fails, run() it (step 1 succeeds and gets checkpointed, step 2 fails),
then build a *fresh* Workflow object with the same step 1 tool and a now-
fixed step 2, and run() it again with the same checkpoint_store/run_id --
this is exactly what rerunning the same script after fixing a bug (or
after a real crash) looks like from aircore's point of view: a brand new
Workflow instance, same declared steps, same run_id.
"""

import pytest

from aircore import CheckpointError, FileCheckpointStore, InMemoryCheckpointStore, Tool, Workflow


def test_first_run_with_checkpointing_behaves_normally_and_records_progress():
    store = InMemoryCheckpointStore()
    calls = []
    workflow = Workflow("W")
    workflow.step(Tool(lambda: calls.append("a") or "a-output", name="a"))
    workflow.step(Tool(lambda: calls.append("b") or "b-output", name="b"))
    journal = workflow.run(checkpoint_store=store, run_id="run-1")

    assert journal.status == "success"
    assert calls == ["a", "b"]
    assert not any(s.replayed for s in journal.steps)
    assert set(store.load("run-1").keys()) == {0, 1}


def test_resume_skips_already_succeeded_steps_and_does_not_refire_side_effects():
    store = InMemoryCheckpointStore()
    calls = []

    def failing_b():
        calls.append("b-attempt")
        raise RuntimeError("boom")

    workflow = Workflow("W")
    workflow.step(Tool(lambda: calls.append("a") or "a-output", name="a"))
    workflow.step(Tool(failing_b, name="b"))
    journal = workflow.run(checkpoint_store=store, run_id="run-1")
    assert journal.status == "failed"
    assert calls == ["a", "b-attempt"]

    # "fix" b and rerun the identical script (a fresh Workflow instance,
    # same steps in the same order) with the same run_id
    calls.clear()
    workflow2 = Workflow("W")
    workflow2.step(Tool(lambda: calls.append("a") or "a-output", name="a"))
    workflow2.step(Tool(lambda: calls.append("b") or "b-output", name="b"))
    journal2 = workflow2.run(checkpoint_store=store, run_id="run-1")

    assert journal2.status == "success"
    # "a"'s side effect never refired -- it was replayed, not re-executed
    assert calls == ["b"]
    a_step, b_step = journal2.steps
    assert a_step.replayed is True
    assert a_step.output == "a-output"
    assert b_step.replayed is False


def test_groups_are_never_checkpointed_and_always_rerun_in_full():
    store = InMemoryCheckpointStore()
    calls = []
    workflow = Workflow("W")
    workflow.parallel(Tool(lambda: calls.append("va") or "a", name="va"),
                       Tool(lambda: calls.append("vb") or "b", name="vb"))
    workflow.run(checkpoint_store=store, run_id="run-g")
    assert store.load("run-g") == {}  # a group entry is never recorded

    calls.clear()
    workflow2 = Workflow("W")
    workflow2.parallel(Tool(lambda: calls.append("va") or "a", name="va"),
                        Tool(lambda: calls.append("vb") or "b", name="vb"))
    workflow2.run(checkpoint_store=store, run_id="run-g")
    # ran again in full -- no replay skip for a parallel/consensus entry
    assert calls == ["va", "vb"] or calls == ["vb", "va"]


def test_non_json_serializable_output_raises_checkpoint_error():
    store = InMemoryCheckpointStore()
    workflow = Workflow("W")
    workflow.step(Tool(lambda: object(), name="weird"))
    with pytest.raises(CheckpointError, match="JSON-serializable"):
        workflow.run(checkpoint_store=store, run_id="run-weird")


def test_resume_with_a_changed_workflow_definition_raises_checkpoint_error():
    store = InMemoryCheckpointStore()
    workflow = Workflow("W")
    workflow.step(Tool(lambda: "a", name="a"))
    workflow.step(Tool(lambda: "b", name="b"))
    workflow.run(checkpoint_store=store, run_id="run-mismatch")

    # rerun with a *different* step at position 0 -- e.g. the script
    # changed between the checkpointed run and this resume attempt
    workflow2 = Workflow("W")
    workflow2.step(Tool(lambda: "x", name="different_tool"))
    workflow2.step(Tool(lambda: "b", name="b"))
    with pytest.raises(CheckpointError, match="workflow definition changed"):
        workflow2.run(checkpoint_store=store, run_id="run-mismatch")


def test_checkpoint_store_and_run_id_must_be_given_together():
    store = InMemoryCheckpointStore()
    workflow = Workflow("W")
    workflow.step(Tool(lambda: "a", name="a"))
    with pytest.raises(ValueError, match="together"):
        workflow.run(checkpoint_store=store)

    workflow2 = Workflow("W")
    workflow2.step(Tool(lambda: "a", name="a"))
    with pytest.raises(ValueError, match="together"):
        workflow2.run(run_id="orphan-run-id")


def test_file_checkpoint_store_persists_across_a_simulated_process_restart(tmp_path):
    path = str(tmp_path / "checkpoint.json")
    calls = []

    def failing_b():
        calls.append("b-attempt")
        raise RuntimeError("boom")

    store1 = FileCheckpointStore(path)
    workflow = Workflow("W")
    workflow.step(Tool(lambda: calls.append("a") or "a-output", name="a"))
    workflow.step(Tool(failing_b, name="b"))
    workflow.run(checkpoint_store=store1, run_id="run-file")

    # A brand new FileCheckpointStore instance pointed at the same path --
    # nothing in memory survived, only what's on disk, same as a real
    # process restart.
    calls.clear()
    store2 = FileCheckpointStore(path)
    workflow2 = Workflow("W")
    workflow2.step(Tool(lambda: calls.append("a") or "a-output", name="a"))
    workflow2.step(Tool(lambda: calls.append("b") or "b-output", name="b"))
    journal2 = workflow2.run(checkpoint_store=store2, run_id="run-file")

    assert journal2.status == "success"
    assert calls == ["b"]


def test_pretty_renders_the_replayed_annotation():
    store = InMemoryCheckpointStore()
    workflow = Workflow("W")
    workflow.step(Tool(lambda: "a", name="a"))
    workflow.run(checkpoint_store=store, run_id="run-pretty")

    workflow2 = Workflow("W")
    workflow2.step(Tool(lambda: "a", name="a"))
    journal2 = workflow2.run(checkpoint_store=store, run_id="run-pretty")
    assert "(replayed from checkpoint)" in journal2.pretty()
