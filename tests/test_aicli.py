"""aircli's `ai run`/`ai trace`: covers what "finishing" the CLI meant --
catching it up to Session (added well after the CLI was first built,
never wired in), plus the --json trace output that Journal.to_json()
already supported but the CLI never exposed. Runs main() directly with an
argv list against small script files written to tmp_path, capturing
stdout via pytest's capsys -- no subprocess needed since main() is a
plain function.
"""

import json

import pytest

from aircli.__main__ import main

WORKFLOW_SCRIPT = """
from aircore import Workflow, tool

@tool
def hello():
    return "hi"

if __name__ == "__main__":
    workflow = Workflow("Greeting")
    workflow.step(hello)
    workflow.run()
"""

UNRUN_WORKFLOW_SCRIPT = """
from aircore import Workflow, tool

@tool
def hello():
    return "hi"

if __name__ == "__main__":
    workflow = Workflow("NeverRun")
    workflow.step(hello)
    # .run() deliberately not called
"""

SESSION_SCRIPT = """
from airpy import Session, MockProvider

if __name__ == "__main__":
    session = Session("assistant", MockProvider(response="hello there"))
    session.send("hi")
    session.send("how are you")
"""

MIXED_SCRIPT = """
from aircore import Workflow, tool
from airpy import Session, MockProvider

@tool
def hello():
    return "hi"

if __name__ == "__main__":
    workflow = Workflow("Greeting")
    workflow.step(hello)
    workflow.run()

    session = Session("assistant", MockProvider(response="hello there"))
    session.send("hi")
"""


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return str(path)


def test_run_reports_a_completed_workflow(tmp_path, capsys):
    script = _write(tmp_path, "wf.py", WORKFLOW_SCRIPT)
    main(["run", script])
    out = capsys.readouterr().out
    assert "ai run summary" in out
    assert "workflow (Greeting): success  1 steps" in out


def test_run_skips_a_workflow_that_never_called_run(tmp_path, capsys):
    script = _write(tmp_path, "unrun.py", UNRUN_WORKFLOW_SCRIPT)
    main(["run", script])
    out = capsys.readouterr().out
    assert "ai run summary" not in out


def test_run_reports_a_used_session(tmp_path, capsys):
    script = _write(tmp_path, "session.py", SESSION_SCRIPT)
    main(["run", script])
    out = capsys.readouterr().out
    assert "ai run summary" in out
    assert "2 turns, last turn success" in out
    assert "session-" in out


def test_run_reports_both_workflow_and_session_in_one_script(tmp_path, capsys):
    script = _write(tmp_path, "mixed.py", MIXED_SCRIPT)
    main(["run", script])
    out = capsys.readouterr().out
    assert "workflow (Greeting): success" in out
    assert "1 turns, last turn success" in out


def test_trace_renders_the_workflow_graph(tmp_path, capsys):
    script = _write(tmp_path, "wf.py", WORKFLOW_SCRIPT)
    main(["trace", script])
    out = capsys.readouterr().out
    assert "workflow (Greeting)" in out
    assert "Greeting" in out
    assert "hello" in out


def test_trace_json_prints_valid_parseable_journal(tmp_path, capsys):
    script = _write(tmp_path, "wf.py", WORKFLOW_SCRIPT)
    main(["trace", script, "--json"])
    out = capsys.readouterr().out

    # Each "--- header ---\n{json}" block: split on the header marker, then
    # drop the header line itself, leaving just the JSON body.
    chunk = out.split("\n--- ")[-1]
    json_text = chunk.split("\n", 1)[1].strip()
    parsed = json.loads(json_text)
    assert parsed["workflow"] == "Greeting"
    assert parsed["status"] == "success"


def test_trace_reports_one_block_per_session_turn(tmp_path, capsys):
    script = _write(tmp_path, "session.py", SESSION_SCRIPT)
    main(["trace", script])
    out = capsys.readouterr().out
    assert "turn 1" in out
    assert "turn 2" in out


def test_trace_json_for_a_session_prints_one_valid_journal_per_turn(tmp_path, capsys):
    script = _write(tmp_path, "session.py", SESSION_SCRIPT)
    main(["trace", script, "--json"])
    out = capsys.readouterr().out

    chunks = out.split("\n--- ")[1:]  # each: "header ---\n{json}\n"
    json_chunks = [c.split("\n", 1)[1].strip() for c in chunks if "turn" in c.split("\n", 1)[0]]
    assert len(json_chunks) == 2
    for chunk in json_chunks:
        parsed = json.loads(chunk)
        assert parsed["status"] == "success"


def test_trace_with_nothing_to_report_prints_a_clear_message(tmp_path, capsys):
    script = _write(tmp_path, "unrun.py", UNRUN_WORKFLOW_SCRIPT)
    main(["trace", script])
    err = capsys.readouterr().err
    assert "no completed Workflow or used Session" in err


def test_run_on_missing_script_exits_with_error(tmp_path, capsys):
    missing = str(tmp_path / "does_not_exist.py")
    with pytest.raises(SystemExit) as excinfo:
        main(["run", missing])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "no such file" in err
