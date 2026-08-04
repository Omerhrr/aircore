"""AirLang-M2: `ai run`/`ai trace` accepting a .airlang file instead of a .py
script (aircli/__main__.py's `_run_ail`/`_is_ail`). Covers the happy path
(a workflow using only AirLang-M1's non-blocked subset), that --json/--html
work identically to the .py path (both go through the same
_print_and_write_trace() now), and that every one of AirLang's own error
types (syntax, not-yet-supported, binding, policy) surfaces as one clear
stderr line + exit 1, not a raw traceback -- consistent with how
`ai run`/`ai trace` already handle a missing .py script.

BLOCKED_IF_AIL uses a standalone `if` (not immediately after `consensus`)
-- since AirLang-M3 (see test_ail_fallback.py), `if confidence < X {
Reviewer }` right after a `consensus judge { confidence true }` block
actually runs; it's `if` anywhere else that's still the general-branching
case with no runtime primitive.
"""

import json

import pytest

from aircli.__main__ import main

RESEARCH_AIL = """
provider mock

agent Literature {
    provider mock
}

agent Reddit {
    provider mock
}

workflow Research {
    parallel {
        Literature
        Reddit
    }
    consensus majority
    artifact Report
}
"""

BLOCKED_IF_AIL = """
provider mock
agent A { provider mock }
agent Reviewer { provider mock }
workflow W {
    step A
    if confidence < 0.85 { Reviewer }
}
"""

MISSING_TOOL_AIL = """
workflow W {
    step clone_repo
}
"""

SYNTAX_ERROR_AIL = "workflow W { step "

APPROVAL_AIL = """
provider mock
agent A { provider mock }
policy { approval A }
workflow W {
    step A
}
"""


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return str(path)


def test_run_executes_an_ail_workflow_end_to_end(tmp_path, capsys):
    script = _write(tmp_path, "research.airlang", RESEARCH_AIL)
    main(["run", script])
    out = capsys.readouterr().out
    assert "ai run summary" in out
    assert "workflow (Research): success  3 steps" in out


def test_trace_renders_an_ail_workflow_graph(tmp_path, capsys):
    script = _write(tmp_path, "research.airlang", RESEARCH_AIL)
    main(["trace", script])
    out = capsys.readouterr().out
    assert "workflow (Research)" in out
    assert "Literature" in out
    assert "consensus" in out


def test_trace_json_for_an_ail_workflow(tmp_path, capsys):
    script = _write(tmp_path, "research.airlang", RESEARCH_AIL)
    main(["trace", script, "--json"])
    out = capsys.readouterr().out
    chunk = out.split("\n--- ")[-1]
    json_text = chunk.split("\n", 1)[1].strip()
    parsed = json.loads(json_text)
    assert parsed["workflow"] == "Research"
    assert parsed["status"] == "success"


def test_trace_html_for_an_ail_workflow(tmp_path, capsys):
    script = _write(tmp_path, "research.airlang", RESEARCH_AIL)
    output = str(tmp_path / "out.html")
    main(["trace", script, "--html", "--output", output])
    out = capsys.readouterr().out
    assert f"wrote trace viewer to {output}" in out
    html_text = open(output, encoding="utf-8").read()
    assert "Research" in html_text


def test_run_on_ail_file_with_policy_approval_prompts_and_succeeds(tmp_path, capsys, monkeypatch):
    # `ai run` wires aircore.approval.cli_approval_callback in by default
    # for .airlang files (aircli/__main__.py's _run_ail) -- this is what lets a
    # `policy { approval <tool> }` line actually run interactively instead
    # of hitting the pre-flight PolicyViolation every other caller of
    # airlang.execute_file() gets without an approval_callback.
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    script = _write(tmp_path, "approval.airlang", APPROVAL_AIL)
    main(["run", script])
    out = capsys.readouterr().out
    assert "workflow (W): success  1 steps" in out


def test_run_on_ail_file_with_policy_approval_denied_fails_that_step(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    script = _write(tmp_path, "approval.airlang", APPROVAL_AIL)
    main(["run", script])
    out = capsys.readouterr().out
    assert "workflow (W): failed  1 steps" in out


def test_run_on_ail_file_with_unsupported_if_reports_a_clear_error(tmp_path, capsys):
    script = _write(tmp_path, "blocked.airlang", BLOCKED_IF_AIL)
    with pytest.raises(SystemExit) as excinfo:
        main(["run", script])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "not executable yet" in err


def test_run_on_ail_file_with_unbound_tool_reports_a_clear_error(tmp_path, capsys):
    script = _write(tmp_path, "missing_tool.airlang", MISSING_TOOL_AIL)
    with pytest.raises(SystemExit) as excinfo:
        main(["run", script])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "clone_repo" in err and "no implementation" in err


def test_run_on_ail_syntax_error_reports_a_clear_error(tmp_path, capsys):
    script = _write(tmp_path, "bad.airlang", SYNTAX_ERROR_AIL)
    with pytest.raises(SystemExit) as excinfo:
        main(["run", script])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "syntax error" in err


def test_run_on_missing_ail_file_reports_a_clear_error(tmp_path, capsys):
    missing = str(tmp_path / "nope.airlang")
    with pytest.raises(SystemExit) as excinfo:
        main(["run", missing])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "no such file" in err
