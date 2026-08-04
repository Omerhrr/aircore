"""html_trace.py (`ai trace --html`): the static, self-contained HTML
trace viewer. Covers:

- render_trace_html() embeds valid, parseable JSON for every run
- the CLI wires --html into `ai trace`, writing a file in addition to
  (not instead of) the normal text/--json stdout output
- --output overrides the default <script>.trace.html path
- a run with a failed step still renders (no template crash on error/
  None fields)

Not tested here (out of scope for an offline suite): that the generated
HTML actually renders/behaves correctly in a real browser -- there's no
browser automation in this project's test suite. What IS verified is
that the document is well-formed enough to matter: the embedded JSON
parses, and the file contains the data a real browser's JS would need.
"""

import json
import re

import pytest

from aircli.__main__ import main
from aircli.html_trace import render_trace_html
from aircore import Workflow, tool


def _embedded_runs(html_text):
    match = re.search(r"const RUNS = (\[.*?\]);\n", html_text, re.S)
    assert match, "couldn't find embedded RUNS payload in generated HTML"
    return json.loads(match.group(1))


def test_render_trace_html_embeds_parseable_journal_data():
    @tool
    def hello():
        return "hi"

    workflow = Workflow("Greeting")
    workflow.step(hello)
    journal = workflow.run()

    html_text = render_trace_html([("workflow (Greeting)", journal.to_dict())])

    assert "<html" in html_text
    runs = _embedded_runs(html_text)
    assert len(runs) == 1
    label, journal_dict = runs[0]
    assert label == "workflow (Greeting)"
    assert journal_dict["status"] == "success"
    assert journal_dict["steps"][0]["tool"] == "hello"
    assert journal_dict["steps"][0]["output"] == "hi"


def test_render_trace_html_handles_a_failed_step():
    @tool
    def boom():
        raise ValueError("simulated failure")

    workflow = Workflow("Boom")
    workflow.step(boom)
    journal = workflow.run()

    html_text = render_trace_html([("workflow (Boom)", journal.to_dict())])
    runs = _embedded_runs(html_text)
    assert runs[0][1]["status"] == "failed"
    assert "simulated failure" in runs[0][1]["steps"][0]["error"]


def test_render_trace_html_with_multiple_runs():
    @tool
    def hello():
        return "hi"

    w1 = Workflow("First")
    w1.step(hello)
    j1 = w1.run()

    w2 = Workflow("Second")
    w2.step(hello)
    j2 = w2.run()

    html_text = render_trace_html([
        ("workflow (First)", j1.to_dict()),
        ("workflow (Second)", j2.to_dict()),
    ])
    runs = _embedded_runs(html_text)
    assert [label for label, _ in runs] == ["workflow (First)", "workflow (Second)"]


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


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return str(path)


def test_cli_trace_html_writes_a_file_and_keeps_printing_the_text_graph(tmp_path, capsys):
    script = _write(tmp_path, "wf.py", WORKFLOW_SCRIPT)
    output = str(tmp_path / "out.html")
    main(["trace", script, "--html", "--output", output])

    out = capsys.readouterr().out
    assert "workflow (Greeting)" in out  # text graph still printed
    assert f"wrote trace viewer to {output}" in out

    html_text = open(output, encoding="utf-8").read()
    runs = _embedded_runs(html_text)
    assert runs[0][1]["steps"][0]["output"] == "hi"


def test_cli_trace_html_default_output_path_is_derived_from_script(tmp_path, capsys):
    script = _write(tmp_path, "wf.py", WORKFLOW_SCRIPT)
    main(["trace", script, "--html"])

    expected = str(tmp_path / "wf.trace.html")
    out = capsys.readouterr().out
    assert f"wrote trace viewer to {expected}" in out
    assert (tmp_path / "wf.trace.html").exists()


def test_cli_trace_without_html_does_not_write_a_file(tmp_path, capsys):
    script = _write(tmp_path, "wf.py", WORKFLOW_SCRIPT)
    main(["trace", script])
    capsys.readouterr()
    assert not (tmp_path / "wf.trace.html").exists()
