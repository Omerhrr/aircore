"""`ail parse <file> --ir` -- AirLang-M0's only CLI surface (a
debugging tool, not an executor -- see airlang/__main__.py's docstring).
Runs main() directly with an argv list, same pattern tests/test_aicli.py
already uses for `ai`, no subprocess needed."""

import json

import pytest

from airlang.__main__ import main


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return str(path)


def test_parse_prints_ir_as_json(tmp_path, capsys):
    script = _write(tmp_path, "hello.airlang", "workflow Hello {\n  step greet\n}\n")
    main(["parse", script, "--ir"])
    out = capsys.readouterr().out
    ir = json.loads(out)
    assert ir["workflow"]["name"] == "Hello"
    assert ir["workflow"]["body"] == [{"kind": "step", "ref": "greet"}]


def test_parse_on_missing_file_exits_with_error(tmp_path, capsys):
    missing = str(tmp_path / "nope.airlang")
    with pytest.raises(SystemExit) as excinfo:
        main(["parse", missing, "--ir"])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "no such file" in err


def test_parse_on_syntax_error_exits_with_error_and_position(tmp_path, capsys):
    script = _write(tmp_path, "bad.airlang", "workflow W { step ")
    with pytest.raises(SystemExit) as excinfo:
        main(["parse", script, "--ir"])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "line" in err and "column" in err
