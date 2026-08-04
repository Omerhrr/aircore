"""Sandboxed tool execution (aircore/sandbox.py) and the Tool.timeout fix
(scheduler.py's _execute_with_timeout) -- see sandbox.py's module
docstring for exactly what is and isn't covered: real process isolation
via a subprocess, a best-effort (socket-module-level, not kernel-level)
network egress allowlist, Unix-only best-effort memory limits, and a
genuine forced-kill timeout -- as opposed to Tool.timeout, which only
stops the scheduler from *waiting*, not the call itself.

Every function used as a sandboxed Tool's `fn` in this file is defined at
module scope, never as a lambda or a nested function -- multiprocessing's
"spawn" start method (used by run_sandboxed for portability) pickles `fn`
by import path, and only a real, importable module-level function works
that way. This is the same constraint sandbox.py documents for any real
caller of Tool(sandbox=...).
"""

import os
import socket
import time

import pytest

from aircore import (
    EgressDenied, Sandbox, SandboxedToolError, SandboxTimeout, Tool,
    ToolTimeout, Workflow, run_sandboxed,
)


def _return_pid():
    return os.getpid()


def _sleep_forever():
    time.sleep(30)
    return "should never get here"


def _raise_value_error():
    raise ValueError("boom")


def _connect_to_a_denied_host():
    s = socket.create_connection(("example.com", 80), timeout=2)
    s.close()
    return "connected"


def _connect_to_an_allowed_but_closed_port():
    # 127.0.0.1:1 is expected to have nothing listening in any normal
    # test/CI environment -- this proves the allowlist let the attempt
    # through to the real socket layer (a real ConnectionRefusedError
    # comes back, not EgressDenied), without needing actual internet
    # access or a real listening server.
    s = socket.create_connection(("127.0.0.1", 1), timeout=2)
    s.close()
    return "connected"


def test_sandboxed_call_runs_in_a_separate_process():
    result = run_sandboxed(_return_pid, Sandbox())
    assert result != os.getpid()


def test_sandbox_max_runtime_kills_a_hanging_call():
    with pytest.raises(SandboxTimeout, match="max_runtime"):
        run_sandboxed(_sleep_forever, Sandbox(max_runtime=0.3))


def test_sandbox_reraises_the_tools_own_exception():
    with pytest.raises(SandboxedToolError) as excinfo:
        run_sandboxed(_raise_value_error, Sandbox())
    assert excinfo.value.original_type == "ValueError"
    assert "boom" in excinfo.value.original_message


def test_egress_allowlist_denies_a_non_allowed_host():
    sandbox = Sandbox(allowed_hosts={"other.example.com"}, max_runtime=5)
    with pytest.raises(EgressDenied, match="example.com"):
        run_sandboxed(_connect_to_a_denied_host, sandbox)


def test_egress_allowlist_permits_a_listed_host_through_to_the_real_connect():
    sandbox = Sandbox(allowed_hosts={"127.0.0.1"}, max_runtime=5)
    with pytest.raises(SandboxedToolError) as excinfo:
        run_sandboxed(_connect_to_an_allowed_but_closed_port, sandbox)
    # got past the allowlist check -- the failure is a real socket-layer
    # refusal, not our own EgressDenied
    assert "Refused" in excinfo.value.original_type


def test_tool_with_sandbox_delegates_execution_to_the_subprocess():
    t = Tool(_return_pid, name="pid_tool", sandbox=Sandbox())
    result = t.execute()
    assert result != os.getpid()


def test_tool_without_sandbox_still_runs_in_process():
    t = Tool(_return_pid, name="pid_tool")
    assert t.execute() == os.getpid()


def test_tool_timeout_now_actually_stops_the_scheduler_from_waiting():
    # The pre-existing bug: Tool.timeout was stored on every Tool but the
    # scheduler never read it anywhere. This proves it's enforced now --
    # a thread-based wait, not a forced kill (see ToolTimeout's docstring
    # in executable.py), so this only asserts the *step* fails promptly,
    # not that the background call actually stopped running.
    workflow = Workflow("W")
    workflow.step(Tool(lambda: time.sleep(2) or "done", name="slow", timeout=0.1))
    journal = workflow.run()
    assert journal.status == "failed"
    assert "ToolTimeout" in journal.steps[0].error


def test_tool_without_a_timeout_is_completely_unaffected():
    workflow = Workflow("W")
    workflow.step(Tool(lambda: "fast", name="fast"))
    journal = workflow.run()
    assert journal.status == "success"
    assert journal.steps[0].output == "fast"
