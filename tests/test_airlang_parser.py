"""AirLang-M0's parser: .airlang source -> IR (a plain, JSON-serializable dict
matching airlang-spec-v1.md section 7). Covers every construct in
airlang-spec-v1.md section 4, the "one file, one workflow" rule, and that
`if`/`let`/`approval` -- flagged in section 5 as having no runtime
equivalent yet -- still parse cleanly (rejecting them is the future
executor's job, not the parser's, per parser.py's docstring).
"""

import pytest

from airlang.lexer import AirLangSyntaxError
from airlang.parser import parse


def test_minimal_workflow_parses():
    ir = parse("""
    workflow Hello {
        step greet
    }
    """)
    assert ir["airlang_version"] == "0.1"
    assert ir["workflow"]["name"] == "Hello"
    assert ir["workflow"]["body"] == [{"kind": "step", "ref": "greet"}]


def test_import_tool_capability_provider_memory_declarations():
    ir = parse("""
    import github
    import slack
    tool clone_repo
    capability Network
    provider deepseek
    memory session
    workflow W { step clone_repo }
    """)
    assert ir["imports"] == ["github", "slack"]
    assert ir["tools"] == ["clone_repo"]
    assert ir["capabilities"] == ["Network"]
    assert ir["provider_default"] == "deepseek"
    assert ir["memory"] == "session"


def test_agent_block_full():
    ir = parse("""
    agent Researcher {
        provider deepseek
        model deepseek-chat
        capabilities { Network Filesystem }
        tools { clone_repo search_docs }
        prompt "Investigate the codebase."
    }
    workflow W { step Researcher }
    """)
    agent = ir["agents"][0]
    assert agent["name"] == "Researcher"
    assert agent["provider"] == "deepseek"
    assert agent["model"] == "deepseek-chat"
    assert agent["capabilities"] == ["Network", "Filesystem"]
    assert agent["tools"] == ["clone_repo", "search_docs"]
    assert agent["prompt"] == "Investigate the codebase."


def test_agent_block_defaults_when_fields_omitted():
    ir = parse("agent Bare { } \n workflow W { step Bare }")
    agent = ir["agents"][0]
    assert agent == {
        "name": "Bare", "provider": None, "model": None,
        "capabilities": [], "tools": [], "prompt": None,
    }


def test_policy_block():
    ir = parse("""
    policy {
        max_cost $2
        max_parallel 8
        timeout 5m
        approval deploy
    }
    workflow W { step x }
    """)
    assert ir["policy"] == {
        "max_cost": 2.0, "max_parallel": 8, "max_runtime": 300.0, "approval_for": ["deploy"],
    }


def test_timeout_units():
    assert parse("policy { timeout 500ms } \n workflow W { step x }")["policy"]["max_runtime"] == 0.5
    assert parse("policy { timeout 30s } \n workflow W { step x }")["policy"]["max_runtime"] == 30.0
    assert parse("policy { timeout 1h } \n workflow W { step x }")["policy"]["max_runtime"] == 3600.0


def test_parallel_block():
    ir = parse("""
    workflow W {
        parallel { Researcher Reviewer Verifier }
    }
    """)
    assert ir["workflow"]["body"] == [
        {"kind": "parallel", "members": ["Researcher", "Reviewer", "Verifier"]}
    ]


def test_parallel_requires_at_least_two_members():
    with pytest.raises(AirLangSyntaxError, match="at least 2 members"):
        parse("workflow W { parallel { Solo } }")


def test_consensus_bare_form():
    ir = parse("workflow W { consensus judge }")
    assert ir["workflow"]["body"] == [
        {"kind": "consensus", "strategy": "judge", "mode": None, "confidence": False}
    ]


def test_consensus_block_form():
    ir = parse("""
    workflow W {
        consensus {
            strategy judge
            mode synthesize
            confidence true
        }
    }
    """)
    assert ir["workflow"]["body"] == [
        {"kind": "consensus", "strategy": "judge", "mode": "synthesize", "confidence": True}
    ]


def test_consensus_rejects_unknown_strategy():
    with pytest.raises(AirLangSyntaxError, match="unknown consensus strategy"):
        parse("workflow W { consensus vibes }")


def test_consensus_block_requires_strategy():
    with pytest.raises(AirLangSyntaxError, match="must set strategy"):
        parse("workflow W { consensus { mode select } }")


def test_artifact_bare_and_with_fields():
    ir = parse("""
    workflow W {
        artifact Report
        artifact Findings { schema AuditFinding }
        artifact Summary { type markdown }
    }
    """)
    assert ir["workflow"]["body"] == [
        {"kind": "artifact", "name": "Report", "type": None, "schema": None},
        {"kind": "artifact", "name": "Findings", "type": None, "schema": "AuditFinding"},
        {"kind": "artifact", "name": "Summary", "type": "markdown", "schema": None},
    ]


def test_if_block_with_bare_ref_body():
    ir = parse("""
    workflow W {
        if confidence < 0.85 {
            HumanReviewer
        }
    }
    """)
    assert ir["workflow"]["body"] == [{
        "kind": "if", "field": "confidence", "op": "<", "value": 0.85,
        "then": [{"kind": "ref", "name": "HumanReviewer"}],
    }]


def test_if_supports_every_comparator():
    for op in ["<", ">", "<=", ">=", "==", "!="]:
        ir = parse(f"workflow W {{ if confidence {op} 0.5 {{ Foo }} }}")
        assert ir["workflow"]["body"][0]["op"] == op


def test_let_binds_an_artifact_reference():
    ir = parse("""
    workflow W {
        artifact Report
        let report = artifact Report
    }
    """)
    assert ir["workflow"]["body"][1] == {
        "kind": "let", "name": "report", "value": {"kind": "artifact_ref", "name": "Report"},
    }


def test_approval_block_inside_workflow():
    ir = parse("""
    workflow W {
        approval { message "Deploy?" }
    }
    """)
    assert ir["workflow"]["body"] == [{"kind": "approval", "message": "Deploy?"}]


def test_bare_identifier_is_sugar_for_a_ref_step():
    ir = parse("workflow W { HumanReviewer }")
    assert ir["workflow"]["body"] == [{"kind": "ref", "name": "HumanReviewer"}]


def test_full_audit_example_matches_spec_shape():
    ir = parse("""
    import github
    tool clone_repo
    capability Network
    provider deepseek

    agent Researcher {
        provider deepseek
        model deepseek-chat
        capabilities { Network }
        tools { clone_repo }
    }

    policy {
        max_cost $2
        max_parallel 8
        timeout 5m
    }

    workflow Audit {
        step clone_repo
        parallel { Researcher Researcher Researcher }
        consensus { strategy judge mode synthesize confidence true }
        if confidence < 0.85 { HumanReviewer }
        artifact AuditReport { type markdown }
    }
    """)
    assert ir["workflow"]["name"] == "Audit"
    assert [n["kind"] for n in ir["workflow"]["body"]] == [
        "step", "parallel", "consensus", "if", "artifact",
    ]


def test_zero_workflows_is_a_syntax_error():
    with pytest.raises(AirLangSyntaxError, match="exactly one workflow"):
        parse("import github")


def test_two_workflows_is_a_syntax_error():
    with pytest.raises(AirLangSyntaxError, match="exactly one workflow"):
        parse("workflow A { step x } \n workflow B { step y }")


def test_unknown_top_level_keyword_is_a_syntax_error():
    with pytest.raises(AirLangSyntaxError, match="top-level declaration"):
        parse("banana foo \n workflow W { step x }")


def test_unclosed_brace_is_a_syntax_error():
    with pytest.raises(AirLangSyntaxError):
        parse("workflow W { step x")


def test_ir_round_trips_through_json():
    import json
    ir = parse("workflow W { step x }")
    assert json.loads(json.dumps(ir)) == ir
