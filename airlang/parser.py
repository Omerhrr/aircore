"""AirLang's parser: token stream (lexer.py) -> IR (a plain, JSON-serializable
dict matching airlang-spec-v1.md section 7 exactly).

Hand-written recursive descent, per airlang-spec-v1.md section 9. Keywords
(`agent`, `step`, `parallel`, ...) are not their own token kind -- the
lexer only knows IDENT; this parser tells them apart by comparing an
IDENT token's value against the fixed keyword set below. That keeps the
lexer generic and puts all of AirLang's actual grammar in one place.

This module ONLY produces IR. It has no dependency on aircore or airpy --
per airlang-spec-v1.md section 3, parsing and execution are deliberately
separate concerns, so a future non-Python executor could consume the same
IR this produces without this module changing at all. It also does not
reject constructs airlang-spec-v1.md section 5 flags as not-yet-executable
(`if`, `approval`, `let`) -- the parser's job is to accept the full
grammar; deciding what can actually run is the executor's job, once one
exists (AirLang-M1).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .lexer import AirLangSyntaxError, Token, tokenize

IR_VERSION = "0.1"

_TOP_LEVEL_KEYWORDS = {"import", "tool", "capability", "provider", "memory", "agent", "policy", "workflow"}
_CONSENSUS_STRATEGIES = {"majority", "unanimous", "judge"}
_CONSENSUS_MODES = {"select", "synthesize"}
_MEMORY_SCOPES = {"session", "project", "temporary"}
_COMPARATORS = {"<", ">", "<=", ">=", "==", "!="}
_DURATION_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse(source: str) -> Dict[str, Any]:
    """Parses one .airlang file's source text into IR. Raises AirLangSyntaxError
    on any grammar violation, with a line/column pointing at the actual
    offending token."""
    tokens = tokenize(source)
    return _Parser(tokens).parse_program()


def parse_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return parse(f.read())


class _Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # -- token stream helpers -------------------------------------------------

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        token = self._tokens[self._pos]
        if token.kind != "EOF":
            self._pos += 1
        return token

    def _error(self, message: str, token: Optional[Token] = None) -> AirLangSyntaxError:
        token = token or self._peek()
        return AirLangSyntaxError(message, token.line, token.column)

    def _expect(self, kind: str) -> Token:
        token = self._peek()
        if token.kind != kind:
            raise self._error(f"expected {kind}, got {token.kind} {token.value!r}")
        return self._advance()

    def _expect_ident(self, value: Optional[str] = None) -> Token:
        token = self._peek()
        if token.kind != "IDENT":
            raise self._error(f"expected an identifier, got {token.kind} {token.value!r}")
        if value is not None and token.value != value:
            raise self._error(f"expected '{value}', got '{token.value}'")
        return self._advance()

    def _at_ident(self, *values: str) -> bool:
        token = self._peek()
        return token.kind == "IDENT" and token.value in values

    def _ident_list_until_rbrace(self) -> List[str]:
        names = []
        while not self._at_kind("RBRACE"):
            names.append(self._expect("IDENT").value)
        return names

    def _at_kind(self, kind: str) -> bool:
        return self._peek().kind == kind

    # -- top level --------------------------------------------------------

    def parse_program(self) -> Dict[str, Any]:
        ir: Dict[str, Any] = {
            "airlang_version": IR_VERSION,
            "imports": [],
            "tools": [],
            "capabilities": [],
            "provider_default": None,
            "memory": None,
            "agents": [],
            "policy": {"max_cost": None, "max_parallel": None, "max_runtime": None, "approval_for": []},
            "workflow": None,
        }
        workflow_count = 0

        while not self._at_kind("EOF"):
            token = self._peek()
            if token.kind != "IDENT" or token.value not in _TOP_LEVEL_KEYWORDS:
                raise self._error(
                    f"expected a top-level declaration (import/tool/capability/provider/"
                    f"memory/agent/policy/workflow), got {token.kind} {token.value!r}"
                )
            keyword = token.value
            if keyword == "import":
                self._advance()
                ir["imports"].append(self._expect("IDENT").value)
            elif keyword == "tool":
                self._advance()
                ir["tools"].append(self._expect("IDENT").value)
            elif keyword == "capability":
                self._advance()
                ir["capabilities"].append(self._expect("IDENT").value)
            elif keyword == "provider":
                self._advance()
                ir["provider_default"] = self._expect("IDENT").value
            elif keyword == "memory":
                self._advance()
                scope = self._expect("IDENT").value
                if scope not in _MEMORY_SCOPES:
                    raise self._error(f"unknown memory scope '{scope}' (expected one of {sorted(_MEMORY_SCOPES)})")
                ir["memory"] = scope
            elif keyword == "agent":
                ir["agents"].append(self._parse_agent())
            elif keyword == "policy":
                ir["policy"] = self._parse_policy()
            elif keyword == "workflow":
                if workflow_count:
                    raise self._error(
                        "a .airlang file may declare exactly one workflow (see airlang-spec-v1.md "
                        "section 1, 'one file, one workflow') -- this is a second one"
                    )
                ir["workflow"] = self._parse_workflow()
                workflow_count += 1

        if workflow_count == 0:
            raise self._error("a .airlang file must declare exactly one workflow block", self._peek())

        return ir

    # -- agent --------------------------------------------------------------

    def _parse_agent(self) -> Dict[str, Any]:
        self._expect_ident("agent")
        name = self._expect("IDENT").value
        self._expect("LBRACE")

        agent: Dict[str, Any] = {
            "name": name, "provider": None, "model": None,
            "capabilities": [], "tools": [], "prompt": None,
        }
        while not self._at_kind("RBRACE"):
            field = self._expect("IDENT").value
            if field == "provider":
                agent["provider"] = self._expect("IDENT").value
            elif field == "model":
                agent["model"] = self._expect("IDENT").value
            elif field == "prompt":
                agent["prompt"] = _unquote(self._expect("STRING").value)
            elif field == "capabilities":
                self._expect("LBRACE")
                agent["capabilities"] = self._ident_list_until_rbrace()
                self._expect("RBRACE")
            elif field == "tools":
                self._expect("LBRACE")
                agent["tools"] = self._ident_list_until_rbrace()
                self._expect("RBRACE")
            else:
                raise self._error(f"unknown agent field '{field}'")
        self._expect("RBRACE")
        return agent

    # -- policy ---------------------------------------------------------------

    def _parse_policy(self) -> Dict[str, Any]:
        self._expect_ident("policy")
        self._expect("LBRACE")
        policy: Dict[str, Any] = {"max_cost": None, "max_parallel": None, "max_runtime": None, "approval_for": []}
        while not self._at_kind("RBRACE"):
            field = self._expect("IDENT").value
            if field == "max_cost":
                policy["max_cost"] = _parse_dollar(self._expect("DOLLAR"))
            elif field == "max_parallel":
                policy["max_parallel"] = int(_parse_number(self._expect("NUMBER")))
            elif field == "timeout":
                policy["max_runtime"] = _parse_duration(self._expect("DURATION"))
            elif field == "approval":
                policy["approval_for"].append(self._expect("IDENT").value)
            else:
                raise self._error(f"unknown policy field '{field}'")
        self._expect("RBRACE")
        return policy

    # -- workflow -------------------------------------------------------------

    def _parse_workflow(self) -> Dict[str, Any]:
        self._expect_ident("workflow")
        name = self._expect("IDENT").value
        self._expect("LBRACE")
        body = self._parse_workflow_body()
        self._expect("RBRACE")
        return {"name": name, "body": body}

    def _parse_workflow_body(self) -> List[Dict[str, Any]]:
        body: List[Dict[str, Any]] = []
        while not self._at_kind("RBRACE"):
            body.append(self._parse_workflow_stmt())
        return body

    def _parse_workflow_stmt(self) -> Dict[str, Any]:
        if self._at_ident("step"):
            self._advance()
            return {"kind": "step", "ref": self._expect("IDENT").value}
        if self._at_ident("parallel"):
            self._advance()
            self._expect("LBRACE")
            members = self._ident_list_until_rbrace()
            self._expect("RBRACE")
            if len(members) < 2:
                raise self._error("parallel { } needs at least 2 members to be meaningful")
            return {"kind": "parallel", "members": members}
        if self._at_ident("consensus"):
            return self._parse_consensus()
        if self._at_ident("artifact"):
            return self._parse_artifact()
        if self._at_ident("if"):
            return self._parse_if()
        if self._at_ident("let"):
            return self._parse_let()
        if self._at_ident("approval"):
            return self._parse_approval()
        if self._at_ident("memory"):
            self._advance()
            scope = self._expect("IDENT").value
            if scope not in _MEMORY_SCOPES:
                raise self._error(f"unknown memory scope '{scope}' (expected one of {sorted(_MEMORY_SCOPES)})")
            return {"kind": "memory", "scope": scope}

        # Bare identifier sugar: `HumanReviewer` alone means "run this
        # agent/tool as a step" -- see airlang-spec-v1.md section 7's `if`
        # example ({"kind": "ref", "name": "HumanReviewer"}).
        if self._at_kind("IDENT"):
            return {"kind": "ref", "name": self._advance().value}

        raise self._error(
            f"expected a workflow statement (step/parallel/consensus/artifact/if/let/"
            f"approval/memory, or a bare agent/tool name), got {self._peek().kind} "
            f"{self._peek().value!r}"
        )

    def _parse_consensus(self) -> Dict[str, Any]:
        self._expect_ident("consensus")
        if self._at_kind("LBRACE"):
            self._advance()
            node = {"kind": "consensus", "strategy": None, "mode": None, "confidence": False}
            while not self._at_kind("RBRACE"):
                field = self._expect("IDENT").value
                if field == "strategy":
                    strategy = self._expect("IDENT").value
                    if strategy not in _CONSENSUS_STRATEGIES:
                        raise self._error(
                            f"unknown consensus strategy '{strategy}' (expected one of "
                            f"{sorted(_CONSENSUS_STRATEGIES)})"
                        )
                    node["strategy"] = strategy
                elif field == "mode":
                    mode = self._expect("IDENT").value
                    if mode not in _CONSENSUS_MODES:
                        raise self._error(f"unknown consensus mode '{mode}' (expected one of {sorted(_CONSENSUS_MODES)})")
                    node["mode"] = mode
                elif field == "confidence":
                    value = self._expect("IDENT").value
                    if value not in ("true", "false"):
                        raise self._error(f"confidence expects true or false, got '{value}'")
                    node["confidence"] = value == "true"
                else:
                    raise self._error(f"unknown consensus field '{field}'")
            self._expect("RBRACE")
            if node["strategy"] is None:
                raise self._error("consensus { } block must set strategy")
            return node

        # Bare form: `consensus judge`
        strategy = self._expect("IDENT").value
        if strategy not in _CONSENSUS_STRATEGIES:
            raise self._error(f"unknown consensus strategy '{strategy}' (expected one of {sorted(_CONSENSUS_STRATEGIES)})")
        return {"kind": "consensus", "strategy": strategy, "mode": None, "confidence": False}

    def _parse_artifact(self) -> Dict[str, Any]:
        self._expect_ident("artifact")
        name = self._expect("IDENT").value
        node = {"kind": "artifact", "name": name, "type": None, "schema": None}
        if self._at_kind("LBRACE"):
            self._advance()
            while not self._at_kind("RBRACE"):
                field = self._expect("IDENT").value
                if field == "type":
                    node["type"] = self._expect("IDENT").value
                elif field == "schema":
                    node["schema"] = self._expect("IDENT").value
                else:
                    raise self._error(f"unknown artifact field '{field}'")
            self._expect("RBRACE")
        return node

    def _parse_if(self) -> Dict[str, Any]:
        self._expect_ident("if")
        field = self._expect("IDENT").value
        op_token = self._peek()
        if op_token.kind == "COMPARATOR":
            op = self._advance().value
        elif op_token.kind in ("LT", "GT"):
            op = self._advance().value
        else:
            raise self._error(f"expected a comparator (< > <= >= == !=), got {op_token.kind} {op_token.value!r}")
        if op not in _COMPARATORS:
            raise self._error(f"unknown comparator '{op}'")
        value = _parse_number(self._expect("NUMBER"))
        self._expect("LBRACE")
        then_body = self._parse_workflow_body()
        self._expect("RBRACE")
        return {"kind": "if", "field": field, "op": op, "value": value, "then": then_body}

    def _parse_let(self) -> Dict[str, Any]:
        self._expect_ident("let")
        name = self._expect("IDENT").value
        self._expect("EQUALS")
        self._expect_ident("artifact")
        artifact_name = self._expect("IDENT").value
        return {"kind": "let", "name": name, "value": {"kind": "artifact_ref", "name": artifact_name}}

    def _parse_approval(self) -> Dict[str, Any]:
        self._expect_ident("approval")
        self._expect("LBRACE")
        node = {"kind": "approval", "message": None}
        while not self._at_kind("RBRACE"):
            field = self._expect("IDENT").value
            if field == "message":
                node["message"] = _unquote(self._expect("STRING").value)
            else:
                raise self._error(f"unknown approval field '{field}'")
        self._expect("RBRACE")
        return node


def _unquote(raw: str) -> str:
    # raw includes the surrounding quotes, e.g. '"Deploy?"'
    inner = raw[1:-1]
    return inner.replace('\\"', '"').replace("\\\\", "\\")


def _parse_number(token: Token) -> float:
    value = float(token.value)
    return int(value) if value.is_integer() else value


def _parse_dollar(token: Token) -> float:
    return float(token.value[1:])


def _parse_duration(token: Token) -> float:
    for unit in ("ms", "s", "m", "h"):
        if token.value.endswith(unit):
            number = float(token.value[: -len(unit)])
            return number * _DURATION_UNITS[unit]
    raise AirLangSyntaxError(f"malformed duration '{token.value}'", token.line, token.column)
