"""AirLang's lexer: turns .airlang source text into a flat token stream.

Regex-based, single master pattern, longest-match-first via ordering (not
length) -- e.g. DURATION ("5m") must be tried before NUMBER so a duration
literal doesn't get split into NUMBER("5") + IDENT("m"). Whitespace and
`# ...` comments are skipped, never emitted as tokens -- AirLang is brace-
delimited (airlang-spec-v1.md section 4), so newlines carry no meaning.

Kept deliberately dependency-free and hand-written, same choice
airlang-spec-v1.md section 9 made for the parser and for the same reason:
AirLang's grammar is small and fixed, and owning the tokenizer means owning
the error messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


class AirLangSyntaxError(Exception):
    """Raised by the lexer or parser. Always carries a line/column so a
    real .airlang file's error points at the actual offending text, not just
    'somewhere in your file'."""

    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(f"line {line}, column {column}: {message}")
        self.message = message
        self.line = line
        self.column = column


@dataclass
class Token:
    kind: str
    value: str
    line: int
    column: int


# Order matters: earlier patterns win on a tie at the same position.
_TOKEN_SPEC = [
    ("COMMENT", r"#[^\n]*"),
    ("WHITESPACE", r"[ \t\r\n]+"),
    ("STRING", r'"(?:\\.|[^"\\])*"'),
    ("DURATION", r"\d+(?:\.\d+)?(?:ms|[smh])\b"),
    ("DOLLAR", r"\$\d+(?:\.\d+)?"),
    ("COMPARATOR", r"<=|>=|==|!="),
    ("LT", r"<"),
    ("GT", r">"),
    ("NUMBER", r"\d+(?:\.\d+)?"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("EQUALS", r"="),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_\-\.]*"),
]
_MASTER_RE = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPEC))


def tokenize(source: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    line = 1
    line_start = 0

    while pos < len(source):
        match = _MASTER_RE.match(source, pos)
        if match is None:
            column = pos - line_start + 1
            raise AirLangSyntaxError(f"unexpected character {source[pos]!r}", line, column)

        kind = match.lastgroup
        value = match.group()
        column = match.start() - line_start + 1

        if kind == "STRING" and "\n" in value:
            raise AirLangSyntaxError("strings cannot span multiple lines", line, column)

        if kind not in ("WHITESPACE", "COMMENT"):
            tokens.append(Token(kind=kind, value=value, line=line, column=column))

        newlines = value.count("\n")
        if newlines:
            line += newlines
            line_start = pos + value.rfind("\n") + 1
        pos = match.end()

    tokens.append(Token(kind="EOF", value="", line=line, column=pos - line_start + 1))
    return tokens
