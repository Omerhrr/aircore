"""AirLang-M0's lexer: tokenizing .airlang source text. Covers the token kinds the
parser depends on, ordering ambiguities (duration vs number+ident, string
vs ident), and that syntax errors carry a real line/column."""

import pytest

from airlang.lexer import AirLangSyntaxError, tokenize


def _kinds(tokens):
    return [t.kind for t in tokens if t.kind != "EOF"]


def test_tokenizes_keywords_and_braces_as_ident_and_brace_tokens():
    tokens = tokenize("agent Researcher { }")
    assert _kinds(tokens) == ["IDENT", "IDENT", "LBRACE", "RBRACE"]


def test_model_names_with_dashes_and_dots_are_single_idents():
    tokens = tokenize("model deepseek-chat")
    assert tokens[1].value == "deepseek-chat"
    tokens2 = tokenize("model gemini-1.5-pro")
    assert tokens2[1].value == "gemini-1.5-pro"


def test_duration_is_not_split_into_number_and_ident():
    tokens = tokenize("timeout 5m")
    assert _kinds(tokens) == ["IDENT", "DURATION"]
    assert tokens[1].value == "5m"


def test_dollar_amount():
    tokens = tokenize("max_cost $2.50")
    assert tokens[1].kind == "DOLLAR"
    assert tokens[1].value == "$2.50"


def test_comparators():
    tokens = tokenize("if confidence < 0.85 { }")
    kinds = _kinds(tokens)
    assert kinds == ["IDENT", "IDENT", "LT", "NUMBER", "LBRACE", "RBRACE"]

    tokens2 = tokenize("a <= b >= c == d != e")
    assert _kinds(tokens2) == ["IDENT", "COMPARATOR", "IDENT", "COMPARATOR", "IDENT",
                                "COMPARATOR", "IDENT", "COMPARATOR", "IDENT"]


def test_string_literal():
    tokens = tokenize('message "Deploy?"')
    assert tokens[1].kind == "STRING"
    assert tokens[1].value == '"Deploy?"'


def test_comments_are_skipped():
    tokens = tokenize("import github # a comment\ntool clone_repo")
    assert _kinds(tokens) == ["IDENT", "IDENT", "IDENT", "IDENT"]


def test_unterminated_string_is_a_syntax_error_with_position():
    with pytest.raises(AirLangSyntaxError) as excinfo:
        tokenize('message "oops')
    assert excinfo.value.line == 1


def test_unexpected_character_is_a_syntax_error():
    with pytest.raises(AirLangSyntaxError) as excinfo:
        tokenize("agent @Bad { }")
    assert "@" in str(excinfo.value)


def test_line_and_column_tracking_across_newlines():
    tokens = tokenize("agent Foo {\n  provider deepseek\n}")
    provider_token = next(t for t in tokens if t.value == "provider")
    assert provider_token.line == 2
