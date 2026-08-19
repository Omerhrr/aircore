"""Offline tests for aircore/mindgraph.py (M9-M1): the standalone
Node/MindGraph data structure and its bounded prompt-context renderer, plus
the default deterministic summarizers. No Scheduler/Workflow/ModelAgent
integration is exercised here -- that's M2+."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from aircore.mindgraph import (
    MindGraph, MindGraphError, NodeNotFound,
    estimate_tokens, default_summarize, summarize_number_series,
    summarize_ohlc_candles, summarize_text,
)


# --- Node/graph basics ----------------------------------------------------

def test_add_node_assigns_an_id_and_increasing_seq():
    g = MindGraph()
    a = g.add_node("first fact")
    b = g.add_node("second fact")
    assert a.id != b.id
    assert b.seq > a.seq
    assert len(g) == 2


def test_add_node_with_explicit_id():
    g = MindGraph()
    g.add_node("fact", node_id="my-id")
    assert "my-id" in g
    assert g.get("my-id").summary == "fact"


def test_add_node_rejects_duplicate_explicit_id():
    g = MindGraph()
    g.add_node("fact", node_id="dup")
    with pytest.raises(MindGraphError):
        g.add_node("fact 2", node_id="dup")


def test_add_node_rejects_edge_to_unknown_node():
    g = MindGraph()
    with pytest.raises(NodeNotFound):
        g.add_node("fact", edges=["does-not-exist"])


def test_get_unknown_node_raises():
    g = MindGraph()
    with pytest.raises(NodeNotFound):
        g.get("nope")


def test_full_ref_round_trips_without_being_rendered():
    g = MindGraph()
    raw = {"huge": "payload", "with": list(range(1000))}
    node = g.add_node("compact summary", full_ref=raw)
    assert g.get_full(node.id) is raw
    # the raw payload must never leak into prompt context -- that's the
    # entire point of full_ref being opaque.
    assert "huge" not in g.to_prompt_context(node.id)


def test_link_adds_edge_after_the_fact():
    g = MindGraph()
    a = g.add_node("a")
    b = g.add_node("b")
    g.link(b.id, depends_on=a.id)
    assert a.id in g.get(b.id).edges


def test_link_rejects_unknown_target():
    g = MindGraph()
    a = g.add_node("a")
    with pytest.raises(NodeNotFound):
        g.link(a.id, depends_on="ghost")


def test_remove_drops_a_node():
    g = MindGraph()
    a = g.add_node("a")
    g.remove(a.id)
    assert a.id not in g
    assert len(g) == 0


def test_all_nodes_returns_oldest_first():
    g = MindGraph()
    a = g.add_node("a")
    b = g.add_node("b")
    c = g.add_node("c")
    assert [n.id for n in g.all_nodes()] == [a.id, b.id, c.id]


# --- neighborhood / bounded context ---------------------------------------

def test_neighborhood_walks_dependencies_and_dependents():
    g = MindGraph()
    root = g.add_node("root")
    child = g.add_node("child", edges=[root.id])
    grandchild = g.add_node("grandchild", edges=[child.id])
    unrelated = g.add_node("unrelated")

    # from child, 1 hop should reach root (a dependency) and grandchild
    # (a dependent), but not the unrelated node.
    neighborhood_ids = {n.id for n in g.neighborhood(child.id, hops=1)}
    assert neighborhood_ids == {root.id, child.id, grandchild.id}
    assert unrelated.id not in neighborhood_ids


def test_neighborhood_hops_zero_is_just_the_node_itself():
    g = MindGraph()
    root = g.add_node("root")
    g.add_node("child", edges=[root.id])
    assert [n.id for n in g.neighborhood(root.id, hops=0)] == [root.id]


def test_neighborhood_of_unknown_node_raises():
    g = MindGraph()
    with pytest.raises(NodeNotFound):
        g.neighborhood("ghost")


def test_context_size_stays_bounded_as_graph_grows():
    """The actual point of this whole module: adding many more nodes to
    the graph must NOT grow the token cost of a bounded neighborhood
    around one node, the way a linear transcript would."""
    g = MindGraph()
    root = g.add_node("root fact, fairly short summary text")
    small_context_tokens = g.token_estimate(root.id, hops=1)

    # simulate a long-running loop: 500 more, unrelated nodes added after
    # the root -- a linear transcript approach would have every one of
    # these paid for on every subsequent turn.
    for i in range(500):
        g.add_node(f"unrelated fact number {i}, with some more descriptive words")

    same_context_tokens = g.token_estimate(root.id, hops=1)
    assert same_context_tokens == small_context_tokens

    # but the whole-graph render (hops=None / node_id=None) DOES grow --
    # confirming the bound comes from neighborhood(), not from the data
    # structure silently dropping nodes.
    whole_graph_tokens = g.token_estimate(node_id=None)
    assert whole_graph_tokens > same_context_tokens * 50


# --- to_prompt_context rendering ------------------------------------------

def test_to_prompt_context_renders_kind_and_summary():
    g = MindGraph()
    g.add_node("EURUSD is rising", kind="observation", node_id="n1")
    text = g.to_prompt_context("n1", hops=0)
    assert "(observation)" in text
    assert "EURUSD is rising" in text


def test_to_prompt_context_omits_metadata_by_default():
    g = MindGraph()
    g.add_node("fact", metadata={"ticker": "EURUSD"}, node_id="n1")
    text = g.to_prompt_context("n1", hops=0)
    assert "EURUSD" not in text


def test_to_prompt_context_includes_metadata_when_asked():
    g = MindGraph()
    g.add_node("fact", metadata={"ticker": "EURUSD"}, node_id="n1")
    text = g.to_prompt_context("n1", hops=0, include_metadata=True)
    assert "ticker=EURUSD" in text


def test_to_prompt_context_none_renders_the_whole_graph():
    g = MindGraph()
    g.add_node("a")
    g.add_node("b")
    text = g.to_prompt_context(None)
    assert "a" in text and "b" in text


def test_to_prompt_context_can_hide_ids():
    g = MindGraph()
    g.add_node("fact", node_id="visible-id")
    text = g.to_prompt_context("visible-id", hops=0, include_ids=False)
    assert "visible-id" not in text


# --- estimate_tokens -------------------------------------------------------

def test_estimate_tokens_empty_string_is_zero():
    assert estimate_tokens("") == 0


def test_estimate_tokens_roughly_four_chars_per_token():
    assert estimate_tokens("x" * 400) == 100


# --- default summarizers ---------------------------------------------------

def test_summarize_number_series_reports_last_and_trend():
    text = summarize_number_series([1.10, 1.11, 1.12, 1.13, 1.14, 1.20], label="SMA(10)")
    assert "SMA(10)" in text
    assert "1.2000" in text
    assert "rising" in text


def test_summarize_number_series_falling():
    text = summarize_number_series([1.20, 1.18, 1.15, 1.10], label="close")
    assert "falling" in text


def test_summarize_number_series_empty():
    assert summarize_number_series([]) == "series: (empty)"


def test_summarize_ohlc_candles_dense_one_liner():
    candles = [
        {"time": 1, "open": 1.10, "high": 1.12, "low": 1.09, "close": 1.11},
        {"time": 2, "open": 1.11, "high": 1.13, "low": 1.10, "close": 1.125},
    ]
    text = summarize_ohlc_candles(candles)
    assert "2 bars" in text
    assert "range [1.0900, 1.1300]" in text


def test_summarize_ohlc_candles_empty():
    assert summarize_ohlc_candles([]) == "candles: (empty)"


def test_summarize_text_short_text_passthrough():
    assert summarize_text("short and sweet") == "short and sweet"


def test_summarize_text_truncates_long_text():
    long_text = "word " * 200
    result = summarize_text(long_text, max_chars=50)
    assert len(result) <= 50
    assert result.endswith("…")


def test_summarize_text_collapses_whitespace():
    assert summarize_text("a\n\n\tb   c") == "a b c"


def test_default_summarize_dispatches_on_type():
    assert "42" in default_summarize(42, label="answer")
    assert "list of 3" in default_summarize([1, "a", {}], label="mixed")
    assert "keys" in default_summarize({"a": 1, "b": 2}, label="obj")
    assert "rising" in default_summarize([1.0, 1.1, 1.2], label="trend")


def test_default_summarize_dispatches_ohlc_shaped_dicts():
    candles = [{"close": 1.1, "open": 1.0, "high": 1.2, "low": 0.9}]
    assert "bars" in default_summarize(candles, label="candles")


def test_default_summarize_long_string_is_truncated():
    result = default_summarize("x" * 1000)
    assert len(result) < 1000


# --- token reduction is real, measured end-to-end ---------------------------

def test_summarized_node_is_dramatically_smaller_than_raw_payload():
    """The whole point, proven with real numbers: summarizing a realistic
    tool output (200 OHLC candles, the kind analysis_tools.py's
    get_recent_candles returns) into a MindGraph node costs far fewer
    tokens than the raw payload would if resent on every subsequent
    conversation turn."""
    import json

    candles = [
        {"time": i, "open": 1.10 + i * 0.0001, "high": 1.101 + i * 0.0001,
         "low": 1.099 + i * 0.0001, "close": 1.1005 + i * 0.0001}
        for i in range(200)
    ]
    raw_tokens = estimate_tokens(json.dumps(candles))

    g = MindGraph()
    node = g.add_node(summarize_ohlc_candles(candles), kind="tool_result", full_ref=candles)
    summary_tokens = node.token_estimate()

    assert summary_tokens < raw_tokens / 20
    # and the raw data is still reachable if actually needed
    assert g.get_full(node.id) is candles
