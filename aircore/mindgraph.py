"""MindGraph: a compact, graph-shaped context memory -- the token-reduction
primitive everything else in this module builds on.

The problem this solves: both a single agent's tool-calling loop and a
multi-step Workflow tend to accumulate context linearly -- every tool
result, once produced, gets resent verbatim on every subsequent call for
the rest of the run. A 2,000-token tool result paid for once becomes a
2,000-token tax on every remaining turn. Over a long tool-calling loop, or
a long-running always-on agent, this is the dominant cost, not the actual
"thinking" tokens.

MindGraph's fix: don't carry raw tool/step output in context at all.
Instead, every piece of information becomes a Node -- a short, dense
`summary` (what a prompt actually sees) plus a `full_ref` pointer back to
where the real, verbatim value still lives (the Journal, MemoryScope,
whatever the caller already has). Building a prompt from a MindGraph means
walking a bounded neighborhood of nodes (`to_prompt_context(node_id,
hops=N)`), not replaying the whole history -- so context size stays
roughly constant as a run gets longer, instead of growing with it.

This module is deliberately standalone: a plain data structure plus a
renderer, no dependency on Scheduler, Workflow, or ModelAgent. Nothing
here decides *when* to summarize or *what* counts as "nearby" for a given
caller -- that's each integration's job:
  - ModelAgent's tool-calling loop (airpy) summarizes each tool result
    into a node before it goes back into the conversation, with an
    `expand_node` tool as the escape hatch back to the full value.
  - Workflow/consensus (a later milestone) will let parallel specialists
    read from one shared graph instead of each re-deriving their own copy
    of the same input.
  - TradingOS's AlwaysOnLoop (a later milestone) will use this same graph
    to progressively collapse old nodes so a run that's been going for
    hours doesn't cost more per cycle than one that just started.

No raw data is ever lost by summarizing it -- `full_ref` is an opaque
handle the caller defines (a Journal step id, a MemoryScope key, or the
value itself if the caller wants to keep it in memory); MindGraph never
interprets it, just carries it.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set


class MindGraphError(Exception):
    """Base class for MindGraph errors."""


class NodeNotFound(MindGraphError):
    """Raised by get()/link()/to_prompt_context() for an unknown node id."""


# Rough, provider-agnostic token estimate -- ~4 characters per token is the
# standard rule-of-thumb approximation used across this project wherever an
# exact tokenizer isn't worth the dependency (real usage, when available,
# comes from the provider's own response -- see events.py's UsageReported).
# This exists purely so callers/tests can compare "context size" before and
# after MindGraph without needing a real tokenizer installed.
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class Node:
    id: str
    kind: str
    summary: str
    # Opaque to MindGraph -- whatever the caller wants to be able to get
    # back to the full value with (a Journal step id, a MemoryScope key,
    # or the raw value itself). Never rendered into prompt context; only
    # returned by get_full() for a caller-side expand/lookup.
    full_ref: Any = None
    # Free-form, small metadata (e.g. {"tool": "sma", "ticker": "EURUSD"})
    # -- rendered into prompt context only if the caller opts in via
    # to_prompt_context(include_metadata=True), since most of it is for
    # filtering/debugging, not for the model to read.
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Ids of nodes this one depends on / was derived from. Directed,
    # node -> its inputs, so to_prompt_context can walk "what produced
    # this" outward from any starting node.
    edges: List[str] = field(default_factory=list)
    # Monotonically increasing per-graph sequence number, set by
    # MindGraph.add_node -- this is what compact()/aging (M4) orders by,
    # not insertion order in a dict (which Python does preserve, but a
    # graph that supports removal shouldn't rely on that).
    seq: int = 0

    def token_estimate(self) -> int:
        return estimate_tokens(self.summary)


class MindGraph:
    """A directed graph of Nodes plus a bounded, prompt-ready renderer.

    Not thread-safe by itself -- same expectation as Journal/EventBus in
    this project (single Scheduler/loop iteration owns it at a time); a
    caller sharing one MindGraph across threads (M3's consensus voters)
    is responsible for its own locking, same pattern as
    Scheduler._cost_lock in scheduler.py.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._seq = itertools.count(1)
        self._auto_id = itertools.count(1)
        # dedup_key -> node id. See add_node's `dedup_key` and
        # get_by_dedup_key -- this is what M3 (shared graphs across
        # parallel/consensus specialists) uses to recognize "this is the
        # same call as one already answered" instead of redoing the work.
        self._dedup_index: Dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def add_node(
        self,
        summary: str,
        kind: str = "fact",
        full_ref: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        edges: Optional[Iterable[str]] = None,
        node_id: Optional[str] = None,
        dedup_key: Optional[str] = None,
    ) -> Node:
        """Create and store a node, returning it. `edges` are the ids of
        nodes this one depends on / was derived from -- validated to
        already exist (a node can't depend on something that isn't in the
        graph yet), same "define before you reference" discipline as
        Workflow.bindings (see cross-step-data-flow.md).

        `dedup_key`, if given, registers this node under that key (see
        get_by_dedup_key) -- meant for a caller that wants "the same
        logical call" (e.g. the same tool name + arguments) to resolve to
        one node even if it happens more than once, rather than a fresh
        node each time. add_node() itself does NOT check for an existing
        key and skip creation -- that policy decision (dedup vs. always
        create) belongs to the caller; see get_by_dedup_key() for the
        check-first half of that pattern. A duplicate dedup_key silently
        overwrites the index entry to point at this newer node -- last
        write wins, same as a plain dict."""
        edges = list(edges or [])
        for parent_id in edges:
            if parent_id not in self._nodes:
                raise NodeNotFound(
                    f"add_node: edge references unknown node id {parent_id!r} -- "
                    f"nodes must be added before something can depend on them"
                )
        if node_id is None:
            node_id = f"n{next(self._auto_id)}"
        elif node_id in self._nodes:
            raise MindGraphError(f"add_node: node id {node_id!r} already exists")
        node = Node(
            id=node_id, kind=kind, summary=summary, full_ref=full_ref,
            metadata=dict(metadata or {}), edges=edges, seq=next(self._seq),
        )
        self._nodes[node_id] = node
        if dedup_key is not None:
            self._dedup_index[dedup_key] = node_id
        return node

    def get_by_dedup_key(self, dedup_key: str) -> Optional[Node]:
        """The check-first half of add_node's dedup_key pattern: returns
        the node already registered under this key, or None if nothing's
        been added with it yet (or the node it pointed at was later
        removed -- remove() does not clean up this index, so a stale key
        after remove() also returns None rather than a dangling id)."""
        node_id = self._dedup_index.get(dedup_key)
        if node_id is None or node_id not in self._nodes:
            return None
        return self._nodes[node_id]

    def get(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise NodeNotFound(f"no such node: {node_id!r}") from None

    def get_full(self, node_id: str) -> Any:
        """The escape hatch: return the node's full_ref, for a caller that
        wants to expand a summary back to its verbatim source. MindGraph
        itself never dereferences this -- interpreting it (e.g. looking up
        a Journal step id) is entirely the caller's job."""
        return self.get(node_id).full_ref

    def link(self, node_id: str, depends_on: str) -> None:
        """Add an edge after the fact (both nodes must already exist) --
        for the case where a dependency is only discovered after both
        nodes were created, rather than known at add_node() time."""
        node = self.get(node_id)
        if depends_on not in self._nodes:
            raise NodeNotFound(f"link: unknown node id {depends_on!r}")
        if depends_on not in node.edges:
            node.edges.append(depends_on)

    def remove(self, node_id: str) -> None:
        """Drop a node entirely (used by M4's compaction, once a node has
        been folded into a collapsed summary node and no longer needs to
        exist on its own). Dangling edges from other nodes that pointed at
        it are left as-is and simply skipped by to_prompt_context -- a
        node that referenced something now-collapsed is expected to also
        get relinked to the collapsed node by whoever called remove()."""
        self._nodes.pop(node_id, None)

    def all_nodes(self) -> List[Node]:
        """All nodes, oldest first (by seq) -- the order compaction (M4)
        walks in, and a reasonable default for anything that wants "every
        node" rather than a bounded neighborhood."""
        return sorted(self._nodes.values(), key=lambda n: n.seq)

    def collapse(self, node_ids: Iterable[str], summary: str, kind: str = "collapsed",
                 full_ref: Any = None, metadata: Optional[Dict[str, Any]] = None) -> Node:
        """Folds several existing nodes into one new node and removes the
        originals -- the actual mechanism M4 (long-run compaction) uses:
        a graph that's been running for hours doesn't have to keep every
        individual node forever, it can periodically replace a batch of
        old ones with a single denser summary, keeping to_prompt_context's
        token cost bounded by NODE COUNT staying bounded, not just by the
        neighborhood-walk trick M1 already provides for a single render.

        `summary` is the caller's job to compute (aircore has no opinion
        on how N old nodes should be summarized into one -- see
        compact_oldest() below for the common "just describe them as a
        batch" case, or a caller can write its own summarizer, e.g.
        TradingOS's AlwaysOnLoop aggregating cycle outcomes into
        win/loss/action counts).

        Any OTHER node that had an edge pointing at one of the collapsed
        ids is automatically relinked to point at the new collapsed node
        instead (deduplicated, so multiple collapsed dependencies don't
        create multiple identical edges) -- this is what remove()'s own
        docstring says is "expected to" happen, done here in one place so
        every caller gets it for free instead of reimplementing the
        relink themselves.

        Raises NodeNotFound if any id in node_ids doesn't exist. Returns
        the new collapsed node; its `edges` are empty (it doesn't depend
        on the now-gone originals -- it stands in for the information
        they carried, not for a dependency on them)."""
        ids_to_collapse = list(node_ids)
        for node_id in ids_to_collapse:
            if node_id not in self._nodes:
                raise NodeNotFound(f"collapse: unknown node id {node_id!r}")
        collapsed_set = set(ids_to_collapse)

        new_node = self.add_node(summary, kind=kind, full_ref=full_ref, metadata=metadata)

        for node_id in ids_to_collapse:
            self.remove(node_id)

        for node in self._nodes.values():
            if node.id == new_node.id:
                continue
            relinked = False
            new_edges = []
            seen = set()
            for edge in node.edges:
                target = new_node.id if edge in collapsed_set else edge
                if target in seen:
                    relinked = True  # a duplicate produced by the relink, drop it
                    continue
                seen.add(target)
                new_edges.append(target)
                if edge in collapsed_set:
                    relinked = True
            if relinked:
                node.edges = new_edges

        return new_node

    def compact_oldest(self, n: int, summarize: Callable[[List[Node]], str],
                        kind: str = "collapsed") -> Optional[Node]:
        """The common case built on collapse(): take the `n` oldest nodes
        (by seq -- all_nodes()'s order) and fold them into one, using
        `summarize(nodes) -> str` to describe the batch. Returns None
        (does nothing) if the graph has `n` or fewer nodes total -- there
        would be nothing left to keep individually addressable, and a
        caller looping "compact whenever len(graph) > threshold" every
        cycle shouldn't have to separately guard against
        over-compacting a small graph.

        This is what a long-running caller (M4's actual use case --
        TradingOS's AlwaysOnLoop calling this every cycle once the graph
        passes some size) uses to keep node count bounded forever
        regardless of how long the run goes on: call this once per
        cycle/tick with the same `n`/threshold policy, and the graph's
        size settles into a steady state instead of growing without
        limit."""
        if len(self._nodes) <= n or n <= 0:
            return None
        oldest = self.all_nodes()[:n]
        summary = summarize(oldest)
        return self.collapse([node.id for node in oldest], summary, kind=kind)

    def neighborhood(self, node_id: str, hops: int = 1) -> List[Node]:
        """Breadth-first walk outward from node_id along edges (both
        directions -- a node's dependencies AND anything that depends on
        it), up to `hops` steps, including the starting node. Returned
        oldest-first (by seq), which reads naturally as "what happened,
        in order" when rendered."""
        if node_id not in self._nodes:
            raise NodeNotFound(f"no such node: {node_id!r}")
        # reverse adjacency (who points at this node) computed once, not
        # per hop -- cheap since MindGraph is expected to hold hundreds to
        # low thousands of nodes per run, not millions.
        reverse: Dict[str, List[str]] = {}
        for n in self._nodes.values():
            for parent in n.edges:
                reverse.setdefault(parent, []).append(n.id)

        seen: Set[str] = {node_id}
        frontier: Set[str] = {node_id}
        for _ in range(hops):
            next_frontier: Set[str] = set()
            for nid in frontier:
                node = self._nodes.get(nid)
                if node is None:
                    continue
                for neighbor in itertools.chain(node.edges, reverse.get(nid, [])):
                    if neighbor not in seen and neighbor in self._nodes:
                        next_frontier.add(neighbor)
            seen |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        return sorted((self._nodes[nid] for nid in seen), key=lambda n: n.seq)

    def to_prompt_context(
        self,
        node_id: Optional[str] = None,
        hops: int = 1,
        include_metadata: bool = False,
        include_ids: bool = True,
    ) -> str:
        """Render a bounded neighborhood as prompt-ready text. If node_id
        is None, renders the whole graph (all_nodes()) -- useful for a
        first call in a fresh loop where there's no "current node" yet.

        This is the token-reduction lever in one function: the caller
        controls context size via `hops`, not by however many turns have
        happened so far. Output format is deliberately plain, dense lines
        (not JSON) -- markup costs tokens too."""
        nodes = self.all_nodes() if node_id is None else self.neighborhood(node_id, hops=hops)
        lines: List[str] = []
        for n in nodes:
            prefix = f"[{n.id}] " if include_ids else ""
            lines.append(f"{prefix}({n.kind}) {n.summary}")
            if include_metadata and n.metadata:
                meta_str = ", ".join(f"{k}={v}" for k, v in sorted(n.metadata.items()))
                lines.append(f"    {meta_str}")
        return "\n".join(lines)

    def token_estimate(self, node_id: Optional[str] = None, hops: int = 1) -> int:
        """Estimated token cost of what to_prompt_context() with the same
        arguments would produce -- for callers/tests comparing "context
        size" before and after adopting MindGraph, without needing to
        actually render the text."""
        return estimate_tokens(self.to_prompt_context(node_id, hops=hops))


# --- default summarizers -----------------------------------------------
#
# Deterministic, code-based summarization for common structured tool
# outputs -- zero extra LLM tokens spent, which matters because most tool
# results in a typical workflow (numeric/structured data, not free text)
# can be compressed this way for free. Free-text results (a web page, a
# long document) have no deterministic summary and need an actual
# cheap-model call -- that's left to the caller (M2's ModelAgent
# integration), not built into this module, since it needs a
# ModelProvider and MindGraph has no concept of one.
#
# A summarizer is any Callable[[Any], str]; this registry is a
# convenience default set, not a closed list -- callers can register
# their own per tool name (see M2).

def summarize_number_series(values: List[float], label: str = "series", digits: int = 4) -> str:
    """`[1.1000, 1.1005, ..., 1.1042]` -> `"series: 1.1042 (last of 50), rising over last 5"`.
    Used for indicator arrays (SMA/EMA/RSI over a candle window) -- the
    model almost always only cares about the latest value and the recent
    trend, not every element."""
    if not values:
        return f"{label}: (empty)"
    last = values[-1]
    trend = "flat"
    window = values[-5:] if len(values) >= 2 else values
    if len(window) >= 2:
        if window[-1] > window[0]:
            trend = "rising"
        elif window[-1] < window[0]:
            trend = "falling"
    return f"{label}: {last:.{digits}f} (last of {len(values)}), {trend} over last {len(window)}"


def summarize_ohlc_candles(candles: List[Dict[str, Any]], label: str = "candles") -> str:
    """Real OHLC bars (see market_state.py's recent_candles / iqair's
    Candle) -> one dense line instead of N*5 raw numbers. This is exactly
    the kind of tool output TradingOS's specialists otherwise pay for on
    every single tool-calling turn (see analysis_tools.py's
    get_recent_candles)."""
    if not candles:
        return f"{label}: (empty)"
    closes = [c["close"] for c in candles if "close" in c]
    high = max((c["high"] for c in candles if "high" in c), default=None)
    low = min((c["low"] for c in candles if "low" in c), default=None)
    trend = summarize_number_series(closes, label="close").split(": ", 1)[-1] if closes else "n/a"
    parts = [f"{label}: {len(candles)} bars", f"close {trend}"]
    if high is not None and low is not None:
        parts.append(f"range [{low:.4f}, {high:.4f}]")
    return ", ".join(parts)


def summarize_text(text: str, max_chars: int = 240) -> str:
    """Free-text fallback with no LLM call: collapse whitespace and
    truncate. This is deliberately dumb -- a real summary of long free
    text needs a cheap-model call, which belongs in the M2 integration
    (it needs a ModelProvider); this is only the zero-cost fallback for
    when that's unavailable or the text is already short enough not to
    need one."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 1].rstrip() + "…"


def default_summarize(value: Any, label: str = "value") -> str:
    """Best-effort summary for an arbitrary tool result, dispatching on
    shape. Used by M2's tool-calling integration as the fallback when a
    tool has no summarizer registered for it by name."""
    if isinstance(value, str):
        return summarize_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return f"{label}: {value!r}"
    if isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
        return summarize_number_series(list(value), label=label)
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value) and "close" in value[0]:
        return summarize_ohlc_candles(value, label=label)
    if isinstance(value, dict):
        keys = ", ".join(sorted(value.keys())[:8])
        return f"{label}: dict with keys [{keys}]" + (" (+more)" if len(value) > 8 else "")
    if isinstance(value, list):
        return f"{label}: list of {len(value)} items"
    return summarize_text(str(value))
