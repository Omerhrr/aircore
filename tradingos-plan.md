# TradingOS — Plan

**An autonomous trading operating system that continuously observes, reasons, executes, learns, and improves within explicit risk constraints — built on top of `aircore`/`airpy`/`airlang` (the runtime) and `iqair` (the broker SDK), all consumed as published packages.**

Status: **All 8 milestones built** (see the `tradingos/` project, offline tests passing: 75/75). Milestones 1-7 as before (one manual-trigger cycle -> Market State Engine + Event Bus -> a second specialist with real consensus -> an always-on event-driven loop with reconnect -> a backtesting engine replaying real historical data -> a Learning Engine + Knowledge Base -> a FastAPI gateway). **Milestone 8** turned that gateway into a full multi-instrument system with a real Vue 3 + Bun frontend (`web/`): pick any (ticker, category) iqair actually supports, run "test mode" (manual backtest/trade, both through the exact same risk gate) or "agent mode" (`AlwaysOnLoop`, autonomous, every cycle's reasoning visible), and browse the learned-pattern handbook. A "real mode" -- switching the account to REAL capital -- was explicitly requested during this milestone's scoping and explicitly declined; this is enforced as a hard code gate, not just convention: `TradingOSConfig` itself refuses to construct with `balance_mode` other than `"PRACTICE"` (see `config.py`'s `__post_init__`), and every write route the gateway exposes is enumerated in `gateway.WRITE_ROUTES`, covered by an explicit allowlist test. All 8 milestones from the original build order are complete; remaining work (learning-loop feedback into prompts, a dedicated candle-history endpoint, async backtests) is real but beyond the original plan's scope -- see the project's own README "Future work" section.

~~Known open item: `pip install airlang` currently fails on PyPI...~~ **Fixed in source, pending release.** Root cause confirmed: PyPI's `airlang` 0.4.0 wheel declares `Requires-Dist: airpy`, but `airpy`'s actual PyPI distribution name is `airpyy` -- pip resolves by distribution name, so this points at the wrong (or a nonexistent) project. `packaging/airlang/pyproject.toml` already declared the correct `airpyy` dependency in source; the bug was that PyPI still had the stale 0.4.0 build. Bumped `airlang/__init__.py` to `0.4.1` and verified locally: a clean `python -m build` of the staged package now produces a wheel whose METADATA reads `Requires-Dist: airpyy`. This isn't live on PyPI yet -- that only happens when a new GitHub Release is cut (`.github/workflows/publish.yml`, `skip-existing: true` means the old 0.4.0 publish is a permanent no-op otherwise). Still not a blocker for TradingOS (not a dependency here either way).

---

## 0. Constraints and standing decisions

- **New, separate project.** TradingOS is its own repository (`tradingos`), not a subfolder of the `aircore` monorepo. It depends on the published packages only:
  ```
  pip install aircore airpyy airlang airclii iqair
  ```
  (recall: `airpyy`/`airclii` are the PyPI *distribution* names — you still `import airpy`, run the `ai` command, etc. `import iqair` is unchanged.)
- **Only the published API surface.** No modifications to `aircore`/`airpy`/`airlang` internals for this project. Anything TradingOS needs that those packages don't provide is built as TradingOS's own application code, calling the public API.
- **PRACTICE only, for the entire build.** Every milestone below runs against the IQ Option PRACTICE balance. Switching any part of this to REAL is a deliberate, separate decision made later, by the user — not something any milestone here does automatically.
- **Standing rule, restated:** the assistant (Claude) will not execute trades or move real money on the user's behalf, and will not flip any autonomy switch to REAL capital, regardless of how the system is configured. This applies throughout every phase below, including Phase 8 (Autonomous Mode).
- **News is explicitly out of scope for now.** No news/sentiment specialist agent in any milestone below. May be added later as its own decision (needs a real data source — `iqair` has no news endpoint).
- **Risk Engine is a hard code gate, not an advisory step.** "The AI cannot bypass this" is only true if enforced in the Execution Tool's own Python code (refuse before calling `iqair`), not merely as one workflow step an agent's reasoning could route around. Every milestone that includes execution builds it this way.

---

## 1. Architecture — full 18-layer vision

```
                +----------------------+
                |      Vue.js UI       |
                +----------+-----------+
                           |
                     FastAPI Gateway
                           |
        +------------------+------------------+
        |                                     |
        |                              REST/WebSocket
        |                                     |
+-------v--------+                 +----------v----------+
|   TradingOS    |                 |       iqair          |
|                |                 |                      |
|  aircore       | <-------------> | login                |
|  airpy         |                 | streaming            |
|  airlang       |                 | trading              |
+-------+--------+                 | positions            |
        |                          | balance              |
        |                          | candles              |
        |                          +----------------------+
        |
+-------v------------------------------+
| Multi-Agent Runtime                  |
+---------------------------------------+
```

### Layer 1 — iqair (already built)
Pure SDK, no AI. Authentication, PRACTICE/REAL switching, streaming (candles, price ticks, positions, orders, balance, asset status, payout), all six trading modes (turbo, binary, digital, forex, cfd, crypto), portfolio/history, buy/sell/close.

### Layer 2 — Market State Engine
TradingOS's own in-memory model, kept current by `iqair`'s streams. Agents read from this, never from `iqair` directly.
```
MarketState.EURUSD: price, spread, ATR, trend, EMA, RSI, MACD,
volume, momentum, volatility, liquidity, payout, open/closed,
recent trades, confidence
```

### Layer 3 — Event Bus
Everything becomes an event (`PriceChanged`, `NewCandle`, `TradeOpened`, `TradeClosed`, `BalanceChanged`, `ConnectionLost`, `MarketOpened`, `MarketClosed`, `PayoutChanged`). Subsystems subscribe; nothing polls. Starts as an in-process pub/sub (mirroring the subscriber pattern already in `iqair/ws/client.py`) — no Redis needed until TradingOS is actually deployed as a FastAPI service.

### Layer 4 — Tool Calling
Every `iqair` capability becomes an `aircore.Tool` (`get_balance`, `buy`, `sell`, `get_positions`, `stream_prices`, `stream_candles`, `calculate_indicator`, `find_pattern`, `estimate_volatility`, `risk_assessment`, `portfolio_status`, ...). Agents never touch `iqair` directly — only tools. This is exactly how `aircore` is designed to be used.

### Layer 5 — Agent Team
Specialists, each with one job: Market Analyst, Technical Analyst, Macro Observer, Pattern Finder, Risk Manager, Trader, Execution Agent, Recorder, Critic, Teacher. (News specialist deliberately excluded for now.)

### Layer 6 — Strategy Engine
Strategies as plugins (Momentum, Breakout, Scalping, Mean Reversion, Liquidity Grab, Range, Trend Following, Volatility, Binary-specific, Digital-specific), each emitting `{signal, confidence, risk, reason}`.

### Layer 7 — Consensus
`aircore`'s existing consensus primitive (`Workflow.consensus()` / `JudgeConsensus`), already built and tested — specialists vote, a judge reduces to one decision + confidence.

### Layer 8 — Risk Engine
Owns the money; the AI never does. Hard-coded rules: max loss/day, max drawdown, max consecutive losses, max position size, max leverage, max trades/hour, market blacklist, cooldown, Kelly sizing, volatility adjustment, correlation limits. Built on `aircore.Policy.approval_for` + a deterministic `approval_callback` — but enforced as a real code gate in the Execution Tool (see §0), not just a workflow step.

### Layer 9 — Execution Engine
Turns a decision (`BUY EURUSD`) into `iqair` calls. Handles retries, reconnects, order confirmation, partial failures, slippage, execution logs.

### Layer 10 — Recorder
Every decision stored: market snapshot, reasoning, indicators, prompt, response, tool calls, trade, result, P&L, duration, confidence. Largely already available "for free" via `aircore`'s journal.

### Layer 11 — Learning Engine
Every trade becomes training data: `Before → Decision → Outcome → Reflection → Lesson`, accumulating weighted lessons over time. Needs real trade history to exist first — deferred until later phases.

### Layer 12 — Knowledge Base
The AI's own trading handbook, built from Layer 11's accumulated lessons: `Pattern → Observed → Success rate → Confidence → Conditions → Markets → Time → Volatility → Win rate`.

### Layer 13 — Memory
`aircore.Memory`'s existing scopes: session (today's trades), project (entire account history), permanent (things learned over months).

### Layer 14 — Simulation Engine
Backtest before ever risking money: replay historical candles through the same `Workflow`, observe decisions, compute P&L, at scale (millions of candles).

### Layer 15 — Frontend
Vue dashboard: market heatmap, signals, confidence, P&L, reasoning, risk meter, open/closed trades, agent conversations, consensus, memory, learning, charts, streaming.

### Layer 16 — Observability
Per-run drill-down (market → decision → tool → execution → outcome), LangSmith-style. Largely already available via `aircore`'s journal and `ai trace --json/--html`.

### Layer 17 — aircore Integration
Not a separate build — this is what naturally happens by building TradingOS on `aircore`. Every workflow (parallel specialists → consensus → risk → execution → recorder → learning) is automatically visible in `aircore`'s existing visual debugger.

### Layer 18 — Autonomous Mode
```
User → Deposit → Constraints → Enable Autonomous Mode
```
Example policy: capital $500, markets EURUSD/GBPUSD, risk 2%, max drawdown 10%, no overnight, London/New York sessions only, goal = maximize Sharpe ratio. From then on: Observe → Reason → Research → Analyze → Vote → Consensus → Risk Check → Execute → Record → Reflect → Learn → Repeat, with no manual intervention required while operating within the defined constraints — but never exceeding the Risk Engine's hard limits, and never enabled against real capital by the assistant (see §0).

---

## 2. What's already built vs. what's new

**Already available, reused as-is (no new runtime code):**
- Tool calling (`aircore.Tool`)
- Consensus (`Workflow.consensus()`, `JudgeConsensus`)
- Policy/approval gate (`Policy.approval_for` + `approval_callback`)
- Journal + `ai trace` (covers most of Recorder + Observability)
- Memory scopes (session/project/permanent)
- iqair itself (Layer 1, already live-verified)

**New, but straightforward TradingOS application code:**
- Market State Engine (in-memory model fed by `iqair` streams)
- Event Bus (in-process pub/sub to start)
- Execution Engine (Tool wrapping `api.broker.trade` + retry/verification logic)
- Risk Engine (rule functions wired into an `approval_callback`, enforced as a hard code gate)

**Real, separate undertakings — deferred until the core loop is proven on PRACTICE:**
- Learning Engine + Knowledge Base (needs real trade history first)
- Simulation/Backtesting Engine
- Frontend (Vue app)
- FastAPI gateway / Redis-backed event bus (only once actually deploying as a service)
- Autonomous Mode with real capital (last, and never enabled by the assistant)

---

## 3. Build order

1. **Milestone 1 — one full cycle, one asset, manual trigger.**
   Repo scaffold + Market Data Tool + one specialist (Technical Analyst) + Risk Engine (hard-coded rule gate) + Execution Engine + Recorder (journal-based). One manual-trigger decision cycle on EURUSD, PRACTICE balance, Risk Engine auto-approves (no human pause) but can still refuse. No news, no consensus yet (single specialist).

2. **Milestone 2 — Market State Engine + Event Bus.**
   Replace ad hoc polling with a continuously-updated in-memory market model, kept current by `iqair` streams. Still manually triggered per decision cycle, but backed by live data instead of one-off fetches.

3. **Milestone 3 — second specialist + real consensus.**
   Add a second specialist (Pattern Finder or Macro Observer — not News) and wire `Workflow.consensus()` for real multi-voter agreement.

4. **Milestone 4 — always-on loop.**
   Move from manual trigger to event-driven repeated `Workflow.run()`s (still PRACTICE, still auto-approved by the Risk Engine). This is "Layer 18 minus real money."

5. **Milestone 5 — Simulation/Backtesting Engine.**
   Replay historical candles through the same `Workflow` before any drawdown-sensitive tuning or scaling up.

6. **Milestone 6 — Learning Engine + Knowledge Base.**
   Once there's real trade history from Milestones 1–5 to learn from.

7. **Milestone 7 — Frontend + FastAPI gateway.**
   Vue dashboard, REST/WebSocket gateway, wiring the Event Bus to Redis if/when this becomes an actual deployed service.

8. **Milestone 8 — Autonomous Mode.**
   User-defined capital/risk constraints, full observe→reason→execute→learn loop with no manual intervention within those constraints. Last, only after everything above has run reliably on PRACTICE, and never enabled against real capital by the assistant.

---

## 4. Open decisions (deferred, not blocking Milestone 1)

- News/sentiment data source (deliberately excluded for now — see §0).
- Second specialist's exact focus for Milestone 3 (Pattern Finder vs. Macro Observer vs. something else).
- Redis vs. continued in-process pub/sub for the Event Bus — decide when Milestone 7 (FastAPI gateway) is actually reached.
- Exact Risk Engine rule set/parameters (max loss/day, max drawdown, position sizing method, etc.) — needed before Milestone 1's Risk Engine can be finalized, but the gate mechanism itself doesn't depend on the specific numbers.
