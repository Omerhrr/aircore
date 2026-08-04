"""Self-contained HTML trace viewer: `ai trace <script.py> --html` renders
the same data `ai trace`'s text graph and `--json` already expose, but as
a single .html file you can open in any browser and click through --
node per step/group, a right-hand panel showing whatever the journal
actually recorded for it (status, latency, retries, usage/cost/tokens,
consensus metadata like confidence/reasoning, output or error).

Deliberately NOT built as a live-updating dashboard (no local server, no
websocket, no polling) -- that's real infrastructure (a process, a port,
a refresh strategy) with no proven need yet; this is a static snapshot of
a Journal, generated once, after the run, same as --json already is. One
HTML file, no external assets, no CDN scripts -- opens offline, nothing
to install, matching every other artifact this project produces.

Real limitation, not an oversight: a step's *input* (the prompt sent to a
ModelAgent, the arguments a Tool was called with) isn't in the journal at
all -- Journal only ever records what a step's execute() returned, not
what was passed to it (see journal.py; ModelAgent's own docstring notes
the same gap for its internal tool-calling loop). So a node's panel here
shows output/error, not a "prompt" field -- there's currently nothing to
show, and inventing a way to capture it wasn't part of what was asked
for. Fixing that would mean aircore's Executable protocol or ModelAgent
recording its inputs somewhere the journal can see, which is a real
gap worth closing later, not something this viewer can paper over.
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Tuple

Run = Tuple[str, Dict[str, Any]]  # (label, journal.to_dict())


def render_trace_html(runs: List[Run]) -> str:
    """`runs` is a list of (label, journal_dict) pairs -- one per Workflow
    or Session turn discovered in a script, the same set `ai trace`
    already iterates over. Returns one complete HTML document as a
    string; the caller writes it to a file."""
    payload = json.dumps(runs, default=str)
    title = html.escape(runs[0][0]) if runs else "ai trace"
    return _TEMPLATE.replace("__TITLE__", title).replace("__RUNS_JSON__", payload)


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ai trace -- __TITLE__</title>
<style>
  :root {
    --bg: #0f1117; --panel: #171a23; --border: #2a2f3d; --text: #e6e8ef;
    --muted: #8b90a3; --success: #3ecf8e; --failed: #f26d6d; --accent: #6ea8fe;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden;
  }
  #sidebar {
    width: 260px; flex-shrink: 0; background: var(--panel); border-right: 1px solid var(--border);
    overflow-y: auto; padding: 12px 0;
  }
  #sidebar h1 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--muted); padding: 0 16px; margin: 4px 0 12px; }
  .run-item { padding: 8px 16px; cursor: pointer; font-size: 13px; border-left: 3px solid transparent; }
  .run-item:hover { background: rgba(255,255,255,0.04); }
  .run-item.active { background: rgba(110,168,254,0.12); border-left-color: var(--accent); }
  .run-item .status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; }
  #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #header { padding: 16px 24px; border-bottom: 1px solid var(--border); }
  #header h2 { margin: 0 0 4px; font-size: 16px; }
  #header .meta { font-size: 12px; color: var(--muted); }
  #graph { flex: 1; overflow: auto; padding: 24px; display: flex; align-items: flex-start; gap: 16px; }
  .node {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 14px; min-width: 140px; cursor: pointer; font-size: 13px;
  }
  .node:hover { border-color: var(--accent); }
  .node.selected { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .node .name { font-weight: 600; margin-bottom: 4px; }
  .node .sub { color: var(--muted); font-size: 11px; }
  .node .badge { display: inline-block; padding: 1px 6px; border-radius: 10px; font-size: 10px;
    margin-top: 6px; }
  .badge.success { background: rgba(62,207,142,0.15); color: var(--success); }
  .badge.failed { background: rgba(242,109,109,0.15); color: var(--failed); }
  .group-box { border: 1px dashed var(--border); border-radius: 10px; padding: 12px; display: flex;
    flex-direction: column; gap: 8px; }
  .group-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  #panel {
    width: 380px; flex-shrink: 0; border-left: 1px solid var(--border); background: var(--panel);
    overflow-y: auto; padding: 20px;
  }
  #panel h3 { margin-top: 0; font-size: 14px; }
  #panel .field { margin-bottom: 14px; }
  #panel .field .k { font-size: 11px; text-transform: uppercase; color: var(--muted);
    letter-spacing: 0.04em; margin-bottom: 3px; }
  #panel .field .v { font-size: 13px; white-space: pre-wrap; word-break: break-word; }
  #panel .empty { color: var(--muted); font-size: 13px; }
  code, pre { font-family: ui-monospace, Menlo, Consolas, monospace; }
</style>
</head>
<body>
<div id="sidebar">
  <h1>Runs</h1>
  <div id="run-list"></div>
</div>
<div id="main">
  <div id="header"><h2 id="run-title">-</h2><div class="meta" id="run-meta"></div></div>
  <div id="graph"></div>
</div>
<div id="panel"><div class="empty">Click a node to see its details.</div></div>

<script>
const RUNS = __RUNS_JSON__;
let currentRunIndex = 0;

function statusClass(status) {
  return status === "success" ? "success" : (status ? "failed" : "");
}

function renderSidebar() {
  const list = document.getElementById("run-list");
  list.innerHTML = "";
  RUNS.forEach(([label, journal], i) => {
    const el = document.createElement("div");
    el.className = "run-item" + (i === currentRunIndex ? " active" : "");
    const dotColor = journal.status === "success" ? "var(--success))" : "var(--failed)";
    el.innerHTML = `<span class="status-dot" style="background:${journal.status === "success" ? "var(--success)" : "var(--failed)"}"></span>${label}`;
    el.onclick = () => { currentRunIndex = i; renderAll(); };
    list.appendChild(el);
  });
}

function nodeEl(labelHtml, subHtml, status, onClick) {
  const el = document.createElement("div");
  el.className = "node";
  const badge = status ? `<span class="badge ${statusClass(status)}">${status}</span>` : "";
  el.innerHTML = `<div class="name">${labelHtml}</div><div class="sub">${subHtml}</div>${badge}`;
  el.onclick = (ev) => { ev.stopPropagation(); onClick(); selectNode(el); };
  return el;
}

function selectNode(el) {
  document.querySelectorAll(".node.selected").forEach(n => n.classList.remove("selected"));
  el.classList.add("selected");
}

function fmtMs(v) { return v === null || v === undefined ? "-" : v.toFixed(2) + "ms"; }

function renderGraph() {
  const [label, journal] = RUNS[currentRunIndex];
  document.getElementById("run-title").textContent = journal.workflow || label;
  document.getElementById("run-meta").textContent =
    `status: ${journal.status || "-"}  |  duration: ${fmtMs(journal.duration_ms)}  |  steps: ${(journal.steps || []).length}`;

  const graph = document.getElementById("graph");
  graph.innerHTML = "";

  const groupsById = {};
  (journal.groups || []).forEach(g => groupsById[g.id] = g);
  const stepsByGroup = {};
  (journal.steps || []).forEach(s => {
    if (s.group_id) { (stepsByGroup[s.group_id] = stepsByGroup[s.group_id] || []).push(s); }
  });

  const seenGroups = new Set();
  (journal.steps || []).slice().sort((a, b) => a.id - b.id).forEach(step => {
    if (step.group_id) {
      if (seenGroups.has(step.group_id)) return;
      seenGroups.add(step.group_id);
      const group = groupsById[step.group_id];
      const box = document.createElement("div");
      box.className = "group-box";
      box.innerHTML = `<div class="group-label">${group.kind} group</div>`;
      (stepsByGroup[step.group_id] || []).sort((a, b) => a.id - b.id).forEach(member => {
        box.appendChild(nodeEl(escapeHtml(member.tool), fmtMs(member.duration_ms), member.status,
          () => showStepDetail(member)));
      });
      const groupWrap = document.createElement("div");
      groupWrap.style.display = "flex";
      groupWrap.style.flexDirection = "column";
      groupWrap.style.gap = "6px";
      const groupHeader = nodeEl(escapeHtml(group.kind + " consensus"), fmtMs(group.duration_ms), group.status,
        () => showGroupDetail(group));
      groupWrap.appendChild(groupHeader);
      groupWrap.appendChild(box);
      graph.appendChild(groupWrap);
    } else {
      graph.appendChild(nodeEl(escapeHtml(step.tool), fmtMs(step.duration_ms), step.status,
        () => showStepDetail(step)));
    }
  });
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s === null || s === undefined ? "" : String(s);
  return d.innerHTML;
}

function field(k, v) {
  if (v === null || v === undefined || v === "") return "";
  return `<div class="field"><div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(typeof v === "object" ? JSON.stringify(v, null, 2) : v)}</div></div>`;
}

function showStepDetail(step) {
  const panel = document.getElementById("panel");
  let html_ = `<h3>${escapeHtml(step.tool)}</h3>`;
  html_ += field("Status", step.status);
  html_ += field("Started", step.started_at);
  html_ += field("Finished", step.finished_at);
  html_ += field("Latency", fmtMs(step.duration_ms));
  if (step.retries) html_ += field("Retries", step.retries);
  if (step.retry_errors && step.retry_errors.length) html_ += field("Retry errors", step.retry_errors.join("\\n"));
  if (step.usage) {
    Object.entries(step.usage).forEach(([k, v]) => { html_ += field(k, v); });
  }
  if (step.metadata) {
    Object.entries(step.metadata).forEach(([k, v]) => { html_ += field(k, v); });
  }
  if (step.status === "success") {
    html_ += field("Output", step.output);
  } else if (step.error) {
    html_ += field("Error", step.error);
  }
  panel.innerHTML = html_;
}

function showGroupDetail(group) {
  const panel = document.getElementById("panel");
  let html_ = `<h3>${escapeHtml(group.kind)} group</h3>`;
  html_ += field("Status", group.status);
  html_ += field("Duration", fmtMs(group.duration_ms));
  html_ += field("Members", (group.tool_names || []).join(", "));
  panel.innerHTML = html_;
}

function renderAll() {
  renderSidebar();
  renderGraph();
  document.getElementById("panel").innerHTML = '<div class="empty">Click a node to see its details.</div>';
}

renderAll();
</script>
</body>
</html>
"""
