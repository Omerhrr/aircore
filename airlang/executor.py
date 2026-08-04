"""AirLang-M1: the IR executor. Walks the IR parser.py produces and makes the
exact same airpy calls a human would type by hand -- `Agent(...)`,
`Workflow(...).step(...)`, `.parallel(...)`, `.consensus(...)`. This is
the first module in the `airlang` package that imports airpy (parser.py and
lexer.py import neither aircore nor airpy -- see their docstrings); per
airlang-spec-v1.md section 3, `airlang` imports `airpy`, `airpy` never imports
`airlang`.

Scope: only the IR node kinds airlang-spec-v1.md section 8 (AirLang-M1) lists as
non-blocked execute here -- step, ref, parallel, consensus (majority/
unanimous/judge), artifact, let, and top-level agent/policy/memory
declarations. A workflow-body-level `memory` statement still raises
AirLangNotYetSupportedError -- per section 5, it has no real runtime primitive
to compile to yet, and silently ignoring it would be worse than refusing.
A body-level `approval { message }` step (pausing unconditionally, not
tied to any one tool) is also still unsupported -- see below for what *is*
now supported.

`let <name> = artifact <ArtifactName>` (§5.4) is now closed. Two pieces
had to land together, both scoped narrowly:

1. Producer linkage: an `artifact <ArtifactName>` node immediately
   following a `step`/`ref`/`consensus` node in the body is now treated
   as that node's producer -- build_workflow() peeks one node ahead (the
   same lookahead shape already used for `if`-after-`consensus`) and
   binds that step's output via `Workflow.step(..., as_=...)` /
   `Workflow.consensus(..., as_=...)` (the latter needed a small, real
   aircore change -- see aircore/workflow.py's Bindings section for why
   consensus groups, uniquely among groups, can bind at all). An
   `artifact` node with no immediately preceding step/consensus (e.g. the
   first statement in a body, or one following a bare `parallel` with no
   `consensus`) stays metadata-only, exactly as before -- there is
   nothing to bind it to.
2. Aliasing: `let <name> = artifact <ArtifactName>` means later `agent`
   prompts should be able to say `{name}`, not `{ArtifactName}`. A
   pre-scan pass over the whole body (before any step is built) collects
   every `let` into an `artifact_name -> let_name` map; when the
   producer-linkage step above binds an artifact's producer, it binds
   under the `let` name if one was declared for that artifact, or the
   artifact's own name otherwise. This means `let` can appear anywhere in
   the file, not just immediately after its artifact -- the alias is
   resolved before generation starts. Referencing an artifact name in
   `let` that never appears as an `artifact` node anywhere in the body is
   an AirLangBindingError, not a silent no-op.

Agent prompts (§4.4/§5.4) that contain a literal `{...}` are now compiled
to `PromptTemplate(prompt_string)` with `prompt_bindings=workflow.bindings`
(see airpy/model_agent.py) instead of a plain string -- so `agent Foo {
prompt "Summarize {report}" }` resolves `{report}` from whatever's bound
so far at execute() time, same as hand-written airpy code. A prompt with
no `{` in it is still passed through as a plain string, completely
unaffected. Referencing a name that's never bound by the time that agent
runs fails loudly at that step (PromptTemplate's existing "missing
template variable" behavior), not silently -- this includes the ordering
mistake of referencing an artifact/let name before its producing step has
run.

Policy-level `approval <tool>` (airlang-spec-v1.md section 4.5) IS now
supported, now that aircore has a real primitive for it (aircore/approval.py,
added alongside this change): each name in `ir["policy"]["approval_for"]`
maps straight onto `Policy(approval_for=...)`. Running an .airlang file whose
policy names an approval-gated tool requires passing an
`approval_callback` to `execute_ir`/`execute_file` (which forward it
to `Workflow.run()`) -- omit it and building still succeeds, but running
raises the same pre-flight PolicyViolation a hand-written airpy workflow
would. `aircli` wires in `aircore.approval.cli_approval_callback` by default
for `ai run`/`ai trace` on `.airlang` files (see aircli/__main__.py), so this
works interactively out of the box; a non-interactive caller of
execute_file() must supply its own.

`if` (AirLang-M3, added once aircore grew a real primitive for it -- see
aircore/consensus.py's `fallback`/`fallback_below`): no longer universally
blocked. An `if <field> < <value> { <single ref> }` node immediately
following a `consensus` node folds into that same consensus step's
confidence-gated fallback -- ConsensusGroup(fallback=..., fallback_below=
..., fallback_field=...) -- rather than becoming a separate step. This is
still the narrow case airlang-spec-v1.md section 5.1 scoped out, not general
branching: `if` in any other position, with any comparator other than
`<`, or with a `then` body that isn't exactly one bare reference, still
raises AirLangNotYetSupportedError. `if confidence < X` additionally requires
the preceding consensus to be `judge` with `confidence true` set -- only
JudgeConsensus reports a `confidence` value at all (majority/unanimous
never do), so anything else fails loudly at build time instead of
silently never triggering.

Two real scope decisions worth being explicit about, since the IR sketch
in airlang-spec-v1.md section 7 doesn't fully resolve them:

1. `consensus` always reduces the immediately preceding `parallel`
   block's results (aircore's result-reuse mechanism -- see
   aircore/workflow.py's ParallelResults) rather than re-running voters.
   AirLang's grammar never lists explicit consensus voters separately from a
   parallel block (airlang-spec-v1.md section 4.6's examples always pair
   them), so this executor treats `consensus` with no immediately
   preceding `parallel` as an error, not as "zero voters."
2. `artifact` still does not construct a ModelAgent(output_schema=...)
   call or validate `type`/`schema` against real output -- that's a
   separate, still-open gap (§5.3's schema-enforcement question) from the
   producer-linkage one this change closed. What changed: `artifact`
   nodes are still recorded on the built Workflow as `.airlang_artifacts` (a
   plain list of dicts) purely for inspection and add no step of their
   own to the journal, but an `artifact` immediately following a
   `step`/`ref`/`consensus` node is now used to bind *that node's*
   output in `workflow.bindings` (see the "let" section above) --
   `artifact` went from purely inert metadata to metadata-plus-a-binding-
   side-effect, without gaining any new step of its own.

A `consensus judge` node backs the judge with the workflow's top-level
default `provider` declaration -- AirLang currently has no syntax for naming
a distinct judge provider (a real, small gap, flagged here rather than
silently defaulted to something surprising).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aircore import (
    ApprovalCallback, Capability, Database, Email, Filesystem, Memory, Network,
    Payments, Policy, Workflow, majority, unanimous,
)
from airpy import Agent, JudgeConsensus, MockProvider, PromptTemplate

from .bindings import Bindings, load_bindings_for
from .parser import parse_file

_BUILTIN_CAPABILITIES: Dict[str, Capability] = {
    "Network": Network, "Filesystem": Filesystem, "Email": Email,
    "Payments": Payments, "Database": Database,
}
# "approval" here is the workflow-body-level `approval { message }` step
# (pause unconditionally, not tied to any one tool) -- NOT the policy-
# level `policy { approval <tool> }` construct, which now maps to
# Policy.approval_for below and is fully supported. `let` used to be here
# too -- see build_workflow()'s producer-linkage pass for why it isn't
# anymore.
_NOT_YET_SUPPORTED_KINDS = {"approval"}


class AirLangNotYetSupportedError(Exception):
    """Raised for an IR node airlang-spec-v1.md section 5 documents as having
    no runtime equivalent yet. Distinguishes "this parsed fine but can't
    run" from a real binding/reference error (AirLangBindingError)."""


class AirLangBindingError(Exception):
    """Raised when a name in the IR (a tool, schema, capability, or
    provider reference) can't be resolved against builtins or the
    supplied Bindings -- see bindings.py."""


def build_workflow(ir: Dict[str, Any], bindings: Optional[Bindings] = None) -> Workflow:
    """Builds (but does not run) a real airpy Workflow from IR. Split out
    from execute_ir() so callers can inspect/modify the Workflow before
    running it, and so tests can assert on construction failures
    (AirLangNotYetSupportedError, AirLangBindingError) without needing a
    Scheduler run to happen first."""
    bindings = bindings or Bindings()

    def resolve_capability(name: str) -> Capability:
        if name in _BUILTIN_CAPABILITIES:
            return _BUILTIN_CAPABILITIES[name]
        if name in bindings.capabilities:
            return bindings.capabilities[name]
        raise AirLangBindingError(
            f"unknown capability '{name}' -- not one of aircore's builtins "
            f"({sorted(_BUILTIN_CAPABILITIES)}) and not in bindings.capabilities"
        )

    def resolve_tool(name: str):
        if name in bindings.tools:
            return bindings.tools[name]
        raise AirLangBindingError(
            f"tool '{name}' has no implementation -- AirLang cannot author tool bodies "
            f"(airlang-spec-v1.md section 2/6), bind it in bindings.tools (or a sibling "
            f"<file>.airlang.py's TOOLS dict)"
        )

    def resolve_provider(name: str):
        if name in bindings.providers:
            return bindings.providers[name]
        if name == "mock":
            return MockProvider()
        import airpy as _airpy
        catalog_fn = getattr(_airpy, name, None)
        if callable(catalog_fn) and name in (
            "openai", "anthropic", "deepseek", "gemini", "qwen", "nvidia",
            "zai", "ollama", "lmstudio", "openrouter",
        ):
            return catalog_fn()
        raise AirLangBindingError(
            f"unknown provider '{name}' -- not 'mock', not in airpy's provider catalog, "
            f"and not in bindings.providers"
        )

    policy = Policy(
        max_parallel=ir["policy"]["max_parallel"],
        max_runtime=ir["policy"]["max_runtime"],
        max_cost=ir["policy"]["max_cost"],
        approval_for=ir["policy"]["approval_for"] or None,
    )

    memory = None
    if ir["memory"] is not None:
        memory = getattr(Memory(), ir["memory"])

    # Built before agents (unlike before this change) so a templated
    # agent prompt can be wired to workflow.bindings at construction time
    # -- see this module's docstring's "let"/prompt-template section.
    workflow = Workflow(ir["workflow"]["name"], policy=policy, memory=memory)

    body = ir["workflow"]["body"]

    # Pre-scan pass (see docstring point 2): `let name = artifact X` can
    # appear anywhere in the body, not just immediately after X's
    # producer, so the alias has to be known before any step is built.
    artifact_names_in_body = {n["name"] for n in body if n["kind"] == "artifact"}
    artifact_to_let_name: Dict[str, str] = {}
    for node in body:
        if node["kind"] != "let":
            continue
        target = node["value"]["name"]
        if target not in artifact_names_in_body:
            raise AirLangBindingError(
                f"'let {node['name']} = artifact {target}' references an artifact that "
                f"never appears as an `artifact {target}` node in this workflow's body"
            )
        artifact_to_let_name[target] = node["name"]  # last `let` for a given artifact wins

    def binding_name_for_following_artifact(next_index: int) -> Optional[str]:
        """If body[next_index] is an `artifact` node, returns the name a
        producer ending here should bind under -- its `let` alias if one
        was declared, otherwise the artifact's own name. None if there's
        no artifact immediately next (nothing to bind)."""
        if next_index >= len(body) or body[next_index]["kind"] != "artifact":
            return None
        artifact_name = body[next_index]["name"]
        return artifact_to_let_name.get(artifact_name, artifact_name)

    agents: Dict[str, Agent] = {}
    for agent_ir in ir["agents"]:
        provider_name = agent_ir["provider"] or ir["provider_default"]
        if provider_name is None:
            raise AirLangBindingError(
                f"agent '{agent_ir['name']}' has no provider (set `provider` in its "
                f"block, or a top-level `provider <name>` default)"
            )
        provider = resolve_provider(provider_name)
        capabilities = [resolve_capability(c) for c in agent_ir["capabilities"]]
        tools = [resolve_tool(t) for t in agent_ir["tools"]]
        # An omitted `prompt` gets this literal placeholder rather than
        # failing, unchanged from before. A prompt containing a literal
        # `{` is now compiled to a PromptTemplate reading workflow.
        # bindings at execute() time -- see this module's docstring.
        prompt_text = agent_ir["prompt"] or f"You are {agent_ir['name']}."
        if "{" in prompt_text:
            agents[agent_ir["name"]] = Agent(
                agent_ir["name"], provider, PromptTemplate(prompt_text),
                model=agent_ir["model"] or "mock",
                requires=capabilities, tools=tools,
                prompt_bindings=workflow.bindings,
            )
        else:
            agents[agent_ir["name"]] = Agent(
                agent_ir["name"], provider, prompt_text,
                model=agent_ir["model"] or "mock",
                requires=capabilities, tools=tools,
            )

    def resolve_ref(name: str):
        if name in agents:
            return agents[name]
        return resolve_tool(name)

    artifacts: List[Dict[str, Any]] = []
    lets: Dict[str, str] = {}
    pending_parallel = None  # the ParallelResults handle from the immediately preceding node

    i = 0
    while i < len(body):
        node = body[i]
        kind = node["kind"]

        if kind in _NOT_YET_SUPPORTED_KINDS:
            raise AirLangNotYetSupportedError(
                f"'{kind}' is not executable yet (see airlang-spec-v1.md section 5): {node}"
            )

        if kind == "if":
            # A standalone `if` (not immediately following a consensus
            # node -- that case is consumed inline by the "consensus"
            # branch below, so control never reaches here for it) is
            # still the general-branching case airlang-spec-v1.md section
            # 5.1 explicitly did not build a runtime primitive for.
            raise AirLangNotYetSupportedError(
                f"'if' is only supported immediately after a 'consensus' node, as a "
                f"confidence-gated fallback (airlang-spec-v1.md section 5.1) -- general "
                f"branching elsewhere is not executable yet: {node}"
            )

        if kind == "step":
            workflow.step(resolve_ref(node["ref"]), as_=binding_name_for_following_artifact(i + 1))
            pending_parallel = None
        elif kind == "ref":
            workflow.step(resolve_ref(node["name"]), as_=binding_name_for_following_artifact(i + 1))
            pending_parallel = None
        elif kind == "parallel":
            members = [resolve_ref(name) for name in node["members"]]
            pending_parallel = workflow.parallel(*members)
        elif kind == "consensus":
            if pending_parallel is None:
                raise AirLangBindingError(
                    "consensus must immediately follow a parallel block -- AirLang always "
                    "reduces the preceding parallel block's results (airlang-spec-v1.md "
                    "section 4.6), it never re-runs voters"
                )
            strategy = _resolve_strategy(node, ir, resolve_provider)

            fallback_kwargs: Dict[str, Any] = {}
            next_node = body[i + 1] if i + 1 < len(body) else None
            if next_node is not None and next_node["kind"] == "if":
                fallback_kwargs = _resolve_fallback(next_node, node, resolve_ref)
                i += 1  # the `if` node was consumed as part of this consensus step

            # An `artifact` immediately after this consensus (or after the
            # `if` fallback it just consumed, since that's this step's
            # real next position) is this consensus's producer -- see
            # aircore/workflow.py's Bindings section for why a consensus
            # group, uniquely among groups, can bind at all.
            pending_parallel.consensus(strategy=strategy,
                                        as_=binding_name_for_following_artifact(i + 1),
                                        **fallback_kwargs)
            pending_parallel = None
        elif kind == "artifact":
            artifacts.append({"name": node["name"], "type": node["type"], "schema": node["schema"]})
            pending_parallel = None
        elif kind == "let":
            # Producer linkage and aliasing already happened in the
            # pre-scan pass above (artifact_to_let_name) and at the
            # producing step's binding_name_for_following_artifact() call
            # -- by the time this node is reached there's nothing left to
            # do at runtime except record it for introspection, the same
            # role airlang_artifacts plays for `artifact`.
            lets[node["name"]] = node["value"]["name"]
            pending_parallel = None
        elif kind == "memory":
            raise AirLangNotYetSupportedError(
                "a `memory` statement inside the workflow body is not yet supported -- "
                "use a top-level `memory <scope>` declaration instead"
            )
        else:
            raise AirLangBindingError(f"unknown IR node kind '{kind}'")

        i += 1

    workflow.airlang_artifacts = artifacts
    workflow.airlang_lets = lets
    return workflow


def _resolve_fallback(if_node: Dict[str, Any], consensus_node: Dict[str, Any], resolve_ref) -> Dict[str, Any]:
    """Folds an `if` node immediately following a `consensus` node into
    that consensus step's confidence-gated fallback kwargs -- see this
    module's docstring for exactly what's in and out of scope here."""
    if if_node["op"] != "<":
        raise AirLangNotYetSupportedError(
            f"'if' as a consensus fallback only supports '<' (airlang-spec-v1.md section "
            f"5.1) -- got '{if_node['op']}'"
        )
    then_body = if_node["then"]
    if len(then_body) != 1 or then_body[0]["kind"] != "ref":
        raise AirLangNotYetSupportedError(
            f"'if' as a consensus fallback must have exactly one bare agent/tool "
            f"reference in its body (e.g. `if confidence < 0.85 {{ HumanReviewer }}`), "
            f"got: {then_body}"
        )
    if if_node["field"] == "confidence":
        if consensus_node["strategy"] != "judge":
            raise AirLangBindingError(
                "`if confidence < X` needs a `consensus judge` (only JudgeConsensus "
                f"reports confidence) -- this consensus uses strategy "
                f"'{consensus_node['strategy']}'"
            )
        if not consensus_node["confidence"]:
            raise AirLangBindingError(
                "`if confidence < X` needs `consensus { ... confidence true }` -- this "
                "consensus never requested confidence from the judge, so it has none "
                "to check"
            )

    fallback_ref_name = then_body[0]["name"]
    return {
        "fallback": resolve_ref(fallback_ref_name),
        "fallback_below": if_node["value"],
        "fallback_field": if_node["field"],
    }


def _resolve_strategy(node: Dict[str, Any], ir: Dict[str, Any], resolve_provider):
    name = node["strategy"]
    if name == "majority":
        return majority
    if name == "unanimous":
        return unanimous
    if name == "judge":
        provider_name = ir["provider_default"]
        if provider_name is None:
            raise AirLangBindingError(
                "consensus judge needs a provider to back the judge -- set a top-level "
                "`provider <name>` default"
            )
        provider = resolve_provider(provider_name)
        return JudgeConsensus(provider, mode=node["mode"] or "synthesize", confidence=node["confidence"])
    raise AirLangBindingError(f"unknown consensus strategy '{name}'")


def execute_ir(ir: Dict[str, Any], bindings: Optional[Bindings] = None,
                approval_callback: "ApprovalCallback | None" = None) -> Workflow:
    """Builds and runs the Workflow. Returns it already run (`.journal`
    populated), same convention as calling `.run()` on a hand-written
    airpy Workflow yourself.

    `approval_callback` is forwarded to `Workflow.run()` unchanged -- only
    matters if `ir["policy"]["approval_for"]` is non-empty (i.e. the .airlang
    file had a `policy { approval <tool> }` line); everything else ignores
    it. Omitting it on a workflow that needs it surfaces the same
    pre-flight PolicyViolation airpy would raise directly."""
    workflow = build_workflow(ir, bindings)
    workflow.run(approval_callback=approval_callback)
    return workflow


def execute_file(path: str, bindings: Optional[Bindings] = None,
                  approval_callback: "ApprovalCallback | None" = None) -> Workflow:
    """Parses and runs a .airlang file in one call. `bindings`, if not given,
    is loaded via the `<path>.py` sibling-file convention (bindings.py) --
    pass one explicitly to override or to avoid the filesystem lookup
    entirely (e.g. from a test). `approval_callback` is forwarded to
    execute_ir() unchanged -- see its docstring."""
    ir = parse_file(path)
    resolved_bindings = bindings if bindings is not None else load_bindings_for(path)
    return execute_ir(ir, resolved_bindings, approval_callback=approval_callback)
