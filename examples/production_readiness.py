"""Live, end-to-end validation of every primitive built to close the
"can bite in a real project" gaps -- against a real provider (DeepSeek,
via LiteLLM), not MockProvider. Each of these has offline test coverage
already (tests/test_approval.py, test_checkpoint.py, test_sandbox.py,
test_bindings.py, test_persistent_memory.py); this is the thing offline
tests can't prove -- that they actually work *together*, in one workflow,
against a real API call, the way a real project would combine them.

Deliberately NOT part of the automated test suite (tests/), same reason
examples/live_deepseek.py isn't: needs real network access and a real API
key, run manually on your own machine.

Setup:
    pip install litellm
    export DEEPSEEK_API_KEY=sk-...           (macOS/Linux)
    setx DEEPSEEK_API_KEY "sk-..."            (Windows, then reopen terminal)

Run:
    python examples/production_readiness.py

What this proves, in one workflow:
  1. Sandboxed execution -- the research step runs in a real subprocess
     (Sandbox()), proven by it reporting a different PID than this
     process.
  2. Cross-step data flow -- the sandboxed step's real output is bound
     (as_="research_result") and a second, tiny step extracts just the
     part the next prompt needs (as_="topic"), which a PromptTemplate
     then reads and renders fresh at execute() time -- not a fixed string
     decided up front -- right before it's sent to the real DeepSeek API.
  3. Approval -- the real (paid) API call is gated behind
     Policy.approval_for; auto_approve is used by default so this runs
     unattended -- see the comment below for how to make it a genuine
     interactive y/n prompt instead.
  4. Durable resume -- the whole run uses a FileCheckpointStore. Run this
     script twice with the same run_id (the default) and the second run
     skips the sandboxed research step, the extraction step, AND the
     already-answered DeepSeek call entirely -- provably not re-spending
     real API cost on a step that already succeeded.
  5. Real cost/usage flowing into Metrics, exactly as M8 always promised,
     now proven alongside every primitive built since.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not os.environ.get("DEEPSEEK_API_KEY"):
    print("DEEPSEEK_API_KEY is not set. Set it and re-run:")
    print("  export DEEPSEEK_API_KEY=sk-...")
    sys.exit(1)

from aircore import FileCheckpointStore, Policy, Sandbox, Tool, Workflow, auto_approve
from airpy import ModelAgent, LiteLLMProvider, PromptTemplate

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "production_readiness_checkpoint.json")
RUN_ID = "production-readiness-demo"


def _produce_research_topic():
    """Runs inside a real subprocess (Sandbox()) -- must be a module-level
    function, not a closure/lambda, per sandbox.py's picklability
    constraint. Returns its own PID alongside the topic so the printed
    output can prove this really ran in a different process."""
    import os as _os
    return {"topic": "the CAP theorem", "produced_by_pid": _os.getpid()}


if __name__ == "__main__":
    provider = LiteLLMProvider(model="deepseek/deepseek-chat")

    print(f"This process's PID: {os.getpid()}")
    print(f"Checkpoint file:     {CHECKPOINT_PATH}")
    print(f"run_id:              {RUN_ID}")
    print("(run this script a second time with the same run_id to see resume skip the "
          "sandboxed step, the extraction step, and the already-answered DeepSeek call)\n")

    policy = Policy(approval_for={"explainer"})
    workflow = Workflow("ProductionReadinessDemo", policy=policy)

    # Step 1: sandboxed. A real subprocess, not in-process.
    workflow.step(
        Tool(_produce_research_topic, name="research", sandbox=Sandbox(max_runtime=30)),
        as_="research_result",
    )

    # Step 2: a tiny sequential step extracting just the field the next
    # prompt needs, under the exact name the template declares -- this is
    # honest about what workflow.bindings actually does (binds whatever a
    # step returns, under one name), not a nested-key-access feature that
    # doesn't exist.
    workflow.step(
        Tool(lambda: workflow.bindings["research_result"]["topic"], name="extract_topic"),
        as_="topic",
    )

    # Step 3: cross-step data flow, against a real provider. This
    # PromptTemplate isn't rendered until execute() -- it reads
    # workflow.bindings["topic"], which only exists once step 2 above has
    # actually finished. `model` is deliberately left at ModelAgent's
    # default ("mock") -- LiteLLMProvider.generate() treats that as "use
    # whatever model I was constructed with" (see litellm_provider.py),
    # which is the correctly-prefixed "deepseek/deepseek-chat" from
    # `provider` above.
    template = PromptTemplate("In two sentences, explain {topic} to a junior engineer.")
    explainer = ModelAgent("explainer", provider, template, prompt_bindings=workflow.bindings)
    workflow.step(explainer)

    # Step 4: approval, gating the real (paid) API call above.
    # auto_approve makes this runnable unattended; swap in
    # `from aircore import cli_approval_callback` for a genuine interactive
    # y/n prompt before the real call fires.
    approval_callback = auto_approve

    checkpoint_store = FileCheckpointStore(CHECKPOINT_PATH)
    journal = workflow.run(
        approval_callback=approval_callback,
        checkpoint_store=checkpoint_store,
        run_id=RUN_ID,
    )

    print(journal.pretty())
    print(workflow.metrics.summary())

    research_step = next(s for s in journal.steps if s.tool == "research")
    if research_step.replayed:
        print("\nResearch step was replayed from checkpoint -- it ran in a subprocess on a "
              "prior invocation of this script, not this one.")
    else:
        produced_by_pid = research_step.output.get("produced_by_pid")
        print(f"\nResearch step ran in PID {produced_by_pid}, this process is PID {os.getpid()} "
              f"-- {'proves subprocess isolation' if produced_by_pid != os.getpid() else 'UNEXPECTED: same PID'}")
