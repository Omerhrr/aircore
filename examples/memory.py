"""M5 acceptance example: temporary scope shares data between steps within
one run, then is wiped automatically after the run finishes; session
persists across multiple runs on the same Memory object; project is
shared between two entirely separate Memory instances.

Run with: python examples/memory.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool, Memory


if __name__ == "__main__":
    print("=== temporary: shared within a run, cleared after ===")
    mem = Memory()

    @tool
    def search():
        mem.temporary.set("results", ["paper A", "paper B"])
        return "searched"

    @tool
    def summarize():
        results = mem.temporary.get("results", [])
        return f"summary of {len(results)} results"

    workflow = Workflow("Research", memory=mem)
    workflow.step(search)
    workflow.step(summarize)
    journal = workflow.run()
    print(f"step 2 output: {journal.steps[1].output!r}")
    print(f"temporary after run(): {mem.temporary.snapshot()}\n")

    print("=== session: persists across multiple run() calls on the same Memory ===")

    @tool
    def remember_name():
        mem.session.set("user_name", "Ada")
        return "remembered"

    @tool
    def greet():
        name = mem.session.get("user_name", "stranger")
        return f"Hello, {name}"

    wf1 = Workflow("Turn1", memory=mem)
    wf1.step(remember_name)
    wf1.run()

    wf2 = Workflow("Turn2", memory=mem)  # separate workflow, same Memory object
    wf2.step(greet)
    j2 = wf2.run()
    print(f"turn 2 output: {j2.steps[0].output!r}\n")

    print("=== project: shared across two independent Memory instances ===")
    researcher_mem = Memory(project="acme-corp")
    auditor_mem = Memory(project="acme-corp")  # different Memory, same project

    @tool
    def researcher_logs_finding():
        researcher_mem.project.set("finding", "unpatched CVE in dependency X")
        return "logged"

    @tool
    def auditor_reads_finding():
        return auditor_mem.project.get("finding", "nothing found")

    wf3 = Workflow("Research", memory=researcher_mem)
    wf3.step(researcher_logs_finding)
    wf3.run()

    wf4 = Workflow("Audit", memory=auditor_mem)
    wf4.step(auditor_reads_finding)
    j4 = wf4.run()
    print(f"auditor sees: {j4.steps[0].output!r}")
