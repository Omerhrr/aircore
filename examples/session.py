"""Session: a long-running conversation with real per-turn journaling and
capability enforcement -- the difference between "a chat loop" and an
agent whose every action is audited and permission-checked, sketched here
as a small customer-support exchange. Runs offline against MockProvider.

Run with: python examples/session.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Agent, Capability, Policy
from airpy import MockProvider, Session, SessionTurnFailed


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


if __name__ == "__main__":
    section("A support session remembers the customer across turns")

    scripted = iter([
        "Hi! I'm sorry to hear that. Can you tell me your order number?",
        "Thanks -- I can see order #9988 shipped 3 days ago and hasn't arrived. "
        "I've logged a delivery issue and a replacement is on its way.",
    ])
    support = Session("support_agent", MockProvider(response=lambda request: next(scripted)))

    customer_message_1 = "My order never arrived, this is really frustrating."
    customer_message_2 = "It's order #9988."

    print(f"customer: {customer_message_1}")
    print(f"agent: {support.send(customer_message_1)}")
    print(f"\ncustomer: {customer_message_2}")
    print(f"agent: {support.send(customer_message_2)}")

    section("Every turn is a real, journaled Workflow step")
    for i, journal in enumerate(support.journals, start=1):
        step = journal.steps[0]
        print(f"turn {i}: status={step.status} latency={step.duration_ms:.2f}ms")

    section("Capabilities: a support agent can't approve a refund on its own")

    refund_capability = Capability("PaymentRefund")
    refund_agent = Session(
        "refund_agent", MockProvider(response="Refund approved for $45.00."),
        requires=refund_capability,
        policy=Policy(require_agent=True),  # every turn must have an identified caller
    )

    unverified_caller = Agent("support_agent")  # no PaymentRefund capability
    try:
        refund_agent.send("Please refund $45.00 to this customer.", agent=unverified_caller)
    except SessionTurnFailed as exc:
        print(f"denied, as expected: {exc}")

    verified_caller = Agent("refund_desk", capabilities=[refund_capability])
    result = refund_agent.send("Please refund $45.00 to this customer.", agent=verified_caller)
    print(f"approved: {result!r}")

    section("Sessions close explicitly, but history stays inspectable")
    support.close()
    print(f"{support!r}")
    print("full transcript:")
    for message in support.history:
        print(f"  {message['role']}: {message['content']}")
