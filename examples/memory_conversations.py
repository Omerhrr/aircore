"""ModelAgent(memory=..., conversation_id=...): a real multi-turn
conversation, built entirely out of the Memory primitive M5 already
shipped -- no aircore changes were needed for this. Runs offline against
MockProvider; swap in LiteLLMProvider for a real model.

Run with: python examples/memory_conversations.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Memory
from airpy import ModelAgent, MockProvider


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


if __name__ == "__main__":
    section("Same agent, two calls: the second sees the first as history")

    turns = iter([
        "Nice to meet you, Ada!",
        "You told me your name is Ada.",
    ])

    def scripted_reply(request):
        # A real model would actually use the conversation history in
        # `request.messages` -- this just proves it's really being sent.
        print(f"  (model saw {len(request.messages)} prior message(s) this turn)")
        return next(turns)

    memory = Memory()
    assistant = ModelAgent(
        "assistant", MockProvider(response=scripted_reply), prompt="Hi, my name is Ada.",
        memory=memory.session, conversation_id="ada-chat",
    )
    print(f"turn 1: {assistant.execute()!r}")

    # A second call re-uses the same prompt in this example for simplicity,
    # but in a real chat loop you'd construct a new ModelAgent per turn with
    # the next thing the user said, same memory/conversation_id.
    follow_up = ModelAgent(
        "assistant", MockProvider(response=scripted_reply), prompt="What's my name again?",
        memory=memory.session, conversation_id="ada-chat",
    )
    print(f"turn 2: {follow_up.execute()!r}")

    section("Full conversation now stored in memory")
    for message in follow_up.conversation_history():
        print(f"  {message['role']}: {message['content']}")

    section("Two different agents sharing one conversation_id see the same history")

    researcher = ModelAgent(
        "researcher", MockProvider(response="Noted: the deadline is Friday."),
        prompt="The deadline is Friday.", memory=memory.session, conversation_id="shared-thread",
    )
    researcher.execute()

    reviewer_prompts = []

    def reviewer_reply(request):
        reviewer_prompts.append(request.messages)
        return "Got it, I'll review before Friday."

    reviewer = ModelAgent(
        "reviewer", MockProvider(response=reviewer_reply),
        prompt="When's the deadline?", memory=memory.session, conversation_id="shared-thread",
    )
    reviewer.execute()

    print("reviewer's view of the conversation before replying:")
    for message in reviewer_prompts[0]:
        print(f"  {message['role']}: {message['content']}")
