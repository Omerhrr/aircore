"""Tool-calling acceptance example: a ModelAgent decides to call an aircore
Tool, gets the result back, and answers -- the gap flagged at the end of
the M8 summary is now closed. Uses a scripted MockProvider so this needs
no API key; see tests/test_litellm_tool_calling.py for the same flow
through a (faked) real provider.

Run with: python examples/tool_calling.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircore import Workflow, tool, Agent, Network
from airpy import ModelAgent, MockProvider, ModelResponse, ToolCallRequest


@tool(requires=Network, description="Look up the current weather for a city")
def get_weather(city: str):
    fake_data = {"Lagos": "31C, sunny", "London": "14C, rainy"}
    return fake_data.get(city, "no data")


if __name__ == "__main__":
    print("=== A model deciding to call a tool, then answering with the result ===")

    scripted = MockProvider(responses=[
        # turn 1: model asks to call get_weather
        ModelResponse(content="", tool_calls=[
            ToolCallRequest(id="1", name="get_weather", arguments={"city": "Lagos"}),
        ]),
        # turn 2: model has the tool result and gives a final answer
        "It's 31C and sunny in Lagos right now.",
    ])

    researcher = Agent("Researcher", capabilities=[Network])  # identity granting Network
    weather_bot = ModelAgent(
        "weather_bot", scripted, prompt="What's the weather in Lagos?",
        tools=[get_weather], identity=researcher,
    )

    answer = weather_bot.execute()
    print(f"Final answer: {answer!r}")
    print(f"Tool calls made: {[(r.name, r.arguments, r.result) for r in weather_bot.tool_call_log]}")

    print("\n=== Same agent, but as a workflow step -- internal tool call is invisible to the journal ===")
    scripted2 = MockProvider(responses=[
        ModelResponse(content="", tool_calls=[
            ToolCallRequest(id="1", name="get_weather", arguments={"city": "London"}),
        ]),
        "London is 14C and rainy.",
    ])
    weather_bot2 = ModelAgent("weather_bot2", scripted2, prompt="Weather in London?",
                               tools=[get_weather], identity=researcher)
    workflow = Workflow("WeatherCheck")
    workflow.step(weather_bot2)
    journal = workflow.run()
    print(journal.pretty())
    print(f"(the get_weather call itself only shows up in weather_bot2.tool_call_log, not here)")

    print("\n=== identity without Network capability -- tool call gets denied, model told why ===")
    scripted3 = MockProvider(responses=[
        ModelResponse(content="", tool_calls=[
            ToolCallRequest(id="1", name="get_weather", arguments={"city": "Lagos"}),
        ]),
        "I wasn't able to check the weather due to a permissions issue.",
    ])
    sandboxed = Agent("Sandboxed", capabilities=[])
    restricted_bot = ModelAgent("restricted_bot", scripted3, prompt="Weather in Lagos?",
                                 tools=[get_weather], identity=sandboxed)
    print(f"Final answer: {restricted_bot.execute()!r}")
    print(f"Error recorded: {restricted_bot.tool_call_log[0].error}")
