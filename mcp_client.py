import asyncio
import json

import weave
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from openai import OpenAI
from weave.scorers import HallucinationFreeScorer


SYSTEM_PROMPT = """
You are a helpful agent who always answers in English.

When a user asks you something, consider whether you can answer the question
using one or more of the available tools.

With math problems YOU MUST USE the tools available to you.

You may need to break down the problem into smaller problems and solve each
smaller problem with a different tool. ONLY USE ONE TOOL AT A TIME.

The final answer to the user question MUST START WITH "Final answer:"
"""


@weave.op()
async def main(openai_client: OpenAI, mcp_client: Client, model_id: str, messages: list[dict], **kwargs):
    # Connection is established here
    async with mcp_client:
        print(f"Client connected: {mcp_client.is_connected()}")

        # Make MCP calls within the context
        tools = await mcp_client.list_tools()
        
        print(f"Available tools:")
        available_tools = []
        for tool in tools:
            _tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
            }
            print(f"Tool:\n{json.dumps(_tool, indent=2)}")
            available_tools.append(_tool)

        _messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + messages

        final_output = None

        for m in _messages: print(m)

        while final_output is None:
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=_messages,
                tools=available_tools,
                **kwargs,
            )

            message = response.choices[0].message
            _message = {"role": "assistant", "content": message.content, "tool_calls": message.tool_calls}
            print(_message)
            _messages.append(_message)

            if message.tool_calls:
                # Handle multiple tool calls in sequence
                for tool_call in message.tool_calls:
                    result = await mcp_client.call_tool(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments),
                    )
                    _messages.append(
                        {
                            "role": "tool",
                            "content": result[0].text,
                            "tool_call_id": tool_call.id,
                        }
                    )
            elif message.content.startswith("Final answer:"):
                final_output = message.content.replace("Final answer:", "").strip()
            else:
                print(f"Intermediate message: {message.content}")

        print(f"Final answer: {final_output}")
        return final_output


if __name__ == "__main__":
    import os

    endpoint = None
    api_key = os.getenv("OPENAI_API_KEY")
    model_id = "o4-mini"

    mcp_client = Client(transport=SSETransport("https://curly-recipe-96d91.apps.demo.hosted.unionai.cloud/sse"))
    openai_client = OpenAI(base_url=endpoint, api_key=api_key)

    prompt = input("Enter a prompt: ")
    if prompt == "":
        prompt = "Add 67 and 26, then multiply the result by 38."
        print(f"Using default prompt: {prompt}")

    weave.init("union-chat")

    asyncio.run(main(openai_client, mcp_client, model_id, [{"role": "user", "content": prompt}]))
