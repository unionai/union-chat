import asyncio
import json

import weave
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from openai import OpenAI
from weave import EvaluationLogger
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

RELEVANCE_PROMPT = """
You are a helpful assistant that assesses the relevance of a response to a user question.

The user question is:
{question}

The response is:
{response}

Is the response relevant to the question? Output 0 if it is not relevant, and 1 if it is relevant.

Response:
"""


async def assess_relevance(openai_client: OpenAI, model_id: str, messages: list[dict], output: str):
    response = openai_client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": RELEVANCE_PROMPT.format(question=messages[-1]['content'], response=output),
            },
        ],
    )
    return response.choices[0].message.content


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

        original_prompt = messages[-1]['content']
        _messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + messages

        final_output = None

        print("Messages:")
        for m in _messages:
            print(m)

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

        eval_logger = EvaluationLogger(name="union-chat-eval-harness")
        relevance = await assess_relevance(openai_client, model_id, messages, final_output)
        pred_logger = eval_logger.log_prediction(
            inputs={
                "original_prompt": original_prompt,
                "output": final_output,
            },
            output=relevance,
        )
        pred_logger.log_score(
            scorer="relevance",
            score=relevance,
        )

        return final_output


if __name__ == "__main__":
    import os

    endpoint = None
    api_key = os.getenv("OPENAI_API_KEY")
    model_id = "o4-mini"

    mcp_client = Client(transport=SSETransport(os.getenv("MCP_CLIENT_URL")))
    openai_client = OpenAI(base_url=endpoint, api_key=api_key)

    prompt = input("Enter a prompt: ")
    if prompt == "":
        prompt = "Add 67 and 26, then multiply the result by 38."
        print(f"Using default prompt: {prompt}")

    weave.init("union-chat")

    asyncio.run(main(openai_client, mcp_client, model_id, [{"role": "user", "content": prompt}]))
