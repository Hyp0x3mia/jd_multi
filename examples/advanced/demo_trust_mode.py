import asyncio
import os

from oxygent import MAS, oxy

oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    oxy.StdioMCPClient(
        name="time_tools",
        params={
            "command": "uvx",
            "args": ["mcp-server-time", "--local-timezone=Asia/Shanghai"],
        },
    ),
    # normal mode ReActAgent
    oxy.ReActAgent(
        name="normal_agent",
        tools=["time_tools"],
        llm_model="default_llm",
        trust_mode=False,  # disable trust mode
    ),
    # trust mode ReActAgent
    oxy.ReActAgent(
        name="trust_agent",
        tools=["time_tools"],
        llm_model="default_llm",
        trust_mode=True,  # enable trust mode
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        query = "What is the current time"

        print("=== normal mode test ===")
        normal_result = await mas.call("normal_agent", {"query": query})
        print(f"normal mode output: {normal_result}")

        print("=== trust mode test ===")
        trust_result = await mas.call("trust_agent", {"query": query})
        print(f"trust mode output: {trust_result}")


if __name__ == "__main__":
    asyncio.run(main())
