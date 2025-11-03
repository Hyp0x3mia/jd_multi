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
    oxy.StdioMCPClient(
        name="file_tools",
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "./local_file"],
        },
    ),
    oxy.ReActAgent(
        name="master_agent",
        is_master=True,
        sub_agents=["time_agent", "file_agent"],
        llm_model="default_llm",
    ),
    oxy.ReActAgent(
        name="time_agent",
        desc="A tool for time query",
        tools=["time_tools"],
        llm_model="default_llm",
    ),
    oxy.ReActAgent(
        name="file_agent",
        desc="A tool for file operation.",
        tools=["file_tools"],
        llm_model="default_llm",
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(
            first_query="Get what time it is and save in `log.txt` under `/local_file`",
        )


if __name__ == "__main__":
    asyncio.run(main())
