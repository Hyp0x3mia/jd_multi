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
    oxy.ReActAgent(
        name="time_agent",
        tools=["time_tools"],
        llm_model="default_llm",
    ),
]


async def get_mas_object():
    mas = await MAS.create(oxy_space=oxy_space)
    await mas.start_web_service()


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        # 调用 oxy
        await mas.call(
            callee="time_agent",
            arguments={"query": "What time it is?"},
        )
        await mas.call(
            callee="get_current_time",
            arguments={"timezone": "Asia/Shanghai"},
        )
        await mas.call(
            callee="default_llm",
            arguments={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "hello"},
                ],
                "llm_params": {"temperature": 0.2},
            },
        )

        # 调用 Master Agent
        payload = {"query": "What time it is?"}
        oxy_response = await mas.chat_with_agent(payload=payload)
        print(oxy_response)

        # 启动
        await mas.start_cli_mode(first_query="What time it is?")
        await mas.start_batch_processing(["What time it is?"] * 10)
        await mas.start_web_service(first_query="What time it is?")


if __name__ == "__main__":
    asyncio.run(main())
