import asyncio
import os

from oxygent import MAS, oxy

oxy_space1 = [
    oxy.StreamableMCPClient(
        name="time_tools",
        server_url="http://127.0.0.1:8000/mcp",
        headers={"key1": "value1"},
    ),
]


oxy_space2 = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    oxy.StreamableMCPClient(
        name="time_tools",
        server_url="http://127.0.0.1:8000/mcp",
        headers={"key1": "value1"},
        is_dynamic_headers=True,
    ),
    oxy.ReActAgent(
        name="master_agent",
        tools=["time_tools"],
        llm_model="default_llm",
    ),
]

oxy_space3 = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    oxy.StreamableMCPClient(
        name="time_tools",
        server_url="http://127.0.0.1:8000/mcp",
        headers={"key1": "value1"},
        is_dynamic_headers=True,
        is_inherit_headers=True,
    ),
    oxy.ReActAgent(
        name="master_agent",
        tools=["time_tools"],
        llm_model="default_llm",
    ),
]


async def main():
    # xxxMCPClient传入headers属性，会在调用MCP工具时附带
    async with MAS(oxy_space=oxy_space1) as mas:
        await mas.call(callee="power", arguments={"n": 2, "m": 3})

    # xxxMCPClient设置is_dynamic_headers=True后
    # 在调用MCP工具时会将shared_data里headers属性的值作为headers传入
    async with MAS(oxy_space=oxy_space2) as mas:
        payload = {
            "query": "2的3次方是多少",
            "shared_data": {"headers": {"key2": "value2"}},
        }
        await mas.chat_with_agent(payload=payload)

    # xxxMCPClient设置is_inherit_headers=True后
    # 在调用MCP工具时会透传前端请求的headers
    async with MAS(oxy_space=oxy_space3) as mas:
        await mas.start_web_service(first_query="2的3次方是多少")

    # 所以总共有三个地方传入headers，允许属性重名
    # 优先级为：前端请求headers > 覆盖shared_data的headers > 覆盖xxxMCPClient的headers


if __name__ == "__main__":
    asyncio.run(main())
