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
        desc="A tool for time query.",
        tools=["time_tools"],
        llm_model="default_llm",
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        """first"""
        payload = {
            "query": "Get what time it is Asia/Shanghai",
        }
        oxy_response = await mas.chat_with_agent(payload=payload)
        print("LLM-first: ", oxy_response.output)

        """second"""
        # payload = {
        #     "query": "Get what time it is Asia/Shanghai",
        #     "restart_node_id": "wYF2EqBj3RYKiRK7",  # 传入第一次调用的中间节点node_id
        #     "restart_node_output": """{
        #         "timezone": "Asia/Shanghai",
        #         "datetime": "2024-10-14T06:18:00+08:00",
        #         "day_of_week": "Tuesday",
        #         "is_dst": false
        #     }""",
        # }
        # oxy_response = await mas.chat_with_agent(payload=payload)
        # print("LLM-second: ", oxy_response.output)


if __name__ == "__main__":
    asyncio.run(main())
