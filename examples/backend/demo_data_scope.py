"""Demo for testing group_id behavior in OxyGent MAS."""

import asyncio
import os

from oxygent import MAS, OxyRequest, oxy


def process_input(oxy_request: OxyRequest) -> OxyRequest:
    print("--- agent name --- :", oxy_request.callee)
    print("--- arguments --- :", oxy_request.get_arguments())
    print("--- shared_data --- :", oxy_request.get_shared_data())
    print("--- group_data --- :", oxy_request.get_group_data())
    print("--- global_data --- :", oxy_request.get_global_data())
    return oxy_request


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
        name="master_agent",
        is_master=True,
        llm_model="default_llm",
        sub_agents=["time_agent"],
        func_process_input=process_input,
    ),
    oxy.ReActAgent(
        name="time_agent",
        desc="A tool for time query.",
        tools=["time_tools"],
        func_process_input=process_input,
    ),
]


async def main():
    global_data = {"global_key": "global_value"}
    async with MAS(oxy_space=oxy_space, global_data=global_data) as mas:
        # round 1-1
        oxy_response = await mas.chat_with_agent(
            {
                "query": "What time is it",
            },
        )
        # round 2-1
        oxy_response = await mas.chat_with_agent(
            {
                "query": "What time is it",
                "shared_data": {"shared_key": "shared_value"},
                "group_data": {"group_key": "group_value"},
            },
        )
        # round 2-2
        trace_id = oxy_response.oxy_request.current_trace_id
        oxy_response = await mas.chat_with_agent(
            {
                "query": "What time is it",
                "from_trace_id": trace_id,
            },
        )


if __name__ == "__main__":
    asyncio.run(main())
