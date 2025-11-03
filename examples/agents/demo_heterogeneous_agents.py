import asyncio
import os

from oxygent import MAS, Config, OxyRequest, oxy

Config.set_agent_llm_model("default_llm")


async def workflow(oxy_request: OxyRequest):
    oxy_response = await oxy_request.call(
        callee="get_current_time",
        arguments={"timezone": "Asia/Shanghai"},
    )
    return oxy_response.output


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
        llm_model="default_llm",
        sub_agents=["QA_agent", "time_agent"],
    ),
    oxy.ChatAgent(
        name="QA_agent",
        desc="A tool for knowledge.",
        llm_model="default_llm",
    ),
    oxy.WorkflowAgent(
        name="time_agent",
        desc="A tool for time query.",
        tools=["time_tools"],
        func_workflow=workflow,
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(
            first_query="What time it is?",
        )


if __name__ == "__main__":
    asyncio.run(main())
