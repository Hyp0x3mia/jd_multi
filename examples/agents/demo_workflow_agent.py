import asyncio
import os

from oxygent import MAS, OxyRequest, oxy


async def workflow(oxy_request: OxyRequest):
    # 获取memory
    current_short_memory = oxy_request.get_short_memory()
    print("--- Current Short Memory --- :", current_short_memory)
    master_short_memory = oxy_request.get_short_memory(master_level=True)
    print("--- Master Short Memory --- :", master_short_memory)

    # 获取query
    current_query = oxy_request.get_query()
    print("--- Current query ---:", current_query)
    master_query = oxy_request.get_query(master_level=True)
    print("--- Master query ---:", master_query)

    # 发消息
    await oxy_request.send_message({"type": "msg_type", "content": "msg_content"})

    # 调用Agent
    oxy_response = await oxy_request.call(
        callee="chat_agent",
        arguments={"query": current_query},
    )
    print("--- Direct answer ---:", oxy_response.output)

    # 调用LLM
    question = (
        f"用户的问题是{current_query}，用户想要小数点后多少位圆周率？直接回答数字"
    )
    oxy_response = await oxy_request.call(
        callee="default_llm",
        arguments={
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            "llm_params": {"temperature": 0.2},
        },
    )
    print("--- Precision ---:", oxy_response.output)

    n = oxy_response.output
    # 调用Tool
    oxy_response = await oxy_request.call(
        callee="calc_pi",
        arguments={"prec": n},
    )
    return f"Save {n} positions: {oxy_response.output}"


oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    oxy.StdioMCPClient(
        name="math_tools",
        params={
            "command": "uv",
            "args": ["--directory", "./mcp_servers", "run", "math_tools.py"],
        },
    ),
    oxy.ChatAgent(
        name="chat_agent",
        llm_model="default_llm",
    ),
    oxy.WorkflowAgent(
        name="math_agent",
        is_master=True,
        sub_agents=["chat_agent"],
        tools=["math_tools"],
        func_workflow=workflow,
        llm_model="default_llm",
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(
            first_query="Please calculate the 20 positions of Pi",
        )


if __name__ == "__main__":
    asyncio.run(main())
