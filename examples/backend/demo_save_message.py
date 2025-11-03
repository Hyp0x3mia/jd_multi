import asyncio
import os

from oxygent import MAS, Config, OxyRequest, oxy

Config.set_message_is_show_in_terminal(True)
Config.set_message_is_stored(True)


async def update_query(oxy_request: OxyRequest) -> OxyRequest:
    await oxy_request.send_message(
        {"type": "test1", "content": "test1", "_is_stored": False, "_is_send": False}
    )
    await oxy_request.send_message(
        {"type": "test2", "content": "test2", "_is_stored": False, "_is_send": True}
    )
    await oxy_request.send_message(
        {"type": "test3", "content": "test3", "_is_stored": True, "_is_send": False}
    )
    await oxy_request.send_message(
        {"type": "test4", "content": "test4", "_is_stored": True, "_is_send": True}
    )
    return oxy_request


oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
        llm_params={"stream": True},
    ),
    oxy.ChatAgent(
        name="qa_agent",
        llm_model="default_llm",
        func_process_input=update_query,
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(first_query="hello")


if __name__ == "__main__":
    asyncio.run(main())
