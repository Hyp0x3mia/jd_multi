import asyncio
import os

from oxygent import MAS, Config, oxy

Config.set_agent_llm_model("default_llm")


def func_filter(payload):
    print(payload)
    payload["group_data"] = {"user_pin": "123456"}
    return payload


def func_interceptor(payload):
    print(payload)
    return {"code": 403, "message": "Permission denied."}


oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    oxy.ChatAgent(
        name="master_agent",
        llm_model="default_llm",
    ),
]


async def main():
    async with MAS(
        oxy_space=oxy_space,
        func_filter=func_filter,
        func_interceptor=func_interceptor,
    ) as mas:
        await mas.start_web_service(first_query="hello")


if __name__ == "__main__":
    asyncio.run(main())
