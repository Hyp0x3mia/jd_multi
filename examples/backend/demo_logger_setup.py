import asyncio
import os

from oxygent import MAS, OxyRequest, oxy
from oxygent.log_setup import setup_logging

logger = setup_logging()


def update_query(oxy_request: OxyRequest) -> OxyRequest:
    query = oxy_request.get_query()
    logger.info(f"The current query is: {query}")
    return oxy_request


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
        func_process_input=update_query,
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(first_query="hello")


if __name__ == "__main__":
    asyncio.run(main())
