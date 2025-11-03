"""Demo for using OxyGent with multiple LLMs and an agent."""

import asyncio
import os

from oxygent import MAS, Config, OxyRequest, OxyResponse, oxy

Config.set_agent_short_memory_size(7)


def update_query(oxy_request: OxyRequest) -> OxyRequest:
    query = oxy_request.get_query()
    oxy_request.set_query(query + " Please answer in detail.")
    return oxy_request


def format_output(oxy_response: OxyResponse) -> OxyResponse:
    oxy_response.output = "Answer: " + oxy_response.output
    return oxy_response


oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
        llm_params={"temperature": 0.01},
        semaphore=4,
        timeout=300,
        retries=3,
    ),
    oxy.ChatAgent(
        name="master_agent",
        llm_model="default_llm",
        prompt="You are a helpful assistant.",
        func_process_input=update_query,
        func_format_output=format_output,
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(
            first_query="Hello",
            welcome_message="Hi, I’m OxyGent. How can I assist you?",
        )


if __name__ == "__main__":
    asyncio.run(main())
