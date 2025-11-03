import asyncio
import os

from oxygent import MAS, oxy

oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
        semaphore=4,  # limit concurrency to 4
    ),
    oxy.ChatAgent(
        name="chat_agent",
        llm_model="default_llm",
        semaphore=6,  # limit concurrency to 6
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        outs = await mas.start_batch_processing(["hello"] * 10, return_trace_id=True)
        [print(out) for out in outs]


if __name__ == "__main__":
    asyncio.run(main())
