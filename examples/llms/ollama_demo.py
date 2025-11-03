"""gemma_cli_demo.py"""

import asyncio
import os

from oxygent import MAS, oxy

oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        base_url="http://localhost:11434/api/chat",
        model_name=os.getenv("DEFAULT_OLLAMA_MODEL"),
        semaphore=1,
    ),
    oxy.ChatAgent(
        name="master_agent",
        is_master=True,
        llm_model="default_llm",
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        history = [{"role": "system", "content": "You are a helpful assistant."}]

        while True:
            user_in = input("User: ").strip()
            if user_in.lower() in {"exit", "quit", "q"}:
                break

            history.append({"role": "user", "content": user_in})
            result = await mas.call(
                callee="master_agent",
                arguments={"messages": history},
            )
            assistant_out = result
            print(f"Assistant: {assistant_out}\n")
            history.append({"role": "assistant", "content": assistant_out})


if __name__ == "__main__":
    asyncio.run(main())
