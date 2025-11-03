"""
global_data_demo.py
───────────────────
A minimal example (modelled after single_demo.py) that shows how to
read/write the new MAS.global_data store from inside an agent.

• The agent keeps a simple “call counter” in global_data["counter"].
• Every time you call the agent it increments the counter and
  returns the current value.
"""

import asyncio
import os

from oxygent import MAS, OxyRequest, OxyResponse, oxy
from oxygent.oxy.agents.base_agent import BaseAgent
from oxygent.schemas import OxyState


class CounterAgent(BaseAgent):
    async def execute(self, oxy_request: OxyRequest):
        cnt = oxy_request.get_global_data("counter", 0) + 1
        oxy_request.set_global_data("counter", cnt)

        return OxyResponse(
            state=OxyState.COMPLETED,
            output=f"This MAS has been called {cnt} time(s).",
            oxy_request=oxy_request,
        )


oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    CounterAgent(
        name="master_agent",  # mark as master so chat_with_agent works
        is_master=True,
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        # first call → counter = 1
        r1 = await mas.chat_with_agent({"query": "first"})
        print(r1.output)

        # second call → counter = 2 (global_data persisted inside MAS)
        r2 = await mas.chat_with_agent({"query": "second"})
        print(r2.output)

        # you can also inspect the entire dict:
        print("Current global_data:", mas.global_data)


if __name__ == "__main__":
    asyncio.run(main())
