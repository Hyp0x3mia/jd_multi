import asyncio
import os

from pydantic import Field

from oxygent import MAS, oxy

fh_joke_tools = oxy.FunctionHub(name="joke_tools")


@fh_joke_tools.tool(description="a tool for telling jokes")
async def joke_tool(joke_type: str = Field(description="The type of the jokes")):
    import random

    jokes = [
        "Teacher: Can you use the word 'because' in a sentence? \n Student: I didn't do my homework because… because I didn't do my homework.",
        "Patient: Doctor, I feel like a pair of curtains.\nDoctor: Pull yourself together!",
        "How many software engineers does it take to change a light bulb?\nNone. That's a hardware problem.",
    ]
    print("The type of the jokes", joke_type)
    return random.choice(jokes)


oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    fh_joke_tools,
    oxy.ReActAgent(
        name="joke_agent",
        tools=["joke_tools"],
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(first_query="Please tell a joke")


if __name__ == "__main__":
    asyncio.run(main())
