import asyncio
import os

from oxygent import MAS, OxyRequest, oxy


async def workflow(oxy_request: OxyRequest):
    print(oxy_request.get_arguments("query"))
    PI = "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998"
    return PI[: int(oxy_request.get_arguments("precision"))]


oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    oxy.ReActAgent(
        name="master_agent",
        sub_agents=["math_agent"],
        is_master=True,
        llm_model="default_llm",
    ),
    oxy.WorkflowAgent(
        name="math_agent",
        desc="A tool for pi query",
        input_schema={
            "properties": {
                "query": {"description": "Query question"},
                "precision": {"description": "How many decimal places are there"},
            },
            "required": ["query", "precision"],
        },
        func_workflow=workflow,
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(
            first_query="Please calculate the 20 positions of Pi",
        )


if __name__ == "__main__":
    asyncio.run(main())
