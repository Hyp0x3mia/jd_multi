import asyncio
import os

from oxygent import MAS, OxyRequest, oxy


# 自定义reflexion函数
def master_reflexion(response: str, oxy_request: OxyRequest) -> str:
    import re

    pattern = r"^[-+]?(\d+(\.\d*)?|\.\d+)$"
    if not bool(re.match(pattern, response)):
        return "仅回答数字"


oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    oxy.ReActAgent(
        name="master_agent",
        llm_model="default_llm",
        func_reflexion=master_reflexion,  # 注册reflexion函数
        additional_prompt="请你根据我的问题，给出最优的回答",  # 补充prompt
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(first_query="1+1等于几")


if __name__ == "__main__":
    asyncio.run(main())
