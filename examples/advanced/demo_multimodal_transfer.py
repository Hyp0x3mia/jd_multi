import asyncio
import os

from oxygent import MAS, oxy, preset_tools

oxy_space = [
    oxy.HttpLLM(
        name="default_vlm",
        api_key=os.getenv("DEFAULT_VLM_API_KEY"),
        base_url=os.getenv("DEFAULT_VLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_VLM_MODEL_NAME"),
        is_multimodal_supported=True,
        is_convert_url_to_base64=True,
    ),
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    preset_tools.image_gen_tools,
    oxy.ChatAgent(
        name="vision_agent",
        desc="一个图片理解工具",
        llm_model="default_vlm",
    ),
    oxy.ReActAgent(
        name="master_agent",
        is_master=True,
        sub_agents=["vision_agent"],
        tools=["image_gen_tools"],
        llm_model="default_llm",
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(first_query="这是什么，生成一张卡通的")


if __name__ == "__main__":
    asyncio.run(main())
