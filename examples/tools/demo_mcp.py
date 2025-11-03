import asyncio
import os

from oxygent import MAS, oxy

oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    oxy.StdioMCPClient(
        name="time_tools",
        params={
            "command": "uvx",
            "args": ["mcp-server-time", "--local-timezone=Asia/Shanghai"],
        },
    ),
    oxy.StdioMCPClient(
        name="map_tools",
        params={
            "command": "npx",
            "args": ["-y", "@amap/amap-maps-mcp-server"],
            "env": {"AMAP_MAPS_API_KEY": "API_KEY"},  # 配置高德地图API密钥
        },
    ),
    oxy.StdioMCPClient(
        name="math_tools",
        params={
            "command": "uv",
            "args": ["--directory", "./mcp_servers", "run", "math_tools.py"],
        },
    ),
    # oxy.SSEMCPClient(
    #     name="sse_mcp_tools",
    #     sse_url="http://127.0.0.1:8000/sse",
    # ),
    # oxy.StreamableMCPClient(
    #     name="stream_mcp_tools",
    #     server_url="http://127.0.0.1:8000/mcp",
    # ),
    oxy.ReActAgent(
        name="master_agent",
        tools=["time_tools", "math_tools"],
        llm_model="default_llm",
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(first_query="What time is it")


if __name__ == "__main__":
    asyncio.run(main())
