import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client

url = "http://localhost:8000/sse"


async def main():
    async with sse_client(url) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tool_list_result = await session.list_tools()
            print(tool_list_result)
            tool_call_result = await session.call_tool("calc_pi", {"prec": 20})
            print(tool_call_result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
