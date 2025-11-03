import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

url = "http://localhost:8000/mcp"


async def main():
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_list_result = await session.list_tools()
            print(tool_list_result)
            tool_call_result = await session.call_tool("calc_pi", {"prec": 20})
            print(tool_call_result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
