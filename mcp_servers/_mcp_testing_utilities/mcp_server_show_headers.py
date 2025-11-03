import math

import anyio
import uvicorn
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

mcp = FastMCP("auth-mcp-server")


class PrintHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        print("----------------")
        for k, v in dict(request.headers).items():
            print(k, v)
        print("----------------")
        return await call_next(request)


@mcp.tool(description="A tool for calculating powers")
def power(
    n: int = Field(description="base"),
    m: int = Field(description="index", default=2),
) -> int:
    return math.pow(n, m)


app = mcp.streamable_http_app()
app.add_middleware(PrintHeaderMiddleware)


async def run():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    anyio.run(run)
