"""
音频转写 MCP 服务（SiliconFlow SenseVoice via OpenAI SDK）
"""

import os
from typing import Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP()


@mcp.tool(description="将音频文件转写为文本（使用 SiliconFlow SenseVoice）")
async def audio_transcribe(
    path: str = Field(description="本地音频文件路径，如 .wav/.mp3/.m4a/.flac"),
    model_name: Optional[str] = Field(default=None, description="可选：模型名，默认读取环境变量 SENSEVOICE_MODEL 或 SenseVoiceSmall"),
    prompt: Optional[str] = Field(default=None, description="可选：转写提示词，用于限定领域或人名"),
    language: Optional[str] = Field(default=None, description="可选：目标语言（若模型支持）"),
):

    try:
        from openai import OpenAI
    except Exception:
        return "缺少依赖 openai，请先安装：pip install openai"

    if not os.path.exists(path):
        return {"error": f"音频文件不存在: {path}"}

    # 读取 SiliconFlow API 配置
    api_key = os.getenv("SENSEVOICE_KEY")
    base_url = os.getenv("SENSEVOICE_URL")
    model =os.getenv("SENSEVOICE_MODEL")

    if not api_key or not base_url or not model:
        return {"error": "音频转写配置不完整，请设置 SENSEVOICE_KEY/SENSEVOICE_URL/SENSEVOICE_MODEL 或对应 VL_/DEFAULT_ 环境变量"}

    # OpenAI 兼容 SDK 客户端
    client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))

    try:
        with open(path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=model,
                file=f,
                prompt=prompt,
                # 可根据服务支持情况添加：temperature、response_format、language 等
            )
        # SDK 返回对象不同版本可能不同，兼容 text/content 两种取法
        text = None
        if hasattr(resp, "text") and resp.text:
            text = resp.text
        elif hasattr(resp, "content") and resp.content:
            text = resp.content
        else:
            # 某些实现返回 dict-like
            try:
                text = resp.get("text") or resp.get("content")
            except Exception:
                text = None

        if not text:
            return {"error": "ASR返回为空，请检查模型与配额/文件格式", "raw": str(resp)}

        return {
            "status": "success",
            "note": "【音频转写】以下为音频识别文本",
            "text": text,
            "path": path,
            "model": model,
        }
    except Exception as e:
        return {"error": f"调用ASR失败: {str(e)}"}


if __name__ == "__main__":
    mcp.run()


