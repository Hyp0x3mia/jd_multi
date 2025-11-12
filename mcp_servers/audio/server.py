"""
音频转写 MCP 服务（SiliconFlow SenseVoice via OpenAI SDK）
"""

import os
from typing import Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP()

'''
pip install mutagen
'''

import os
import pathlib


@mcp.tool(description="获取音频文件的基本信息，包括文件大小、时长、比特率、采样率和通道数")
async def audio_file_info(file_path: str):
    """
    获取音频文件的基本信息
    
    Args:
        file_path: 音频文件路径
        
    Returns:
        dict: 包含文件大小(MB)、时长(秒)、比特率(kbps)、采样率(Hz)、通道数等信息
    """
    try:
        from mutagen.mp3 import MP3
        import os
        import pathlib
    except ImportError:
        return {"error": "缺少依赖 mutagen，请先安装：pip install mutagen"}

    # 解析文件路径
    path = pathlib.Path(file_path).expanduser().resolve()
    if not path.exists():
        return {
            "status": "failed",
            "analysis": "文件路径不存在",
            "file_path": str(path)
        }

    try:
        # 解析 MP3 文件
        audio = MP3(path)
        file_size_mb = os.path.getsize(path) / (1024 * 1024)

        return {
            "status": "success",
            "analysis": {
                "file_path": str(path),
                "file_size_MB": round(file_size_mb, 2),
                "duration_sec": round(audio.info.length, 2),
                "bitrate_kbps": int(audio.info.bitrate // 1000),
                "sample_rate_Hz": audio.info.sample_rate,
                "channels": audio.info.channels
            }
        }
    except Exception as e:
        return {
            "status": "failed",
            "analysis": f"解析音频文件时出错: {str(e)}"
        }


@mcp.tool(description="将音频文件转写为文本,纯音乐则返回未检测到任何的文本")
async def audio_transcribe(
    path: str = Field(description="本地音频文件路径，如 .wav/.mp3/.m4a/.flac"),
    model_name: Optional[str] = Field(default=None, description="可选：模型名，默认读取环境变量 SENSEVOICE_MODEL 或 SenseVoiceSmall"),
    prompt: Optional[str] = Field(default=None, description="可选：转写提示词，用于限定领域或人名"),
    language: Optional[str] = Field(default=None, description="可选：目标语言（若模型支持）"),
):
    try:
        from openai import OpenAI
        import asyncio
        import re
        import io
        import json
    except Exception:
        return {"error": "缺少依赖 openai，请先安装：pip install openai"}

    # # 清理路径：移除引号、方括号等
    # path = path.strip()
    # # 移除可能的引号
    # path = path.strip('"').strip("'")
    # # 移除可能的方括号和列表标记
    # if path.startswith('[') and path.endswith(']'):
    #     # 尝试解析为列表并取第一个
    #     try:
    #         path_list = json.loads(path)
    #         if isinstance(path_list, list) and len(path_list) > 0:
    #             path = str(path_list[0]).strip('"').strip("'")
    #     except:
    #         # 如果 JSON 解析失败，直接用正则提取路径
    #         match = re.search(r"['\"]([^'\"]+)['\"]", path)
    #         if match:
    #             path = match.group(1)
    #         else:
    #             # 移除方括号
    #             path = path.strip('[]').strip()
    
    # # 进一步清理可能的转义字符
    # path = path.replace('\\', '').strip()

    if not path:
        return {"error": "路径为空"}
    
    if not os.path.exists(path):
        return {"error": f"音频文件不存在: {path}"}

    # 读取 SiliconFlow API 配置
    api_key = os.getenv("SENSEVOICE_KEY")
    base_url = os.getenv("SENSEVOICE_URL")
    model = os.getenv("SENSEVOICE_MODEL") or model_name

    if not api_key or not base_url or not model:
        return {"error": "音频转写配置不完整，请设置 SENSEVOICE_KEY/SENSEVOICE_URL/SENSEVOICE_MODEL 或对应 VL_/DEFAULT_ 环境变量"}

    # OpenAI 兼容 SDK 客户端，设置超时
    client = OpenAI(
        api_key=api_key, 
        base_url=base_url.rstrip("/"),
        timeout=300.0  # 5分钟超时
    )

    try:
        # 先读取文件内容，然后在 executor 中调用 API
        try:
            with open(path, "rb") as f:
                file_content = f.read()
        except Exception as e:
            return {"error": f"无法读取音频文件: {str(e)}"}
        
        # 检查文件大小（避免过大文件导致内存问题）
        if len(file_content) > 100 * 1024 * 1024:  # 100MB
            return {"error": f"音频文件过大（{len(file_content) / 1024 / 1024:.1f}MB），超过100MB限制"}
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # 如果没有事件循环，创建一个新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 在 executor 中运行同步 API 调用，避免阻塞事件循环
        def _transcribe():
            # 创建临时文件对象
            file_obj = io.BytesIO(file_content)
            file_obj.name = os.path.basename(path)
            return client.audio.transcriptions.create(
                model=model,
                file=file_obj,
                prompt=prompt,
                # 可根据服务支持情况添加：temperature、response_format、language 等
            )
        
        resp = await loop.run_in_executor(None, _transcribe)
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
            return  {
            "status": "success",
            "analysis": "ASR未检测到任何文本,文件可能是纯音乐，请使用audio_recognize_song工具识别音乐名后交叉确认"
        }

        return {
            "status": "success",
            "analysis": f"【音频转写】以下为音频识别文本：{text}"
        }
    except asyncio.TimeoutError:
        return {
            "status": "success",
            "analysis": "ASR未检测到任何文本,可能文件是纯音乐，请使用audio_recognize_song工具识别音乐名后交叉确认"
        }
    except Exception as e:
        error_str = str(e)
        # 区分不同类型的错误
        return {
            "status": "success",
            "analysis": "ASR未检测到任何的文本,可能文件是纯音乐，请使用audio_recognize_song工具识别音乐名后交叉确认"
        }



@mcp.tool(description="通过音频指纹识别歌曲（使用 Chromaprint 和 pyacoustid）")
async def audio_recognize_song(
    path: str = Field(description="本地音频文件路径，如 .wav/.mp3/.m4a/.flac"),
):
    """通过音频指纹识别歌曲名称和艺术家"""
    try:

        import json
        import shlex
        import subprocess

    except ImportError as e:
        missing = str(e)
        if "acoustid" in missing:
            return {"error": "缺少依赖库，请先安装：pip install subprocess shlex json 库"}
        return {"error": f"缺少依赖库: {missing}"}
    
    # 检查 fpcalc 是否可用（pyacoustid 需要）
    try:
        result = subprocess.run(['fpcalc', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=2,
                              stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            raise FileNotFoundError("fpcalc not found")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return {
            "error": "缺少 fpcalc 工具（Chromaprint 命令行工具）。请在系统中安装 Chromaprint:\n"
                     "  - macOS: brew install chromaprint\n"
                     "  - Ubuntu/Debian: sudo apt-get install libchromaprint-tools\n"
                     "  - 或从 https://acoustid.org/chromaprint 下载安装"
        }
    except Exception:
        # 如果检测失败，继续执行（可能在 _recognize 中会有更详细的错误信息）
        pass
    
    # 清理路径：移除引号、方括号等（复用 audio_transcribe 的逻辑）
    # path = path.strip()
    # path = path.strip('"').strip("'")
    # if path.startswith('[') and path.endswith(']'):
    #     try:
    #         path_list = json.loads(path)
    #         if isinstance(path_list, list) and len(path_list) > 0:
    #             path = str(path_list[0]).strip('"').strip("'")
    #     except:
    #         match = re.search(r"['\"]([^'\"]+)['\"]", path)
    #         if match:
    #             path = match.group(1)
    #         else:
    #             path = path.strip('[]').strip()
    
    # path = path.replace('\\', '').strip()
    
    if not path:
        return {"error": "路径为空"}
    
    if not os.path.exists(path):
        return {"error": f"音频文件不存在: {path}"}
    
    try:
        # 让 ffprobe 吐出 json，方便直接丢给 Python
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-show_format', '-show_streams',
            '-of', 'json', path
        ]
        info = subprocess.check_output(cmd, encoding='utf-8')
        meta = json.loads(info)['format']['tags']   # 纯 dict

        title  = meta.get('title',  '')
        artist = meta.get('artist', '')
        album  = meta.get('album',  '')
        # 现在全是 str，随便拼接/赋值
        whole = f'{artist} - {title} ({album})'
        return {
        "status": "success",
        "analysis": whole,
        }
            
    except Exception as e:
        return {"error": f"调用音频识别失败: {str(e)}"}

import asyncio
if __name__ == "__main__":
    mcp.run()


