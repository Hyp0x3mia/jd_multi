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
        import asyncio
        import re
        import io
        import json
    except Exception:
        return {"error": "缺少依赖 openai，请先安装：pip install openai"}

    # 清理路径：移除引号、方括号等
    path = path.strip()
    # 移除可能的引号
    path = path.strip('"').strip("'")
    # 移除可能的方括号和列表标记
    if path.startswith('[') and path.endswith(']'):
        # 尝试解析为列表并取第一个
        try:
            path_list = json.loads(path)
            if isinstance(path_list, list) and len(path_list) > 0:
                path = str(path_list[0]).strip('"').strip("'")
        except:
            # 如果 JSON 解析失败，直接用正则提取路径
            match = re.search(r"['\"]([^'\"]+)['\"]", path)
            if match:
                path = match.group(1)
            else:
                # 移除方括号
                path = path.strip('[]').strip()
    
    # 进一步清理可能的转义字符
    path = path.replace('\\', '').strip()

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
            return {"error": "ASR返回为空，请检查模型与配额/文件格式", "raw": str(resp)}

        return {
            "status": "success",
            "note": "【音频转写】以下为音频识别文本",
            "text": text,
            "path": path,
            "model": model,
        }
    except asyncio.TimeoutError:
        return {"error": "音频转写超时（超过5分钟），文件可能过大或网络问题"}
    except Exception as e:
        error_str = str(e)
        # 区分不同类型的错误
        if "504" in error_str or "Gateway Timeout" in error_str or "timeout" in error_str.lower():
            return {"error": f"ASR服务超时（504 Gateway Timeout），可能是文件过大或服务繁忙，请稍后重试: {error_str}"}
        elif "429" in error_str or "rate limit" in error_str.lower():
            return {"error": f"ASR服务限流，请稍后重试: {error_str}"}
        elif "401" in error_str or "unauthorized" in error_str.lower():
            return {"error": f"ASR认证失败，请检查 API Key: {error_str}"}
        else:
            return {"error": f"调用ASR失败: {error_str}"}


@mcp.tool(description="通过音频指纹识别歌曲（使用 Chromaprint 和 pyacoustid）")
async def audio_recognize_song(
    path: str = Field(description="本地音频文件路径，如 .wav/.mp3/.m4a/.flac"),
):
    """通过音频指纹识别歌曲名称和艺术家"""
    try:
        import acoustid
        import asyncio
        import re
        import io
        import json
        import subprocess
    except ImportError as e:
        missing = str(e)
        if "acoustid" in missing:
            return {"error": "缺少依赖库，请先安装：pip install pyacoustid。注意：pyacoustid 需要系统安装 chromaprint 库"}
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
    path = path.strip()
    path = path.strip('"').strip("'")
    if path.startswith('[') and path.endswith(']'):
        try:
            path_list = json.loads(path)
            if isinstance(path_list, list) and len(path_list) > 0:
                path = str(path_list[0]).strip('"').strip("'")
        except:
            match = re.search(r"['\"]([^'\"]+)['\"]", path)
            if match:
                path = match.group(1)
            else:
                path = path.strip('[]').strip()
    
    path = path.replace('\\', '').strip()
    
    if not path:
        return {"error": "路径为空"}
    
    if not os.path.exists(path):
        return {"error": f"音频文件不存在: {path}"}
    
    # 读取 AcoustID API Key
    api_key = os.getenv("ACOUSTID_API_KEY")
    if not api_key:
        return {"error": "音频指纹识别配置不完整，请设置 ACOUSTID_API_KEY 环境变量"}
    
    try:
        # 获取事件循环
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 在 executor 中运行同步指纹提取和识别
        def _recognize():
            try:
                # 提取音频指纹（使用 pyacoustid，返回 (duration, fingerprint)）
                duration, fingerprint = acoustid.fingerprint_file(path)
                if not fingerprint:
                    return {"error": "无法提取音频指纹，可能是文件格式不支持或文件损坏"}
                
                # 使用 AcoustID API 查找匹配的歌曲
                results = acoustid.lookup(api_key, fingerprint, duration)
                
                if not results or not results.get('results') or len(results.get('results', [])) == 0:
                    return {"error": "未找到匹配的歌曲", "status": "not_found"}
                
                # 提取最佳匹配结果（按 score 排序，取第一个）
                results_list = results.get('results', [])
                # 按 score 降序排序（如果有 score 字段）
                if results_list and 'score' in results_list[0]:
                    results_list = sorted(results_list, key=lambda x: x.get('score', 0), reverse=True)
                
                best_match = results_list[0]
                
                if not best_match.get('recordings') or len(best_match.get('recordings', [])) == 0:
                    return {"error": "找到匹配但无录制信息", "status": "no_recordings"}
                
                recording = best_match['recordings'][0]
                title = recording.get('title', 'Unknown')
                
                # 提取艺术家信息
                artists = []
                if 'artists' in recording and recording.get('artists'):
                    for artist in recording['artists']:
                        if isinstance(artist, dict) and 'name' in artist:
                            artists.append(artist['name'])
                        elif isinstance(artist, str):
                            artists.append(artist)
                
                artist_name = ', '.join(artists) if artists else 'Unknown Artist'
                
                return {
                    "status": "success",
                    "note": "【音频识别】以下为歌曲识别结果",
                    "title": title,
                    "artist": artist_name,
                    "duration": duration,
                    "path": path,
                }
            except acoustid.WebServiceError as e:
                return {"error": f"AcoustID API 错误: {str(e)}"}
            except FileNotFoundError as e:
                if "fpcalc" in str(e).lower():
                    return {
                        "error": "缺少 fpcalc 工具（Chromaprint 命令行工具）。请在系统中安装 Chromaprint:\n"
                                "  - macOS: brew install chromaprint\n"
                                "  - Ubuntu/Debian: sudo apt-get install libchromaprint-tools\n"
                                "  - 或从 https://acoustid.org/chromaprint 下载安装"
                    }
                return {"error": f"文件未找到: {str(e)}"}
            except OSError as e:
                if "fpcalc" in str(e).lower() or "No such file" in str(e):
                    return {
                        "error": "缺少 fpcalc 工具（Chromaprint 命令行工具）。请在系统中安装 Chromaprint:\n"
                                "  - macOS: brew install chromaprint\n"
                                "  - Ubuntu/Debian: sudo apt-get install libchromaprint-tools\n"
                                "  - 或从 https://acoustid.org/chromaprint 下载安装"
                    }
                return {"error": f"系统错误: {str(e)}"}
            except Exception as e:
                error_str = str(e)
                if "fpcalc" in error_str.lower() or "fpcalc not found" in error_str.lower():
                    return {
                        "error": "缺少 fpcalc 工具（Chromaprint 命令行工具）。请在系统中安装 Chromaprint:\n"
                                "  - macOS: brew install chromaprint\n"
                                "  - Ubuntu/Debian: sudo apt-get install libchromaprint-tools\n"
                                "  - 或从 https://acoustid.org/chromaprint 下载安装"
                    }
                return {"error": f"音频识别失败: {error_str}"}
        
        result = await loop.run_in_executor(None, _recognize)
        return result
        
    except Exception as e:
        return {"error": f"调用音频识别失败: {str(e)}"}


if __name__ == "__main__":
    mcp.run()


