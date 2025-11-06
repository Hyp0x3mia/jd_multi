"""
浏览器内容提取和截图功能

提供捕获页面的可访问性快照和截取页面截图等功能
"""

import os
import sys
import uuid
from datetime import datetime
import base64
import json
from typing import Optional

from pydantic import Field

from .core import (
    _ensure_page,
    _set_operation_status,
    _verify_data_ready,
    check_dependencies,
    mcp,
)


@mcp.tool(description="捕获页面的可访问性快照")
async def browser_snapshot():
    """
    捕获页面的可访问性快照，包括页面标题、URL和内容
    """
    # 检查依赖
    missing_deps = check_dependencies()
    if missing_deps:
        return f"缺少必要的库: {', '.join(missing_deps)}。请使用pip安装: pip install {' '.join(missing_deps)}"

    try:
        await _set_operation_status(True)
        page = await _ensure_page()

        # 获取页面信息
        title = await page.title()
        url = page.url

        # 获取页面内容
        content = await page.content()

        # 提取页面文本
        text = await page.evaluate("""() => {
            return document.body.innerText;
        }""")

        # 如果文本太长，截取前2000个字符
        if len(text) > 2000:
            text = text[:2000] + "...(内容已截断)"

        snapshot = {"title": title, "url": url, "text": text, "data_complete": True}

        await _verify_data_ready()
        await _set_operation_status(False)
        return str(snapshot)
    except Exception as e:
        await _set_operation_status(False)
        return f"捕获页面快照时发生错误: {str(e)}"


@mcp.tool(description="截取页面截图")
async def browser_take_screenshot(
    path: str = Field(
        default="", description="保存截图的路径，如果为空则保存到cache_dir目录"
    ),
    full_page: bool = Field(
        default=False, description="是否截取整个页面，而不仅仅是可见区域"
    ),
):
    """
    截取当前页面的截图，保存为文件并返回文件路径

    如果未指定路径，将自动保存到项目根目录下的cache_dir目录中
    """
    # 检查依赖
    missing_deps = check_dependencies()
    if missing_deps:
        return f"缺少必要的库: {', '.join(missing_deps)}。请使用pip安装: pip install {' '.join(missing_deps)}"

    try:
        await _set_operation_status(True)
        page = await _ensure_page()

        # 等待页面稳定
        import asyncio

        await asyncio.sleep(0.5)

        # 截取截图
        screenshot_bytes = await page.screenshot(full_page=full_page)

        # 计算图片大小（用于信息展示）
        size_mb = len(screenshot_bytes) / (1024 * 1024)

        # 确定保存路径
        save_path = path
        if not save_path:
            # 创建cache_dir目录（如果不存在）
            cache_dir = os.path.join(os.getcwd(), "../", "cache_dir")
            os.makedirs(cache_dir, exist_ok=True)

            # 创建screenshot子目录（如果不存在）
            screenshot_dir = os.path.join(cache_dir, "screenshot")
            os.makedirs(screenshot_dir, exist_ok=True)

            # 生成唯一的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}_{unique_id}.png"

            # 完整的保存路径
            save_path = os.path.join(screenshot_dir, filename)
        else:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        # 保存截图到文件
        with open(save_path, "wb") as f:
            f.write(screenshot_bytes)

        await _verify_data_ready()
        await _set_operation_status(False)
        return {"path": save_path, "size_mb": round(size_mb, 2)}
    except Exception as e:
        await _set_operation_status(False)
        return f"截取页面截图时发生错误: {str(e)}"


@mcp.tool(description="对页面截图进行视觉理解（调用外部视觉LLM），支持长页多切片并行送入")
async def browser_analyze_screenshot(
    prompt: str = Field(description="需要从截图中提取/理解的指令，如：'读取页面主按钮文字'"),
    path: str = Field(default="", description="已有截图路径；为空则自动截取当前页面"),
    full_page: bool = Field(default=False, description="是否截取整页（当 path 为空时生效)"),
    slice_if_tall: bool = Field(default=True, description="当整页过高时是否自动切片后多图送入"),
    segment_height: int = Field(default=2200, description="单切片高度像素（建议2000-2600）"),
    max_segments: int = Field(default=4, description="最长页最多切片数（避免超长上下文）"),
    overlap: int = Field(default=80, description="相邻切片重叠像素，利于跨切片字段识别"),
    model_name: Optional[str] = Field(default=None, description="可选，覆盖环境变量中的视觉模型名"),
):
    """
    使用官方openai-py SDK调用多模态视觉大模型
    """
    try:
        import requests
        import base64
        import json
        from openai import OpenAI
    except Exception:
        return "缺少必要的库: requests 和 openai，请执行 pip install requests openai"

    # 截图逻辑
    try:
        await _set_operation_status(True)
        screenshot_path = path
        if not screenshot_path:
            page = await _ensure_page()
            import asyncio
            await asyncio.sleep(0.5)
            bytes_png = await page.screenshot(full_page=full_page)
            cache_dir = os.path.join(os.getcwd(), "../", "cache_dir")
            os.makedirs(cache_dir, exist_ok=True)
            screenshot_dir = os.path.join(cache_dir, "screenshot")
            os.makedirs(screenshot_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            uid = str(uuid.uuid4())[:8]
            filename = f"{ts}_{uid}.png"
            screenshot_path = os.path.join(screenshot_dir, filename)
            with open(screenshot_path, "wb") as f:
                f.write(bytes_png)
        else:
            if not os.path.exists(screenshot_path):
                await _set_operation_status(False)
                return f"提供的截图路径不存在: {screenshot_path}"

        # 读取图片，并按需切片
        from PIL import Image
        image_paths: List[str] = []
        if slice_if_tall:
            try:
                im = Image.open(screenshot_path).convert("RGB")
                w, h = im.size
                if h > segment_height:
                    top = 0
                    segs = 0
                    while top < h and segs < max_segments:
                        bottom = min(h, top + segment_height)
                        crop = im.crop((0, top, w, bottom))
                        # 保存临时切片文件
                        cache_dir = os.path.join(os.getcwd(), "../", "cache_dir")
                        os.makedirs(cache_dir, exist_ok=True)
                        seg_dir = os.path.join(cache_dir, "screenshot")
                        os.makedirs(seg_dir, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        uid = str(uuid.uuid4())[:8]
                        seg_name = f"{ts}_{uid}_seg{segs+1}.png"
                        seg_path = os.path.join(seg_dir, seg_name)
                        crop.save(seg_path, format="PNG", optimize=True)
                        image_paths.append(seg_path)
                        segs += 1
                        if bottom >= h:
                            break
                        top = bottom - max(0, overlap)
                else:
                    image_paths.append(screenshot_path)
            except Exception:
                # 回退为单图
                image_paths.append(screenshot_path)
        else:
            image_paths.append(screenshot_path)

        # base64编码图片，拼接data-url（多图）
        img_data_urls: List[str] = []
        for p in image_paths:
            with open(p, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            img_data_urls.append(f"data:image/png;base64,{img_b64}")

        # 视觉模型参数
        api_key = os.getenv("VL_KEY")
        base_url = os.getenv("VL_URL")
        model = (model_name or os.getenv("VL_MODEL"))
        if not api_key or not base_url or not model:
            await _set_operation_status(False)
            return "视觉LLM配置不完整：请配置 VL_KEY、VL_URL 和 VL_MODEL"

        # SDK初始化
        client = OpenAI(api_key=api_key, base_url=base_url)

        contents = []
        for url in img_data_urls:
            contents.append({"type": "image_url", "image_url": {"url": url}})
        contents.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": contents}]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.01,
            )
            # 返回体输出（使用 stderr 避免干扰 MCP 协议）
            sys.stderr.write(f"VLM接口返回response: {response}\n")
            sys.stderr.flush()
            content = ""
            raw_result = None
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                # openai-py 1.x/2.x结构可能不同
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    content = choice.message.content
                    raw_result = choice.message
                elif isinstance(choice, dict) and "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    raw_result = choice["message"]
            if not content:
                sys.stderr.write(f"VLM解析返回空content！response: {str(response)[:500]}\n")
                sys.stderr.flush()
                await _set_operation_status(False)
                return {
                    "error": "VLM返回空内容，请检查模型和配额或图片/指令是否合理。", 
                    "raw": str(response),
                    "path": screenshot_path
                }
            await _verify_data_ready()
            await _set_operation_status(False)
            return {"result": str(content), "raw": str(raw_result), "paths": image_paths if len(image_paths)>1 else [screenshot_path]}
        except Exception as e:
            sys.stderr.write(f"VLM调用异常：{str(e)}\n")
            sys.stderr.flush()
            await _set_operation_status(False)
            return {"error": f"VLM SDK调用异常: {str(e)}", "path": screenshot_path}
    except Exception as e:
        await _set_operation_status(False)
        return f"处理截图/视觉LLM时发生异常: {str(e)}"
