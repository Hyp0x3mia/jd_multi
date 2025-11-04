"""
PDF / Image OCR + LLM 内容分析 MCP 服务
依赖安装：
pip install mcp-server-fastmcp pdfplumber pillow cnocr openai
（如需英文/多语言，可把 cnocr 换成 easyocr）
"""

import os
import io
import time
import random
from typing import Optional, List
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from pathlib import Path
import base64
from openai import OpenAI
import fitz
from PIL import Image
import sys, traceback
from dotenv import load_dotenv


_DEEP_ANALYZER_INSTRUCTION = """You should step-by-step analyze the task and/or the attached content.
* When the task involves playing a game or performing calculations. Please consider the conditions imposed by the game or calculation rules. You may take extreme conditions into account.
* When the task involves spelling words, you must ensure that the spelling rules are followed and that the resulting word is meaningful.
* When the task involves compute the area in a specific polygon. You should separate the polygon into sub-polygons and ensure that the area of each sub-polygon is computable (e.g, rectangle, circle, triangle, etc.). Step-by-step to compute the area of each sub-polygon and sum them up to get the final area.
* When the task involves calculation and statistics, it is essential to consider all constraints. Failing to account for these constraints can easily lead to statistical errors.

Here is the task:
"""
# -------------------- 初始化 --------------------
mcp = FastMCP()

load_dotenv("./.env")
# 读取 VLM 配置（环境变量优先，带安全默认值，避免 NameError）
_VL_URL = (os.getenv("VL_URL")).rstrip("/")
_VL_KEY = os.getenv("VL_KEY") 
_VL_MODEL = os.getenv("VL_MODEL")
_DOC_MAX_WIDTH = int(os.getenv("DOC_MAX_WIDTH", "1600"))
_PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "220"))

client = OpenAI(base_url=f"{_VL_URL}", api_key=_VL_KEY)
model_name = _VL_MODEL

# -------------------- 通用工具 --------------------
def _resize_image_bytes(img_bytes: bytes, max_width: int = _DOC_MAX_WIDTH) -> bytes:
    """将图片按最长边等比缩放到不超过 max_width，返回PNG字节。"""
    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w <= max_width:
                buf = io.BytesIO()
                im.save(buf, format="PNG", optimize=True)
                return buf.getvalue()
            scale = max_width / float(w)
            new_size = (max_width, int(h * scale))
            im = im.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
    except Exception:
        return img_bytes


def _call_vlm(messages: List[dict], max_retries: int = 4, base_sleep: float = 0.6, timeout: int = 120) -> str:
    """稳定调用 VLM，带指数退避重试。"""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                top_p=0.8,
                max_tokens=512,
                stream=False,
                timeout=timeout,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            # 指数退避
            sleep_s = base_sleep * (1.6 ** (attempt - 1)) + random.uniform(0, 0.2)
            time.sleep(sleep_s)
    raise RuntimeError(f"VLM call failed after {max_retries} attempts: {last_err}")

# -------------------- 通用 OCR --------------------
def _analyze_image(path: str, prompt: str) -> str:
    try:
        with open(path, "rb") as f:
            raw = f.read()
        img_bytes = _resize_image_bytes(raw, max_width=_DOC_MAX_WIDTH)
        b64_image = base64.b64encode(img_bytes).decode("utf-8")

        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
            {"type": "text", "text": (_DEEP_ANALYZER_INSTRUCTION + (prompt or "请只输出答案，禁止解释。"))},
        ]
        messages = [{"role": "user", "content": content}]
        return _call_vlm(messages)
    except Exception as e:
        return f"[IMAGE analyze failed: {e}]"

# -------------------- PDF 文字提取 --------------------
def _analyze_pdf(path: str, prompt: str) -> str:
    try:
        doc = fitz.open(path)
        content: List[dict] = []
        # 渲染每页为图片（较高 DPI），单页 PDF 则仅一张
        for page in doc:
            pix = page.get_pixmap(dpi=_PDF_RENDER_DPI, alpha=False)
            png_bytes = pix.tobytes("png")
            png_bytes = _resize_image_bytes(png_bytes, max_width=_DOC_MAX_WIDTH)
            b64_image = base64.b64encode(png_bytes).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}})
        doc.close()

        content.append({"type": "text", "text": (_DEEP_ANALYZER_INSTRUCTION + (prompt or "请只输出答案，禁止解释。"))})
        messages = [{"role": "user", "content": content}]
        return _call_vlm(messages)
    except Exception as e:
        return f"[PDF analyze failed: {e}]"




# -------------------- MCP 工具 --------------------
@mcp.tool(description="读取本地 PDF 或图片，提取文字并做内容分析")
async def doc_analyze(
# def doc_analyze(
    path: str = Field(description="本地文件路径（pdf / jpg / jpeg / png / bmp）"),
    prompt: Optional[str] = Field(
        default=None, description="可选：给 LLM 的额外提示，如“重点检查合同金额”,“pdf里框里文字什么”"
    ),
):  
    path_str = str(path).strip()
    if path_str:
        if path_str.startswith('./') or (not os.path.isabs(path_str) and '/' not in path_str):
            here = Path(__file__).resolve()   # /a/b/c/d/file.py
            root_path = here.parent.parent.parent           # /a/b
            if root_path.exists():
                path = str(root_path / path_str.lstrip('./'))
    if not os.path.exists(path):
        return {"error": f"文件不存在: {path}"}

    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        analysis = _analyze_pdf(path, prompt)
    elif ext in {".jpg", ".jpeg", ".png", ".bmp"}:
        analysis = _analyze_image(path, prompt)
    else:
        return {"error": "仅支持 pdf / jpg / jpeg / png / bmp"}

    if not analysis or analysis.startswith("[") and "failed" in analysis:
        return {"error": "分析失败", "raw": analysis}


    return {
        "status": "success",
        "path": path,
        "analysis": analysis,
    }

# -------------------- 启动 --------------------
if __name__ == "__main__":
    # print(doc_analyze("/Users/hyp0x3mia/BUPT_Master/Master/2025/开源之夏/京东/OxyGent/valid/hqwqa3.pdf", "查找京东历史规则中关于评价内容被折叠的条件，特别是位于'乱码'之前的评价内容类型词是什么"))
    mcp.run()

