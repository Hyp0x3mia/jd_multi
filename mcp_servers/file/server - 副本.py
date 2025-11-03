"""
PDF / Image OCR + LLM 内容分析 MCP 服务
依赖安装：
pip install mcp-server-fastmcp pdfplumber pillow cnocr openai
（如需英文/多语言，可把 cnocr 换成 easyocr）
"""

import os
import io
from typing import Optional, List
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from pathlib import Path
import fitz
import base64
from openai import OpenAI


_DEEP_ANALYZER_INSTRUCTION = """You should step-by-step analyze the task and/or the attached content.
* When the task involves playing a game or performing calculations. Please consider the conditions imposed by the game or calculation rules. You may take extreme conditions into account.
* When the task involves spelling words, you must ensure that the spelling rules are followed and that the resulting word is meaningful.
* When the task involves compute the area in a specific polygon. You should separate the polygon into sub-polygons and ensure that the area of each sub-polygon is computable (e.g, rectangle, circle, triangle, etc.). Step-by-step to compute the area of each sub-polygon and sum them up to get the final area.
* When the task involves calculation and statistics, it is essential to consider all constraints. Failing to account for these constraints can easily lead to statistical errors.

Here is the task:
"""
# -------------------- 初始化 --------------------
mcp = FastMCP()

client = OpenAI(
    base_url="https://api.siliconflow.cn/v1" ,
    api_key="sk-ouloabxhwnsirhhvsbkokapsjkqyhtwsizuxlgnibbshnfxe",
)

model_name = "Qwen/Qwen3-VL-32B-Thinking"
# -------------------- 通用 OCR --------------------
def _analyze_image(path: str, prompt: str) -> str:
    try:
        task = _DEEP_ANALYZER_INSTRUCTION + prompt
        content = [
            {"type": "text", "text": task},
        ]
        with open(path, "rb") as f:
            b64_image =  base64.b64encode(f.read()).decode("utf-8")
        # 2. 每页当成一张图塞进 content
        content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
        })


        messages = [
            {
                "role": 'user',
                "content": content,
            }
        ]

        response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=False
                )
        output = response.choices[0].message.content
        return output
    except Exception as e:
        return f"[PDF extract failed: {e}]"

# -------------------- PDF 文字提取 --------------------
def _analyze_pdf(path: str, prompt: str) -> str:
    try:
        task = _DEEP_ANALYZER_INSTRUCTION + prompt
        content = [
            {"type": "text", "text": task},
        ]

        doc = fitz.open(path)

        for page in doc:
            pix = page.get_pixmap(dpi=300)          # 200 dpi 足够看清
            png_bytes = pix.tobytes("png")
            b64_image = base64.b64encode(png_bytes).decode()
            # 2. 每页当成一张图塞进 content
            content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_image}"},
            })
        doc.close()
        print( path)

        messages = [
            {
                "role": 'user',
                "content": content,
            }
        ]

        response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=False
                )
        output = response.choices[0].message.content
        return output
    except Exception as e:
        print(e)
        return f"[PDF extract failed: {e}]"




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
    # print(doc_analyze("./cache_dir/uploads/20251102213731_help_1756089021301.pdf", "查找京东历史规则中关于评价内容被折叠的条件，特别是位于'乱码'之前的评价内容类型词是什么"))

    mcp.run()
