"""
流程图API模块，提供保存和生成流程图的API端点
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
import os
import time
import datetime
import asyncio

# 创建路由器
router = APIRouter()

# 请求模型
class SaveFlowchartRequest(BaseModel):
    mermaid_code: str
    file_name: str = None

class GenerateFlowchartRequest(BaseModel):
    description: str
    file_name: str = None

@router.post("/save-flowchart")
async def save_flowchart(request: SaveFlowchartRequest):
    """保存编辑后的流程图"""
    try:
        # 生成文件名
        if request.file_name:
            file_name = request.file_name
            if not file_name.endswith('.html'):
                file_name += '.html'
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            file_name = f"flowchart-{timestamp}.html"
        
        # 创建HTML内容
        from oxygent.chart.create_flow_image import create_html_with_mermaid
        
        success = create_html_with_mermaid(request.mermaid_code, file_name)
        
        if success:
            return {"success": True, "file_path": os.path.abspath(file_name)}
        else:
            return {"success": False, "error": "保存文件失败"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/generate")
async def generate_flowchart(request: GenerateFlowchartRequest):
    """根据描述生成流程图"""
    try:
        # 导入生成函数
        from oxygent.chart.create_flow_image import generate_flow_chart, call_ollama_api, create_html_with_mermaid
        
        # 生成文件名
        if request.file_name:
            file_name = request.file_name
            if not file_name.endswith('.html'):
                file_name += '.html'
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            file_name = f"flowchart-{timestamp}.html"
            
        # 确保使用绝对路径
        file_name = os.path.abspath(file_name)
        
        # 调用 Ollama API 生成 Mermaid 代码
        mermaid_code = call_ollama_api(request.description)
        
        # 创建 HTML 文件
        success = create_html_with_mermaid(mermaid_code, file_name)
        
        if success:
            abs_path = os.path.abspath(file_name)
            
            # 每次请求都打开浏览器
            from oxygent.chart.open_chart_tools import open_chart_tools
            print(f"准备打开文件: {abs_path}")
            await open_chart_tools.open_html_chart(abs_path)
            
            return {
                "success": True,
                "file_path": abs_path,
                "message": f"流程图已生成并在浏览器中打开: {abs_path}"
            }
        else:
            return {"success": False, "error": "生成流程图时出错"}
    except Exception as e:
        return {"success": False, "error": str(e)}