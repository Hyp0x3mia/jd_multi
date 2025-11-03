"""
流程图打开工具模块
"""

from oxygent.oxy import FunctionHub
import os
import webbrowser
import asyncio

# 初始化 FunctionHub
open_chart_tools = FunctionHub(name="open_chart_tools")

@open_chart_tools.tool(
    description="在浏览器中打开生成的流程图 HTML 文件"
)
async def open_html_chart(file_path: str) -> str:
    """
    在浏览器中打开生成的流程图 HTML 文件
    
    Args:
        file_path: HTML 文件路径
        
    Returns:
        str: 操作结果消息
    """
    try:
        # 确保使用绝对路径
        if not os.path.isabs(file_path):
            # 如果是相对路径，需要考虑当前工作目录
            current_dir = os.getcwd()
            
            # 如果当前在 examples/other 目录，并且文件路径以 output/ 开头，需要调整路径
            if (current_dir.endswith('examples/other') or current_dir.endswith('examples\\other')) and file_path.startswith('output/'):
                # 回到项目根目录来解析路径
                project_root = os.path.abspath(os.path.join(current_dir, '../..'))
                output_path_abs = os.path.join(project_root, file_path)
            else:
                output_path_abs = os.path.abspath(file_path)
        else:
            output_path_abs = file_path
            
        # 检查文件是否存在
        if not os.path.exists(output_path_abs):
            return f"错误：文件不存在: {output_path_abs}"
            
        print(f"正在打开文件: {output_path_abs}")
        webbrowser.open(f"file://{output_path_abs}")
        return f"已在浏览器中打开流程图: {output_path_abs}"
    except Exception as e:
        return f"打开浏览器时出错: {e}\n请手动打开生成的文件: {file_path}"

# 导出 open_chart_tools 供其他模块使用
__all__ = ['open_chart_tools']