"""
Directory Operations MCP 服务
文件目录操作工具 - 提供文件夹内容检索、文件操作等功能
依赖安装：
pip install mcp-server-fastmcp
"""

import os
import shutil
from typing import List, Optional, Dict, Any
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from pathlib import Path
import stat

# -------------------- 初始化 --------------------
mcp = FastMCP("")

# -------------------- MCP 工具 --------------------

@mcp.tool(description="列出目录内容，支持文件类型过滤和递归搜索")
async def list_directory(
    path: str = Field(description="目录路径"),
    file_type: Optional[str] = Field(default=None, description="文件类型过滤，如 '.txt', '.py' 等"),
    recursive: bool = Field(default=False, description="是否递归搜索子目录")
) -> Dict[str, Any]:
    """列出目录内容，返回文件和文件夹的详细信息"""
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            return {"error": f"目录不存在: {path}"}
        if not path_obj.is_dir():
            return {"error": f"路径不是目录: {path}"}

        items = []
        total_files = 0
        total_dirs = 0
        total_size = 0

        def scan_directory(current_path: Path, current_depth: int = 0):
            nonlocal total_files, total_dirs, total_size
            try:
                for item in current_path.iterdir():
                    try:
                        item_stat = item.stat()
                        is_dir = item.is_dir()
                        
                        # 文件类型过滤
                        if file_type and not is_dir:
                            if not item.suffix.lower() == file_type.lower():
                                continue
                        
                        item_info = {
                            "name": item.name,
                            "path": str(item),
                            "type": "directory" if is_dir else "file",
                            "size": item_stat.st_size if not is_dir else 0,
                            "modified_time": item_stat.st_mtime,
                            "permissions": stat.filemode(item_stat.st_mode),
                            "depth": current_depth
                        }
                        
                        items.append(item_info)
                        
                        if is_dir:
                            total_dirs += 1
                            if recursive:
                                scan_directory(item, current_depth + 1)
                        else:
                            total_files += 1
                            total_size += item_stat.st_size
                            
                    except (PermissionError, OSError) as e:
                        # 跳过无权限访问的文件/目录
                        continue
                        
            except (PermissionError, OSError) as e:
                return

        scan_directory(path_obj)
        
        return {
            "status": "success",
            "path": path,
            "items": items,
            "statistics": {
                "total_files": total_files,
                "total_directories": total_dirs,
                "total_size": total_size,
                "filter_type": file_type,
                "recursive": recursive
            }
        }
        
    except Exception as e:
        return {"error": f"列出目录失败: {str(e)}"}

@mcp.tool(description="统计目录中特定类型文件的数量")
async def count_files(
    path: str = Field(description="目录路径"),
    file_type: Optional[str] = Field(default=None, description="文件类型，如 '.txt', '.py'，不提供则统计所有文件"),
    recursive: bool = Field(default=False, description="是否递归统计子目录")
) -> Dict[str, Any]:
    """统计目录中文件数量"""
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            return {"error": f"目录不存在: {path}"}
        if not path_obj.is_dir():
            return {"error": f"路径不是目录: {path}"}

        count = 0
        total_size = 0

        def count_in_directory(current_path: Path):
            nonlocal count, total_size
            try:
                for item in current_path.iterdir():
                    try:
                        if item.is_file():
                            # 文件类型过滤
                            if file_type:
                                if item.suffix.lower() == file_type.lower():
                                    count += 1
                                    total_size += item.stat().st_size
                            else:
                                count += 1
                                total_size += item.stat().st_size
                        elif item.is_dir() and recursive:
                            count_in_directory(item)
                    except (PermissionError, OSError):
                        continue
            except (PermissionError, OSError):
                return

        count_in_directory(path_obj)
        
        return {
            "status": "success",
            "path": path,
            "file_type": file_type or "all files",
            "count": count,
            "total_size": total_size,
            "recursive": recursive
        }
        
    except Exception as e:
        return {"error": f"统计文件失败: {str(e)}"}



@mcp.tool(description="获取文件/目录详细信息")
async def get_file_info(
    path: str = Field(description="文件或目录路径")
) -> Dict[str, Any]:
    """获取文件或目录的详细信息"""
    try:
        path_obj = Path(path)
        
        if not path_obj.exists():
            return {"error": f"路径不存在: {path}"}
        
        stat_info = path_obj.stat()
        
        info = {
            "name": path_obj.name,
            "path": str(path_obj),
            "type": "directory" if path_obj.is_dir() else "file",
            "size": stat_info.st_size,
            "created_time": stat_info.st_ctime,
            "modified_time": stat_info.st_mtime,
            "accessed_time": stat_info.st_atime,
            "permissions": stat.filemode(stat_info.st_mode),
            "absolute_path": str(path_obj.resolve())
        }
        
        # 如果是目录，添加额外信息
        if path_obj.is_dir():
            file_count = 0
            dir_count = 0
            total_size = 0
            
            try:
                for item in path_obj.iterdir():
                    if item.is_file():
                        file_count += 1
                        total_size += item.stat().st_size
                    elif item.is_dir():
                        dir_count += 1
            except (PermissionError, OSError):
                pass
            
            info.update({
                "file_count": file_count,
                "directory_count": dir_count,
                "total_size": total_size
            })
        
        return {
            "status": "success",
            "info": info
        }
        
    except Exception as e:
        return {"error": f"获取文件信息失败: {str(e)}"}

# -------------------- 启动 --------------------
if __name__ == "__main__":
    mcp.run()