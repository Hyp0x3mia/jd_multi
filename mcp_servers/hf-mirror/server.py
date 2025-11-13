from huggingface_hub import snapshot_download
import pandas as pd
import asyncio
from mcp.server.fastmcp import FastMCP
import hashlib
import os

mcp = FastMCP()

@mcp.tool(description="Download huggingface file.Requires the **repo (Repository ID, like openai/gsm8k)**")  
async def hf_download(repo='openai/gsm8k'):
    """
    从 Hugging Face Hub 下载数据集的配置快照。
    
    参数:
        repo (str): 仓库ID，如 'openai/gsm8k'。
        repotype (str): 仓库类型，'dataset' 或 'model'。

    返回:
        dict: 包含 '状态' 和 '路径' 或 '错误原因' 的字典。
    """
    
    # 设定下载目录
    repo_path = repo.replace('/','_')
    local_dir = "./huggingface_download/"+repo_path


        
    try:
        # 1. 执行下载操作
        downloaded_path = snapshot_download(
            repo_id=repo,
            repo_type='dataset',
            local_dir=local_dir,
            max_workers=8,
            cache_dir=None,
            ignore_patterns=[
                "*.bin", 
                "*.safetensors",
                "*.h5",
                "*.msgpack",
                "*.pt",
                "*.ckpt",
                ".gitattributes"
            ]
        )
        
        # 2. 下载成功，返回成功字典
        return {
            "状态": "成功",
            "路径": downloaded_path  # snapshot_download 成功后返回下载的本地路径
        }

    except Exception as e:
        # 5. 处理所有其他未预期的错误
        return {
            "状态": "失败",
            "错误原因": f"发生未知错误: {e}"
        }

@mcp.tool(description="reads data from a Parquet file.Requires the **row number (1-based, not the index—the code will subtract 1)** and **repo(Repository ID, like openai/gsm8k)** and **parquet_path(parquet specific path, like main/train-00000-of-00001.parquet)**")  
async def parquet_read(number: int, repo='openai/gsm8k',  parquet_path = 'main/train-00000-of-00001.parquet') -> dict:
    """
    读取 Parquet 文件，并根据指定的行号（基于1）提取 'question' 和 'answer' 数据。
    返回字典包含处理状态和提取的数据。

    Args:
        number (int): 基于1的行索引（例如：1代表第一行）。
        file_path (str): Parquet 文件的路径。

    Returns:
        dict: 包含 'state', 'question', 'answer' 或 'error' 信息的字典。
    """
    repo_path = repo.replace('/','_')

    file_path = "./huggingface_download/"+repo_path +'/' +  parquet_path

    if number <= 0:
        return {
            "state": "failure",
            "error": f"行号必须是正整数，但接收到: {number}"
        }

    try:
        # 1. 文件读取
        df = pd.read_parquet(file_path)
        # print(f"数据集形状: {df.shape}")
        # print(f"列名: {list(df.columns)}")
        
        # 2. 索引越界检查 (基于0索引)
        idx = number - 1
        if idx >= len(df):
            return {
                "state": "failure",
                "error": f"请求的行号({number})超出数据集范围。数据集总行数: {len(df)}"
            }

        # 3. 提取指定的行数据 (Pandas Series)
        row_data = df.iloc[idx]
        
        # 4. 检查 'question' 键是否存在
        if 'question' not in row_data:
             # 如果 'question' 都不存在，则无法提供有意义的输出
            return {
                "state": "failure",
                "error": "指定的行中缺少必需的 'question' 列。"
            }
        
        # 5. 获取 'question' 的值
        question_value = row_data['question']
        print(question_value)
        # 6. 检查 'answer' 键是否存在
        if 'answer' in row_data:
            # 方案 A: question 和 answer 都存在
            return {
                "state": "success",
                "question": question_value,
                "answer": row_data['answer']
            }
        else:
            # 方案 B: 仅 question 存在
            # 尽管您要求 question['question'] 和 question['answer'] 都不存在时返回 question，
            # 但根据逻辑，如果 question 存在而 answer 不存在，这是最贴切的返回结构。
            # 如果 answer 列缺失，我们设置 answer 为 None
            return {
                "state": "success",
                "question": question_value,
                "answer": None
            }

    except FileNotFoundError:
        return {
            "state": "failure",
            "error": f"文件未找到: {file_path}"
        }
    except Exception as e:
        # 捕获其他读取或解析错误（如 Parquet 格式错误）
        return {
            "state": "failure",
            "error": f"读取文件时发生未知错误: {e}"
        }



@mcp.tool(description="Calculate the SHA256 hash digest of a specified file.Requires the **file_path**")
async def calculate_sha256(repo='openai/gsm8k',  parquet_path = 'main/train-00000-of-00001.parquet') -> dict:

    """
    计算指定文件的 SHA256 哈希值。

    Args:
        file_path (str): 文件的路径。

    Returns:
        Dict[str, Any]: 包含 'state' 和 'hash' 或 'error' 信息的字典。
    """
    repo_path = repo.replace('/','_')

    file_path = "./huggingface_download/"+repo_path +'/' +  parquet_path
    # 1. 初始化 SHA256 算法对象
    sha256_hash = hashlib.sha256()

    try:
        # 2. 检查文件是否存在
        if not os.path.exists(file_path):
            return {
                "state": "failure",
                "error": f"文件未找到: {file_path}"
            }
        
        # 3. 以二进制只读模式打开文件并分块读取
        # 使用 'rb' 模式确保文件内容按字节读取
        with open(file_path, "rb") as f:
            # 持续读取文件直到结束
            while True:
                data = f.read(65536)
                if not data:
                    break
                # 更新哈希对象
                sha256_hash.update(data)
        
        # 4. 获取最终的十六进制哈希值
        final_hash = sha256_hash.hexdigest()
        
        # 5. 成功返回
        return {
            "state": "success",
            "hash": final_hash
        }

    except IOError as e:
        # 捕获文件读取权限或其他 I/O 错误
        return {
            "state": "failure",
            "error": f"文件读取 I/O 错误: {e}"
        }
    except Exception as e:
        # 捕获其他未知错误
        return {
            "state": "failure",
            "error": f"计算哈希时发生未知错误: {e}"
        }



if __name__ == "__main__":
    mcp.run()

    # import os
    # import asyncio

    # # # 1. 下载测试
    # # print(">>> 开始下载 gsm8k...")
    # loop = asyncio.get_event_loop()
    # down_res = loop.run_until_complete(hf_download("openai/gsm8k"))
    # print("下载结果：", down_res)
    # if down_res["状态"] != "成功":
    #     print("下载失败，测试终止。")
    #     exit(1)

    # # 2. 拼装默认路径（与你工具里的默认逻辑保持一致）

    # # 3. 读取第 1 行
    # print("\n>>> 读取第 1 条样本...")
    # row1 = loop.run_until_complete(parquet_read(1, 'openai/gsm8k', 'main/train-00000-of-00001.parquet'))
    # print(row1)
    # print("\n>>> 读取第 10 条样本...")

    # row1000 = loop.run_until_complete(parquet_read(10, 'openai/gsm8k', 'main/train-00000-of-00001.parquet'))
    # print(row1000)
    # # 4. 读取第 1000 行（数据集共 7473 行，1000 合法）
    # print("\n>>> 读取第 20 条样本...")
    # row1000 = loop.run_until_complete(parquet_read(20, 'openai/gsm8k', 'main/train-00000-of-00001.parquet'))
    # print(row1000)
    # print("\n>>> 读取第 30 条样本...")

    # row1000 = loop.run_until_complete(parquet_read(30, 'openai/gsm8k', 'main/train-00000-of-00001.parquet'))
    # print(row1000)

    # row1000 = loop.run_until_complete(calculate_sha256( 'openai/gsm8k', 'main/train-00000-of-00001.parquet'))

    # print(row1000)

    # print("\n>>> 功能测试全部完成！")