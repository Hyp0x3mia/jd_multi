import requests
import hashlib
import os

def download_file(url, filename):
    """下载文件"""
    print(f"正在下载 {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"文件已下载到: {filename}")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def calculate_sha256(filename):
    """计算文件的SHA256值"""
    print(f"正在计算 {filename} 的SHA256值...")
    sha256_hash = hashlib.sha256()
    
    try:
        with open(filename, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        sha256_value = sha256_hash.hexdigest()
        print(f"SHA256值: {sha256_value}")
        return sha256_value
    except Exception as e:
        print(f"计算SHA256失败: {e}")
        return None

def main():
    # hf-mirror的URL
    base_url = "https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/"
    filename = "train-00000-of-00001.parquet"
    url = base_url + filename
    
    print("开始下载gsm-8k数据集...")
    
    # 下载文件
    if download_file(url, filename):
        # 获取文件信息
        file_size = os.path.getsize(filename)
        print(f"文件大小: {file_size} 字节")
        
        # 计算SHA256
        sha256_value = calculate_sha256(filename)
        
        if sha256_value:
            print("\n下载和SHA256计算完成!")
            print(f"文件名: {filename}")
            print(f"文件大小: {file_size} 字节")
            print(f"SHA256: {sha256_value}")
    else:
        print("下载失败，请检查网络连接或URL是否正确。")

if __name__ == "__main__":
    main()