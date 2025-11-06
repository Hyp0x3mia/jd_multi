import requests
import pandas as pd
import os

# 下载gsm-8k数据集的parquet文件
def download_gsm8k():
    url = "https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/train-00000-of-00001.parquet"
    filename = "train-00000-of-00001.parquet"
    
    print(f"正在下载 {filename}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"文件下载完成: {filename}")
        return filename
    except Exception as e:
        print(f"下载失败: {e}")
        return None

# 读取parquet文件并提取第30个问题
def extract_30th_question(filename):
    try:
        # 读取parquet文件
        df = pd.read_parquet(filename)
        print(f"数据集包含 {len(df)} 条记录")
        
        # 获取第30个问题（索引为29，因为从0开始）
        if len(df) > 29:
            question_30 = df.iloc[29]
            print("\n第30个问题:")
            print(f"问题: {question_30['question']}")
            print(f"答案: {question_30['answer']}")
            return question_30
        else:
            print("数据集不足30条记录")
            return None
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

if __name__ == "__main__":
    # 下载文件
    filename = download_gsm8k()
    
    if filename:
        # 提取第30个问题
        extract_30th_question(filename)