import os
import requests
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
import pandas as pd

def download_gsm8k_dataset():
    """从hf-mirror下载GSM-8K数据集"""
    print("正在下载GSM-8K数据集...")
    try:
        # 使用hf-mirror镜像下载
        file_path = hf_hub_download(
            repo_id="openai/gsm8k",
            filename="main/train-00000-of-00001.parquet",
            repo_type="dataset",
            endpoint="https://hf-mirror.com"
        )
        print(f"GSM-8K数据集下载完成: {file_path}")
        return file_path
    except Exception as e:
        print(f"下载GSM-8K数据集失败: {e}")
        return None

def download_timesfm_model():
    """从hf-mirror下载TimesFM模型"""
    print("正在下载TimesFM模型...")
    try:
        # 使用hf-mirror镜像下载模型配置
        config_path = hf_hub_download(
            repo_id="google/timesfm-2.0-500m-pytorch",
            filename="config.json",
            endpoint="https://hf-mirror.com"
        )
        print(f"TimesFM模型配置下载完成: {config_path}")
        return config_path
    except Exception as e:
        print(f"下载TimesFM模型失败: {e}")
        return None

def load_model_config(config_path):
    """加载模型配置并获取注意力头数量和隐藏层数量"""
    print("正在加载模型配置...")
    try:
        config = AutoConfig.from_pretrained(os.path.dirname(config_path))
        
        # 获取注意力头数量
        num_attention_heads = getattr(config, 'num_attention_heads', 'N/A')
        
        # 获取隐藏层数量
        num_hidden_layers = getattr(config, 'num_hidden_layers', 'N/A')
        if num_hidden_layers == 'N/A':
            num_hidden_layers = getattr(config, 'num_layers', 'N/A')
        
        print(f"模型配置加载成功!")
        print(f"注意力头数量: {num_attention_heads}")
        print(f"隐藏层数量: {num_hidden_layers}")
        
        return {
            'num_attention_heads': num_attention_heads,
            'num_hidden_layers': num_hidden_layers
        }
    except Exception as e:
        print(f"加载模型配置失败: {e}")
        return None

def read_parquet_data(file_path):
    """读取parquet文件数据"""
    print("正在读取GSM-8K数据...")
    try:
        df = pd.read_parquet(file_path)
        print(f"数据读取成功，共 {len(df)} 条记录")
        print(f"列名: {list(df.columns)}")
        if len(df) > 0:
            print("第一条数据示例:")
            print(df.iloc[0])
        return df
    except Exception as e:
        print(f"读取数据失败: {e}")
        return None

def main():
    print("开始执行下载和加载任务...")
    
    # 下载GSM-8K数据集
    gsm8k_path = download_gsm8k_dataset()
    
    # 下载TimesFM模型
    timesfm_config_path = download_timesfm_model()
    
    # 加载模型配置
    if timesfm_config_path:
        model_config = load_model_config(timesfm_config_path)
    
    # 读取GSM-8K数据
    if gsm8k_path:
        gsm8k_data = read_parquet_data(gsm8k_path)
    
    print("任务完成!")

if __name__ == "__main__":
    main()