import json
import time

# --- 1. 配置您的文件路径 ---

# 包含 task_id 的 txt 文件
file_a_path = '重跑题目.txt'

# 原始的 jsonl 数据文件
file_b_path = 'test/data.jsonl'

# 筛选后输出的 jsonl 文件
file_c_path = 'test/rerun.jsonl'

# jsonl 文件中存储 task_id 的字段名
task_id_field_name = 'task_id'

# --- 1b. 新增配置：排除规则 ---
# 您不希望加入的字段名
level_field_name = 'level' 
# 您不希望加入的字段值 (注意：这里是整数 3，不是字符串 "3")
level_value_to_exclude = "3"

# --- 2. 脚本执行 ---

def filter_jsonl_by_task_ids():
    start_time = time.time()
    
    # 步骤 1: 读取 A.txt 中的所有 task_id 到一个 set 中
    print(f"正在从 {file_a_path} 中加载 task_id...")
    task_id_set = set()
    try:
        with open(file_a_path, 'r', encoding='utf-8') as f_a:
            for line in f_a:
                task_id_set.add(line.strip())
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_a_path}")
        return
    
    if not task_id_set:
        print(f"警告：{file_a_path} 中没有找到任何 task_id。")
        return
        
    print(f"成功加载 {len(task_id_set)} 个唯一的 task_id。")

    # 步骤 2: 逐行读取 B.jsonl，筛选并写入 C.jsonl
    print(f"正在处理 {file_b_path} 并写入 {file_c_path}...")
    match_count = 0
    total_lines = 0
    excluded_by_level = 0
    
    try:
        with open(file_b_path, 'r', encoding='utf-8') as f_b, \
             open(file_c_path, 'w', encoding='utf-8') as f_c:
            
            for line in f_b:
                total_lines += 1
                try:
                    # 解析 JSON 行
                    sample = json.loads(line)
                    
                    # 获取 task_id
                    task_id = sample.get(task_id_field_name)
                    
                    if task_id is None:
                        if total_lines == 1:
                            print(f"警告：文件B中某些行可能缺少 '{task_id_field_name}' 字段。")
                        continue

                    # --- 核心筛选逻辑 (已更新) ---

                    # 条件1：检查 task_id 是否在集合中
                    task_id_matches = str(task_id) in task_id_set
                    
                    if task_id_matches:
                        # 条件2：如果 task_id 匹配，再检查 level 字段
                        level_value = sample.get(level_field_name)
                        
                        # .get() 在字段不存在时返回 None, None != 3 为 True
                        # 所以字段不存在的样本也会被保留
                        level_is_valid = (level_value != level_value_to_exclude)

                        if level_is_valid:
                            # 两个条件都满足，写入文件 C
                            f_c.write(line)
                            match_count += 1
                        else:
                            # task_id 匹配，但 level 为 3，被排除
                            excluded_by_level += 1
                    
                except json.JSONDecodeError:
                    print(f"警告：第 {total_lines} 行不是有效的JSON。已跳过。")
    
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_b_path}")
        return

    end_time = time.time()
    
    print("\n--- 处理完成 ---")
    print(f"总共处理 {file_b_path} 中的 {total_lines} 行。")
    print(f"找到 {match_count + excluded_by_level} 条匹配的 task_id。")
    print(f"  - 因 '{level_field_name}' 字段为 {level_value_to_exclude} 而排除了 {excluded_by_level} 条。")
    print(f"  - 最终提取了 {match_count} 条样本到 {file_c_path}。")
    print(f"总耗时: {end_time - start_time:.2f} 秒。")

# --- 运行脚本 ---
if __name__ == "__main__":
    filter_jsonl_by_task_ids()