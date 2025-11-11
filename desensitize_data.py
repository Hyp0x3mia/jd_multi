import os
import re
import argparse
from datetime import datetime
import logging
import platform

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def desensitize_content(content, sensitive_fields):
    """对内容中的敏感字段进行脱敏处理"""
    # 处理每个敏感字段
    patterns = []
    for field in sensitive_fields:
        # 构建正则表达式模式，匹配键值对
        # 处理普通引号格式 "key": "value"
        patterns.append(rf'"{field}":\s*"[^"]*"')
        # 处理转义引号格式 \"key\": \"value\"
        patterns.append(rf'\\"{field}\\\":\s*\\\"[^"]*\\\"')

    # 组合所有模式
    pattern = r'(' + r'|'.join(patterns) + r')'

    # 定义替换函数
    def replace_match(match):
        match_str = match.group(0)
        # 保留键和引号结构，只替换值部分
        if '\\"' in match_str:
            return re.sub(r':\s*\\\"[^"]*\\\"', r': \\"\\"', match_str)
        else:
            return re.sub(r':\s*"[^"]*"', r': ""', match_str)

    # 执行替换
    content = re.sub(pattern, replace_match, content)
    return content


def desensitize_file(input_file_path, prefix, sensitive_fields, output_dir):
    """对单个文件进行脱敏处理"""
    file_name = os.path.basename(input_file_path)

    # 检查文件是否以指定前缀开头
    if not file_name.startswith(prefix):
        logger.debug(f"文件 {file_name} 不以指定前缀 {prefix} 开头，跳过处理")
        return False

    try:
        # 尝试多种编码读取文件
        encodings = ['utf-8', 'gbk', 'latin-1', 'utf-16']
        content = None

        for encoding in encodings:
            try:
                with open(input_file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"使用编码 {encoding} 读取文件 {input_file_path} 时出错: {str(e)}")

        if content is None:
            logger.error(f"无法解码文件 {input_file_path}，尝试了多种编码")
            return False

        # 脱敏处理
        desensitized_content = desensitize_content(content, sensitive_fields)

        # 如果内容没有变化，可能是没有找到敏感字段
        if desensitized_content == content:
            # logger.info(f"文件 {file_name} 中未发现需要脱敏的字段")
            1
        else:
            logger.info(f"文件 {file_name} 中发现需要脱敏的字段")
        # 获取当前时间到秒，格式化为字符串
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")

        # 构建输出文件名: 保留原文件名，在前缀后添加时间戳
        base_name = os.path.splitext(file_name)[0]
        ext = os.path.splitext(file_name)[1]

        # 处理前缀后的文件名部分
        if base_name.startswith(prefix):
            suffix = base_name[len(prefix):]
            # 移除可能存在的分隔符
            if suffix.startswith(('_', '-')):
                suffix = suffix[1:]
            new_base_name = f"{prefix}_{current_time}_{suffix}"
        else:
            new_base_name = f"{prefix}_{current_time}_{base_name}"

        output_file_name = f"{new_base_name}{ext}"
        output_file_path = os.path.join(output_dir, output_file_name)

        # 检查输出文件是否已存在
        counter = 1
        while os.path.exists(output_file_path):
            output_file_name = f"{new_base_name}_{counter}{ext}"
            output_file_path = os.path.join(output_dir, output_file_name)
            counter += 1

        # 写入脱敏后的内容，统一使用utf-8编码
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(desensitized_content)

        logger.info(f"脱敏完成，新文件保存至：{output_file_path}")
        return True

    except Exception as e:
        logger.error(f"处理文件 {input_file_path} 时出错：{str(e)}", exc_info=True)
        return False


def process_directory(directory, prefix, sensitive_fields, output_dir, recursive=False):
    """处理目录中的文件，可选择是否递归处理子目录"""
    if not os.path.isdir(directory):
        logger.error(f"目录 {directory} 不存在或不是一个有效的目录")
        return 0

    processed_count = 0

    # 遍历目录下的所有条目
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)

        # 处理系统特定的特殊文件
        if platform.system() == "Windows" and entry in ("System Volume Information",):
            continue
        if platform.system() in ("Linux", "Darwin") and entry.startswith("."):
            continue  # 跳过隐藏文件

        if os.path.isfile(entry_path):
            if desensitize_file(entry_path, prefix, sensitive_fields, output_dir):
                processed_count += 1
        elif os.path.isdir(entry_path) and recursive:
            # 如果是目录且需要递归处理，则递归调用
            logger.info(f"进入子目录：{entry_path}")
            sub_count = process_directory(entry_path, prefix, sensitive_fields, output_dir, recursive)
            processed_count += sub_count

    return processed_count


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='对指定目录下的文件进行脱敏处理')
    parser.add_argument('--directory', required=True, help='文件所在目录')
    parser.add_argument('--prefix', required=True, help='要处理的文件前缀')
    parser.add_argument('--output_dir', help='脱敏后文件的输出目录，默认为输入目录下的desensitized子目录')
    parser.add_argument('--sensitive_fields', default=["api_key"],
                        help='需要脱敏的字段列表，默认为 ["api_key"]')
    parser.add_argument('--recursive', action='store_true', help='是否递归处理子目录', default=False)
    parser.add_argument('--verbose', action='store_true', help='显示详细日志信息', default=True)

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # 确定脱敏后输出目录
    output_dir = args.output_dir if args.output_dir else os.path.join(args.directory, 'local_es_data')

    # 确保输出目录存在
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"输出目录设置为：{output_dir}")
    except Exception as e:
        logger.error(f"无法创建输出目录 {output_dir}：{str(e)}")
        return

    # 处理目录中的文件
    processed_count = process_directory(args.directory, args.prefix, args.sensitive_fields, output_dir, args.recursive)

    if processed_count == 0:
        logger.info("未找到符合条件的文件进行处理")
    else:
        logger.info(f"脱敏处理完成，共处理了 {processed_count} 个文件")


if __name__ == "__main__":
    main()

