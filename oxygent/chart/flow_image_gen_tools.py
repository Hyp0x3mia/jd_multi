"""
流程图生成工具模块
"""

from oxygent.oxy import FunctionHub
import os
import json
import requests
import webbrowser
from pathlib import Path
import time
import datetime
from dotenv import load_dotenv

# 初始化 FunctionHub
flow_image_gen_tools = FunctionHub(name="flow_image_gen_tools")

# API 配置 - 使用 OpenAI 兼容接口
API_BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

# 默认提示词模板
DEFAULT_PROMPT_TEMPLATE = """
你是一个专业的流程图设计师，请根据以下解析出的步骤结构生成一个简洁清晰的 Mermaid 流程图代码。

步骤结构：{description}

## 流程图设计规则：
1. **节点类型选择**：
   - 普通步骤使用方框：`A[步骤名称]`
   - 判断条件使用菱形：`B{{是否通过?}}`
   - 开始/结束使用圆角：`Start([开始])` 或 `End([结束])`

2. **连接线标注**：
   - 判断分支要标注条件：`-->|是| 或 -->|否|`
   - 重要流向可以添加说明：`-->|提交审核|`

3. **样式美化**：
   - 开始节点：`style Start fill:#c8e6c9,stroke:#4caf50,stroke-width:2px`
   - 结束节点：`style End fill:#e3f2fd,stroke:#2196f3,stroke-width:2px`
   - 决策节点：`style Decision fill:#fff3e0,stroke:#ff9800,stroke-width:2px`

## 示例参考：
```mermaid
flowchart TD
    Start([开始]) --> A[需求分析]
    A --> B[系统设计]
    B --> C[编码实现]
    C --> D[系统测试]
    D --> E{{测试通过?}}
    E -->|是| F[部署上线]
    E -->|否| C
    F --> End([结束])
    
    style Start fill:#c8e6c9,stroke:#4caf50,stroke-width:2px
    style End fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    style E fill:#fff3e0,stroke:#ff9800,stroke-width:2px
```

要求：
- 只返回 Mermaid 代码，以 ```mermaid 开头，以 ``` 结尾
- 严格按照提供的步骤顺序生成流程
- 保持流程图简洁清晰，不添加额外的详细描述框
- 确保代码语法正确且逻辑清晰
- 不要包含任何解释或说明文字
"""

@flow_image_gen_tools.tool(
    description="根据文本描述生成 Mermaid 流程图并返回 HTML 文件路径。此工具使用 open API 将文本描述转换为 Mermaid 流程图代码，然后生成可视化的 HTML 文件并在浏览器中打开。"
)
async def generate_flow_chart(description: str, output_path: str = None) -> str:
    """
    根据文本描述生成 Mermaid 流程图并在浏览器中打开
    
    Args:
        description: 流程图的文本描述
        output_path: 输出的 HTML 文件路径，默认为 "flowchart.html"
        
    Returns:
        str: 生成的 HTML 文件的路径
    """
    try:
        # 如果没有提供输出路径，则生成带有时间戳的默认文件名
        if output_path is None:
            # 使用项目根目录下的 output 文件夹
            # 从当前工作目录开始查找项目根目录
            current_dir = os.getcwd()
            project_root = current_dir
            
            # 如果当前在 examples/other 目录，需要回到项目根目录
            if current_dir.endswith('examples/other') or current_dir.endswith('examples\\other'):
                project_root = os.path.abspath(os.path.join(current_dir, '../..'))
            
            output_dir = os.path.join(project_root, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成带时间戳的文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flowchart_{timestamp}.html"
            output_path = os.path.join(output_dir, filename)
        else:
            # 确保输出路径是绝对路径
            # 如果用户提供的是相对路径，需要考虑当前工作目录
            if not os.path.isabs(output_path):
                current_dir = os.getcwd()
                
                # 如果当前在 examples/other 目录，将文件保存到项目根目录的 output 文件夹
                if current_dir.endswith('examples/other') or current_dir.endswith('examples\\other'):
                    project_root = os.path.abspath(os.path.join(current_dir, '../..'))
                    output_dir = os.path.join(project_root, "output")
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, os.path.basename(output_path))
                else:
                    output_path = os.path.abspath(output_path)
            else:
                output_path = output_path
            
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 调用 OpenAI 兼容 API 生成 Mermaid 代码
        mermaid_code = call_openai_api(description)
        
        # 确保输出路径是文件而不是目录
        if os.path.isdir(output_path):
            # 如果output_path是目录，添加默认文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flowchart_{timestamp}.html"
            output_path = os.path.join(output_path, filename)
            print(f"输出路径是目录，已调整为: {output_path}")
        
        # 创建 HTML 文件并渲染流程图
        if create_html_with_mermaid(mermaid_code, output_path):
            # 自动在浏览器中打开生成的文件
            import webbrowser
            try:
                webbrowser.open(f"file://{output_path}")
                return f"✅ 流程图已生成并在浏览器中打开: {output_path}"
            except Exception as e:
                print(f"打开浏览器时出错: {e}")
                return f"✅ 流程图已生成并保存到: {output_path}，请手动打开文件"
        else:
            return "❌ 生成流程图时出错"
    except Exception as e:
        print(f"generate_flow_chart 函数执行出错: {e}")
        return f"❌ 生成流程图时出错: {str(e)}"

def call_openai_api(description):
    """调用 OpenAI 兼容 API 生成 Mermaid 代码"""
    # 检查 API 配置
    if not API_KEY  or not API_BASE_URL or not MODEL_NAME:
        print("API 配置不完整，使用示例流程图")
        return generate_sample_mermaid(description)
    
    try:
        prompt = DEFAULT_PROMPT_TEMPLATE.format(description=description)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        # OpenAI API 格式
        data = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000,
            "stream": False
        }
        
        print("正在调用 OpenAI 兼容 API 生成流程图代码...")
        print(f"请求URL: {API_BASE_URL}/chat/completions")
        print(f"请求模型: {MODEL_NAME}")
        
        response = requests.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=data, timeout=30)
        
        print(f"API 响应状态: {response.status_code}")
        
        if response.status_code != 200:
            print(f"API 请求失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            print("使用示例流程图")
            return generate_sample_mermaid(description)
        
        result = response.json()
        
        # OpenAI API 标准格式
        if "choices" in result and len(result.get("choices", [])) > 0:
            content = result["choices"][0]["message"]["content"]
            print(f"API 调用成功，内容长度: {len(content)}")
        else:
            print(f"无法识别的 API 响应格式: {result}")
            print("使用示例流程图")
            return generate_sample_mermaid(description)
            
        # 提取 Mermaid 代码
        mermaid_code = extract_mermaid_code(content)
        if not mermaid_code:
            print("未能从 API 响应中提取有效的 Mermaid 代码，将使用示例流程图")
            return generate_sample_mermaid(description)
        
        print("成功提取 Mermaid 代码")
        return mermaid_code
        
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
        print("使用示例流程图")
        return generate_sample_mermaid(description)
    except Exception as e:
        print(f"调用 API 时出错: {e}")
        print("使用示例流程图")
        return generate_sample_mermaid(description)

def extract_mermaid_code(content):
    """从 API 响应中提取 Mermaid 代码"""
    # 尝试提取 ```mermaid ... ``` 格式的代码块
    if "```mermaid" in content and "```" in content.split("```mermaid", 1)[1]:
        return content.split("```mermaid", 1)[1].split("```", 1)[0].strip()
    # 如果没有明确的标记，尝试提取看起来像 Mermaid 代码的部分
    elif any(keyword in content.lower() for keyword in ["graph ", "flowchart ", "sequencediagram", "classDiagram"]):
        lines = content.split('\n')
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if start_idx is None and any(keyword in line.lower() for keyword in ["graph ", "flowchart ", "sequencediagram", "classDiagram"]):
                start_idx = i
            elif start_idx is not None and line.strip() == "" and i > start_idx + 3:
                end_idx = i
                break
        
        if start_idx is not None:
            end_idx = end_idx or len(lines)
            return '\n'.join(lines[start_idx:end_idx]).strip()
    
    return None

def generate_sample_mermaid(description=""):
    """根据描述生成相应的 Mermaid 流程图代码"""
    
    # 解析描述中的关键词和步骤
    steps = parse_description_to_steps(description)
    
    if not steps:
        # 如果无法解析描述，返回默认的软件开发流程图
        return """flowchart TD
    A[需求分析] --> B[系统设计]
    B --> C[技术选型]
    C --> D[架构设计]
    D --> E[编码实现]
    E --> F[单元测试]
    F --> G[集成测试]
    G --> H{测试通过?}
    H -->|是| I[代码审查]
    H -->|否| E
    I --> J[部署准备]
    J --> K[生产部署]
    K --> L[监控运维]
    L --> M[用户反馈]
    M --> N{需要优化?}
    N -->|是| A
    N -->|否| O[项目完成]
    
    style A fill:#e1f5fe
    style O fill:#c8e6c9
    style H fill:#fff3e0
    style N fill:#fff3e0"""
    
    # 根据解析的步骤生成流程图
    return generate_flowchart_from_steps(steps)

def parse_description_to_steps(description):
    """解析描述文本，提取流程步骤，支持编号识别"""
    if not description:
        return []
    
    import re
    
    # 清理描述文本
    description = description.strip()
    
    steps = []
    
    # 优先方法：识别编号格式的步骤描述
    numbered_patterns = [
        # 数字编号：1. 2. 3. 或 1、2、3、
        r'(?:^|\n)\s*(\d+)[.、]\s*([^\n\d]+?)(?=\s*\d+[.、]|\s*$)',
        # 中文编号：第一步、第二步、第三步
        r'(?:^|\n)\s*第([一二三四五六七八九十]+)步[：:]?\s*([^\n]+?)(?=\s*第[一二三四五六七八九十]+步|\s*$)',
        # 步骤编号：步骤1、步骤2、步骤3
        r'(?:^|\n)\s*步骤(\d+)[：:]?\s*([^\n]+?)(?=\s*步骤\d+|\s*$)',
        # 阶段编号：阶段1、阶段2、阶段3
        r'(?:^|\n)\s*阶段(\d+)[：:]?\s*([^\n]+?)(?=\s*阶段\d+|\s*$)',
    ]
    
    for pattern in numbered_patterns:
        matches = re.findall(pattern, description, re.MULTILINE | re.DOTALL)
        if matches:
            # 提取步骤描述并清理
            for match in matches:
                if len(match) == 2:  # (编号, 描述)
                    step_desc = match[1].strip()
                    # 清理描述中的无用词汇
                    step_desc = re.sub(r'[，。；：]$', '', step_desc)
                    step_desc = re.sub(r'^[：:]', '', step_desc)
                    if step_desc:
                        steps.append(step_desc)
            if steps:
                return clean_and_standardize_steps(steps)
    
    # 预处理：移除常见的无用词汇
    cleanup_patterns = [
        r'请生成.*?流程图[，,]?',
        r'包括',
        r'阶段',
        r'步骤',
        r'流程',
        r'过程'
    ]
    
    cleaned_desc = description
    for pattern in cleanup_patterns:
        cleaned_desc = re.sub(pattern, '', cleaned_desc, flags=re.IGNORECASE)
    
    # 方法1: 按箭头分割（最精确）
    if '→' in cleaned_desc or '->' in cleaned_desc:
        text = cleaned_desc.replace('→', '->').replace(' ', '')
        parts = text.split('->')
        steps = [part.strip() for part in parts if part.strip()]
    
    # 方法2: 按中文标点分割
    elif any(sep in cleaned_desc for sep in ['，', '、', '；', '和']):
        # 使用更精确的分割
        parts = re.split('[，、；和]', cleaned_desc)
        steps = [part.strip() for part in parts if part.strip()]
    
    # 方法3: 按英文标点分割
    elif ',' in cleaned_desc:
        parts = cleaned_desc.split(',')
        steps = [part.strip() for part in parts if part.strip()]
    
    # 方法4: 按顺序关键词分割
    elif any(keyword in cleaned_desc for keyword in ['然后', '接着', '之后', '最后', '其次', '再次']):
        parts = re.split('然后|接着|之后|最后|其次|再次', cleaned_desc)
        steps = [part.strip() for part in parts if part.strip()]
    
    # 方法5: 智能识别常见流程模式
    else:
        # 软件开发流程识别
        if any(keyword in description for keyword in ['软件', '开发', '系统']):
            dev_steps = []
            if '需求' in description: dev_steps.append('需求分析')
            if '设计' in description: dev_steps.append('系统设计')
            if '编码' in description or '开发' in description: dev_steps.append('编码实现')
            if '测试' in description: dev_steps.append('测试验证')
            if '部署' in description: dev_steps.append('系统部署')
            if dev_steps: return dev_steps
        
        # 项目管理流程识别
        elif any(keyword in description for keyword in ['项目', '管理']):
            pm_steps = []
            if '启动' in description: pm_steps.append('项目启动')
            if '规划' in description: pm_steps.append('项目规划')
            if '执行' in description: pm_steps.append('项目执行')
            if '监控' in description: pm_steps.append('监控控制')
            if '收尾' in description or '结束' in description: pm_steps.append('项目收尾')
            if pm_steps: return pm_steps
        
        # 业务流程识别
        elif any(keyword in description for keyword in ['业务', '订单', '申请', '审批']):
            business_steps = []
            if '提交' in description or '申请' in description: business_steps.append('提交申请')
            if '审核' in description or '审批' in description: business_steps.append('审核审批')
            if '处理' in description: business_steps.append('业务处理')
            if '完成' in description or '结束' in description: business_steps.append('流程完成')
            if business_steps: return business_steps
        
        # 通用流程关键词提取
        common_keywords = {
            '开始': '开始',
            '启动': '启动',
            '初始化': '初始化',
            '需求分析': '需求分析',
            '需求': '需求收集',
            '分析': '需求分析',
            '设计': '系统设计',
            '架构': '架构设计',
            '编码': '编码实现',
            '开发': '开发实现',
            '实现': '功能实现',
            '测试': '测试验证',
            '验证': '验证确认',
            '部署': '系统部署',
            '发布': '系统发布',
            '上线': '系统上线',
            '监控': '监控运维',
            '维护': '系统维护',
            '审核': '审核确认',
            '审批': '审批处理',
            '检查': '检查验证',
            '确认': '确认处理',
            '完成': '流程完成',
            '结束': '结束'
        }
        
        # 在描述中查找关键词
        found_steps = []
        for keyword, step_name in common_keywords.items():
            if keyword in description:
                found_steps.append(step_name)
        
        # 去重并保持顺序
        steps = list(dict.fromkeys(found_steps))
        
        # 如果没找到任何关键词，使用默认流程
        if not steps:
            steps = ['开始', '处理', '结束']
    
    return clean_and_standardize_steps(steps)

def clean_and_standardize_steps(steps):
    """清理和标准化步骤名称"""
    import re
    
    # 后处理：清理和标准化步骤名称
    cleaned_steps = []
    for step in steps:
        # 移除多余的词汇和标点
        step = re.sub(r'^(第\d+步|步骤\d+|阶段\d+)[：:]?', '', step)
        step = re.sub(r'[：:，。；]$', '', step)
        step = re.sub(r'^[：:]', '', step)
        step = step.strip()
        
        # 标准化常见步骤名称
        standardizations = {
            '需求': '需求分析',
            '分析': '需求分析',
            '设计': '系统设计',
            '编码': '编码实现',
            '开发': '开发实现',
            '测试': '测试验证',
            '部署': '系统部署',
            '发布': '系统发布'
        }
        
        if step in standardizations:
            step = standardizations[step]
        
        if step and len(step) > 0:
            cleaned_steps.append(step)
    
    return cleaned_steps if cleaned_steps else ['开始', '处理', '结束']

def is_decision_node(step):
    """判断步骤是否为决策节点"""
    return ('?' in step or
            any(word in step for word in ['是否', '检查', '判断', '确认', '审核', '验证', '测试']))

def generate_flowchart_from_steps(steps):
    """根据步骤列表生成 Mermaid 流程图代码"""
    if not steps:
        return ""
    
    # 构建流程图代码
    flowchart_lines = ["flowchart TD"]
    
    # 处理单步骤情况
    if len(steps) == 1:
        step = steps[0]
        if is_decision_node(step):
            flowchart_lines.append(f"    A{{{step}}}")
        else:
            flowchart_lines.append(f"    A[{step}]")
        flowchart_lines.append("    style A fill:#e1f5fe")
        return "\n".join(flowchart_lines)
    
    # 多步骤处理
    first_step = steps[0]
    last_step = steps[-1]
    
    has_start = any(word in first_step.lower() for word in ['开始', '启动', 'start', '初始'])
    has_end = any(word in last_step.lower() for word in ['结束', '完成', '结果', 'end', '完毕'])
    
    # 生成连接
    connections = []
    
    # 生成节点连接
    for i in range(len(steps)):
        current_id = chr(65 + i)
        step = steps[i]
        
        # 第一个步骤：从开始节点连接（如果需要）
        if i == 0 and not has_start:
            if is_decision_node(step):
                connections.append(f"    Start([开始]) --> {current_id}{{{step}}}")
            else:
                connections.append(f"    Start([开始]) --> {current_id}[{step}]")
        
        # 中间步骤连接
        if i < len(steps) - 1:
            next_id = chr(65 + i + 1)
            next_step = steps[i + 1]
            
            # 确定当前节点和下一个节点的类型
            current_node = f"{current_id}{{{step}}}" if is_decision_node(step) else f"{current_id}[{step}]"
            next_node = f"{next_id}{{{next_step}}}" if is_decision_node(next_step) else f"{next_id}[{next_step}]"
            
            # 如果当前是决策节点，可能需要添加条件标签
            if is_decision_node(step):
                if any(word in next_step for word in ['通过', '成功', '是', '正确', '批准']):
                    connections.append(f"    {current_node} -->|是| {next_node}")
                elif any(word in next_step for word in ['失败', '否', '错误', '拒绝', '重试']):
                    connections.append(f"    {current_node} -->|否| {next_node}")
                else:
                    connections.append(f"    {current_node} --> {next_node}")
            else:
                connections.append(f"    {current_node} --> {next_node}")
        
        # 最后一个步骤：连接到结束节点（如果需要）
        if i == len(steps) - 1 and not has_end:
            current_node = f"{current_id}{{{step}}}" if is_decision_node(step) else f"{current_id}[{step}]"
            connections.append(f"    {current_node} --> End([完成])")
    
    # 组合所有内容
    flowchart_lines.extend(connections)
    
    # 添加样式
    style_lines = [""]
    
    # 开始节点样式
    if not has_start:
        style_lines.append("    style Start fill:#c8e6c9")
    else:
        style_lines.append("    style A fill:#c8e6c9")
    
    # 结束节点样式
    if not has_end:
        style_lines.append("    style End fill:#e3f2fd")
    else:
        last_id = chr(65 + len(steps) - 1)
        style_lines.append(f"    style {last_id} fill:#e3f2fd")
    
    # 决策节点样式
    for i, step in enumerate(steps):
        if ('?' in step or
            any(word in step for word in ['是否', '检查', '判断', '确认', '审核', '验证', '测试'])):
            node_id = chr(65 + i)
            style_lines.append(f"    style {node_id} fill:#fff3e0")
    
    flowchart_lines.extend(style_lines)
    return "\n".join(flowchart_lines)

def create_html_with_mermaid(mermaid_code, output_path):
    """创建包含可交互编辑的 Mermaid 流程图的 HTML 文件"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>交互式 Mermaid 流程图编辑器</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/mode/javascript/javascript.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/theme/dracula.min.css">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1, h2 {{
                color: #333;
                text-align: center;
            }}
            .editor-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin: 20px 0;
            }}
            .editor-panel {{
                flex: 1;
                min-width: 300px;
            }}
            .preview-panel {{
                flex: 1;
                min-width: 300px;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                background-color: white;
            }}
            .CodeMirror {{
                height: 400px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .button-group {{
                margin: 20px 0;
                text-align: center;
            }}
            button {{
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                margin: 0 5px;
                font-size: 14px;
            }}
            button:hover {{
                background-color: #45a049;
            }}
            .template-selector {{
                margin: 20px 0;
                text-align: center;
            }}
            select {{
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #ddd;
                font-size: 14px;
                min-width: 200px;
            }}
            .mermaid {{
                display: flex;
                justify-content: center;
                margin: 20px 0;
                min-height: 200px;
                border: 1px solid #eee;
                padding: 10px;
                border-radius: 4px;
                background-color: #fafafa;
            }}
            .mermaid svg {{
                max-width: 100%;
                height: auto !important;
            }}
            .error-message {{
                color: #d32f2f;
                background-color: #ffebee;
                padding: 10px;
                border-radius: 4px;
                margin-top: 10px;
                border-left: 4px solid #d32f2f;
                font-family: monospace;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                color: #666;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>交互式 Mermaid 流程图编辑器</h1>
            
            <div class="template-selector">
                <label for="template-select">选择模板：</label>
                <select id="template-select" onchange="loadTemplate()">
                    <option value="custom">自定义</option>
                    <option value="software-dev">软件开发流程</option>
                    <option value="project-management">项目管理流程</option>
                    <option value="business-process">业务流程</option>
                    <option value="decision-tree">决策树</option>
                </select>
            </div>
            
            <div class="editor-container">
                <div class="editor-panel">
                    <h2>Mermaid 代码编辑器</h2>
                    <textarea id="code-editor">{mermaid_code}</textarea>
                </div>
                
                <div class="preview-panel">
                    <h2>实时预览</h2>
                    <div id="preview" class="mermaid">
{mermaid_code}
                    </div>
                    <div id="render-error" class="error-message" style="display:none;"></div>
                </div>
            </div>
            
            <div class="button-group">
                <button onclick="updatePreview()">更新预览</button>
                <button onclick="exportSVG()">导出 SVG</button>
                <button onclick="exportPNG()">导出 PNG</button>
            </div>
            
            <div class="footer">
                <p>由 OxyGent 和 Mermaid.js 提供支持 | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
        
        <script>
            // 初始化 Mermaid
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'default',
                securityLevel: 'loose',
                flowchart: {{
                    htmlLabels: true,
                    curve: 'basis'
                }},
                er: {{
                    useMaxWidth: false
                }},
                gantt: {{
                    useMaxWidth: false
                }}
            }});
            
            // 初始化代码编辑器
            var editor = CodeMirror.fromTextArea(document.getElementById("code-editor"), {{
                mode: "javascript",
                theme: "dracula",
                lineNumbers: true,
                autoCloseBrackets: true,
                matchBrackets: true,
                tabSize: 2,
                indentWithTabs: true
            }});
            
            // 模板库
            const templates = {{
                "software-dev": `flowchart TD
    A[需求分析] --> B[系统设计]
    B --> C[编码实现]
    C --> D[测试]
    D --> E{"测试通过?"}
    E -->|是| F[部署]
    E -->|否| C
    F --> G[维护]`,
                "project-management": `flowchart TD
    A[项目启动] --> B[需求收集]
    B --> C[项目规划]
    C --> D[执行]
    D --> E[监控与控制]
    E --> F{"需要变更?"}
    F -->|是| G[变更控制]
    G --> D
    F -->|否| H[项目收尾]`,
                "business-process": `flowchart TD
    A[开始] --> B[接收订单]
    B --> C[库存检查]
    C --> D{"库存充足?"}
    D -->|是| E[处理付款]
    D -->|否| F[通知缺货]
    F --> G[补充库存]
    G --> E
    E --> H[发货]
    H --> I[结束]`,
                "decision-tree": `flowchart TD
    A[问题] --> B{"条件1?"}
    B -->|是| C[结果1]
    B -->|否| D{"条件2?"}
    D -->|是| E[结果2]
    D -->|否| F[结果3]`
            }};
            
            // 加载模板
            function loadTemplate() {{
                const select = document.getElementById("template-select");
                const templateName = select.value;
                
                if (templateName !== "custom") {{
                    editor.setValue(templates[templateName]);
                    updatePreview();
                }}
            }}
            
            // 更新预览
            function updatePreview() {{
                const code = editor.getValue();
                const previewDiv = document.getElementById("preview");
                const errorDiv = document.getElementById("render-error");
                
                try {{
                    // 完全重新创建预览区域，避免渲染问题
                    const previewContainer = previewDiv.parentNode;
                    const oldPreview = previewDiv;
                    
                    // 创建新的预览div
                    const newPreview = document.createElement("div");
                    newPreview.id = "preview";
                    newPreview.className = "mermaid";
                    newPreview.textContent = code;
                    
                    // 替换旧的预览div
                    previewContainer.replaceChild(newPreview, oldPreview);
                    
                    // 重新初始化Mermaid
                    mermaid.initialize({{
                        startOnLoad: false,  // 不自动启动
                        theme: 'default',
                        securityLevel: 'loose',
                        flowchart: {{
                            htmlLabels: true,
                            curve: 'basis'
                        }}
                    }});
                    
                    // 手动渲染
                    setTimeout(() => {{
                        try {{
                            mermaid.init(undefined, "#preview");
                            // 隐藏错误信息
                            errorDiv.style.display = "none";
                        }} catch (innerError) {{
                            console.error("Mermaid渲染错误:", innerError);
                            errorDiv.textContent = "图表渲染错误: " + innerError.message;
                            errorDiv.style.display = "block";
                        }}
                    }}, 300);
                }} catch (e) {{
                    console.error("Mermaid渲染错误:", e);
                    errorDiv.textContent = "图表渲染错误: " + e.message;
                    errorDiv.style.display = "block";
                }}
            }}
            
            // 保存更改
            function saveChanges() {{
                const code = editor.getValue();
                
                // 发送到服务器保存
                fetch('/api/save-flowchart', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ mermaid_code: code }}),
                }})
                .then(response => response.json())
                .then(data => {{
                    if (data.success) {{
                        alert('流程图已保存！文件路径: ' + data.file_path);
                    }} else {{
                        alert('保存失败: ' + data.error);
                    }}
                }})
                .catch(error => {{
                    console.error('保存出错:', error);
                    alert('保存出错，请查看控制台获取详细信息');
                }});
            }}
            
            // 导出 SVG
            function exportSVG() {{
                const svgCode = document.querySelector(".mermaid svg");
                if (svgCode) {{
                    const svgData = new XMLSerializer().serializeToString(svgCode);
                    const svgBlob = new Blob([svgData], {{type: 'image/svg+xml;charset=utf-8'}});
                    const svgUrl = URL.createObjectURL(svgBlob);
                    
                    const downloadLink = document.createElement("a");
                    downloadLink.href = svgUrl;
                    downloadLink.download = "flowchart.svg";
                    document.body.appendChild(downloadLink);
                    downloadLink.click();
                    document.body.removeChild(downloadLink);
                }} else {{
                    alert("无法导出 SVG，请先更新预览");
                }}
            }}
            
            // 导出 PNG
            function exportPNG() {{
                const svgElement = document.querySelector(".mermaid svg");
                if (svgElement) {{
                    const canvas = document.createElement("canvas");
                    const ctx = canvas.getContext("2d");
                    
                    // 创建图像
                    const svgData = new XMLSerializer().serializeToString(svgElement);
                    const img = new Image();
                    
                    img.onload = function() {{
                        canvas.width = img.width;
                        canvas.height = img.height;
                        ctx.drawImage(img, 0, 0);
                        
                        const pngUrl = canvas.toDataURL("image/png");
                        
                        const downloadLink = document.createElement("a");
                        downloadLink.href = pngUrl;
                        downloadLink.download = "flowchart.png";
                        document.body.appendChild(downloadLink);
                        downloadLink.click();
                        document.body.removeChild(downloadLink);
                    }};
                    
                    img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
                }} else {{
                    alert("无法导出 PNG，请先更新预览");
                }}
            }}
            
            // 页面加载完成后初始化预览
            document.addEventListener('DOMContentLoaded', function() {{
                // 延迟一下以确保所有组件已加载
                setTimeout(function() {{
                    // 确保编辑器内容已加载
                    if (editor.getValue().trim() === '') {{
                        // 如果编辑器为空，尝试从预览区域获取内容
                        const previewContent = document.getElementById("preview").textContent.trim();
                        if (previewContent) {{
                            editor.setValue(previewContent);
                        }}
                    }}
                    
                    // 确保mermaid已完全加载
                    if (typeof mermaid !== 'undefined') {{
                        // 配置mermaid
                        mermaid.initialize({{
                            startOnLoad: false,
                            theme: 'default',
                            securityLevel: 'loose',
                            flowchart: {{
                                htmlLabels: true,
                                curve: 'basis'
                            }}
                        }});
                        
                        // 完全重新创建预览区域，避免渲染问题
                        const previewDiv = document.getElementById("preview");
                        const previewContainer = previewDiv.parentNode;
                        const code = editor.getValue();
                        
                        // 创建新的预览div
                        const newPreview = document.createElement("div");
                        newPreview.id = "preview";
                        newPreview.className = "mermaid";
                        newPreview.textContent = code;
                        
                        // 替换旧的预览div
                        previewContainer.replaceChild(newPreview, previewDiv);
                        
                        // 手动渲染
                        mermaid.init(undefined, "#preview");
                    }} else {{
                        console.error("Mermaid库未加载");
                        document.getElementById("render-error").textContent = "图表库未加载，请刷新页面";
                        document.getElementById("render-error").style.display = "block";
                    }}
                }}, 500);  // 增加延迟时间，确保所有资源加载完成
            }});
        </script>
    </body>
    </html>
    """
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"已保存流程图到: {output_path}")
        return True
    except Exception as e:
        print(f"保存 HTML 文件时出错: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(main())