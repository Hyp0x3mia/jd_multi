# 基于OxyGent和JoyCode实现智能流程图生成工具案例实践

## 概述

`create_flow_image_demo.py` 是一个基于 OxyGent 框架的多代理系统（Multi-Agent System, MAS）示例，展示了如何利用多代理协作实现智能流程图自动生成任务。该示例主要演示了通过 OpenAI 兼容 API 生成 Mermaid 流程图代码，并将其渲染为可视化 HTML 文件的功能，同时展示了多代理系统中的任务分配、协作和执行流程。

<html>
    <h2 align="center">
      <img src="https://storage.jd.com/opponent/AI/5.png" width="50%"/>
      <img src="https://storage.jd.com/opponent/AI/6.png" width="50%"/>
    </h2>
</html>

### 核心特点

- **多代理协作**：主控代理（Master Agent）智能协调流程图生成和打开操作
- **Mermaid 流程图**：使用 Mermaid 语法生成专业的流程图
- **OpenAI 兼容 API**：调用 OpenAI 兼容接口将文本描述转换为 Mermaid 代码
- **Web 服务接口**：基于 FastAPI 提供用户交互界面
- **模块化设计**：将流程图生成、打开功能和静态文件管理分离为独立模块
- **智能决策**：主控代理能够根据用户输入智能选择合适的工具和策略
- **实时预览功能**：支持编辑 Mermaid 代码并实时预览流程图
- **导出功能**：支持将流程图导出为 SVG 和 PNG 格式
- **错误降级机制**：API 调用失败时自动使用高质量备用流程图

### 技术栈

- **OxyGent**：多代理系统框架
- **OpenAI 兼容 API**：大语言模型接口
- **Mermaid**：流程图生成库
- **FastAPI**：Web 服务框架
- **FunctionHub**：工具函数组织框架

## 环境准备

### 系统要求

- Python 3.10+
- OpenAI 兼容的 LLM API 服务

### 依赖安装

1. 安装 Python 依赖：

```bash
# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 环境变量配置

在项目根目录创建 `.env` 文件，配置以下环境变量：

```
OPENAI_BASE_URL=your_openai_base_url
OPENAI_MODEL_NAME=your_model_name
OPENAI_API_KEY=your_api_key
```

## 快速启动

### 运行示例

1. 确保已激活虚拟环境：

```bash
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows
```

2. 确保 API 服务可访问：

```bash
# 测试 API 连接
curl -X POST "your_openai_base_url/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{"model": "your_model_name", "messages": [{"role": "user", "content": "Hello"}]}'
```

3. 运行示例程序：

```bash
python examples/other/create_flow_image_demo.py
```

4. 程序启动后，会自动打开浏览器访问 Web 界面：

```
http://127.0.0.1:8080/web/index.html
```
<html>
    <h2 align="center">
      <img src="https://storage.jd.com/opponent/AI/5.png" width="50%"/>
      <img src="https://storage.jd.com/opponent/AI/6.png" width="50%"/>
    </h2>
</html>

### 使用方法

1. **默认查询**：程序启动后会自动执行"请生成一个软件开发流程图，包括需求分析、设计、编码、测试和部署阶段"
2. **自定义查询**：在 Web 界面输入框中输入自定义流程图描述，例如：
   - "生成一个项目管理流程图，包括启动、规划、执行、监控和收尾阶段"
   - "创建一个用户注册流程图"
   - "画一个订单处理的流程图"
3. **查看结果**：
   - 流程图会自动在浏览器中打开
   - HTML 文件会保存在项目目录中，文件名格式为 `software-development-workflow.html`
4. **编辑和预览**：
   - 在打开的流程图页面中，可以直接编辑 Mermaid 代码
   - 点击"更新预览"按钮可以实时预览修改后的流程图
   - 支持多种预设模板选择
5. **导出图表**：
   - 点击"导出 SVG"按钮可以将流程图导出为 SVG 格式
   - 点击"导出 PNG"按钮可以将流程图导出为 PNG 格式

### 停止程序

在终端中按 `Ctrl+C` 停止程序，或使用以下命令终止进程：

```bash
pkill -f create_flow_image_demo.py
```

## 关键功能实现

### 多代理系统架构

系统由一个主控代理和两个子代理组成：

1. **Master Agent**：负责智能任务协调和分发，具备完整的决策能力
2. **Image Gen Agent**：专门负责流程图生成任务
3. **Open Chart Agent**：专门负责在浏览器中打开流程图文件
4. **直接工具访问**：主控代理也可以直接调用工具函数

### 主控代理智能决策系统

主控代理使用详细的提示模板来实现智能决策：

```python
MASTER_AGENT_PROMPT = """
你是一个智能流程图助手，专门负责分析用户需求并智能选择合适的代理来完成任务。

## 核心职责
1. 智能分析用户输入的意图和需求
2. 根据关键词和语义判断应该使用哪个代理
3. 合理调度子代理或直接使用工具
4. 提供清晰的任务执行反馈

## 可用代理和工具

### image_gen_agent (流程图生成代理)
- **工具**: generate_flow_chart
- **功能**: 根据文本描述生成 Mermaid 流程图并自动在浏览器中打开
- **参数**:
  - description (必需): 流程图的详细文本描述
  - output_path (可选): 输出HTML文件路径
- **触发关键词**: 生成、创建、制作、画、设计、新建、开发流程图等

### open_chart_agent (图表打开代理)
- **工具**: open_html_chart  
- **功能**: 在浏览器中打开已存在的流程图HTML文件
- **参数**:
  - file_path (必需): HTML文件的完整路径
- **触发关键词**: 打开、查看、显示、浏览已有/现有的流程图等

## 智能决策规则

### 规则1: 生成新流程图
**触发条件**: 用户提到"生成"、"创建"、"制作"、"画"、"设计"、"新的"等词汇，并描述了流程内容
**执行策略**: 
- 优先使用 image_gen_agent 或直接调用 generate_flow_chart
- 提取用户描述的流程内容作为 description 参数
- 自动生成合适的文件名作为 output_path

### 规则2: 打开现有流程图
**触发条件**: 用户提到"打开"、"查看"、"显示"现有的/已有的流程图文件
**执行策略**:
- 使用 open_chart_agent 或直接调用 open_html_chart
- 需要用户提供文件路径，如果没有提供则询问

### 规则3: 编辑操作指导
**触发条件**: 用户询问如何编辑、修改、保存、导出流程图
**执行策略**:
- 不需要调用代理，直接提供操作指导
- 确保流程图已在浏览器中打开
"""
```

### 系统配置和初始化

```python
# 配置管理
Config.set_agent_llm_model("default_llm")

# OxyGent 空间配置
oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model_name=os.getenv("OPENAI_MODEL_NAME"),
    ),
    flow_image_gen_tools,
    open_chart_tools,
    oxy.ReActAgent(
        name="image_gen_agent",
        tools=["flow_image_gen_tools"],
        desc="流程图生成代理,"
    ),
    oxy.ReActAgent(
        name="open_chart_agent",
        tools=["open_chart_tools"],
        desc="在浏览器中打开流程图代理"
    ),
    oxy.ReActAgent(
        name="master_agent",
        llm_model="default_llm",
        is_master=True,
        sub_agents=["image_gen_agent","open_chart_agent"],
        prompt_template=MASTER_AGENT_PROMPT,
        tools=["flow_image_gen_tools", "open_chart_tools"],
    ),
]
```

### FastAPI Web 服务集成

系统集成了 FastAPI Web 服务，提供完整的用户界面：

```python
# 创建FastAPI应用
app = FastAPI(
    title="Mermaid流程图交互式编辑器", 
    description="基于OxyGent和Mermaid的流程图生成与编辑工具"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加API路由
from oxygent.chart.flowchart_api import router as flowchart_router
app.include_router(flowchart_router, prefix="/api")

# 添加根路径路由
@app.get("/")
async def read_root():
    return RedirectResponse(url="/static/index.html")

async def main():
    # 创建web目录（如果不存在）
    os.makedirs("../../oxygent/chart/web", exist_ok=True)
    os.makedirs("../../oxygent/chart/web/css", exist_ok=True)
    os.makedirs("../../oxygent/chart/web/js", exist_ok=True)
    
    # 创建静态文件
    create_static_files("../../oxygent/chart")
    
    # 使用MAS启动Web服务
    app.mount("/static", StaticFiles(directory="../../oxygent/chart/web"), name="web")
    
    async with MAS(oxy_space=oxy_space) as mas:
        # 启动Web服务
        await mas.start_web_service(
            first_query="请生成一个软件开发流程图，包括需求分析、设计、编码、测试和部署阶段",
            port=8081
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### 模块化工具系统

系统使用模块化的工具设计：

1. **flow_image_gen_tools**：流程图生成工具模块
   - `generate_flow_chart`：核心流程图生成功能
   - 支持 OpenAI 兼容 API 调用
   - 自动 HTML 文件生成和浏览器打开

2. **open_chart_tools**：图表打开工具模块
   - `open_html_chart`：在浏览器中打开 HTML 文件
   - 支持本地文件路径处理

3. **static_files_utils**：静态文件管理模块
   - `create_static_files`：创建 Web 界面所需的静态文件
   - 包括 HTML、CSS、JavaScript 文件

## 示例用法

### 基本流程图生成

1. **软件开发流程**：
   - 输入："生成一个软件开发流程图，包括需求分析、设计、编码、测试和部署阶段"
   - 输出：包含完整软件开发生命周期的 Mermaid 流程图

2. **项目管理流程**：
   - 输入："创建一个项目管理流程图"
   - 输出：项目管理标准流程的可视化图表

3. **业务流程**：
   - 输入："画一个用户注册流程图"
   - 输出：用户注册业务流程的详细图表

### 高级功能使用

1. **智能代理调度**：
   ```
   用户输入："生成一个订单处理流程图"
   → Master Agent 分析 → 调用 image_gen_agent → 执行 generate_flow_chart
   ```

2. **打开已有流程图**：
   ```
   用户输入："打开之前生成的流程图"
   → Master Agent 分析 → 调用 open_chart_agent → 执行 open_html_chart
   ```

3. **操作指导**：
   ```
   用户输入："如何编辑这个流程图？"
   → Master Agent 直接提供操作指导，无需调用子代理
   ```

### Web 界面功能

1. **实时编辑**：
   - 在代码编辑选项卡中修改 Mermaid 代码
   - 实时预览功能，即时查看修改效果

2. **模板支持**：
   - 提供多种预设流程图模板
   - 支持快速选择和自定义

3. **导出功能**：
   - SVG 格式导出：矢量图形，适合印刷
   - PNG 格式导出：位图格式，适合网页使用

## 故障排除

### 常见问题

1. **API 连接失败**：
   - 检查 `.env` 文件中的 API 配置
   - 验证网络连接和 API 服务状态
   - 系统会自动降级到示例流程图

2. **端口占用**：
   - 默认端口 8081 被占用时，修改 `port` 参数
   - 或使用 `pkill -f create_flow_image_demo.py` 清理进程

3. **文件路径问题**：
   - 确保有足够的文件系统权限
   - 检查输出目录是否存在

4. **静态文件缺失**：
   - 系统会自动创建必要的静态文件
   - 如有问题，检查 `create_static_files` 函数执行

### 调试模式

启用详细日志输出：

```bash
# 设置日志级别
export LOG_LEVEL=DEBUG
python examples/other/create_flow_image_demo.py
```

### 性能优化

1. **API 调用优化**：
   - 设置合理的超时时间
   - 实现错误重试机制

2. **文件管理**：
   - 定期清理生成的 HTML 文件
   - 使用时间戳避免文件名冲突

## 扩展开发

### 添加新的代理

1. 创建新的工具函数模块
2. 在 `oxy_space` 中注册新代理
3. 更新主控代理的提示模板，添加新的决策规则

示例：
```python
# 新增图表分析代理
oxy.ReActAgent(
    name="chart_analysis_agent",
    tools=["chart_analysis_tools"],
    desc="流程图分析和优化代理"
)
```

### 自定义流程图模板

修改流程图生成工具中的模板系统：

```python
# 在 flow_image_gen_tools.py 中添加新模板
CUSTOM_TEMPLATES = {
    "agile_development": "敏捷开发流程模板",
    "data_pipeline": "数据处理管道模板",
    "user_journey": "用户旅程图模板"
}
```

### 集成其他图表类型

扩展系统以支持其他类型的图表生成：

1. **时序图**：展示时间顺序的交互过程
2. **类图**：显示系统的类结构关系
3. **甘特图**：项目进度和时间管理
4. **思维导图**：概念和想法的层次结构

### API 扩展

添加更多 REST API 端点：

```python
# 添加批量生成功能
@app.post("/api/batch-generate")
async def batch_generate_charts(requests: List[ChartRequest]):
    # 批量生成多个流程图
    pass

# 添加图表模板管理
@app.get("/api/templates")
async def get_templates():
    # 获取可用模板列表
    pass
```

## 最佳实践

### 代码组织

1. **模块化设计**：每个功能模块独立，便于测试和维护
2. **配置管理**：使用环境变量管理敏感信息
3. **错误处理**：实现完善的异常处理和降级机制

### 性能考虑

1. **异步处理**：使用 asyncio 提高并发性能
2. **缓存机制**：缓存常用的流程图模板
3. **资源管理**：及时清理临时文件和资源

### 安全性

1. **输入验证**：验证用户输入，防止注入攻击
2. **文件路径安全**：限制文件访问范围
3. **API 安全**：实现适当的认证和授权机制

## 总结

本示例展示了如何使用 OxyGent 框架构建一个完整的多代理流程图生成系统，具备以下优势：

- **智能化**：主控代理能够理解用户意图并智能调度
- **模块化**：各功能模块独立，易于维护和扩展
- **鲁棒性**：具备错误处理和降级机制
- **用户友好**：提供完整的 Web 界面和实时预览
- **可扩展**：架构设计支持添加新功能和代理

该示例为构建更复杂的多代理系统提供了良好的参考和基础，展现了 OxyGent 框架在实际应用中的强大能力和灵活性。通过合理的架构设计和模块化实现，可以快速构建出功能丰富、性能优良的智能应用系统。