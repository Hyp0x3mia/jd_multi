# Intelligent Flow Chart Generation Tool Case Study Based on OxyGent and JoyCode

## Overview

`create_flow_image_demo.py` is a Multi-Agent System (MAS) example based on the OxyGent framework, demonstrating how to leverage multi-agent collaboration to implement intelligent flow chart automatic generation tasks. This example primarily showcases the functionality of generating Mermaid flow chart code through OpenAI-compatible APIs and rendering them as visualized HTML files, while demonstrating task allocation, collaboration, and execution processes in multi-agent systems.

### Core Features

- **Multi-Agent Collaboration**: Master Agent intelligently coordinates flow chart generation and opening operations
- **Mermaid Flow Charts**: Uses Mermaid syntax to generate professional flow charts
- **OpenAI-Compatible API**: Calls OpenAI-compatible interfaces to convert text descriptions into Mermaid code
- **Web Service Interface**: Provides user interaction interface based on FastAPI
- **Modular Design**: Separates flow chart generation, opening functionality, and static file management into independent modules
- **Intelligent Decision Making**: Master agent can intelligently select appropriate tools and strategies based on user input
- **Real-time Preview**: Supports editing Mermaid code with real-time flow chart preview
- **Export Functionality**: Supports exporting flow charts to SVG and PNG formats
- **Error Fallback Mechanism**: Automatically uses high-quality backup flow charts when API calls fail

### Technology Stack

- **OxyGent**: Multi-agent system framework
- **OpenAI-Compatible API**: Large language model interface
- **Mermaid**: Flow chart generation library
- **FastAPI**: Web service framework
- **FunctionHub**: Tool function organization framework

## Environment Setup

### System Requirements

- Python 3.10+
- OpenAI-compatible LLM API service

### Dependency Installation

1. Install Python dependencies:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Variable Configuration

Create a `.env` file in the project root directory and configure the following environment variables:

```
OPENAI_BASE_URL=your_openai_base_url
OPENAI_MODEL_NAME=your_model_name
OPENAI_API_KEY=your_api_key
```

## Quick Start

### Running the Example

1. Ensure the virtual environment is activated:

```bash
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

2. Ensure API service is accessible:

```bash
# Test API connection
curl -X POST "your_openai_base_url/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{"model": "your_model_name", "messages": [{"role": "user", "content": "Hello"}]}'
```

3. Run the example program:

```bash
python examples/other/create_flow_image_demo.py
```

4. After the program starts, access the web interface in your browser:

```
http://127.0.0.1:8080/web/index.html
```

### Usage Instructions

1. **Default Query**: The program automatically executes "Please generate a software development flow chart including requirements analysis, design, coding, testing, and deployment phases" upon startup
2. **Custom Queries**: Enter custom flow chart descriptions in the web interface input box, such as:
   - "Generate a project management flow chart including initiation, planning, execution, monitoring, and closing phases"
   - "Create a user registration flow chart"
   - "Draw an order processing flow chart"
3. **View Results**:
   - Flow charts will automatically open in the browser
   - HTML files are saved in the project directory with filename format `software-development-workflow.html`
4. **Edit and Preview**:
   - In the opened flow chart page, you can directly edit Mermaid code
   - Click the "Update Preview" button to preview modified flow charts in real-time
   - Supports multiple preset template selections
5. **Export Charts**:
   - Click the "Export SVG" button to export flow charts in SVG format
   - Click the "Export PNG" button to export flow charts in PNG format

### Stop Program

Press `Ctrl+C` in the terminal to stop the program, or use the following command to terminate the process:

```bash
pkill -f create_flow_image_demo.py
```

## Key Feature Implementation

### Multi-Agent System Architecture


The system consists of one master agent and two sub-agents:

1. **Master Agent**: Responsible for intelligent task coordination and distribution with complete decision-making capabilities
2. **Image Gen Agent**: Specialized in flow chart generation tasks
3. **Open Chart Agent**: Specialized in opening flow chart files in browsers
4. **Direct Tool Access**: Master agent can also directly call tool functions

### Master Agent Intelligent Decision System

The master agent uses detailed prompt templates to implement intelligent decision-making:

```python
MASTER_AGENT_PROMPT = """
You are an intelligent flow chart assistant, specialized in analyzing user requirements and intelligently selecting appropriate agents to complete tasks.

## Core Responsibilities
1. Intelligently analyze user input intentions and requirements
2. Determine which agent to use based on keywords and semantics
3. Reasonably schedule sub-agents or directly use tools
4. Provide clear task execution feedback

## Available Agents and Tools

### image_gen_agent (Flow Chart Generation Agent)
- **Tool**: generate_flow_chart
- **Function**: Generate Mermaid flow charts based on text descriptions and automatically open in browser
- **Parameters**:
  - description (required): Detailed text description of the flow chart
  - output_path (optional): Output HTML file path
- **Trigger Keywords**: generate, create, make, draw, design, new, develop flow charts, etc.

### open_chart_agent (Chart Opening Agent)
- **Tool**: open_html_chart  
- **Function**: Open existing flow chart HTML files in browser
- **Parameters**:
  - file_path (required): Complete path to HTML file
- **Trigger Keywords**: open, view, display, browse existing/current flow charts, etc.

## Intelligent Decision Rules

### Rule 1: Generate New Flow Charts
**Trigger Condition**: User mentions "generate", "create", "make", "draw", "design", "new" and other words, and describes flow content
**Execution Strategy**: 
- Prioritize using image_gen_agent or directly call generate_flow_chart
- Extract flow content described by user as description parameter
- Automatically generate appropriate filename as output_path

### Rule 2: Open Existing Flow Charts
**Trigger Condition**: User mentions "open", "view", "display" existing/current flow chart files
**Execution Strategy**:
- Use open_chart_agent or directly call open_html_chart
- Requires user to provide file path, ask if not provided

### Rule 3: Edit Operation Guidance
**Trigger Condition**: User asks how to edit, modify, save, or export flow charts
**Execution Strategy**:
- No need to call agents, directly provide operation guidance
- Ensure flow chart is already open in browser
"""
```

### System Configuration and Initialization

```python
# Configuration management
Config.set_agent_llm_model("default_llm")
# Config.set_server_auto_open_webpage(False)  # Optional: disable auto browser opening

# OxyGent space configuration
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
        desc="Flow chart generation agent,"
    ),
    oxy.ReActAgent(
        name="open_chart_agent",
        tools=["open_chart_tools"],
        desc="Agent for opening flow charts in browser"
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

### FastAPI Web Service Integration

The system integrates FastAPI web services to provide a complete user interface:

```python
# Create FastAPI application
app = FastAPI(
    title="Mermaid Flow Chart Interactive Editor", 
    description="Flow chart generation and editing tool based on OxyGent and Mermaid"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API routes
from oxygent.chart.flowchart_api import router as flowchart_router
app.include_router(flowchart_router, prefix="/api")

# Add root path route
@app.get("/")
async def read_root():
    return RedirectResponse(url="/static/index.html")

async def main():
    # Create web directories (if they don't exist)
    os.makedirs("../../oxygent/chart/web", exist_ok=True)
    os.makedirs("../../oxygent/chart/web/css", exist_ok=True)
    os.makedirs("../../oxygent/chart/web/js", exist_ok=True)
    
    # Create static files
    create_static_files("../../oxygent/chart")
    
    # Start web service using MAS
    app.mount("/static", StaticFiles(directory="../../oxygent/chart/web"), name="web")
    
    async with MAS(oxy_space=oxy_space) as mas:
        # Start web service
        await mas.start_web_service(
            first_query="Please generate a software development flow chart including requirements analysis, design, coding, testing, and deployment phases",
            port=8081
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### Modular Tool System

The system uses modular tool design:

1. **flow_image_gen_tools**: Flow chart generation tool module
   - `generate_flow_chart`: Core flow chart generation functionality
   - Supports OpenAI-compatible API calls
   - Automatic HTML file generation and browser opening

2. **open_chart_tools**: Chart opening tool module
   - `open_html_chart`: Open HTML files in browser
   - Supports local file path handling

3. **static_files_utils**: Static file management module
   - `create_static_files`: Create static files required for web interface
   - Includes HTML, CSS, JavaScript files

## Example Usage

### Basic Flow Chart Generation

1. **Software Development Process**:
   - Input: "Generate a software development flow chart including requirements analysis, design, coding, testing, and deployment phases"
   - Output: Mermaid flow chart containing complete software development lifecycle

2. **Project Management Process**:
   - Input: "Create a project management flow chart"
   - Output: Visualization chart of standard project management processes

3. **Business Process**:
   - Input: "Draw a user registration flow chart"
   - Output: Detailed chart of user registration business process

### Advanced Feature Usage

1. **Intelligent Agent Scheduling**:
   ```
   User Input: "Generate an order processing flow chart"
   → Master Agent Analysis → Call image_gen_agent → Execute generate_flow_chart
   ```

2. **Open Existing Flow Charts**:
   ```
   User Input: "Open the previously generated flow chart"
   → Master Agent Analysis → Call open_chart_agent → Execute open_html_chart
   ```

3. **Operation Guidance**:
   ```
   User Input: "How to edit this flow chart?"
   → Master Agent directly provides operation guidance without calling sub-agents
   ```

### Web Interface Features

1. **Real-time Editing**:
   - Modify Mermaid code in the code editing tab
   - Real-time preview functionality for instant viewing of modifications

2. **Template Support**:
   - Provides multiple preset flow chart templates
   - Supports quick selection and customization

3. **Export Functionality**:
   - SVG format export: Vector graphics, suitable for printing
   - PNG format export: Bitmap format, suitable for web use

## Troubleshooting

### Common Issues

1. **API Connection Failure**:
   - Check API configuration in `.env` file
   - Verify network connection and API service status
   - System automatically falls back to example flow charts

2. **Port Occupied**:
   - When default port 8081 is occupied, modify the `port` parameter
   - Or use `pkill -f create_flow_image_demo.py` to clean processes

3. **File Path Issues**:
   - Ensure sufficient file system permissions
   - Check if output directory exists

4. **Missing Static Files**:
   - System automatically creates necessary static files
   - If issues occur, check `create_static_files` function execution

### Debug Mode

Enable verbose log output:

```bash
# Set log level
export LOG_LEVEL=DEBUG
python examples/other/create_flow_image_demo.py
```

### Performance Optimization

1. **API Call Optimization**:
   - Set reasonable timeout periods
   - Implement error retry mechanisms

2. **File Management**:
   - Regularly clean generated HTML files
   - Use timestamps to avoid filename conflicts

## Extension Development

### Adding New Agents

1. Create new tool function modules
2. Register new agents in `oxy_space`
3. Update master agent prompt template, add new decision rules

Example:
```python
# Add new chart analysis agent
oxy.ReActAgent(
    name="chart_analysis_agent",
    tools=["chart_analysis_tools"],
    desc="Flow chart analysis and optimization agent"
)
```

### Custom Flow Chart Templates

Modify template system in flow chart generation tools:

```python
# Add new templates in flow_image_gen_tools.py
CUSTOM_TEMPLATES = {
    "agile_development": "Agile development process template",
    "data_pipeline": "Data processing pipeline template",
    "user_journey": "User journey map template"
}
```

### Integrate Other Chart Types

Extend system to support other types of chart generation:

1. **Sequence Diagrams**: Show time-ordered interaction processes
2. **Class Diagrams**: Display system class structure relationships
3. **Gantt Charts**: Project progress and time management
4. **Mind Maps**: Hierarchical structure of concepts and ideas

### API Extensions

Add more REST API endpoints:

```python
# Add batch generation functionality
@app.post("/api/batch-generate")
async def batch_generate_charts(requests: List[ChartRequest]):
    # Batch generate multiple flow charts
    pass

# Add chart template management
@app.get("/api/templates")
async def get_templates():
    # Get available template list
    pass
```

## Best Practices

### Code Organization

1. **Modular Design**: Each functional module is independent, facilitating testing and maintenance
2. **Configuration Management**: Use environment variables to manage sensitive information
3. **Error Handling**: Implement comprehensive exception handling and fallback mechanisms

### Performance Considerations

1. **Asynchronous Processing**: Use asyncio to improve concurrent performance
2. **Caching Mechanisms**: Cache commonly used flow chart templates
3. **Resource Management**: Timely cleanup of temporary files and resources

### Security

1. **Input Validation**: Validate user input to prevent injection attacks
2. **File Path Security**: Limit file access scope
3. **API Security**: Implement appropriate authentication and authorization mechanisms

## Summary

This example demonstrates how to use the OxyGent framework to build a complete multi-agent flow chart generation system with the following advantages:

- **Intelligence**: Master agent can understand user intentions and intelligently schedule
- **Modularity**: Each functional module is independent, easy to maintain and extend
- **Robustness**: Features error handling and fallback mechanisms
- **User-Friendly**: Provides complete web interface and real-time preview
- **Extensibility**: Architecture design supports adding new features and agents

This example provides a good reference and foundation for building more complex multi-agent systems, showcasing the powerful capabilities and flexibility of the OxyGent framework in practical applications. Through reasonable architectural design and modular implementation, it's possible to quickly build intelligent application systems with rich functionality and excellent performance.