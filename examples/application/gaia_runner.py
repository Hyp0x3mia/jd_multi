import asyncio
import argparse
import json
import os
import unicodedata
from typing import Any, Dict, List, Optional

from oxygent import MAS, Config, oxy, preset_tools
import importlib.util
import logging
logger = logging.getLogger(__name__)
"""
简化的GAIA Runner - 直接使用内置智能体

环境变量配置说明：
需要配置以下环境变量以支持多模型调用：

# 默认LLM（备用）
DEFAULT_LLM_API_KEY=your_default_api_key
DEFAULT_LLM_BASE_URL=your_default_base_url
DEFAULT_LLM_MODEL_NAME=your_default_model_name

# DeepSeek V3（复杂推理、主控协调）
DEEPSEEK_KEY=your_deepseek_key
DEEPSEEK_URL=https://api.deepseek.com
DEEPSEEK_V3=deepseek-chat

# DeepSeek R1（推理分析）
DEEPSEEK_R1_KEY=your_deepseek_r1_key
DEEPSEEK_R1_URL=https://api.deepseek.com
DEEPSEEK_R1=deepseek-reasoner

# GPT-4o（数学、代码）
OPEN_AI_KEY=your_openai_key
OPEN_AI_URL=https://api.openai.com/v1
GPT_4O=gpt-4o

# Claude Sonnet（文件处理）
CLAUDE_KEY=your_claude_key
CLAUDE_URL=https://api.anthropic.com/v1
CLAUDE_SONNET=claude-3-5-sonnet-20241022

# Claude Model（通用任务）
CLAUDE_MODEL=claude-3-5-haiku-20241022
"""


def _repo_root() -> str:
	# examples/application -> repo root
	return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _clean_final_answer(text: str) -> str:
	"""清理最终答案，移除无关信息"""
	if not text:
		return ""
	
	text = str(text).strip()
	
	# 移除常见的前缀
	prefixes_to_remove = [
		"答案：", "Answer:", "Result:", "结果：", "输出：", "Output:",
		"最终答案：", "Final Answer:", "答案是：", "The answer is:",
		"根据查询结果：", "Based on the search:", "经过分析："
	]
	
	for prefix in prefixes_to_remove:
		if text.startswith(prefix):
			text = text[len(prefix):].strip()
			break
	
	# 移除常见的后缀
	suffixes_to_remove = [
		"以上就是答案", "This is the answer", "答案如上", "Answer as above"
	]
	
	for suffix in suffixes_to_remove:
		if text.endswith(suffix):
			text = text[:-len(suffix)].strip()
			break
	
	# 如果包含多行，只取第一行（通常是最终答案）
	if '\n' in text:
		lines = text.split('\n')
		# 找到包含实际答案的行
		for line in lines:
			line = line.strip()
			if line and not any(keyword in line.lower() for keyword in 
				['explanation', 'explain', 'reasoning', 'because', '由于', '因为', '解释', '说明']):
				text = line
				break
	
	return text


def _normalize_answer(text: str) -> str:
	"""强规范化答案：NFKC归一、数字处理、逗号分割排序等"""
	if not text:
		return ""

	# NFKC 归一化
	text = unicodedata.normalize('NFKC', str(text).strip())

	# 尝试数字转换
	try:
		if '.' in text:
			num = float(text)
			if num == int(num):
				return str(int(num))
			return str(num)
		else:
			return str(int(text))
	except ValueError:
		pass

	# 逗号分割处理
	if ',' in text:
		parts = [p.strip() for p in text.split(',')]
		# 去重并排序
		unique_parts = list(dict.fromkeys(parts))  # 保持顺序去重
		return ', '.join(unique_parts)

	return text


def _classify_task(query: str) -> str:
	"""硬规则Router：关键词匹配分发任务类型"""
	query_lower = query.lower()
	
	# Math 关键词
	math_keywords = [
		'最少', '多少', '期望', '概率', '天', '小时', '分钟', '秒',
		'组合', '排列', '平均', '计算', '切', '段', '枚', '枚硬币',
		'桶', '毒', '试验', '兔子', '称重', '天平', '假币', '真币',
		'硬币', '最少需要', '最多', '期望值', '概率', '百分比',
		'相加', '相乘', '相减', '相除', '平方', '立方', '根号'
	]
	
	# Time 关键词
	time_keywords = [
		'时间', '日期', '几点', '什么时候', '年月日', '星期', '周',
		'今天', '明天', '昨天', '现在', '当前时间', '时区', '年', '月', '日'
	]
	
	# Web 关键词（包含浏览器功能）
	web_keywords = [
		'github', 'youtube', 'wikipedia', '链接', '网页',
		'仓库', '提交', 'pr', '第', '季', '集', '节目名', '截至',
		'官方', '网站', '页面', '百科', '百度', '搜索', '查找', '寻找', '检索', '查询',
		'item.jd.com', 'http', 'https', 'www', '.com', '.cn',
		'在', '关于', '什么', '哪个', '谁', '哪里', '如何',
		# 浏览器相关关键词
		'浏览器', '自动化', '点击', '填写', '表单', '滚动', '下载', '上传',
		'登录', '注册', '提交', '选择', '下拉', '菜单', '按钮', '输入框',
		'多步', '交互', '操作', '导航', '页面跳转', '弹窗', '确认', '取消',
		'浏览器操作', '网页操作', '自动化操作', '模拟操作', '网页内容', '提取信息'
	]
	
	
	# Code 关键词
	code_keywords = [
		'代码', '编程', 'python', 'java', 'javascript', '算法',
		'函数', '变量', '循环', '条件', '数据结构', '编程语言',
		'计算', '运行', '执行', '脚本', '程序'
	]
	
	# Terminal 关键词
	terminal_keywords = [
		'命令', '终端', 'shell', 'bash', 'cmd', '执行', '运行',
		'安装', '下载', '创建', '删除', '移动', '复制', '权限',
		'文件', '目录', '路径', '系统'
	]
	
	# Analysis 关键词
	analysis_keywords = [
		'分析', '推理', '判断', '比较', '评估', '解释', '原因',
		'为什么', '怎么', '如何', '是否', '能否', '应该', '建议'
	]

	# Audio 关键词
	audio_keywords = [
		'音频', '语音', '录音', 'asr', '转写', '转录',
		'wav', 'mp3', 'm4a', 'flac'
	]
	
	# File 关键词
	file_keywords = [
		'pdf', 'jpg', 'png', 'mp3', 'mp4', '附件', '截图', '表格',
		'图片', '视频', '音频', '文档', '文件', 'excel', 'ppt'
	]
	
	# 硬规则匹配（按优先级）
	for keyword in math_keywords:
		if keyword in query_lower:
			return 'math'
	
	for keyword in time_keywords:
		if keyword in query_lower:
			return 'time'
			
	for keyword in code_keywords:
		if keyword in query_lower:
			return 'code'
			
	for keyword in terminal_keywords:
		if keyword in query_lower:
			return 'terminal'
			
	for keyword in analysis_keywords:
		if keyword in query_lower:
			return 'analysis'
			
	for keyword in file_keywords:
		if keyword in query_lower:
			return 'file'

	for keyword in audio_keywords:
		if keyword in query_lower:
			return 'audio'
	
	for keyword in web_keywords:
		if keyword in query_lower:
			return 'web'
	
	# 默认返回 web（大部分事实题）
	return 'web'


def build_oxy_space(enable_mcp: bool = False) -> List[Any]:
	"""构建OxyGent空间，包含所有智能体和工具"""
	Config.set_agent_llm_model("default_llm")
	oxy_space: List[Any] = [
		# 默认LLM
		oxy.HttpLLM(
			name="default_llm",
			api_key=os.getenv("DEFAULT_LLM_API_KEY"),
			base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
			model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
			llm_params={"temperature": 0.01},
			semaphore=4,
		),
		oxy.HttpLLM(
        name='zai-org/GLM-4.6',
        api_key=os.getenv('DEEPSEEK_KEY'),
        base_url=os.getenv('DEEPSEEK_URL'),
        model_name=os.getenv('DEEPSEEK_V3'),
        llm_params={'temperature': 0.01},
        semaphore=4,
        timeout=240
    ),
        oxy.HttpLLM(
        name='inclusionAI/Ring-1T',
        api_key=os.getenv('DEEPSEEK_R1_KEY'),
        base_url=os.getenv('DEEPSEEK_URL'),
        model_name=os.getenv('DEEPSEEK_R1'),
        llm_params={'temperature': 0.01},
        semaphore=4,
        timeout=240
    ),
	    oxy.HttpLLM(
        name='Qwen/Qwen3-Coder-480B-A35B-Instruct',
        api_key=os.getenv('OPEN_AI_KEY'),
        base_url=os.getenv('OPEN_AI_URL'),
        model_name=os.getenv('GPT_4O'),
        llm_params={'temperature': 0.01},
        semaphore=4,
        timeout=240
    ),
	    oxy.HttpLLM(
        name="zai-org/GLM-4.5",
        api_key=os.getenv('CLAUDE_KEY'),
        base_url=os.getenv('CLAUDE_URL'),
        model_name=os.getenv('CLAUDE_SONNET'),
        llm_params={'temperature': 0.01},
        semaphore=4,
        timeout=240
    ),
		oxy.HttpLLM(
		name="vision_llm",
		api_key=os.getenv("VL_KEY"),  # 替换成你用的模型的KEY
		base_url=os.getenv("VL_URL"),
		model_name=os.getenv("VL_MODEL"),
		llm_params={"temperature": 0.01},
		semaphore=2,
		desc="视觉大模型（支持图像理解、截图问答）",
		is_multimodal_supported=True,
		),
		# 预置工具
		preset_tools.time_tools,
		preset_tools.file_tools,
		preset_tools.math_tools,
		preset_tools.python_tools,
		preset_tools.shell_tools,
		preset_tools.http_tools,
		preset_tools.string_tools,
		preset_tools.system_tools,
	]

	# 尝试添加百度搜索工具
	try:
		oxy_space.append(preset_tools.baidu_search_tools)
	except Exception as e:
		print(f"Warning: Could not add baidu_search_tools: {e}")

	# PDF analyze tool removed per user request

	# 本地 PDF 工具已通过 preset_tools.file_tools 注册为 file_tools.tool

	# 暂停加载浏览器MCP（按需再启用）
	print("ℹ Browser MCP disabled temporarily")

	# 引入浏览器 MCP 工具（参考 examples/mcp_tools/browser_demo.py）
	browser_mcp = oxy.StdioMCPClient(
		name="browser_tools",
		params={
			"command": "uv",
			"args": ["--directory", "./mcp_servers", "run", "browser/server.py"],
		},
	)
	oxy_space.append(browser_mcp)

	# 引入音频 MCP 工具（SenseVoice ASR）
	audio_mcp = oxy.StdioMCPClient(
		name="audio_tools",
		params={
			"command": "uv",
			"args": ["--directory", "./mcp_servers", "run", "audio/server.py"],
		},
	)
	oxy_space.append(audio_mcp)


	# File MCP 工具（SenseVoice ASR）
	file_mcp = oxy.StdioMCPClient(
		name="file_content_tools",
		params={
			"command": "uv",
			"args": ["--directory", "./mcp_servers", "run", "file/server.py"],
		},
	)
	oxy_space.append(file_mcp)


	# Dictionary MCP 工具
	dictionary_mcp = oxy.StdioMCPClient(
		name="dictionary_tools",
		params={
			"command": "uv",
			"args": ["--directory", "./mcp_servers", "run", "dictionary/server.py"],
		},
	)
	oxy_space.append(dictionary_mcp)

	# 过滤掉可能为 None 的工具，避免后续 MAS 初始化时报 'NoneType' 错误
	oxy_space = [t for t in oxy_space if t is not None]

	# 任务型智能体
	available_tools = {getattr(t, "name", None) for t in oxy_space if hasattr(t, "name")}
	
	# MathAgent：数学/逻辑计算
	oxy_space.append(
		oxy.ReActAgent(
			name="math_agent",
			desc="Math/Logic Specialist – deterministically solve problems via Python.",
			tools=["python_tools", "math_tools"],
			llm_model="default_llm",
			additional_prompt=(
				"ROLE: Math/Logic Specialist\n"
				"CAPABILITIES: Arithmetic, combinatorics, averages, date arithmetics, small enumeration using Python.\n"
				"POLICY:\n"
				"- Prefer exact computation with python_tools/math_tools.\n"
				"- Keep chain-of-thought internal; only output results.\n"
				"OUTPUT:\n"
				"- Return ONLY the final number/string needed by the question (no units unless requested).\n"
			),
			timeout=60,
		)
	)
	
	# TimeAgent：时间/日期处理
	oxy_space.append(
		oxy.ReActAgent(
			name="time_agent",
			desc="Time/Date Specialist – parse and compute temporal queries.",
			tools=["time_tools", "python_tools"],
			llm_model="default_llm",
			additional_prompt=(
				"ROLE: Time/Date Specialist\n"
				"POLICY: Use datetime arithmetics; avoid ambiguous natural language.\n"
				"OUTPUT: If format not specified, prefer YYYY-MM-DD or HH:MM as appropriate. Return ONLY the final value.\n"
			),
			timeout=30,
		)
	)
	
	# WebAgent：网页信息提取与搜索（集成浏览器工具）
	web_tools = ["http_tools"]
	if "baidu_search_tools" in available_tools:
		web_tools.append("baidu_search_tools")
	if "browser_tools" in available_tools:
		web_tools.append("browser_tools")
	
	oxy_space.append(
		oxy.ReActAgent(
			name="web_agent", 
			desc="Web Specialist – search, fetch, browse, and extract deterministic facts.",
			tools=web_tools,
			llm_model="default_llm",
			additional_prompt=(
				"ROLE: Web + Browser Specialist\n"
				"TOOLS: http_tools, baidu_search_tools (if available), browser_tools (navigate/click/snapshot), browser_analyze_screenshot (VLM).\n"
				"FLOW:\n"
				"1) Search authoritative sources (official sites, Wikipedia, GitHub).\n"
				"2) Open page and extract target facts deterministically.\n"
				"3) If visual content required (text in image, buttons, tables), call browser_analyze_screenshot with a precise, minimal prompt.\n"
				"POLICY:\n"
				"- Keep steps minimal: search → open → extract.\n"
				"- Prefer primary sources and exact matches.\n"
				"OUTPUT FORMAT RULES (CRITICAL):\n"
				"- If question explicitly requests a specific format (e.g., 'comma-separated', '仅用英文逗号间隔', '以逗号分隔'), output ONLY in that exact format.\n"
				"- If question asks for a list (e.g., 'six items', '六个板块'), and requests comma separation, output: 'item1,item2,item3,...' (NO descriptions, NO explanations).\n"
				"- If question asks for names/titles only, extract ONLY the names, join with requested separator (usually comma).\n"
				"- Examples:\n"
				"  * Question: 'List names, comma-separated' → Output: 'Name1,Name2,Name3'\n"
				"  * Question: '六个板块叫什么？请仅用英文逗号间隔输出' → Output: '京东金条,白条,京东小金库,基金,保险,更多服务'\n"
				"- Return ONLY the requested fact/value in the requested format; no extra commentary, no descriptions.\n"
			),
			timeout=120,
		)
	)

	# 移除 browser_agent，合并为 web_agent 单入口；保留视觉工具由 MCP 内部调用
	
	# FileAgent：文件处理
	file_tools_list = ["file_content_tools"]
	if "analyze_pdf_character_tool" in available_tools:
		file_tools_list.append("analyze_pdf_character_tool")

	oxy_space.append(
		oxy.ReActAgent(
			name="file_agent",
			desc="File Specialist – identify, parse and extract from local files.",
			tools=file_tools_list,
			llm_model="zai-org/GLM-4.5",
			additional_prompt=(
				"ROLE: File Specialist\n"
				"SCOPE: Extract content FROM files (text from PDFs, images, documents).\n"
				"LIMITATIONS: This agent does NOT count files or list directories. For counting/statistics tasks, return 'UNABLE_TO_PROCESS: This task requires counting/statistics, please use math_agent with Python code.'\n"
				"POLICY:\n"
				"- Detect file type first; extract only requested fields.\n"
				"- If task asks to count files/list directory items, return the UNABLE_TO_PROCESS message above.\n"
				"OUTPUT: Return only the minimal text/value(s) required by the question, OR the UNABLE_TO_PROCESS message if task is outside scope.\n"
			),
			timeout=120,
		)
	)



	dictionary_tools_list = ["file_tools"]
	if "dictionary_tools" in available_tools:
		dictionary_tools_list.append("dictionary_tools")

	oxy_space.append(
		oxy.ReActAgent(
			name="directory_agent",
			desc="Directory Operations Specialist – handle file/directory creation, reading, writing, deletion, info query, counting, and listing.",
			tools=dictionary_tools_list,
			llm_model="zai-org/GLM-4.5",
			additional_prompt=(
           "ROLE: File Operations Specialist\n"
            "SCOPE: Perform file/directory operations including: create/write/overwrite, read content, delete, get details, count specific file types, and list directory contents.\n"
            "LIMITATIONS:\n"
            "- Only operates within allowed directories; cannot access restricted paths.\n"
            "- Overwriting/deleting files is irreversible – confirm path accuracy before using write_file/delete_file.\n"
            "POLICY:\n"
            "1. For content extraction: Use read_file first; if parsing PDFs, ensure tool is available (if added).\n"
            "2. For writing: Use write_file with explicit path and content; warn implicitly (via tool's nature) about overwrites.\n"
            "3. For deletion: Verify path exists with get_file_info before calling delete_file to avoid errors.\n"
            "4. For counting: Use count_files with path, optional file_type (e.g., '.txt'), and recursive flag as needed.\n"
            "5. For listing: Use list_directory with filters (file_type) and recursion as requested.\n"
            "6. For file details (size, modified time, etc.): Use get_file_info.\n"
            "OUTPUT: Return tool results directly, or error messages if operations fail. Be concise but include critical details (e.g., 'Deleted: /path' or 'Count: 5 .txt files').\n"
			),
			timeout=120,
		)
	)

	# AudioAgent：音频转写（SenseVoice）
	if "audio_tools" in available_tools:
		oxy_space.append(
			oxy.ReActAgent(
				name="audio_agent",
				desc="Audio Transcription Agent – SenseVoice (SiliconFlow).",
				tools=["audio_tools"],
				llm_model="default_llm",
				prompt=(
					"ROLE: ASR Specialist\n"
					"\n"
					"Your ONLY job is to transcribe audio files to text using the audio_transcribe tool.\n"
					"\n"
					"TOOL AVAILABLE:\n"
					"- Tool name: audio_transcribe\n"
					"- Required parameter: path (the full file path to the audio file)\n"
					"- Supported formats: .wav, .mp3, .m4a, .flac\n"
					"\n"
					"FILE PATH EXTRACTION:\n"
					"The query may contain file path information in one of these formats:\n"
					"1) Explicit file path in the query text (e.g., '/path/to/audio.wav')\n"
					"2) File_Name field: The query may include a line like 'File_Name: /path/to/file.wav' or 'File_Name: /path1.wav /path2.wav'\n"
					"3) Extract the FIRST audio file path you find (if multiple, use the first one).\n"
					"\n"
					"PROCESS:\n"
					"1) Search the query for audio file paths (ending in .wav/.mp3/.m4a/.flac).\n"
					"2) If query contains 'File_Name:' line, extract the path(s) after the colon.\n"
					"3) If File_Name contains multiple paths (space-separated), use the first audio file.\n"
					"4) Extract the complete, absolute file path.\n"
					"5) Call audio_transcribe with the extracted path.\n"
					"\n"
					"TOOL CALL FORMAT:\n"
					"When you need to call the tool, respond with JSON:\n"
					'{"think": "Extracted audio file path: /path/to/audio.wav", "tool_name": "audio_transcribe", "arguments": {"path": "/path/to/audio.wav"}}\n'
					"\n"
					"EXAMPLES:\n"
					"- Query: 'Transcribe this audio\\nFile_Name: /Users/.../audio.wav' → Extract '/Users/.../audio.wav'\n"
					"- Query: 'What does this audio say? File_Name: /path/to/sound.mp3' → Extract '/path/to/sound.mp3'\n"
					"\n"
					"After receiving tool response:\n"
					"- If response contains 'text' field: extract it and return as '【音频转写】' + text\n"
					"- If response contains 'error': return the error message\n"
					"\n"
					"OUTPUT FORMAT:\n"
					"Return ONLY: '【音频转写】' + transcribed text (no extra commentary, no explanations).\n"
					"\n"
					"You have access to these tools:\n"
					"${tools_description}\n"
					"\n"
					"CRITICAL: You MUST call the audio_transcribe tool when an audio file path is detected. Do not skip tool calling."
				),
				timeout=120,
			)
		)
	
	# Normalizer：强规范化输出
	oxy_space.append(
		oxy.ReActAgent(
			name="normalizer_agent",
			desc="Strict Normalizer – enforce final-answer formatting.",
			tools=["string_tools"],
			llm_model="zai-org/GLM-4.5",
			prompt=(
				"ROLE: STRICT Answer Normalizer\n"
				"\n"
				"INPUT: {question, candidate_answer}\n"
				"\n"
				"CRITICAL RULE - NO RECALCULATION:\n"
				"- The candidate_answer is ALREADY the final computed result from specialist agents.\n"
				"- Your job is ONLY formatting/normalization, NOT re-computing or re-calculating.\n"
				"- If candidate_answer is a number (e.g., '11'), return it AS-IS. Do NOT calculate 1+1=2 or any further operations.\n"
				"- If candidate_answer is already correct, return it unchanged (only clean formatting).\n"
				"\n"
				"RULES:\n"
				"- Remove explanations/reasoning/units not requested/decoration.\n"
				"- If answer is a single number and no special format is required: return digits ONLY (no re-calculation).\n"
				"- Normalize whitespace + NFKC; trim trailing zeros when appropriate.\n"
				"- FORMAT EXTRACTION (CRITICAL):\n"
				"  * If question requests a specific format (e.g., 'comma-separated', '仅用英文逗号间隔', '以逗号分隔'), extract items from candidate_answer and output in that format.\n"
				"  * If candidate_answer contains a list/description but question asks for 'names only' or 'comma-separated', extract ONLY the item names/titles.\n"
				"  * Example: candidate_answer='1. 京东金条 - 信贷服务\\n2. 白条 - 信用支付', question='六个板块，仅用英文逗号间隔输出' → Output: '京东金条,白条'\n"
				"- Lists: split → trim → deduplicate → preserve order (unless sorting required) → join with requested separator (default: ', ').\n"
				"  * If question specifies separator (e.g., '英文逗号', 'comma'), use that exact separator.\n"
				"  * If question says '仅用英文逗号间隔' or 'comma-separated', use ',' (no space).\n"
				"- Dates: honor requested format; default to YYYY-MM-DD.\n"
				"- Chinese answers: preserve wording strictly.\n"
				"- If specific form requested (word/URL/name), output ONLY that content.\n"
				"\n"
				"EXAMPLES:\n"
				"- Input: candidate_answer='11', question='sum of digits'. Output: '11' (NOT '2', because 11 is already the final answer).\n"
				"- Input: candidate_answer='The answer is 42'. Output: '42' (remove prefix).\n"
				"- Input: candidate_answer='42.0'. Output: '42' (trim trailing zero).\n"
				"- Input: candidate_answer='第一行：1. 京东金条...\\n2. 白条...', question='仅用英文逗号间隔输出六个板块名称'. Output: '京东金条,白条,京东小金库,基金,保险,更多服务'\n"
				"\n"
				"OUTPUT: Return ONLY the normalized answer, no prefixes/suffixes/extra lines. NO recalculation.\n"
				"\n"
				"You have access to these tools:\n"
				"${tools_description}\n"
				"\n"
				"Use tools only if needed for normalization (e.g., extract URLs/emails if specifically requested). "
				"Otherwise, return the normalized candidate_answer directly WITHOUT any recalculation."
			),
			timeout=30,
		)
	)

	# Router：智能分发
	oxy_space.append(
		oxy.ReActAgent(
			name="router_agent",
			desc="Router – classify tasks and delegate to specialists.",
			tools=["string_tools"],
			llm_model="zai-org/GLM-4.5",
			prompt=(
				"ROLE: Task Router\n"
				"\n"
				"Your job is to classify the user's task by analyzing ALL available information:\n"
				"- The main query/question\n"
				"- File_Name field (if present: directories, file paths, etc.)\n"
				"- Any context clues in the query\n"
				"\n"
				"CLASSIFICATION CATEGORIES:\n"
				"- 'math': Math/logic problems, counting/statistics tasks (最少, 多少, 数量, 概率, 计算, 统计, 切, 桶, 硬币, 平均).\n"
				"  * CRITICAL: If query asks to 'count files' / '统计文件数量' AND File_Name is a directory path, classify as 'math' (requires Python code: os.listdir/counting).\n"
				"  * CRITICAL: If task involves counting/listing items in a directory, classify as 'math'.\n"
				"- 'time': Time/date queries (时间, 日期, 几点, 什么时候, 年月日, 星期)\n"
				"- 'web': Web search, browser automation, and extraction (github, youtube, wikipedia, 链接, 网页, 仓库, 搜索, 查找, 浏览器, 自动化, 点击, 填写, 表单)\n"
				"- 'audio': Audio transcription (音频/语音/录音/asr/转写: wav/mp3/m4a/flac). File_Name must be an audio file.\n"
				"- 'file': File content extraction/analysis (read pdf, extract text from images, analyze document content).\n"
				"- 'directory': Directory Operations  – handle file/directory creation, reading, writing, deletion, info query, counting, and listing.\n"
				"  * NOT for counting files or listing directories - that's 'math'.\n"
				"  * Use 'file' when task asks to extract/read content FROM a file (not count files).\n"
				"\n"
				"ANALYSIS PROCESS:\n"
				"1) Read the complete query (including File_Name if present).\n"
				"2) Identify the core task (counting vs extracting vs searching vs calculating).\n"
				"3) If File_Name is a directory path AND task asks to count/list items → 'math'.\n"
				"4) If File_Name is a single file AND task asks to extract/read content → 'file'.\n"
				"5) Otherwise, match based on query keywords.\n"
				"\n"
				"OUTPUT: Return ONLY the classification token (math/time/web/audio/file).\n"
				"\n"
				"You have access to these tools:\n"
				"${tools_description}\n"
				"\n"
				"Use tools ONLY if necessary to disambiguate. Otherwise, analyze the full context and return classification directly."
			),
			timeout=15,
		)
	)

	# 主控智能体：Router → Specialist Agents → Normalizer
	oxy_space.append(
		oxy.ReActAgent(
			is_master=True,
			name="master_agent",
			sub_agents=[
				"router_agent", "math_agent", "time_agent", 
				"web_agent", "audio_agent", "file_agent","directory_agent", "normalizer_agent"
			],
			llm_model="zai-org/GLM-4.6",
			additional_prompt=(
				"ROLE: Master Coordinator\n"
				"WORKFLOW:\n"
				"1) Route: Call router_agent with the COMPLETE query (including File_Name if present). Router needs full context to classify correctly.\n"
				"2) Delegate: Call the appropriate specialist based on router_agent's classification.\n"
				"   CRITICAL for file/audio agents: When calling audio_agent or file_agent, you MUST pass the FULL query string (including the File_Name field). These agents need the file path information.\n"
				"   Example: If query contains 'File_Name: /path/to/audio.wav', pass the entire query including that line to audio_agent.\n"
				"3) Fallback Logic (CRITICAL):\n"
				"   - If file_agent returns error/empty/unable_to_process (especially for counting/statistics tasks), automatically retry with math_agent.\n"
				"   - If any agent fails but task involves counting/statistics/calculation, use math_agent as fallback.\n"
				"4) Normalize (MANDATORY): After any specialist returns a candidate, ALWAYS call normalizer_agent with {question, candidate_answer}.\n"
				"5) Finalize: Output ONLY the normalized answer.\n"
				"RULES:\n"
				"- When calling router_agent, pass the FULL query string (including any File_Name context).\n"
				"- When calling audio_agent/file_agent, ALWAYS pass the FULL query string (including File_Name) so they can extract file paths.\n"
				"- Keep chain-of-thought internal; do not output reasoning.\n"
				"- No prefixes (Answer:/Result:) or extra lines.\n"
				"- The final output must be the normalized answer string only.\n"
			),
			timeout=180,
		)
	)

	return oxy_space


async def run_single(question: str, enable_mcp: bool = False) -> str:
	"""运行单个问题"""
	oxy_space = build_oxy_space(enable_mcp=enable_mcp)
	async with MAS(oxy_space=oxy_space) as mas:
		resp = await mas.call(callee="master_agent", arguments={"query": question})
		answer = resp.output if hasattr(resp, "output") else str(resp)
		
		# 清理输出，只保留最终答案
		cleaned_answer = _clean_final_answer(answer)
		return _normalize_answer(cleaned_answer)


# def compose_question(item: Dict[str, Any]) -> str:
# 	"""组合问题，处理 file_name 字段"""
# 	q = item.get("query") or item.get("question") or item.get("input") or ""
# 	file_name = item.get("file_name")
# 	if file_name:
# 		# Handle file_name: can be a string, list of strings, or list with single item
# 		if isinstance(file_name, list):
# 			# Extract actual file paths from list (filter out empty strings)
# 			paths = [str(f).strip() for f in file_name if f and str(f).strip()]
# 			if paths:
# 				# Join multiple paths with space for clarity
# 				file_paths_str = " ".join(paths)
# 				q += f"\nFile_Name: {file_paths_str}"
# 		else:
# 			# file_name is a string - use as-is if it's already a full path
# 			file_name_str = str(file_name).strip()
# 			if file_name_str:
# 				# If it's a relative path starting with './' or just filename, construct full path件夹最后有/
# 				if file_name_str.startswith('./') or (not os.path.isabs(file_name_str) ):
# 					# Construct path relative to valid dataset directory
# 					base_dir = Path("./valid")
# 					if base_dir.exists():
# 						file_name_str = str(base_dir / file_name_str.lstrip('./'))
# 				q += f"\nFile_Name: {file_name_str}"
# 	return q
from typing import Any, Dict, List, Union

def _parse_maybe_list(file_name: Union[str, List[str], None]) -> List[str]:
    """统一把 file_name 转成 List[str]，兼容字符串/类 list 字符串/真正 list。"""
    if file_name is None:
        return []
    if isinstance(file_name, list):
        return [str(f).strip() for f in file_name if f and str(f).strip()]

    s = str(file_name).strip()
    if not s:
        return []
    # 尝试解析 "['a','b']"
    if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
        try:
            return [str(i) for i in json.loads(s.replace("'", '"'))]
        except Exception:
            pass
    return [s]


def compose_question(item: Dict[str, Any], input_jsonl_path: str) -> str:
    """
    组合问题
    :param item: 原始数据行
    :param input_jsonl_path: 形如 "./valid/data2.jsonl" 的字符串，用来提取 base_dir（valid）
    """
    q = item.get("query") or item.get("question") or item.get("input") or ""

    paths = _parse_maybe_list(item.get("file_name"))
    if not paths:
        return q

    # 从 input_jsonl_path 提取 base_dir
    base_dir = Path(input_jsonl_path).resolve().parent   # 取出 ./valid

    # 相对路径 -> 绝对路径
    def to_abs(p: str) -> str:
        return str(base_dir / p.lstrip('./')) if not os.path.isabs(p) else p

    q += f"\nFile_Name: {' '.join(to_abs(p) for p in paths)}"
    return q
def load_jsonl_dataset(file_path):
    """Load dataset from JSONL file"""
    datasets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                # 确保数据结构与原来一致
                dataset_item = {
                    'task_id': data.get('task_id', ''),
                    'question': data.get('query', ''),
                    'file_name': data.get('file_name', ''),
                    'answer': data.get('answer', ''),
                    'steps': data.get('steps', ''),
                    'level': data.get('level', '')
                }
                datasets.append(dataset_item)
    return datasets
from pathlib import Path
import csv, time 
import pandas as pd


def init_files(result_dir, checkpoint_file):
    """初始化存储目录和检查点文件"""
    result_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_file.exists():
        with open(checkpoint_file, 'w', encoding='utf-8-sig', newline='') as f:  # 添加编码参数
            writer = csv.writer(f)
            writer.writerow(['task_id','response'])


def load_processed(checkpoint_file):
    """增强版已处理记录加载"""
    processed = {}
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                unique_key = f"{row['task_id']}"
                processed[unique_key] = row['response']
    return processed


def save_result(question_data, response, result_dir, checkpoint_file, failed_checkpoint_file, is_error=False):
    """原子化保存结果"""
    timestamp = time.strftime("%Y%m%d%H%M%S")
    filename = "errors.parquet" if is_error else "results.parquet"

    # 构建完整数据记录
    record = {
        **question_data.to_dict(),  # 保留原始数据所有字段
        "task_success": True if is_error else False,
        'record_result': response
    }

    # 使用pandas追加模式写入Parquet（比CSV/JSON更高效）
    df = pd.DataFrame([record])
    output_path = result_dir / filename

    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        df = pd.concat([existing_df, df])
    df = df.map(str)
    df.to_parquet(output_path, index=False)

    # 更新检查点文件（新增原始数据索引）
    if not is_error:
        with open(checkpoint_file, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                question_data['task_id'],  # 假设原始数据有唯一标识列
                response
            ])
    else:
        with open(failed_checkpoint_file, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                question_data['task_id'],  # 假设原始数据有唯一标识列
                response
            ])

async def run_batch(
	input_jsonl_path: str, output_jsonl_path: str, enable_mcp: bool = False, validate: bool = False
) -> None:
	"""批量处理任务，支持断点续传"""
	result_dir = Path("./res/")
	checkpoint_file = result_dir / "processed.csv"
	failed_checkpoint_file = result_dir / "failed_processed.csv"
	
	# 初始化目录和文件
	init_files(result_dir, checkpoint_file)
	init_files(result_dir, failed_checkpoint_file)
	processed = load_processed(checkpoint_file)
	logger.info(f"Loaded {len(processed)} already processed tasks from checkpoint.")

	# 加载数据集 - 使用 input_jsonl_path 参数
	jsonl_file_path = input_jsonl_path if input_jsonl_path else "./valid/data.jsonl"
	if not Path(jsonl_file_path).exists():
		logger.error(f"Dataset file not found: {jsonl_file_path}")
		return
	
	try:
		datasets = load_jsonl_dataset(jsonl_file_path)
		logger.info(f"Loaded dataset: {len(datasets)} examples.")
	except Exception as e:
		logger.error(f"Failed to load dataset: {e}")
		return

	# 过滤已处理的任务
	datasets = [
		dataset for dataset in datasets 
		if not (dataset['task_id'] in processed and processed[dataset['task_id']] != '')
	]
	
	if not datasets:
		logger.info("All tasks have been processed. Exiting.")
		return
	
	datasets_df = pd.DataFrame(datasets)
	logger.info(f"To process: {len(datasets)} tasks.")

	correct_count = 0
	total_count = 0

	# 打开 JSONL 文件用于追加写入（每条任务完成后立即保存）
	jsonl_file = open(output_jsonl_path, "w", encoding="utf-8")
	
	try:
		oxy_space = build_oxy_space(enable_mcp=enable_mcp)
		async with MAS(oxy_space=oxy_space) as mas:
			for idx, item in datasets_df.iterrows():
				answer = ""
				result_record = {}
				try:
					question = compose_question(item, input_jsonl_path)
					if not question:
						logger.warning(f"Task {item.get('task_id', 'N/A')} has no question field, skipping.")
						result_record = {"error": "no question field", **item.to_dict()}
						# 立即保存到 JSONL
						jsonl_file.write(json.dumps(result_record, ensure_ascii=False) + "\n")
						jsonl_file.flush()
						continue
					
					# 实际调用 master_agent
					logger.info(f"Processing task {item.get('task_id', 'N/A')}: {question[:50]}...")
					resp = await mas.call(callee="master_agent", arguments={"query": question})
					answer = resp.output if hasattr(resp, "output") else str(resp)
					
					# 保存结果到 checkpoint 和 Parquet
					save_result(item, answer, result_dir, checkpoint_file, failed_checkpoint_file, is_error=False)
					
					# 构建结果记录
					result_record = {"task_id": item.get('task_id'), "question": question, "answer": answer}
					
					# 立即保存到 JSONL 文件
					jsonl_file.write(json.dumps(result_record, ensure_ascii=False) + "\n")
					jsonl_file.flush()
					
					# 验证（如果启用）
					if validate and "answer" in item:
						total_count += 1
						expected_answer = item["answer"]
						if answer == expected_answer:
							correct_count += 1
						else:
							logger.info(f"Task ID: {item.get('task_id', 'N/A')}")
							logger.info(f"  Expected: {expected_answer}")
							logger.info(f"  Got:      {answer}")
					
				except Exception as e:
					logger.error(f"Error processing task {item.get('task_id', 'N/A')}: {e}", exc_info=True)
					save_result(item, str(e), result_dir, checkpoint_file, failed_checkpoint_file, is_error=True)
					result_record = {"task_id": item.get('task_id'), "error": str(e)}
					
					# 立即保存错误结果到 JSONL 文件
					jsonl_file.write(json.dumps(result_record, ensure_ascii=False) + "\n")
					jsonl_file.flush()
	finally:
		jsonl_file.close()

	# 输出验证统计
	logger.info(f"Completed processing. Results saved to {output_jsonl_path}")
	if validate and total_count > 0:
		accuracy = correct_count / total_count
		logger.info(f"\n{'='*50}")
		logger.info(f"Validation Results:")
		logger.info(f"  Correct: {correct_count}/{total_count}")
		logger.info(f"  Accuracy: {accuracy:.4f}")
		logger.info(f"{'='*50}")


async def start_web(first_query: Optional[str] = None, enable_mcp: bool = False) -> None:
	"""启动Web服务"""
	oxy_space = build_oxy_space(enable_mcp=enable_mcp)
	async with MAS(oxy_space=oxy_space) as mas:
		await mas.start_web_service(
			first_query=first_query,
			welcome_message="Hi, I'm OxyGent GAIA Runner. How can I assist you?",
		)


def main():
	# 配置日志
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S'
	)
	
	parser = argparse.ArgumentParser(description="OxyGent GAIA Runner (Simplified)")
	parser.add_argument("--question", type=str, help="Run a single question.")
	parser.add_argument("--input_jsonl", type=str, help="Path to input JSONL file for batch processing.")
	parser.add_argument("--output", type=str, default="results.jsonl", help="Path to output JSONL file.")
	parser.add_argument("--web", action="store_true", help="Start in web service mode.")
	parser.add_argument("--first_query", type=str, help="Initial query for web service mode.")
	parser.add_argument("--app_name", type=str, default="app", help="Set application name for experiment isolation.")
	parser.add_argument("--validate", action="store_true", help="Validate results against 'answer' field in input_jsonl.")
	parser.add_argument("--enable_mcp", action="store_true", help="Enable MCP tools (requires Node.js and MCP servers).")
	
	args = parser.parse_args()

	Config.set_app_name(args.app_name)

	if args.question:
		answer = asyncio.run(run_single(args.question, enable_mcp=args.enable_mcp))
		print(f"Answer: {answer}")
	elif args.input_jsonl:
		asyncio.run(run_batch(args.input_jsonl, args.output, args.enable_mcp, args.validate))
	elif args.web:
		asyncio.run(start_web(args.first_query, enable_mcp=args.enable_mcp))
	else:
		print("Please provide --question, --input_jsonl, or --web argument.")


if __name__ == "__main__":
	main()