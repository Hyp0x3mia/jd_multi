import asyncio
import argparse
import json
import os
import re
import unicodedata
import time
import requests
from typing import Any, Dict, List, Optional, Union, Literal, Tuple
from pathlib import Path

from oxygent import MAS, Config, oxy, preset_tools
from pydantic import Field
import importlib.util
import logging
Config.set_app_name('1108_v5')

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
    """极简清理：仅去首尾空白与NFKC；不做启发式多行选择。"""
    if not text:
        return ""
    s = str(text).strip()
    return unicodedata.normalize('NFKC', s)


def _extract_final_answer(text: str, question: str = "") -> str:
    """只提取最终答案，忽略所有附加说明。支持定界符、数字、百分比、URL、日期、CSV等类型。"""
    if not text:
        return ""
    
    import re
    s = str(text).strip()
    
    # 1. 先尝试使用确定性流水线（如果问题可用）(保留原代码)
    if question:
        try:
            spec = parse_answer_spec(question)
            extracted = extract_deterministic(spec, s)
            if not isinstance(extracted, dict) or "error" not in extracted:
                normalized = normalize_by_spec(spec, extracted)
                if not isinstance(normalized, dict) or "error" not in normalized:
                    return str(normalized)
        except Exception:
            pass
            
    # 2. 尝试提取定界符内的答案（结构化优先）
    final_answer_match = re.search(r"<[aA]nswer>([\s\S]*?)</[aA]nswer>|<[Ff]inal[Aa]nswer>([\s\S]*?)</[Ff]inal[Aa]nswer>", s)
    if final_answer_match:
        extracted_text = final_answer_match.group(1) or final_answer_match.group(2)
        if extracted_text and extracted_text.strip():
            # 对提取出的内容进行最终清理和 NFKC 归一化
            return _clean_final_answer(extracted_text)

    # 3. 默认返回文本优先 (P1)
    # 将清理后的文本作为基础，用于捕获实体/文本答案 (如 "五星低碳供应商")
    cleaned_s = _clean_final_answer(s)
    
    # --- 通用模式匹配回退（仅在文本中发现更严格的格式时才覆盖 cleaned_s） ---
    
    extracted_candidate = None

    # 3.1. 百分比数字（如 200%）(P2)
    percent_match = re.search(r"(-?\d+(?:\.\d+)?%)", cleaned_s)
    if percent_match:
        extracted_candidate = percent_match.group(1).strip()
    
    # 3.2. URL 提取 (P3)
    elif re.search(r"^https?://", cleaned_s, re.IGNORECASE) or re.search(r"https?://", cleaned_s, re.IGNORECASE):
        # 如果文本以 URL 开头，或者包含 URL，则提取
        url_pattern = r"https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        url_match = re.search(url_pattern, cleaned_s)
        if url_match:
             extracted_candidate = url_match.group(0).strip().rstrip(".,;)")
    
    # 3.3. CSV 类型 (P3)
    elif re.search(r"[,;，；\n]", cleaned_s) and len([p for p in re.split(r"[,;，；\n]", cleaned_s) if p.strip()]) > 1:
        # 提取以通用分隔符分隔的值，仅在存在多项时考虑
        parts = [p.strip() for p in re.split(r"[,;，；\n]", cleaned_s) if p.strip()]
        if len(parts) > 1:
            extracted_candidate = ','.join(parts)

    # 4. 泛化数字/日期提取 (P5 - P6)
    
    # 4.1. 数字（整数、小数、负数）(P5)
    if extracted_candidate is None:
        number_pattern = r"(-?\d+(?:\.\d+)?)"
        all_numbers = list(re.finditer(number_pattern, cleaned_s))
        # 仅在文本较短时进行数字提取，避免从长篇解释中提取数字
        if all_numbers and len(cleaned_s) < 50:
             extracted_candidate = all_numbers[-1].group(1).strip()
        
    # 4.2. 日期提取 (P6)
    if extracted_candidate is None:
        date_pattern = r"(\d{4}[年/-]\d{1,2}[月/-]\d{1,2}[日]?)"
        date_match = re.search(date_pattern, cleaned_s)
        if date_match:
            date_str = date_match.group(1)
            # 标准化日期格式：2025年12月25日 → 2025-12-25
            date_str = re.sub(r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})[日]?", r"\1-\2-\3", date_str)
            extracted_candidate = date_str.strip()

    # 5. 返回最终答案：如果提取到了更严格的类型，则返回它；否则返回最初清理的文本。
    return extracted_candidate if extracted_candidate else cleaned_s


def _extract_final_answer_tag(text: str) -> str:
	"""提取 <FINAL_ANSWER> 标签内的内容，如果不存在标签则返回原文本"""
	if not text:
		return text
	
	import re
	# 记录输入文本（用于调试）
	input_preview = text[:100] if len(text) > 100 else text
	logger.info(f"_extract_final_answer_tag called with input (full): {repr(text)}")
	logger.info(f"_extract_final_answer_tag input length: {len(text)}")
	
	# 方法1：使用正则表达式（支持所有大小写组合和分隔符）
	patterns = [
		(r"<final[_\s-]?answer>([\s\S]*?)</final[_\s-]?answer>", "FINAL_ANSWER/FinalAnswer/final_answer"),
		(r"<answer>([\s\S]*?)</answer>", "Answer/ANSWER"),
	]
	
	for pattern, desc in patterns:
		logger.debug(f"Trying pattern: {pattern} ({desc})")
		match = re.search(pattern, text, re.IGNORECASE)
		if match:
			extracted = match.group(1).strip()
			logger.info(f"✅ Extracted from tag pattern '{desc}': '{extracted[:50]}...' (original length: {len(text)}, extracted length: {len(extracted)})")
			return extracted
		else:
			logger.debug(f"Pattern '{desc}' did not match")
	
	# 方法2：如果正则表达式失败，尝试简单的字符串查找（更健壮）
	# 查找 <FINAL_ANSWER> 或 <FinalAnswer> 等变体（不区分大小写）
	text_lower = text.lower()
	start_markers = ["<final_answer>", "<finalanswer>", "<final-answer>", "<answer>"]
	end_markers = ["</final_answer>", "</finalanswer>", "</final-answer>", "</answer>"]
	
	for start_marker, end_marker in zip(start_markers, end_markers):
		start_idx = text_lower.find(start_marker)
		if start_idx != -1:
			# 找到开始标签，查找对应的结束标签
			end_idx = text_lower.find(end_marker, start_idx + len(start_marker))
			if end_idx != -1:
				# 提取标签内的内容
				extracted = text[start_idx + len(start_marker):end_idx].strip()
				logger.info(f"✅ Extracted using string search '{start_marker}...{end_marker}': '{extracted[:50]}...'")
				return extracted
	
	# 如果没有找到标签，检查是否包含类似标签的文本（用于调试）
	if "<" in text and ">" in text:
		logger.warning(f"⚠️  Text contains '<' and '>' but no matching tag pattern found. Text preview: {input_preview}...")
	
	# 如果没有找到标签，返回原文本
	logger.debug(f"No tag found, returning original text (length: {len(text)})")
	return text


def _normalize_answer(text: str, question: str = "") -> str:
    """强规范化答案：只保留最终答案，不做过多解释。支持多种类型。"""
    if not text:
        return ""
    
    import re
    
    # 先用提取器得到候选答案（不再在此重复提取，仅做标准化）
    cleaned_answer = _extract_final_answer(text, question)
    
    # 对不同类型的答案进行标准化（仅格式修正，不做二次搜索）
    # 1. 百分比：确保格式正确（如 200%）
    if '%' in cleaned_answer:
        cleaned_answer = cleaned_answer.strip()
        # 确保百分比符号正确
        if not cleaned_answer.endswith('%'):
            # 如果数字后面有%但格式不对，修正
            cleaned_answer = re.sub(r"(\d+(?:\.\d+)?)\s*%?", r"\1%", cleaned_answer)
    
    # 2. 纯数字：移除多余空格，处理负数和小数
    elif re.match(r"^-?\d+(?:\.\d+)?$", cleaned_answer):
        cleaned_answer = cleaned_answer.strip()
        # 如果是整数但写成小数形式（如 12.0），转为整数
        try:
            f = float(cleaned_answer)
            if f.is_integer():
                cleaned_answer = str(int(f))
            else:
                # 去除无意义的尾零
                cleaned_answer = ("%f" % f).rstrip("0").rstrip(".")
        except:
            pass
    
    # 3. 日期：标准化为 YYYY-MM-DD 格式（支持 YYYY-M-D 与 含中文分隔）
    if (
        ('年' in cleaned_answer or '月' in cleaned_answer or '日' in cleaned_answer) or
        re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", cleaned_answer)
    ):
        # 统一替换为连字符
        cleaned_answer = re.sub(r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})[日]?", r"\1-\2-\3", cleaned_answer)
        # 确保月份和日期是两位数
        parts = cleaned_answer.split('-')
        if len(parts) == 3 and all(parts) and parts[0].isdigit():
            parts[1] = parts[1].zfill(2)
            parts[2] = parts[2].zfill(2)
            cleaned_answer = '-'.join(parts)
    
    # 4. URL：清理链接
    if cleaned_answer.startswith("http"):
        cleaned_answer = cleaned_answer.strip()
        # 移除尾部标点
        cleaned_answer = cleaned_answer.rstrip(".,;)")
    
    # 5. CSV：确保格式正确（逗号分隔，无多余空格）
    if ',' in cleaned_answer and len(cleaned_answer.split(',')) > 1:
        parts = [p.strip() for p in cleaned_answer.split(',') if p.strip()]
        cleaned_answer = ','.join(parts)
    
    # 返回最终答案（去除多余空格）
    return cleaned_answer.strip()


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


# ==================== GitHub API Tools ====================

def _get_github_headers():
	"""获取 GitHub API 请求头"""
	headers = {"Accept": "application/vnd.github.v3+json"}
	token = os.getenv('GITHUB_TOKEN')
	if token:
		headers["Authorization"] = f"Bearer {token}"
		logger.debug("GitHub token found in environment")
	else:
		logger.warning("GITHUB_TOKEN not found in environment - API calls may be rate-limited")
	return headers


def _reduce_keywords_by_priority(keywords: List[str], priority_remove: List[str]) -> List[str]:
	"""
	按优先级减少关键词列表
	
	Args:
		keywords: 关键词列表
		priority_remove: 按优先级排序的移除关键词列表（优先级高的在前）
	
	Returns:
		减少后的关键词列表
	"""
	if not keywords:
		return []
	
	import re
	
	# 创建关键词的优先级分数（分数越低越优先移除）
	keyword_scores = {}
	for i, keyword in enumerate(keywords):
		score = 100  # 默认分数
		
		# 检查是否是优先级移除列表中的关键词
		for priority_word in priority_remove:
			if priority_word.lower() in keyword.lower():
				# 优先级越高（在列表中越靠前），分数越低
				score = priority_remove.index(priority_word)
				break
		
		# 如果是 issue 编号，分数最高（最不应该移除）
		if re.match(r'^#?\d+$', keyword):
			score = 999
		
		keyword_scores[keyword] = score
	
	# 按分数排序，分数低的优先移除
	sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1])
	
	# 移除分数最低的一个关键词
	if len(sorted_keywords) > 1:
		# 移除分数最低的关键词
		removed_keyword = sorted_keywords[0][0]
		result = [k for k in keywords if k != removed_keyword]
		logger.debug(f"Removed keyword '{removed_keyword}' (priority: {sorted_keywords[0][1]})")
		return result
	
	return keywords


def _normalize_github_query(query: str, reduce_keywords: bool = False) -> str:
	"""
	规范化 GitHub 搜索查询，提取关键信息，避免关键词过多
	
	策略：
	1. 优先提取 issue 编号（如 #40054 或 40054）
	2. 如果包含编号，只返回编号部分（GitHub API 支持直接搜索编号）
	3. 如果没有编号，提取核心关键词（移除常见停用词）
	4. 限制关键词数量（最多 2-3 个），避免过度限制导致结果过少
	5. 如果 reduce_keywords=True，按优先级减少关键词（优先移除日期等不确定性高的）
	
	Args:
		query: 原始查询字符串
		reduce_keywords: 是否减少关键词（用于重试时减少关键词）
	"""
	if not query:
		return ""
	
	import re
	
	# 1. 提取 issue 编号（最高优先级）
	# 匹配 #40054 或 issue #40054 或 issue 40054 等格式
	# 匹配 4 位以上的数字（通常 issue 编号至少 4 位）
	issue_number_match = re.search(r'#?(\d{4,})', query)
	if issue_number_match:
		issue_number = issue_number_match.group(1)
		# 如果找到编号，直接返回编号（GitHub API 支持直接搜索编号）
		return f"#{issue_number}"
	
	# 2. 如果没有编号，提取核心关键词
	# 保留中文字符、英文字母、数字和 # 符号
	# 移除常见停用词
	stop_words = {
		'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
		'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
		'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
		'could', 'should', 'may', 'might', 'must', 'can', 'about', 'into',
		'through', 'during', 'including', 'against', 'among', 'throughout',
		'despite', 'towards', 'upon', 'concerning',
		# 注意：'bug', 'feature', 'fix' 是重要的搜索关键词，不应该被过滤
		'github', 'issue', 'pr', 'pull', 'request',
		'related', 'concerning', 'search', 'find',
		'look', 'for', 'get', 'show', 'list', 'all', 'open', 'closed'
	}
	
	# 清理查询：保留中文字符、英文字母、数字、# 和空格
	# 移除其他标点符号
	cleaned = re.sub(r'[^\w\s#\u4e00-\u9fa5]', ' ', query)
	words = cleaned.split()
	
	# 过滤停用词，保留有意义的关键词
	# 保留：长度 > 2 的英文单词，或包含中文的词语，或数字
	# 注意：对于 GitHub Search API，通常将关键词转为小写更安全（虽然 API 不区分大小写，但小写更兼容）
	keywords = []
	for w in words:
		w_lower = w.lower()
		if not w:
			continue
		# 保留中文词语（保持原样）
		if re.search(r'[\u4e00-\u9fa5]', w):
			keywords.append(w)
		# 保留不是停用词且长度 > 2 的英文单词（转为小写）
		elif w_lower not in stop_words and len(w) > 2:
			keywords.append(w_lower)  # 转为小写，提高兼容性
		# 保留数字
		elif w.isdigit():
			keywords.append(w)
	
	# 如果需要减少关键词，按优先级移除
	if reduce_keywords and len(keywords) > 1:
		# 定义优先级移除列表（优先级高的在前，优先移除）
		priority_remove = [
			# 日期时间相关（最高优先级移除，不确定性高）
			'2024', '2025', '2023', '2022', '2021', '2020',
			'january', 'february', 'march', 'april', 'may', 'june',
			'july', 'august', 'september', 'october', 'november', 'december',
			'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
			'月', '年', '日', '天', '周', '星期',
			'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
			'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
			'today', 'yesterday', 'tomorrow', 'now', 'current',
			'今天', '昨天', '明天', '现在', '当前',
			# 时间相关
			'hour', 'minute', 'second', 'time', 'when', 'during',
			'小时', '分钟', '秒', '时间', '何时', '期间',
			# 其他不确定性高的词
			'about', 'related', 'concerning', 'regarding',
			'关于', '相关', '有关', '涉及',
		]
		
		keywords = _reduce_keywords_by_priority(keywords, priority_remove)
	
	# 限制关键词数量（最多 4 个核心关键词，避免过度限制导致结果过少）
	# 从 2 个增加到 4 个，因为像 "whisper audio bug" 这样的常见查询需要多个关键词
	if not reduce_keywords and len(keywords) > 4:
		# 优先保留包含中文的词语，然后按长度排序
		chinese_keywords = [k for k in keywords if re.search(r'[\u4e00-\u9fa5]', k)]
		english_keywords = [k for k in keywords if k not in chinese_keywords]
		
		# 保留中文关键词 + 最长的英文关键词（最多 4 个）
		if chinese_keywords:
			keywords = chinese_keywords[:2] + sorted(english_keywords, key=len, reverse=True)[:2]
		else:
			keywords = sorted(english_keywords, key=len, reverse=True)[:4]
	
	# 返回清理后的查询（用空格连接）
	result = " ".join(keywords) if keywords else ""
	return result


def _execute_github_search(
	headers: Dict,
	query_terms: List[str],
	max_results: int
) -> List[Dict]:
	"""
	执行 GitHub API 搜索请求
	
	Args:
		headers: HTTP 请求头
		query_terms: 查询条件列表
		max_results: 最大结果数
	
	Returns:
		搜索结果列表
	"""
	# GitHub Search API 查询格式：repo:owner/name is:issue|is:pull-request 关键词1 关键词2
	# 多个关键词用空格分隔，requests 会自动进行 URL 编码
	search_query = " ".join(query_terms)
	logger.info(f"Executing GitHub search query: {search_query}")
	logger.info(f"Query terms breakdown: {query_terms}")
	
	all_items = []
	per_page = min(100, max_results)
	
	try:
		# GitHub Search API 不支持 page 参数，只返回第一页的结果（最多 1000 个）
		# 但我们可以通过调整 per_page 来获取更多结果
		# requests 库会自动对 params 进行 URL 编码，所以 "whisper audio bug" 会被编码为 "whisper%20audio%20bug"
		params = {
			"q": search_query,
			"per_page": per_page,
			"sort": "updated",
			"order": "desc"
		}
		
		logger.debug(f"Request params: {params}")
		logger.info(f"Equivalent web URL format: https://github.com/{query_terms[0].replace('repo:', '')}/issues?q={'%20'.join(query_terms[1:]) if len(query_terms) > 1 else ''}")
		
		# 构建完整的请求 URL（用于调试）
		url = "https://api.github.com/search/issues"
		response = requests.get(
			url,
			headers=headers,
			params=params,
			timeout=30
		)
		
		# 记录完整的请求 URL（用于调试）
		request_url = response.url if hasattr(response, 'url') else url
		logger.debug(f"Full request URL: {request_url}")
		
		# 记录响应状态和详细信息
		logger.debug(f"GitHub API response status: {response.status_code}")
		logger.debug(f"GitHub API response headers: {dict(response.headers)}")
		
		# 处理速率限制
		if response.status_code == 403:
			reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
			sleep_time = max(reset_time - time.time(), 0) + 5
			logger.warning(f"GitHub rate limit reached, sleeping {sleep_time:.1f} seconds")
			time.sleep(sleep_time)
			# 重试一次
			response = requests.get(
				"https://api.github.com/search/issues",
				headers=headers,
				params=params,
				timeout=30
			)
		
		# 处理其他错误状态码
		if response.status_code != 200:
			error_msg = f"GitHub API returned status {response.status_code}"
			try:
				error_data = response.json()
				if "message" in error_data:
					error_msg += f": {error_data['message']}"
				if "errors" in error_data:
					error_msg += f" - Errors: {error_data['errors']}"
				logger.error(error_msg)
				logger.error(f"Response body: {response.text[:500]}")
			except:
				logger.error(f"{error_msg}. Response: {response.text[:500]}")
			return []
		
		response.raise_for_status()
		data = response.json()
		
		# 记录搜索结果统计
		total_count = data.get("total_count", 0)
		incomplete_results = data.get("incomplete_results", False)
		items = data.get("items", [])
		
		logger.info(f"GitHub search returned {len(items)} items (total: {total_count}, incomplete: {incomplete_results})")
		
		if incomplete_results:
			logger.warning("GitHub search results are incomplete (may have timed out)")
		
		# 返回请求的结果数量
		all_items = items[:max_results]
		
		if not all_items:
			logger.warning(f"No items found for query: {search_query}")
			logger.warning(f"Query terms: {query_terms}")
			logger.warning(f"Total count from API: {total_count}")
			# 如果 total_count > 0 但 items 为空，可能是数据格式问题
			if total_count > 0:
				logger.error(f"API reports {total_count} items but returned empty list - possible API issue")
				logger.debug(f"Full API response: {json.dumps(data, indent=2)[:1000]}")
		else:
			# 记录前几个结果的标题，用于验证
			result_titles = [item.get("title", "N/A")[:50] for item in all_items[:3]]
			logger.info(f"Sample result titles: {result_titles}")
		
		# 确保返回的数据是列表格式，且每个项目都是字典
		if all_items and isinstance(all_items[0], dict):
			logger.debug(f"Returning {len(all_items)} items with correct format")
		else:
			logger.error(f"Unexpected data format: {type(all_items[0]) if all_items else 'empty'}")
		
		return all_items
		
	except requests.RequestException as e:
		logger.error(f"GitHub search request failed: {e}")
		logger.error(f"Query: {search_query}")
		return []
	except Exception as e:
		logger.error(f"GitHub search unexpected error: {e}", exc_info=True)
		logger.error(f"Query: {search_query}")
		return []


def _search_github_issues(
	repo: str,
	query: str = "",
	state: str = "all",
	issue_type: str = "all",
	max_results: int = 10
) -> List[Dict]:
	"""
	使用 GitHub Search API 搜索 issues 和 PRs
	
	如果第一次搜索没有结果，会自动减少关键词（优先移除日期等不确定性高的词）并重试。
	
	Args:
		repo: Repository in owner/repo format
		query: Search query string (e.g., "#40054", "bug", "audio whisper")
		state: Issue state (open/closed/all)
		issue_type: Type (issue/pr/all)
		max_results: Maximum number of results to return
	
	Returns:
		List of issue/PR dictionaries
	"""
	headers = _get_github_headers()
	
	# 构建基础查询条件
	# 注意：GitHub Search API 不支持 state:all，只支持 state:open 和 state:closed
	# 如果要搜索所有状态，不添加 state 条件即可（默认会搜索所有状态）
	base_query_terms = [f"repo:{repo}"]
	
	# 只有当 state 不是 "all" 时才添加 state 条件
	if state != "all":
		base_query_terms.append(f"state:{state}")
	
	if issue_type == "pr":
		base_query_terms.append("is:pull-request")
	elif issue_type == "issue":
		base_query_terms.append("is:issue")
	else:
		# GitHub Search API 要求必须包含 is:issue 或 is:pull-request；
		# 默认按 issue 搜索（常见需求），确保不触发 422。
		base_query_terms.append("is:issue")
	
	# 第一次尝试：使用规范化后的完整查询
	if query:
		normalized_query = _normalize_github_query(query, reduce_keywords=False)
		if normalized_query:
			query_terms = base_query_terms + [normalized_query]
			logger.debug(f"Initial normalized GitHub query: '{query}' -> '{normalized_query}'")
		else:
			query_terms = base_query_terms
			logger.debug(f"GitHub query normalized to empty: '{query}'")
	else:
		query_terms = base_query_terms
	
	# 第一次搜索
	results = _execute_github_search(headers, query_terms, max_results)
	
	# 如果第一次搜索没有结果，且查询包含关键词，尝试减少关键词并重试
	if not results and query:
		original_normalized = _normalize_github_query(query, reduce_keywords=False)
		if original_normalized and len(original_normalized.split()) > 1:
			logger.info(f"No results found with query '{original_normalized}', trying to reduce keywords...")
			
			# 提取关键词列表（用于逐步减少）
			import re
			stop_words = {
				'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
				'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
				'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
				'could', 'should', 'may', 'might', 'must', 'can', 'about', 'into',
				'through', 'during', 'including', 'against', 'among', 'throughout',
				'despite', 'towards', 'upon', 'concerning',
				# 注意：'bug', 'feature', 'fix' 是重要的搜索关键词，不应该被过滤
				'github', 'issue', 'pr', 'pull', 'request',
				'related', 'concerning', 'search', 'find',
				'look', 'for', 'get', 'show', 'list', 'all', 'open', 'closed'
			}
			
			cleaned = re.sub(r'[^\w\s#\u4e00-\u9fa5]', ' ', query)
			words = cleaned.split()
			keywords = []
			for w in words:
				w_lower = w.lower()
				if not w:
					continue
				if re.search(r'[\u4e00-\u9fa5]', w):
					keywords.append(w)
				elif w_lower not in stop_words and len(w) > 2:
					keywords.append(w_lower)  # 转为小写，与主函数保持一致
				elif w.isdigit():
					keywords.append(w)
			
			# 限制初始关键词数量（最多 4 个，与主函数保持一致）
			if len(keywords) > 4:
				chinese_keywords = [k for k in keywords if re.search(r'[\u4e00-\u9fa5]', k)]
				english_keywords = [k for k in keywords if k not in chinese_keywords]
				if chinese_keywords:
					keywords = chinese_keywords[:2] + sorted(english_keywords, key=len, reverse=True)[:2]
				else:
					keywords = sorted(english_keywords, key=len, reverse=True)[:4]
			
			# 优先级移除列表
			priority_remove = [
				'2024', '2025', '2023', '2022', '2021', '2020',
				'january', 'february', 'march', 'april', 'may', 'june',
				'july', 'august', 'september', 'october', 'november', 'december',
				'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
				'月', '年', '日', '天', '周', '星期',
				'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
				'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
				'today', 'yesterday', 'tomorrow', 'now', 'current',
				'今天', '昨天', '明天', '现在', '当前',
				'hour', 'minute', 'second', 'time', 'when', 'during',
				'小时', '分钟', '秒', '时间', '何时', '期间',
				'about', 'related', 'concerning', 'regarding',
				'关于', '相关', '有关', '涉及',
			]
			
			# 逐步减少关键词，最多重试 3 次
			current_keywords = keywords.copy()
			for attempt in range(3):
				if len(current_keywords) <= 1:
					logger.debug(f"Only 1 or 0 keywords left, cannot reduce further")
					break
				
				# 减少关键词
				current_keywords = _reduce_keywords_by_priority(current_keywords, priority_remove)
				
				if not current_keywords:
					logger.debug(f"No keywords left after reduction, stopping retry")
					break
				
				# 构建新的查询
				reduced_query = " ".join(current_keywords)
				query_terms = base_query_terms + [reduced_query]
				logger.info(f"Retry attempt {attempt + 1}: using reduced query '{reduced_query}' (keywords: {current_keywords})")
				
				# 重试搜索
				results = _execute_github_search(headers, query_terms, max_results)
				
				if results:
					logger.info(f"Found {len(results)} results with reduced query '{reduced_query}'")
					break
	
	return results


def _get_github_issue_by_number(
	repo: str,
	issue_number: int
) -> Optional[Dict]:
	"""
	通过 issue 编号获取详细信息
	
	Args:
		repo: Repository in owner/repo format
		issue_number: Issue/PR number
	
	Returns:
		Issue/PR dictionary or None
	"""
	headers = _get_github_headers()
	
	try:
		url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
		response = requests.get(url, headers=headers, timeout=30)
		
		if response.status_code == 403:
			reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
			sleep_time = max(reset_time - time.time(), 0) + 5
			logger.warning(f"GitHub rate limit reached, sleeping {sleep_time:.1f} seconds")
			time.sleep(sleep_time)
			response = requests.get(url, headers=headers, timeout=30)
		
		response.raise_for_status()
		return response.json()
		
	except requests.RequestException as e:
		logger.error(f"Failed to get GitHub issue #{issue_number}: {e}")
		return None


# 创建新版 GitHub API 工具（稳定方案）
github_api_tools = oxy.FunctionHub(name="github_api_tools", timeout=120)


@github_api_tools.tool(description="Github.SearchIssues: Search issues/PRs via REST API. Use 'query' for search keywords (e.g., 'whisper audio bug'), 'repo' for repository, 'created' for date range, 'state' for open/closed/all, 'max_results' for result limit.")
def github_search_issues(
    repo: str = Field("", description="Repository in owner/repo format (e.g., 'huggingface/transformers')"),
    query: str = Field("", description="Search query keywords (e.g., 'whisper audio bug', '#40054'). This is the main parameter for search terms."),
    labels: str = Field("", description="Comma-separated labels (OR semantics)"),
    milestone: str = Field("", description="Milestone title or number"),
    created: str = Field("", description="Created date range, e.g., '2025-08-08..2025-08-08'"),
    updated: str = Field("", description="Updated date range, e.g., '2025-01-01..2025-12-31'"),
    in_: str = Field("", description="Search fields, e.g., 'title,body'"),
    state: Literal["open", "closed", "all"] = Field("all", description="Issue state: open, closed, or all"),
    q_extra: str = Field("", description="[DEPRECATED] Use 'query' instead. Extra free-text query qualifiers."),
    per_page: int = Field(10, description="Results per page (<=100)"),
    max_results: int = Field(10, description="Maximum number of results to return (alias for per_page)"),
    page: int = Field(1, description="Page number")
) -> List[Dict]:
    """
    Build GitHub search query and call /search/issues. Fallback: if only repo provided with no text, add 'is:issue'.
    Returns a list of structured items with minimal fields for downstream use.
    """
    headers = _get_github_headers()
    # Use max_results if provided (and different from default), otherwise use per_page
    # If max_results is explicitly set (not default 10), use it; otherwise use per_page
    if max_results != 10:  # If max_results was explicitly set (not default)
        per_page = max(1, min(100, max_results))
    else:
        per_page = max(1, min(100, per_page))
    
    terms: List[str] = []
    if repo:
        terms.append(f"repo:{repo}")
    # state qualifier (skip if all)
    if state and state.lower() != "all":
        terms.append(f"state:{state.lower()}")
    # labels (OR semantics via multiple label: qualifiers)
    if labels:
        for lb in [s.strip() for s in labels.split(',') if s.strip()]:
            terms.append(f"label:{lb}")
    if milestone:
        terms.append(f"milestone:{milestone}")
    if created:
        terms.append(f"created:{created}")
    if updated:
        terms.append(f"updated:{updated}")
    if in_:
        # support comma list
        for field in [s.strip() for s in in_.split(',') if s.strip()]:
            terms.append(f"in:{field}")
    # Use 'query' parameter first, fallback to 'q_extra' for backward compatibility
    search_text = query.strip() if query else (q_extra.strip() if q_extra else "")
    if search_text:
        terms.append(search_text)
    # API requires is:issue|is:pull-request when no free-text sometimes; add is:issue by default
    if not any(t.startswith("is:") for t in terms):
        terms.append("is:issue")

    search_query = " ".join(terms)
    logger.info(f"Github.SearchIssues → q={search_query}")
    params = {"q": search_query, "per_page": per_page, "page": page, "sort": "updated", "order": "desc"}
    try:
        resp = requests.get("https://api.github.com/search/issues", headers=headers, params=params, timeout=30)
        if resp.status_code == 403:
            logger.warning("Rate limit hit for GitHub API search")
        if resp.status_code != 200:
            logger.error(f"Github.SearchIssues HTTP {resp.status_code}: {resp.text[:300]}")
            return []
        data = resp.json()
        items = data.get("items", [])
        results = []
        for it in items:
            results.append({
                "number": it.get("number"),
                "title": it.get("title"),
                "state": it.get("state"),
                "html_url": it.get("html_url"),
                "created_at": it.get("created_at"),
                "updated_at": it.get("updated_at"),
                "labels": [lb.get("name") for lb in it.get("labels", [])],
                "repository_url": it.get("repository_url"),
            })
        # evidence log
        logger.info(f"Github.SearchIssues results: {[r.get('number') for r in results[:5]]}")
        return results
    except Exception as e:
        logger.error(f"Github.SearchIssues error: {e}")
        return []


@github_api_tools.tool(description="Github.GetIssue: Get full issue/PR by number in a repo")
def github_get_issue(
    repo: str = Field(..., description="owner/repo"),
    number: int = Field(..., description="issue or PR number")
) -> Dict:
    headers = _get_github_headers()
    url = f"https://api.github.com/repos/{repo}/issues/{number}"
    logger.info(f"Github.GetIssue → {url}")
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            logger.error(f"Github.GetIssue HTTP {resp.status_code}: {resp.text[:300]}")
            return {"error": f"HTTP {resp.status_code}"}
        it = resp.json()
        return {
            "number": it.get("number"),
            "title": it.get("title"),
            "state": it.get("state"),
            "html_url": it.get("html_url"),
            "created_at": it.get("created_at"),
            "updated_at": it.get("updated_at"),
            "body": it.get("body", ""),
            "labels": [lb.get("name") for lb in it.get("labels", [])],
            "assignees": [u.get("login") for u in it.get("assignees", [])],
        }
    except Exception as e:
        logger.error(f"Github.GetIssue error: {e}")
        return {"error": str(e)}


# Guardrails: GithubQueryBuilder
github_guardrails = oxy.FunctionHub(name="github_guardrails", timeout=60)


@github_guardrails.tool(description="Guardrails.GithubQueryBuilder: turn user intent into API params and plan")
def github_query_builder(userIntent: str = Field(..., description="natural language intent")) -> Dict:
    """A lightweight heuristic planner that extracts repo, labels, state, in-fields and keywords.
    Returns plan dict with {repo, state, labels, in, q_extra, notes}.
    """
    intent = (userIntent or "").strip()
    repo = ""
    import re
    m = re.search(r"([\w-]+/[\w.-]+)", intent)
    if m:
        repo = m.group(1)
    # labels like label:bug or 关键词 bug
    labels = []
    for lb in re.findall(r"label:([\w-]+)", intent, flags=re.I):
        labels.append(lb)
    # if contains words bug/feature/regression add as label candidates (best-effort)
    for kw, tag in [("bug", "bug"), ("regression", "regression"), ("feature", "feature")]:
        if re.search(rf"\b{kw}\b", intent, flags=re.I):
            if tag not in labels:
                labels.append(tag)
    state = "all"
    if re.search(r"\bclosed\b", intent, flags=re.I):
        state = "closed"
    elif re.search(r"\bopen\b", intent, flags=re.I):
        state = "open"
    in_fields = []
    if re.search(r"title|标题", intent):
        in_fields.append("title")
    if re.search(r"body|正文|内容", intent):
        in_fields.append("body")
    if not in_fields:
        in_fields = ["title"]
    # q_extra: remaining words without repo/label/state tokens
    words = re.sub(r"([\w-]+/[\w.-]+)|label:[\w-]+|\b(open|closed|all)\b", " ", intent, flags=re.I)
    q_extra = " ".join(w for w in words.split() if w)
    plan = {
        "repo": repo,
        "state": state,
        "labels": ",".join(labels),
        "in": ",".join(in_fields),
        "q_extra": q_extra,
        "notes": "Plan built by guardrails to feed Github.SearchIssues."
    }
    logger.info(f"GithubQueryBuilder plan: {plan}")
    return plan


# Sanity tools
sanity_tools = oxy.FunctionHub(name="sanity_tools", timeout=30)


@sanity_tools.tool(description="Sanity.ValidateAnswer: validate and normalize output by schema")
def sanity_validate_answer(answer: str = Field(..., description="raw answer"), schema: str = Field("text", description="one of: url, number, filename, text")) -> Dict:
    import re
    a = (answer or "").strip()
    ok = True
    norm = a
    if schema == "url":
        ok = bool(re.match(r"^https?://[\w\-.:/?#%&=+@~]+$", a))
    elif schema == "number":
        m = re.match(r"^\s*([+-]?(?:\d+\.?\d*|\d*\.?\d+))\s*$", a)
        ok = bool(m)
        if m:
            norm = m.group(1)
    elif schema == "filename":
        ok = bool(re.match(r"^[\w\-_.]+(\.[\w\-_.]+)?$", a))
    return {"ok": ok, "normalized": norm}


@github_api_tools.tool(description="Search GitHub issues and PRs by repository, query, state, and type")
def search_github_issues(
	repo: str = Field(..., description="Repository name in owner/repo format (e.g., 'huggingface/transformers')"),
	query: str = Field("", description="Search query string (e.g., '#40054', 'audio whisper', 'bug')"),
	state: Literal["open", "closed", "all"] = Field("all", description="Issue state"),
	issue_type: Literal["issue", "pr", "all"] = Field("all", description="Type of items"),
	max_results: int = Field(10, description="Maximum number of results to return", ge=1, le=100)
) -> List[Dict]:
	"""
	Search GitHub issues and PRs using GitHub Search API.
	
	This tool is particularly useful for finding specific issues by number (e.g., '#40054'),
	searching by keywords, or filtering by state and type.
	
	Examples:
		- search_github_issues(repo="huggingface/transformers", query="#40054", state="all")
		- search_github_issues(repo="pytorch/pytorch", query="audio whisper", state="closed", issue_type="issue")
	"""
	logger.info(f"search_github_issues called with: repo={repo}, query={query}, state={state}, issue_type={issue_type}, max_results={max_results}")
	
	try:
		results = _search_github_issues(
			repo=repo,
			query=query,
			state=state,
			issue_type=issue_type,
			max_results=max_results
		)
		
		logger.info(f"search_github_issues returning {len(results)} results")
		if results:
			logger.debug(f"First result sample: {json.dumps(results[0], indent=2)[:500]}")
		
		# 确保返回的是可序列化的列表
		return results if results else []
		
	except Exception as e:
		logger.error(f"Error in search_github_issues: {e}", exc_info=True)
		return []


@github_api_tools.tool(description="Get detailed information about a specific GitHub issue or PR by number")
def get_github_issue(
	repo: str = Field(..., description="Repository name in owner/repo format"),
	issue_number: int = Field(..., description="Issue or PR number")
) -> Dict:
	"""
	Get detailed information about a specific GitHub issue or PR by its number.
	
	This tool retrieves full details including title, body, comments, labels, state, etc.
	
	Example:
		- get_github_issue(repo="huggingface/transformers", issue_number=40054)
	"""
	result = _get_github_issue_by_number(repo=repo, issue_number=issue_number)
	if result is None:
		return {"error": f"Issue #{issue_number} not found or API request failed"}
	return result


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
        name='Reasoning Model',
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
		timeout = 600
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
	
	# 禁用 GitHub 搜索工具（按用户要求）
	# try:
	# 	oxy_space.append(github_tools)
	# except Exception as e:
	# 	logger.warning(f"Could not add github_tools: {e}")

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

	# 注册 video_mcp
	video_mcp = oxy.StdioMCPClient(
		name="video_tools",
		params={
			"command": "uv",
			"args": ["--directory", "./mcp_servers", "run", "video/server.py"],
		},
	)	
	oxy_space.append(video_mcp)

	# 注册 Case Bank MCP
	try:
		case_bank_mcp = oxy.StdioMCPClient(
			name="mcp-case-bank",
			params={
				"command": "uv",
				"args": ["--directory", "./mcp_servers/mcp-case-bank", "run", "server.py"],
			},
		)
		oxy_space.append(case_bank_mcp)
	except Exception as e:
		print(f"Warning: Could not add mcp-case-bank: {e}")

	# 注册 To-Do MCP
	try:
		todo_mcp = oxy.StdioMCPClient(
			name="mcp-todo",
			params={
				"command": "uv",
				"args": ["--directory", "./mcp_servers/mcp-todo", "run", "server.py"],
			},
		)
		oxy_space.append(todo_mcp)
	except Exception as e:
		print(f"Warning: Could not add mcp-todo: {e}")

	# 注册新版 GitHub 稳定 API 工具与 Guardrails/Sanity（供 web_agent 优先使用）
	oxy_space.append(github_api_tools)
	oxy_space.append(github_guardrails)
	oxy_space.append(sanity_tools)


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
			llm_model="Reasoning Model",
			additional_prompt=(
				"ROLE: Math/Logic Specialist\n"
				"CAPABILITIES: Arithmetic, combinatorics, averages, date arithmetics, small enumeration, information theory, binary encoding, optimization problems, constraint satisfaction using Python.\n"
				"\n"
				"PROBLEM-SOLVING METHODOLOGY:\n"
				"1) UNDERSTAND CONSTRAINTS FIRST:\n"
				"   - Read the problem carefully and identify ALL constraints (time limits, resource limits, detection rules, etc.).\n"
				"   - For problems involving multiple rounds/steps, identify what can be done in each round and what information is available.\n"
				"   - Example: '48小时内用最少的兔子找出毒桶，无法区分中毒先后，且第二次实验需等待24小时' → \n"
				"     * Round 1: 0-24h (给兔子饮水), 24h (观察结果)\n"
				"     * Round 2: 24-48h (根据Round1结果给兔子饮水), 48h (观察结果)\n"
				"     * Constraint: Cannot distinguish which round caused death\n"
				"\n"
				"2) INFORMATION THEORY APPROACH:\n"
				"   - For detection/identification problems: Calculate the information capacity needed.\n"
				"   - If N buckets and need to identify 1, you need at least log2(N) bits of information.\n"
				"   - Each rabbit can provide 1 bit per round (dead/alive), but with constraints, you may need more.\n"
				"   - Use binary encoding: Assign each bucket a unique binary code, assign rabbits to bit positions.\n"
				"   - For single-round problems: N buckets need ⌈log2(N)⌉ rabbits minimum.\n"
				"   - For multi-round problems: Consider whether additional rounds allow optimization below the single-round bound.\n"
				"   - IMPORTANT: Always verify your solution can actually identify all possible cases - don't just rely on theoretical bounds.\n"
				"\n"
				"3) SYSTEMATIC ANALYSIS:\n"
				"   - For problems with multiple rounds: Model each round separately, then combine.\n"
				"   - For constraint problems: Use Python to enumerate small cases, verify logic, then generalize.\n"
				"   - For optimization problems ('最少/最多'): Try different strategies, compare results, choose optimal.\n"
				"\n"
				"4) VERIFICATION:\n"
				"   - After computing an answer, verify it meets ALL constraints.\n"
				"   - For detection problems: Verify that your solution can indeed identify the target in all cases.\n"
				"   - If answer seems too high, reconsider your strategy or check if you're missing an optimization.\n"
				"\n"
				"POLICY:\n"
				"- Prefer exact computation with python_tools/math_tools.\n"
				"- For complex logic problems, break down into steps:\n"
				"  1) Analyze constraints and information flow\n"
				"  2) Design encoding/strategy\n"
				"  3) Implement and verify with Python\n"
				"  4) Check optimality\n"
				"- Keep chain-of-thought internal; only output results.\n"
				"- When solving '最少需要X' problems, consider:\n"
				"  * Information theory lower bounds\n"
				"  * Alternative strategies that might be more efficient\n"
				"  * Whether constraints allow full information extraction\n"
				"  * Try different numbers and verify which is the minimum that works\n"
				"\n"
				"OUTPUT:\n"
				"- Return ONLY the final number/string needed by the question (no units unless requested).\n"
				"- For '最少需要X' problems, output the minimum number that satisfies all constraints.\n"
			),
			timeout=300,
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
	
	# WebAgent：网页信息提取与搜索（优先 GitHub API/URL，其次浏览器）
	web_tools = ["http_tools", "github_api_tools", "github_guardrails", "sanity_tools"]
	if "baidu_search_tools" in available_tools:
		web_tools.append("baidu_search_tools")
	if "browser_tools" in available_tools:
		web_tools.append("browser_tools")
	# 旧版 github_tools 已禁用，统一使用 github_api_tools
	
	oxy_space.append(
		oxy.ReActAgent(
			name="web_agent", 
			desc="Web Specialist – search, fetch, browse, and extract deterministic facts.",
			tools=web_tools,
			llm_model="default_llm",
			additional_prompt=(
				"ROLE: Web + Browser Specialist\n"
				"TOOLS: http_tools; github_api_tools (Github.SearchIssues, Github.GetIssue); github_guardrails (GithubQueryBuilder); sanity_tools (ValidateAnswer); baidu_search_tools (if available); browser_tools (navigate/click/snapshot); browser_analyze_screenshot (VLM).\n"
				"FLOW:\n"
				"1) For GitHub tasks, FIRST use GithubQueryBuilder → Github.SearchIssues → (optional) Github.GetIssue. Compose URL issues?q=... as needed.\n"
				"2) ANALYZE results BEFORE taking further action:\n"
				"   - If search_github_issues returns results, READ and ANALYZE them immediately.\n"
				"   - Check if the results already contain the answer to the query.\n"
				"   - If yes, extract the answer from the results and STOP (do NOT search again).\n"
				"   - If results reference a specific issue, open that issue page via browser and extract details.\n"
				"   - DO NOT repeat the same search - analyze existing results first.\n"
				"3) Open page(s) and extract target facts deterministically (only if search results don't contain the answer).\n"
				"   - When using browser_navigate, prefer wait_until='domcontentloaded' to reduce waiting time.\n"
				"   - Use full-page screenshots when the target may appear outside the initial viewport; otherwise prefer minimal screenshots.\n"
				"4) If visual content required (text in image, buttons, tables), call browser_analyze_screenshot with a precise, minimal prompt.\n"
				"HIT_POLICY (CRITICAL - HARD STOP):\n"
				"- Once any API/URL/page content yields sufficient, directly usable evidence (exact number/date/name/link/table row), you MUST STOP all further search/navigation.\n"
				"- After HIT: Only one optional local extraction (on current result/page) is allowed; then finalize.\n"
				"- Prohibited after HIT: search_github_issues, search_baidu, browser_navigate to a new site, or any external search.\n"
				"IDEMPOTENCE / DEDUP (STRICT):\n"
				"- Within a single task, you may call each search tool at most once per normalized query signature.\n"
				"- Do NOT repeat identical or trivially similar queries (case-insensitive, trimmed, stop-words removed).\n"
				"- If a search returned results already analyzed, do not call search again — switch to analysis or extraction.\n"
				"BUDGET (LIMITS):\n"
				"- search_* tools: budget = 1 per task unless first returned empty; one retry allowed only if you REDUCE keywords.\n"
				"- If API hit (Github.SearchIssues/GetIssue) already provides the needed value, browser budget = 0 (skip browser).\n"
				"VISUAL-FIRST HEURISTICS (IMAGE/TEXT-IN-IMAGE CASES):\n"
				"- If the query mentions or implies values often embedded in images (e.g., 证书/收藏证书/编号/第…号/海报/公告/票据/榜单/截图/照片), prioritize on-page visual extraction over external search.\n"
				"  1) Navigate/open the candidate page (if already on page, skip search).\n"
				"  2) Take a screenshot (full_page=true when content may be below the fold).\n"
				"  3) Call browser_analyze_screenshot with a precise prompt to OCR the target field only.\n"
				"  4) Use regex-like extraction guidance in the prompt, e.g., 提取\"总第…号/第…号/证书编号/No.\"等字样后的编号，输出数字或编号本体。\n"
				"- After a successful visual hit, STOP all searches and finalize.\n"
				"POLICY:\n"
				"- Keep steps minimal: search → analyze results → extract → STOP.\n"
				"- Prefer primary sources and exact matches.\n"
				"- CRITICAL: After calling search_github_issues and getting results, ANALYZE the results first before taking any further action.\n"
				"  * If the results contain the answer, extract it and return immediately - DO NOT search again.\n"
				"  * If you need more details about a specific issue from the results, use get_github_issue with the issue number.\n"
				"  * NEVER repeat the same search with identical or similar parameters - it wastes time and returns the same results.\n"
				"- For GitHub searches: Prefer API → else encoded URL → else browser.\n"
				"CRITICAL - INFORMATION VERIFICATION:\n"
				"- Information from web pages may be incomplete, outdated, or incorrect.\n"
				"- ALWAYS verify extracted information by:\n"
				"  1) Cross-checking with page source or multiple locations on the same page\n"
				"  2) Using browser_snapshot or browser_analyze_screenshot to visually confirm text matches\n"
				"  3) If multiple sources available, prefer official/authoritative sources\n"
				"  4) If information seems ambiguous, extract from the most prominent/clear location\n"
				"- When extracting specific values (dates, numbers, names, URLs), be precise and verify the exact text.\n"
				"- If the extracted information doesn't seem to match the query, try alternative extraction methods or re-read the page.\n"
				"BROWSER ELEMENT LOCATION STRATEGY:\n"
				"- Use robust locators (getByRole/getByLabel/getByText). Avoid raw CSS/XPath. If needed: \n"
				"  1) Take a screenshot first using browser_take_screenshot to see the current page state\n"
				"  2) Use browser_analyze_screenshot to identify elements by visual description (e.g., 'the search box at the top', 'the filter button labeled All')\n"
				"  3) Try more generic selectors (e.g., 'input[type=\"text\"]' instead of 'input[name=\"q\"]')\n"
				"  4) Try text-based selectors (e.g., 'text=All' or 'button:has-text(\"All\")') if supported\n"
				"  5) Use browser_snapshot to get accessibility tree and find elements by accessible name\n"
				"- When clicking elements, verify success by:\n"
				"  * Taking a screenshot after click to confirm the page state changed\n"
				"  * Checking URL changes or page content updates\n"
				"- If element location fails multiple times:\n"
				"  * Re-examine the page structure with browser_snapshot\n"
				"  * Try browser_analyze_screenshot with a detailed prompt about the element location\n"
				"  * Consider using browser navigation (browser_navigate) with direct URL if possible\n"
				"CRITICAL FOR GITHUB ISSUES:\n"
				"- When searching GitHub issues (e.g., 'issue #12345', 'GitHub issue'), ALWAYS use github_api_tools FIRST if available.\n"
				"- WORKFLOW AFTER SEARCH (CRITICAL - DO NOT REPEAT SEARCHES):\n"
				"  1) Use browser to search ONCE on GitHub issues page with proper filters.\n"
				"  2) If search returns results, ANALYZE the results immediately:\n"
				"     * Check if the results contain the information needed to answer the query.\n"
				"     * If yes, extract the answer from the search results and STOP (do NOT search again).\n"
				"     * If the results mention a specific issue number but lack details, open that issue page and extract details.\n"
				"  3) If search returns empty results, try reducing keywords and retry ONCE.\n"
				"  4) NEVER repeat the same search with identical parameters - it will return the same results.\n"
				"  5) If you need more information about a specific issue from search results, open that issue page directly instead of searching again.\n"
				"- Example workflow:\n"
				"  Query: 'Audio Whisper bug in transformers'\n"
				"  Step 1: search_github_issues(repo='huggingface/transformers', query='audio whisper bug', state='all') → Returns list of issues\n"
				"  Step 2: Analyze returned issues - if they contain the answer, extract and return. If need details, use get_github_issue for specific issue.\n"
				"  Step 3: STOP - do NOT search again with the same or similar query.\n"
				"- FALLBACK METHOD (if github_api_tools not available): Use browser automation with GitHub's search box.\n"
				"- If the query asks about an issue (especially with a number like '#40054') and you cannot find it, it may be CLOSED - use state='all' in search_github_issues.\n"
				"- BROWSER METHOD (if API not available): Use GitHub's search box with filter syntax (RECOMMENDED - avoids clicking multiple buttons).\n"
				"- GitHub Issues search supports filter syntax in the search box:\n"
				"  * 'is:all #40054' - Search all issues (open and closed) with number 40054\n"
				"  * 'is:issue is:closed #40054' - Search closed issues with number 40054\n"
				"  * 'state:all #40054' - Alternative syntax for all states\n"
				"  * 'is:open #40054' - Explicitly search open issues (default)\n"
				"- Steps for GitHub issue pages (USE SEARCH BOX METHOD):\n"
				"  1) Navigate to the repository's issues page (e.g., https://github.com/huggingface/transformers/issues)\n"
				"  2) Locate the search box at the top of the issues page (usually has placeholder like 'Search all issues' or 'Filter issues')\n"
				"  3) Type filter syntax directly into the search box:\n"
				"     - For any issue (open or closed): 'is:all #40054' or 'state:all #40054'\n"
				"     - For specific closed issue: 'is:closed #40054' or 'is:issue is:closed #40054'\n"
				"  4) Press Enter or click search to filter\n"
				"  5) If issue found, click on it to open the issue page\n"
				"  6) Extract the required information from the issue page\n"
				"- ALTERNATIVE METHOD (if search box doesn't work):\n"
				"  - If the search box filter syntax doesn't work, you may need to click filter buttons:\n"
				"    1) Navigate to issues page\n"
				"    2) Look for filter buttons/tabs (Open/Closed/All) near the top\n"
				"    3) Click on 'All' to show all issues\n"
				"    4) Use the search box to find the specific issue number\n"
				"- Example: Query 'GitHub issue #40054' → Navigate to issues page → Type 'is:all #40054' in search box → Press Enter → Click on the issue → Extract info.\n"
				"- IMPORTANT: Always try the search box filter syntax FIRST before clicking buttons, as it's faster and more reliable.\n"
				"OUTPUT FORMAT RULES (CRITICAL):\n"
				"- If question explicitly requests a specific format (e.g., 'comma-separated', '仅用英文逗号间隔', '以逗号分隔'), output ONLY in that exact format.\n"
				"- If question asks for a list (e.g., 'six items', '六个板块'), and requests comma separation, output: 'item1,item2,item3,...' (NO descriptions, NO explanations).\n"
				"- If question asks for names/titles only, extract ONLY the names, join with requested separator (usually comma).\n"
				"- Examples:\n"
				"  * Question: 'List names, comma-separated' → Output: 'Name1,Name2,Name3'\n"
				"  * Question: '六个板块叫什么？请仅用英文逗号间隔输出' → Output: '京东金条,白条,京东小金库,基金,保险,更多服务'\n"
				"- Return ONLY the requested fact/value in the requested format; no extra commentary, no descriptions.\n"
			),
				timeout=300,
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
			llm_model="vision_llm",
			additional_prompt=(
				"ROLE: File Specialist\n"
				"SCOPE: Extract content FROM files (text from PDFs, images, documents).\n"
				"LIMITATIONS: This agent does NOT count files or list directories. For counting/statistics tasks, return 'UNABLE_TO_PROCESS: This task requires counting/statistics, please use math_agent with Python code.'\n"
				"WORKFLOW (DECOMPOSE → EXECUTE → VERIFY):\n"
				"1) READ FULL QUERY: Always read the entire query including any 'File_Name:' line.\n"
				"2) PLAN SHORT STEPS (max 3-5): Break complex tasks into minimal atomic steps (e.g., 'render PDF page→extract field A→extract field B').\n"
				"3) EXECUTE STEP-BY-STEP: For each step, call the appropriate tool with the FULL query string (including 'File_Name: ...'). Use doc_analyze for image/PDF (vision extraction).\n"
				"4) VERIFY PROGRESS: After each step, check if the requested field is obtained with sufficient confidence. If yes, proceed to finalize; if not, do the next smallest step.\n"
				"5) STOP EARLY: As soon as required info is extracted, stop further steps and finalize.\n"
				"TOOL USAGE RULES:\n"
				"- Prefer passing the full original query (contains path context) as the tool 'prompt' to ensure the tool knows the path.\n"
				"- If multiple files under 'File_Name:' are provided, process the FIRST relevant one unless the question requires multiple.\n"
				"- For PDF/image: allow high-DPI render; take full-page if the target may be off-screen.\n"
				"- Avoid repeated identical calls. Change prompt or focus region if a retry is necessary.\n"
				"OUTPUT: Return only the minimal text/value(s) required by the question, OR the UNABLE_TO_PROCESS message if task is outside scope.\n"
			),
			timeout=600,
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
				desc="Audio Processing Agent – ASR transcription and song recognition via audio fingerprint.",
				tools=["audio_tools"],
				llm_model="default_llm",
				prompt=(
					"ROLE: Audio Processing Specialist\n"
					"\n"
					"Your job is to process audio files using TWO tools: audio transcription (ASR) and song recognition (audio fingerprint).\n"
					"\n"
					"TOOLS AVAILABLE:\n"
					"1) audio_transcribe: Transcribe audio to text (lyrics/speech) using ASR\n"
					"   - Parameter: path (full file path)\n"
					"   - Returns: transcribed text or error\n"
					"   - Best for: lyrics, speech, or any audio with text content\n"
					"\n"
					"2) audio_recognize_song: Identify song by audio fingerprint (Chromaprint + AcoustID)\n"
					"   - Parameter: path (full file path)\n"
					"   - Returns: song title, artist, or error\n"
					"   - Best for: music files (especially instrumental/pure music without clear lyrics)\n"
					"\n"
					"FILE PATH EXTRACTION:\n"
					"The query may contain file path information in one of these formats:\n"
					"1) Explicit file path in the query text (e.g., '/path/to/audio.wav')\n"
					"2) File_Name field: The query may include a line like 'File_Name: /path/to/file.wav'\n"
					"3) Extract the FIRST audio file path you find (if multiple, use the first one).\n"
					"\n"
					"PROCESSING STRATEGY (FLEXIBLE - USE BOTH TOOLS):\n"
					"1) Extract the audio file path from the query.\n"
					"2) Try audio_transcribe FIRST to get lyrics/transcription:\n"
					"   - If successful and returns meaningful text (lyrics/speech), return that.\n"
					"   - If returns error or empty/unclear text, proceed to step 3.\n"
					"3) If transcription fails or query asks for song information, try audio_recognize_song:\n"
					"   - If successful, return song title and artist.\n"
					"   - If fails, return error message.\n"
					"\n"
					"DECISION LOGIC:\n"
					"- If query asks for '歌词/lyrics/transcription/text content' → prioritize audio_transcribe\n"
					"- If query asks for '歌曲名/song name/artist/歌手' → try audio_recognize_song first, fallback to audio_transcribe\n"
					"- If query is general ('识别/identify/这是什么/what is this') → try both tools, use whichever succeeds\n"
					"- If one tool succeeds, you can return that result immediately (no need to call the other)\n"
					"- If both tools are needed for complete answer, call both and combine results\n"
					"\n"
					"TOOL CALL FORMAT:\n"
					"When calling tools, respond with JSON:\n"
					'{"think": "Extracted path: /path/to/audio.mp3. Trying audio_transcribe first...", "tool_name": "audio_transcribe", "arguments": {"path": "/path/to/audio.mp3"}}\n'
					'Or: {"think": "Transcription failed, trying song recognition...", "tool_name": "audio_recognize_song", "arguments": {"path": "/path/to/audio.mp3"}}\n'
					"\n"
					"RESPONSE HANDLING:\n"
					"- audio_transcribe response:\n"
					"  * If contains 'text' field with meaningful content → Return '【音频转写】' + text\n"
					"  * If contains 'error' → Try audio_recognize_song as fallback\n"
					"- audio_recognize_song response:\n"
					"  * If contains 'title' and 'artist' → Return '【音频识别】歌曲: {title}, 艺术家: {artist}'\n"
					"  * If contains 'error' → Return error message\n"
					"\n"
					"EXAMPLES:\n"
					"- Query: '识别这首歌的名字' → Try audio_recognize_song first\n"
					"- Query: '转写这段音频的歌词' → Try audio_transcribe first\n"
					"- Query: '这是什么歌曲？File_Name: /path/to/song.mp3' → Try audio_recognize_song, fallback to audio_transcribe if needed\n"
					"\n"
					"OUTPUT FORMAT:\n"
					"- If transcription succeeds: Return '【音频转写】' + transcribed text\n"
					"- If song recognition succeeds: Return '【音频识别】歌曲: {title}, 艺术家: {artist}'\n"
					"- If both succeed: You can combine both results\n"
					"- If both fail: Return error message\n"
					"\n"
					"You have access to these tools:\n"
					"${tools_description}\n"
					"\n"
					"CRITICAL: Use BOTH tools flexibly. If one fails, try the other. If one succeeds, you can return that result."
				),
				timeout=600,
			)
		)
	
	# video
	if "video_tools" in available_tools:
		video_tools_list = []
		video_tools_list.append("video_tools")

	oxy_space.append(
		oxy.ReActAgent(
			name="video_agent",
			desc="Video Understanding Specialist – analyze video content through key-frame extraction and visual question answering.",
			tools=video_tools_list,
			llm_model="zai-org/GLM-4.5",
			additional_prompt=(
				'''
				ROLE: Video Understanding Specialist  
				SCOPE: Perform video analysis including: extract basic metadata (size, duration, resolution, frame-rate), sample key frames, and answer user questions grounded in those frames.  
				LIMITATIONS:  
				- Only processes local video files; remote URLs or streams are not supported.  
				- Frame extraction needs a non-zero time span; single-timestamp queries must not duplicate the value.
				- Answers are inference-based; state uncertainty if visual evidence is insufficient.  

				POLICY:  
				1. For metadata: Always call get_basic_video_info first to confirm video validity and gather context.  
				2. For content questions: Invoke video_understanding with a concise vlm_prompt; supply start_time & end_time if query refers to an exact moment.  
				3. For general video summary: Use video_understanding without vlm_prompt to obtain an auto-generated description.  
				4. Return results verbatim from tools; prepend short context (e.g., 'Detected 120 frames, 30 s clip') but keep the main answer unaltered.  
				5. Time-rule – never invent the missing bound:
				– "At 4 s" → start_time = 4, end_time = None (function auto-ends at next key-frame or EOF).
				– "Until 4 s" → start_time = None, end_time = 4.
				– "From 4 s to 6 s" → start_time = 4, end_time = 6.		
						
				TOOL CALL FORMAT:  
				When you need to call the tool, respond with JSON:  
				{"think": "<brief reason>", "tool_name": "get_basic_video_info", "arguments": {"video_path": "<absolute_path>"}}  
				{"think": "<brief reason>", "tool_name": "video_understanding", "arguments": {"path": "<absolute_path>", "vlm_prompt": "<optional_question>", "start_time": <optional_start>, "end_time": <optional_end>}}  

				EXAMPLES:  
				- Query: 'What is the resolution and duration of /data/clips/demo.mp4?'  
				→ {"think": "User asks for metadata of /data/clips/demo.mp4", "tool_name": "get_basic_video_info", "arguments": {"video_path": "/data/clips/demo.mp4"}}  
				
				- Query: ' 在视频的第23秒，请问有什么出现?'  
				→ {"think": "The user is asking about the visual content at the 23-second mark of the video.", "tool_name": "video_understanding", "arguments": {"path": "/tmp/clip.mp4", "start_time": 30, "end_time": None, "vlm_prompt": "What appears or happens at this moment?"}}  

				- Query: ' Until the 46-second mark, how many people are there?'  
				→ {"think": "The user wants a head-count for everything from the start of the video up to, but not beyond, the 46-second mark.", "tool_name": "video_understanding", "arguments": {"path": "/tmp/clip.mp4", "start_time": None, "end_time": 46, "vlm_prompt": "How many people are visible in this segment?"}}
				
				- Query: 'What appears in the search box between 30 s and 32 s in /tmp/clip.mp4?'  
				→ {"think": "Specific visual question on 30-32 s slice", "tool_name": "video_understanding", "arguments": {"path": "/tmp/clip.mp4", "start_time": 30, "end_time": 32, "vlm_prompt": "What text is in the search box?"}}  

				- Query: 'Summarize the whole video /media/intro.avi'  
				→ {"think": "General summary needed", "tool_name": "video_understanding", "arguments": {"path": "/media/intro.avi"}}  

				OUTPUT: Provide tool outputs directly, or concise error messages if operations fail. Include critical metadata (duration, resolution) when relevant.  
				'''
			),
			timeout=600,
		)
	)

	# Normalizer：强规范化输出
	oxy_space.append(
		oxy.ReActAgent(
			name="normalizer_agent",
			desc="Answer Validator – validate and optionally normalize answer format (no recalculation).",
			tools=["string_tools"],
			llm_model="zai-org/GLM-4.5",
			prompt=(
				"ROLE: Answer Validator (只验不改值，格式标准化)\n\n"
				"INPUT: query (contains question and candidate_answer)\n\n"
				"MANDATE:\n"
				"- DO NOT recalculate or invent new answers. **DO NOT change the semantic meaning.**\n"
				"- Only validate format and return the **PURE, extracted answer** (no surrounding text, no markdown).\n"
				"- If valid, the output MUST be wrapped in a <FINAL_ANSWER>...</FINAL_ANSWER> tag.\n\n"
				"PROCESS:\n"
				"1) Extract question and candidate_answer from the query.\n"
				"2) Based on the question, determine the expected answer type (Number, URL, Date, CSV, Text).\n"
				"3) **CLEAN & EXTRACT:** Remove all extraneous surrounding text, markdown formatting (like **), and punctuation (except within URLs or numbers).\n"
				"4) Check if the CLEANED answer matches the expected format:\n"
				"   - For **Numbers**: Extract ONLY the pure number (e.g., '12.50'). **REMOVE all units** ($, ¥, km) UNLESS it is a percentage (%).\n"
				"   - For **URLs**: Should be a valid http(s) URL.\n"
				"   - For **Dates**: Standardize to strict **YYYY-MM-DD** format.\n"
				"   - For **CSV**: Ensure values are separated by a comma (,).\n"
				"5) If valid, return the standardized, PURE answer.\n"
				"6) If invalid, return a simple error message **'invalid_format'**.\n\n"
				"EXAMPLES:\n"
				"- Input: 'Q: 2+2? C: The answer is 4.' → Output: '<FINAL_ANSWER>4</FINAL_ANSWER>'\n"
				"- Input: 'Q: Growth? C: 12% in 2024' → Output: '<FINAL_ANSWER>12%</FINAL_ANSWER>'\n"
				"- Input: 'Q: What day? C: The date is 2025/8/11.' → Output: '<FINAL_ANSWER>2025-08-11</FINAL_ANSWER>'\n"
				"- Input: 'Q: 发布日期？ C: 发布于2025年8月3日。' → Output: '<FINAL_ANSWER>2025-08-03</FINAL_ANSWER>'\n"
				"- Input: 'Q: Price? C: ￥199.00 (limited offer)' → Output: '<FINAL_ANSWER>199.00</FINAL_ANSWER>'\n"
				"- Input: 'Q: Items? C: 1. Apple; 2. Banana' → Output: '<FINAL_ANSWER>Apple,Banana</FINAL_ANSWER>'\n"
				"- Input: 'Q: 合作伙伴？ C: 阿里巴巴、京东、腾讯。' → Output: '<FINAL_ANSWER>阿里巴巴,京东,腾讯</FINAL_ANSWER>'\n"
				"- Input: 'Q: 官方公告链接? C: 请访问 https://example.com/info 。' → Output: '<FINAL_ANSWER>https://example.com/info</FINAL_ANSWER>'\n"
				"- Input: 'Q: 官方邮箱？ C: 联系 support@example.com 获取帮助。' → Output: '<FINAL_ANSWER>support@example.com</FINAL_ANSWER>'\n"
				"- Input: 'Q: 占比？ C: 约等于 23.5%。' → Output: '<FINAL_ANSWER>23.5%</FINAL_ANSWER>'\n"
				"- Input: 'Q: Color? C: The color is blue.' → Output: '<FINAL_ANSWER>blue</FINAL_ANSWER>'\n"
				"- Input: 'Q: What day? C: I think it's 5' → Output: 'invalid_format'\n\n"
				"OUTPUT:\n"
				"Return ONLY the validated answer wrapped in <FINAL_ANSWER>...</FINAL_ANSWER> or 'invalid_format'. Do NOT use JSON format."
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
				"- 'video': Video Understanding Specialist – analyze video content through key-frame extraction and visual question answering.\n"
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
				"OUTPUT: Return ONLY the classification token (math/time/web/audio/file/directory/video).\n"
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
				"web_agent", "audio_agent", "file_agent","directory_agent", "normalizer_agent", "video_agent"
			],
			tools=[
				# guard & utils
				"python_tools", "final_answer_guard",
				# case bank tools
				"casebank_ping", "case_save", "case_search", "case_update_score", "case_get",
				# todo tools
				"todo_ping", "todo_create", "todo_list", "todo_update", "todo_autogen_from_case", "todo_link_case", "todo_stats"
			],
			llm_model="zai-org/GLM-4.6",
			additional_prompt=(
				"ROLE: Master Coordinator\n"
				"WORKFLOW:\n"
				"0) Information Gap Analysis (CRITICAL - DO THIS FIRST):\n"
				"   - Before routing, analyze the query for missing information that requires tool calls.\n"
				"   - If query involves time/date but doesn't specify current time:\n"
				"     * Call time_agent with a query like 'get current time' or '当前时间是什么' to obtain current date/time.\n"
				"     * time_agent will use get_current_time tool and return the current time.\n"
				"     * Append the current time information to the original query before routing.\n"
				"     * Example: Original query '3天前是什么日期' → Call time_agent → Get 'Current time: 2025-11-03' → Append to query: '3天前是什么日期。当前时间: 2025-11-03' → Then route to time_agent.\n"
				"   - If query involves other dynamic information (e.g., current weather, stock prices, real-time data):\n"
				"     * Use appropriate agents/tools to fetch the missing information first.\n"
				"     * Supplement the query with this information before routing.\n"
				"   - Common scenarios requiring information supplementation:\n"
				"     * Time-relative queries ('3天前', 'last week', '下个月', 'yesterday', '前天', '明天')\n"
				"     * Date calculations without reference point ('今天是星期几', '这个月有多少天')\n"
				"     * '当前'/'current'/'now'/'现在' keywords in time context\n"
				"   - After supplementing information, continue to step 1.\n"
				"   - IMPORTANT: If the query already contains explicit date/time (e.g., '2025-11-03', 'October 15'), you may skip this step.\n"
				"1) Case Lookup (BEFORE planning): Call case_search with {query, signature?}. If items found and score≥SIM_THRESHOLD, load the first item's plan as plan_template (do NOT change the final goal).\n"
				"   - signature may include {task_type, answer_type, level}.\n"
				"   - Keep evidence in logs (case_id, score).\n"
				"2) Route: Call router_agent with the COMPLETE query (including File_Name if present, and any supplemented information). Router needs full context to classify correctly.\n"
				"2) Delegate: Call the appropriate specialist based on router_agent's classification.\n"
				"   CRITICAL for file/audio agents: When calling audio_agent or file_agent, you MUST pass the FULL query string (including the File_Name field). These agents need the file path information.\n"
				"   Example: If query contains 'File_Name: /path/to/audio.wav', pass the entire query including that line to audio_agent.\n"
				"   INFORMATION VERIFICATION (CRITICAL):\n"
				"   - Information from specialist agents (especially web_agent) may be incomplete, outdated, or incorrect.\n"
				"   - If the returned answer seems suspicious or doesn't match the question, consider:\n"
				"     * Re-querying the specialist agent with more specific instructions\n"
				"     * Trying an alternative approach or different agent\n"
				"     * For web tasks: requesting web_agent to verify or re-extract the information\n"
				"   - However, trust specialist agents' outputs by default unless there are clear inconsistencies.\n"
				"3) Fallback Logic (CRITICAL):\n"
				"   - If file_agent returns error/empty/unable_to_process (especially for counting/statistics tasks), automatically retry with math_agent.\n"
				"   - If any agent fails but task involves counting/statistics/calculation, use math_agent as fallback.\n"
				"   - If web_agent returns incomplete or unclear information, you may retry with more specific query or different extraction method.\n"
				"4) Normalize (CRITICAL - MANDATORY STEP): After any specialist returns a candidate answer, you MUST ALWAYS call normalizer_agent before finalizing.\n"
				"   - This step is NOT optional. You MUST call normalizer_agent with format: 'Question: [original question] Candidate: [specialist agent output]'\n"
				"   - normalizer_agent will clean and format the answer, returning it wrapped in <FINAL_ANSWER>...</FINAL_ANSWER> tags.\n"
				"   - Extract the content from <FINAL_ANSWER> tags and use that as your final output.\n"
				"   - If normalizer_agent returns 'invalid_format', you may fall back to the original candidate, but you MUST still attempt the call.\n"
				"   - DO NOT skip this step or return the raw specialist output without normalization.\n"
				"5) Commit Final Answer (GUARD): After normalization, call final_answer_guard('commit', answer, answer_type). If commit succeeds, proceed.\n"
				"6) Persist Case (AFTER success): If guard commit succeeded, call case_save(...) then case_update_score(success:true, latency_ms?).\n"
				"7) On Failure/Timeout: call case_update_score(success:false) and todo_autogen_from_case({case_id, reason, attach}).\n"
				"RULES:\n"
				"- ALWAYS check for missing information (especially time) BEFORE routing.\n"
				"- When calling router_agent, pass the FULL query string (including any File_Name context and supplemented information).\n"
				"- When calling audio_agent/file_agent, ALWAYS pass the FULL query string (including File_Name) so they can extract file paths.\n"
				"- **CRITICAL**: You MUST call normalizer_agent after receiving any specialist agent's output. This is a mandatory step, not optional.\n"
				"- **CRITICAL**: Your final output MUST be the result from normalizer_agent (extracted from <FINAL_ANSWER> tags), not the raw specialist output.\n"
				"- Be aware that information from web sources may not always be accurate - trust but verify when possible.\n"
				"- Keep chain-of-thought internal; do not output reasoning.\n"
				"- No prefixes (Answer:/Result:) or extra lines.\n"
				"- The final output must be the normalized answer string only (from normalizer_agent).\n"
			),
			timeout=180,
		)
	)

	return oxy_space


async def run_single(question: str, enable_mcp: bool = False) -> str:
	"""运行单个问题（简化版，类似 web 模式）"""
	oxy_space = build_oxy_space(enable_mcp=enable_mcp)
	async with MAS(oxy_space=oxy_space) as mas:
		# 禁用历史记录访问，避免干扰模型判断（批量处理中每个任务都是独立的）
		resp = await mas.call(callee="master_agent", arguments={
			"query": question,
			"short_memory": [],
			"master_short_memory": []
		})
		answer = resp.output if hasattr(resp, "output") else str(resp)
		# 只做基本清理（类似 web 模式）
		return _clean_final_answer(answer)


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

# ================= Deterministic Normalization Pipeline =================
class AnswerSpec:
    def __init__(self, type_: str = "text", separator: str = ",", constraints: Dict[str, Any] = None):
        self.type = type_
        self.separator = separator
        self.constraints = constraints or {}

def parse_answer_spec(question: str) -> AnswerSpec:
    q = (question or "").lower()
    import re
    # csv hints
    if any(k in q for k in ["仅用英文逗号间隔", "逗号分隔", "comma-separated", "用英文逗号", "以逗号分隔"]):
        return AnswerSpec("csv", ",")
    # url hints
    if any(k in q for k in ["url", "网址", "链接", "http", "https"]):
        return AnswerSpec("url")
    # fraction explicit
    if any(k in q for k in ["分数", "fraction", "p/q"]):
        return AnswerSpec("fraction")
    # date hints
    if any(k in q for k in ["日期", "时间", "date", "年", "月", "日", "什么时候", "年月日"]):
        return AnswerSpec("date")
    # percentage hints (百分比)
    if any(k in q for k in ["百分比", "percent", "%", "同比增长", "增长", "下降", "上升"]):
        # 如果问题涉及百分比，返回 float 类型（可以包含%符号）
        return AnswerSpec("float")
    # number hints → integer/float
    if any(k in q for k in ["仅输出数字", "只输出数字", "only output number", "仅输出数值", "数值即可", "输出数值", "答案是", "最少", "最多", "共", "总计", "需要", "多少", "数值"]):
        if any(k in q for k in ["小数", "decimal", "保留", "位"]):
            return AnswerSpec("float")
        return AnswerSpec("integer")
    return AnswerSpec("text")

def _strip_markdown(text: str) -> str:
    import re
    if not text:
        return ""
    # trim and normalize quotes
    s = str(text).strip()
    # code fences
    s = re.sub(r"```[ \t]*([A-Za-z0-9_\-]+)?\n?([\s\S]*?)```", lambda m: m.group(2).strip(), s)
    s = re.sub(r"^```[ \t]*[A-Za-z0-9_\-]*", "", s)
    # inline code / bold / italics
    s = re.sub(r"`([^`]+)`", lambda m: m.group(1), s)
    s = re.sub(r"\*\*([^*]+)\*\*", lambda m: m.group(1), s)
    s = re.sub(r"__([^_]+)__", lambda m: m.group(1), s)
    s = re.sub(r"\*([^*]+)\*", lambda m: m.group(1), s)
    # bullets
    s = re.sub(r"^\s*[-•\*]\s+", "", s)
    return s.strip().strip('"').strip("'")

def extract_deterministic(spec: AnswerSpec, text: str) -> Union[str, Dict[str, Any]]:
    import re
    s = _strip_markdown(text)
    if spec.type == "fraction":
        m = list(re.finditer(r"(-?\d+)\s*/\s*(-?\d+)", s))
        if not m:
            return {"error": "no_fraction_found"}
        num = int(m[-1].group(1)); den = int(m[-1].group(2))
        if den == 0:
            return {"error": "division_by_zero"}
        from math import gcd
        g = gcd(abs(num), abs(den)) or 1
        num//=g; den//=g
        if den < 0:
            num, den = -num, -den
        return f"{num}/{den}"
    if spec.type == "date":
        # 日期提取：支持多种格式
        date_patterns = [
            r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})[日]?",  # 2025年12月25日
            r"(\d{4})-(\d{1,2})-(\d{1,2})",  # 2025-12-25
            r"(\d{4})/(\d{1,2})/(\d{1,2})",  # 2025/12/25
        ]
        for pattern in date_patterns:
            m = list(re.finditer(pattern, s))
            if m:
                date_match = m[-1]
                year = date_match.group(1)
                month = date_match.group(2).zfill(2)
                day = date_match.group(3).zfill(2)
                return f"{year}-{month}-{day}"
        return {"error": "no_date_found"}
    if spec.type in ("integer", "float"):
        # 百分比优先（如果问题涉及百分比）
        percent_match = re.search(r"(-?\d+(?:\.\d+)?%)", s)
        if percent_match:
            return percent_match.group(1).strip()
        
        cue_pattern = r"(最少|最多|共|总计|需要|仅输出数字|答案是|同比增长|增长|下降|上升|数值)"
        nums = list(re.finditer(r"-?\d+(?:\.\d+)?", s))
        if not nums:
            return {"error": "no_number_found"}
        val = None
        cue_iter = list(re.finditer(cue_pattern, s))
        if cue_iter:
            last_cue_end = cue_iter[-1].end()
            for m in nums:
                if m.start() >= last_cue_end:
                    val = m.group(0)
                    break
        if val is None:
            val = nums[-1].group(0)
        if spec.type == "integer":
            try:
                f = float(val)
                return str(int(round(f)))
            except:
                return re.sub(r"\.0+$", "", val)
        try:
            f = float(val)
            return ("%f" % f).rstrip("0").rstrip(".")
        except:
            return val
    if spec.type == "url":
        m = list(re.finditer(r"https?://[^\s)\]>]+", s, flags=re.IGNORECASE))
        if not m:
            return {"error": "no_url_found"}
        return m[-1].group(0).rstrip(".,;)")
    if spec.type == "csv":
        parts = re.split(r"[\n,，、;；]+", s)
        parts = [p.strip() for p in parts if p and p.strip()]
        if not parts:
            return {"error": "empty_list"}
        seen = set(); out=[]
        for p in parts:
            if p not in seen:
                seen.add(p); out.append(p)
        return out
    return s

def normalize_by_spec(spec: AnswerSpec, value: Union[str, List[str], Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
    if isinstance(value, dict) and "error" in value:
        return value
    if spec.type == "date":
        # 日期标准化：确保 YYYY-MM-DD 格式
        s = str(value).strip()
        import re
        # 如果已经是标准格式，直接返回
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            return s
        # 尝试转换其他格式
        date_match = re.search(r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})[日]?", s)
        if date_match:
            year = date_match.group(1)
            month = date_match.group(2).zfill(2)
            day = date_match.group(3).zfill(2)
            return f"{year}-{month}-{day}"
        return s
    if spec.type in ("integer", "float", "number"):
        s = str(value).strip()
        # 保留百分比符号（如果存在）
        has_percent = '%' in s
        if has_percent:
            s = s.replace("%", "").strip()
        try:
            f = float(s)
            if spec.type == "integer" or f.is_integer():
                result = str(int(round(f)))
            else:
                result = ("%f" % f).rstrip("0").rstrip(".")
            # 如果有百分比符号，加回去
            if has_percent:
                result += "%"
            return result
        except:
            return {"error": "normalize_number_failed"}
    if spec.type == "fraction":
        return str(value)
    if spec.type == "url":
        s = str(value).strip()
        # 移除尾部标点
        s = s.rstrip(".,;)")
        return s
    if spec.type == "csv":
        items = (value if isinstance(value, list) else [value])
        items = [str(x).strip() for x in items]
        seen=set(); dedup=[]
        for x in items:
            if x not in seen:
                seen.add(x); dedup.append(x)
        return spec.separator.join(dedup)
    return str(value).strip()


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
    try:
        if not input_jsonl_path:
            # 如果没有提供路径，使用默认的 valid 目录
            base_dir = Path("./test").resolve()
        else:
            base_dir = Path(input_jsonl_path).resolve().parent   # 取出 ./valid
    except Exception as e:
        logger.warning(f"Failed to resolve base_dir from {input_jsonl_path}: {e}, using default ./test")
        base_dir = Path("./test").resolve()

    # 相对路径 -> 绝对路径
    def to_abs(p: str) -> str:
        if not p:
            return ""
        p = p.strip()
        if os.path.isabs(p):
            return p
        # 处理相对路径
        return str(base_dir / p.lstrip('./'))

    abs_paths = [to_abs(p) for p in paths if p]
    if abs_paths:
        q += f"\nFile_Name: {' '.join(abs_paths)}"
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
        try:
            # 第一轮尝试：按带表头读取
            with open(checkpoint_file, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and 'task_id' in reader.fieldnames and 'response' in reader.fieldnames:
                    for row in reader:
                        if row.get('task_id'):
                            processed[str(row['task_id'])] = row.get('response', '')
                else:
                    # 兼容无表头的历史文件：按首列=task_id，次列=response 解析
                    f.seek(0)
                    row_reader = csv.reader(f)
                    for row in row_reader:
                        if not row:
                            continue
                        task_id = str(row[0]).strip()
                        if not task_id or task_id.lower() == 'task_id':
                            # 跳过可能的表头或空值
                            continue
                        response = str(row[1]).strip() if len(row) > 1 else ''
                        processed[task_id] = response
        except Exception as e:
            logger.warning(f"Failed to load checkpoint file {checkpoint_file}: {e}, starting fresh")
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

async def process_single_item(
	mas: MAS,
	item: pd.Series,
	jsonl_file_path: str,
	result_dir: Path,
	checkpoint_file: Path,
	failed_checkpoint_file: Path,
	jsonl_file,
	validate: bool = False
) -> Tuple[int, int]:
	"""处理单个任务项，返回 (correct_count, total_count)"""
	answer = ""
	result_record = {}
	correct_count = 0
	total_count = 0
	
	try:
		# 使用处理后的 jsonl_file_path 而不是原始的 input_jsonl_path
		question = compose_question(item, jsonl_file_path)
		if not question:
			logger.warning(f"Task {item.get('task_id', 'N/A')} has no question field, skipping.")
			result_record = {"error": "no question field", **item.to_dict()}
			# 立即保存到 JSONL
			jsonl_file.write(json.dumps(result_record, ensure_ascii=False) + "\n")
			jsonl_file.flush()
			return (correct_count, total_count)
		
		# 实际调用 master_agent（简化版，类似 web 模式）
		logger.info(f"Processing task {item.get('task_id', 'N/A')}: {question[:50]}...")
		# 禁用历史记录访问，避免干扰模型判断（批量处理中每个任务都是独立的）
		resp = await mas.call(callee="master_agent", arguments={
			"query": question,
			"short_memory": [],
			"master_short_memory": []
		})
		answer = resp.output if hasattr(resp, "output") else str(resp)
		# 只做基本清理（类似 web 模式）
		answer = _clean_final_answer(answer)
		
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
	
	return (correct_count, total_count)


async def run_batch(
	input_jsonl_path: str, output_jsonl_path: str, enable_mcp: bool = False, validate: bool = False, batch_size: int = 10
) -> None:
	"""批量处理任务，支持断点续传，每处理 batch_size 条数据后重启进程"""
	result_dir = Path("./res/")
	checkpoint_file = result_dir / "processed.csv"
	failed_checkpoint_file = result_dir / "failed_processed.csv"
	
	# 初始化目录和文件
	init_files(result_dir, checkpoint_file)
	init_files(result_dir, failed_checkpoint_file)
	processed = load_processed(checkpoint_file)
	logger.info(f"Loaded {len(processed)} already processed tasks from checkpoint.")

	# 加载数据集 - 使用 input_jsonl_path 参数
	jsonl_file_path = input_jsonl_path if input_jsonl_path else "./test/data.jsonl"
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
	logger.info(f"To process: {len(datasets)} tasks. Batch size: {batch_size}")

	correct_count = 0
	total_count = 0

	# 打开 JSONL 文件用于追加写入（每条任务完成后立即保存）
	# 使用 "a" 模式以支持从中断处继续运行
	jsonl_file = open(output_jsonl_path, "a", encoding="utf-8")
	
	try:
		# 将数据集分成批次处理
		total_batches = (len(datasets_df) + batch_size - 1) // batch_size
		
		for batch_idx in range(total_batches):
			start_idx = batch_idx * batch_size
			end_idx = min((batch_idx + 1) * batch_size, len(datasets_df))
			batch_df = datasets_df.iloc[start_idx:end_idx]
			
			logger.info(f"Processing batch {batch_idx + 1}/{total_batches} (tasks {start_idx + 1}-{end_idx})")
			
			# 为每个批次创建新的 MAS 实例
			oxy_space = build_oxy_space(enable_mcp=enable_mcp)
			async with MAS(oxy_space=oxy_space) as mas:
				for idx, item in batch_df.iterrows():
					item_correct, item_total = await process_single_item(
						mas, item, jsonl_file_path, result_dir, checkpoint_file, 
						failed_checkpoint_file, jsonl_file, validate
					)
					correct_count += item_correct
					total_count += item_total
			
			# 批次处理完成，MAS 实例会自动关闭
			logger.info(f"Batch {batch_idx + 1}/{total_batches} completed. Process will restart for next batch.")
			# 给系统一点时间释放资源
			await asyncio.sleep(1)
			
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
	parser.add_argument("--batch_size", type=int, default=10, help="Number of tasks to process before restarting the process (default: 10).")
	
	try:
		args = parser.parse_args()
	except SystemExit as e:
		# 如果参数解析失败，提供更友好的错误信息
		if e.code != 0:
			print("\n提示：如果看到 'unrecognized arguments'，请检查命令行末尾是否有多余的空格或特殊字符。")
		raise

	Config.set_app_name(args.app_name)

	if args.question:
		answer = asyncio.run(run_single(args.question, enable_mcp=args.enable_mcp))
		print(f"Answer: {answer}")
	elif args.input_jsonl:
		asyncio.run(run_batch(args.input_jsonl, args.output, args.enable_mcp, args.validate, args.batch_size))
	elif args.web:
		asyncio.run(start_web(args.first_query, enable_mcp=args.enable_mcp))
	else:
		print("Please provide --question, --input_jsonl, or --web argument.")


if __name__ == "__main__":
	main()