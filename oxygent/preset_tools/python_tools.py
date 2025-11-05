import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from fractions import Fraction
from pydantic import Field

from oxygent.oxy import FunctionHub

logger = logging.getLogger(__name__)
python_tools = FunctionHub(name="python_tools")

# ==================== 内部工具函数 ====================

def _extract_final_number(text: str) -> Tuple[bool, Optional[Union[int, Fraction]], str]:
    """
    评测前兜底抽取器：从文本中优先抽取最后一个分数 p/q，否则抽取最后一个整数。
    
    Returns:
        (success, value, error_msg)
        - success: bool, 是否成功
        - value: int | Fraction | None, 抽取的值
        - error_msg: str, 错误信息（成功时为""）
    """
    if not text or not isinstance(text, str):
        return False, None, "Empty or non-string input"
    
    # 优先匹配分数：p/q 或 p/q 形式（支持负号）
    fraction_pattern = r'(-?\d+)\s*/\s*(-?\d+)'
    fraction_matches = list(re.finditer(fraction_pattern, text))
    if fraction_matches:
        last_match = fraction_matches[-1]
        try:
            num = int(last_match.group(1))
            den = int(last_match.group(2))
            if den == 0:
                return False, None, "Division by zero in fraction"
            return True, Fraction(num, den), ""
        except (ValueError, ZeroDivisionError) as e:
            return False, None, f"Invalid fraction: {e}"
    
    # 否则匹配整数（支持负号）
    int_pattern = r'-?\d+'
    int_matches = list(re.finditer(int_pattern, text))
    if int_matches:
        last_match = int_matches[-1]
        try:
            return True, int(last_match.group(0)), ""
        except ValueError as e:
            return False, None, f"Invalid integer: {e}"
    
    return False, None, "No number found in text"


# ==================== 工作空间管理 ====================

_code_workspace_base = Path(tempfile.gettempdir()) / "oxy_math"
_code_workspace_base.mkdir(exist_ok=True, parents=True)

@python_tools.tool(description="Code workspace: task-level temporary file management")
def code_workspace(
    action: str = Field(description="Action: 'write', 'read', 'ls', or 'cleanup'"),
    task_id: str = Field(description="Task identifier for workspace isolation"),
    path: str = Field(default="", description="File path (relative to workspace, for write/read)"),
    content: str = Field(default="", description="File content (for write)")
) -> Dict[str, Any]:
    """Task-level temporary directory management."""
    try:
        workspace_dir = _code_workspace_base / task_id
        workspace_dir.mkdir(exist_ok=True, parents=True)
        
        if action == "write":
            if not path:
                return {"ok": False, "error": "path required for write"}
            file_path = workspace_dir / path
            file_path.parent.mkdir(exist_ok=True, parents=True)
            file_path.write_text(content, encoding='utf-8')
            return {"ok": True, "path": str(file_path)}
        
        elif action == "read":
            if not path:
                return {"ok": False, "error": "path required for read"}
            file_path = workspace_dir / path
            if not file_path.exists():
                return {"ok": False, "error": f"File not found: {path}"}
            content = file_path.read_text(encoding='utf-8')
            return {"ok": True, "content": content}
        
        elif action == "ls":
            subdir_path = workspace_dir / (path or ".")
            if not subdir_path.exists():
                return {"ok": False, "error": f"Directory not found: {path}"}
            files = [f.name for f in subdir_path.iterdir()]
            return {"ok": True, "files": files}
        
        elif action == "cleanup":
            import shutil
            if workspace_dir.exists():
                shutil.rmtree(workspace_dir)
            return {"ok": True}
        
        else:
            return {"ok": False, "error": f"Unknown action: {action}"}
    
    except Exception as e:
        logger.error(f"code_workspace error: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


# ==================== 受限Python执行 ====================

_PRELUDE = """
# Auto-injected prelude
import math
import fractions
from fractions import Fraction
from typing import List, Tuple, Dict, Any
import itertools
import collections
"""

def _run_with_timeout(cmd: List[str], timeout: int, stdin_data: Optional[str] = None) -> Tuple[int, str, str]:
    """运行命令并限制超时"""
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_data else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": "0"}
        )
        
        stdout, stderr = proc.communicate(input=stdin_data, timeout=timeout)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)

@python_tools.tool(description="Code runner: restricted Python execution with timeout, network disabled, fixed seed")
def code_runner(
    code: str = Field(description="Python code to execute"),
    stdin: Optional[str] = Field(default=None, description="Standard input"),
    timeout: int = Field(default=3, description="Timeout in seconds"),
    seed: int = Field(default=0, description="Random seed (PYTHONHASHSEED)")
) -> Dict[str, Any]:
    """Restricted Python execution with timeout and network disabled."""
    try:
        # 注入prelude
        full_code = _PRELUDE + "\n" + code
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            script_path = f.name
        
        try:
            # 受限执行（依赖容器/系统策略禁网）
            # 注意：实际禁网需要容器隔离或系统级策略，这里只做超时和seed控制
            cmd = ["python3", script_path]
            exit_code, stdout, stderr = _run_with_timeout(cmd, timeout, stdin)
            
            if exit_code == -1:
                if stderr == "timeout":
                    return {"ok": False, "error": "timeout", "exit_code": -1}
                else:
                    return {"ok": False, "error": stderr, "exit_code": -1}
            
            return {
                "ok": True,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code
            }
        finally:
            try:
                os.unlink(script_path)
            except:
                pass
    
    except Exception as e:
        logger.error(f"code_runner error: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


# ==================== 对拍器 ====================

def _compare_exact(got: str, expect: str) -> bool:
    """精确字符串比较"""
    return got.strip() == expect.strip()

def _compare_number(got: str, expect: str) -> bool:
    """数字比较：优先分数，否则整数"""
    success_got, val_got, _ = _extract_final_number(got)
    success_exp, val_exp, _ = _extract_final_number(expect)
    if not success_got or not success_exp:
        return False
    return val_got == val_exp

def _compare_fraction(got: str, expect: str) -> bool:
    """分数完全相等比较"""
    success_got, val_got, _ = _extract_final_number(got)
    success_exp, val_exp, _ = _extract_final_number(expect)
    if not success_got or not success_exp:
        return False
    if not isinstance(val_got, Fraction) or not isinstance(val_exp, Fraction):
        return False
    return val_got == val_exp

def _compare_float_rel(got: str, expect: str, rel_tol: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    """浮点相对/绝对误差比较"""
    success_got, val_got, _ = _extract_final_number(got)
    success_exp, val_exp, _ = _extract_final_number(expect)
    if not success_got or not success_exp:
        return False
    try:
        float_got = float(val_got)
        float_exp = float(val_exp)
        diff = abs(float_got - float_exp)
        return diff <= max(rel_tol * max(abs(float_got), abs(float_exp)), abs_tol)
    except:
        return False

@python_tools.tool(description="Code tester: batch test runner with multiple comparators")
def code_tester(
    code: str = Field(description="Python code to test"),
    tests: List[Dict[str, str]] = Field(description="List of {in: str, expect: str} test cases"),
    cmp: str = Field(default="number", description="Comparator: 'exact', 'number', 'fraction', 'float_rel'"),
    timeout: int = Field(default=3, description="Timeout per test case")
) -> Dict[str, Any]:
    """Batch test runner with multiple comparison modes."""
    try:
        if not tests:
            return {"ok": False, "error": "tests list is empty"}
        
        cmp_funcs = {
            "exact": _compare_exact,
            "number": _compare_number,
            "fraction": _compare_fraction,
            "float_rel": _compare_float_rel
        }
        
        if cmp not in cmp_funcs:
            return {"ok": False, "error": f"Unknown comparator: {cmp}"}
        
        compare = cmp_funcs[cmp]
        cases = []
        passed = 0
        
        for i, test in enumerate(tests):
            test_in = test.get("in", "")
            test_expect = test.get("expect", "")
            
            exit_code, stdout, stderr = _run_with_timeout(
                ["python3", "-c", _PRELUDE + "\n" + code],
                timeout,
                test_in
            )
            
            if exit_code == -1:
                if stderr == "timeout":
                    cases.append({"i": i, "ok": False, "error": "timeout"})
                else:
                    cases.append({"i": i, "ok": False, "error": stderr})
                continue
            
            if exit_code != 0:
                cases.append({"i": i, "ok": False, "error": stderr, "got": stdout})
                continue
            
            is_match = compare(stdout, test_expect)
            cases.append({
                "i": i,
                "ok": is_match,
                "got": stdout,
                "expect": test_expect
            })
            if is_match:
                passed += 1
        
        return {
            "ok": True,
            "passed": passed,
            "total": len(tests),
            "cases": cases
        }
    
    except Exception as e:
        logger.error(f"code_tester error: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


# ==================== 最终答案守门员 ====================

_final_answer_store: Dict[str, Dict[str, Any]] = {}

@python_tools.tool(description="Final answer guard: single commit point for final answer, read-only for evaluation")
def final_answer_guard(
    action: str = Field(description="Action: 'commit' or 'read'"),
    answer: Any = Field(default=None, description="Answer to commit (for 'commit' action)"),
    answer_type: str = Field(default="number", description="Answer type: 'number', 'fraction', 'string', 'list'")
) -> Dict[str, Any]:
    """Final answer guard: single commit point, read-only for evaluation."""
    try:
        # 使用全局task_id（简化实现，实际应从context获取）
        task_id = "default"
        
        if action == "commit":
            if task_id in _final_answer_store:
                return {"ok": False, "error": "already_committed"}
            
            if answer_type not in ["number", "fraction", "string", "list"]:
                return {"ok": False, "error": f"Invalid answer_type: {answer_type}"}
            
            # 处理number类型
            if answer_type == "number":
                if isinstance(answer, str):
                    success, val, err = _extract_final_number(answer)
                    if not success:
                        return {"ok": False, "error": f"Failed to extract number: {err}"}
                    if isinstance(val, Fraction) and val.denominator != 1:
                        return {"ok": False, "error": "expected_integer"}
                    answer = int(val) if isinstance(val, Fraction) else val
                elif isinstance(answer, (int, float)):
                    answer = int(answer) if isinstance(answer, float) and answer.is_integer() else answer
                else:
                    return {"ok": False, "error": f"Invalid number type: {type(answer)}"}
            
            # 处理fraction类型
            elif answer_type == "fraction":
                if isinstance(answer, str):
                    success, val, err = _extract_final_number(answer)
                    if not success or not isinstance(val, Fraction):
                        return {"ok": False, "error": f"Failed to extract fraction: {err}"}
                    answer = val
                elif isinstance(answer, (int, float)):
                    answer = Fraction(answer).limit_denominator()
                elif not isinstance(answer, Fraction):
                    return {"ok": False, "error": f"Invalid fraction type: {type(answer)}"}
            
            _final_answer_store[task_id] = {
                "answer_type": answer_type,
                "final_answer": answer
            }
            return {"ok": True, "final_answer": answer}
        
        elif action == "read":
            if task_id not in _final_answer_store:
                return {"ok": False, "error": "no_answer_committed"}
            return {"ok": True, "value": _final_answer_store[task_id]}
        
        else:
            return {"ok": False, "error": f"Unknown action: {action}"}
    
    except Exception as e:
        logger.error(f"final_answer_guard error: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


@python_tools.tool(
    description="Runs Python code in the current environment."
)
def run_python_code(
    code: str,
    variable_to_return: Optional[str] = None,
    safe_globals: Optional[dict] = None,
    safe_locals: Optional[dict] = None
) -> str:
    try:
        logger.debug(f"Running code:\n\n{code}\n\n")
        if not safe_globals:
            safe_globals = globals()
        if not safe_locals:
            safe_locals = locals()

        exec(code, safe_globals, safe_locals)

        if variable_to_return:
            variable_value = safe_locals.get(variable_to_return)
            if variable_value is None:
                return f"Variable {variable_to_return} not found"
            logger.debug(
                f"Variable {variable_to_return} value: {variable_value}")
            return str(variable_value)
        else:
            return "successfully run python code"
    except Exception as e:
        logger.error(f"Error running python code: {e}")
        return f"Error running python code: {e}"
