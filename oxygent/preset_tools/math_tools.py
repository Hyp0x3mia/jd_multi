from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import Field

from oxygent.oxy import FunctionHub

math_tools = FunctionHub(name="math_tools")

# ==================== Symbolic Engine (merged into math_tools) ====================

@math_tools.tool(description="Symbolic engine: solve equations")
def solve_equations(
    exprs: List[str] = Field(description="List of equations/expressions (equations with '=' or default to 0)"),
    vars: List[str] = Field(description="List of variable names to solve for")
) -> Dict[str, Any]:
    """Solve system of equations."""
    try:
        from sympy import symbols, Eq, solve, sympify
        
        sym_vars = symbols(vars)
        equations = []
        for expr in exprs:
            if '=' in expr:
                left, right = expr.split('=', 1)
                eq = Eq(sympify(left.strip()), sympify(right.strip()))
            else:
                eq = Eq(sympify(expr.strip()), 0)
            equations.append(eq)
        
        solutions = solve(equations, sym_vars, dict=True)
        if not solutions:
            return {"ok": True, "solution": {}}
        
        # 取第一个解，转换为字符串
        sol = solutions[0]
        result = {str(k): str(v) for k, v in sol.items()}
        return {"ok": True, "solution": result}
    
    except Exception as e:
        return {"ok": False, "error": str(e)}

@math_tools.tool(description="Symbolic engine: simplify expression")
def simplify_expr(
    expr: str = Field(description="Expression to simplify")
) -> Dict[str, Any]:
    """Simplify mathematical expression."""
    try:
        from sympy import sympify, simplify
        
        sym_expr = sympify(expr)
        result = simplify(sym_expr)
        return {"ok": True, "result": str(result)}
    
    except Exception as e:
        return {"ok": False, "error": str(e)}

@math_tools.tool(description="Symbolic engine: Chinese Remainder Theorem")
def crt(
    pairs: List[Tuple[int, int]] = Field(description="List of (remainder, modulus) pairs")
) -> Dict[str, Any]:
    """Chinese Remainder Theorem solver."""
    try:
        from sympy.ntheory.modular import crt
        
        moduli = [m for _, m in pairs]
        remainders = [r for r, _ in pairs]
        
        result, mod = crt(moduli, remainders)
        return {"ok": True, "result": int(result), "mod": int(mod)}
    
    except Exception as e:
        return {"ok": False, "error": str(e)}

@math_tools.tool(description="Symbolic engine: compute limit")
def limit(
    expr: str = Field(description="Expression"),
    var: str = Field(description="Variable name"),
    to: str = Field(description="Limit point (e.g., 'inf', '0', 'oo')")
) -> Dict[str, Any]:
    """Compute limit of expression."""
    try:
        from sympy import symbols, sympify, limit as sympy_limit, oo
        
        x = symbols(var)
        sym_expr = sympify(expr)
        
        if to.lower() in ['inf', 'oo', 'infinity']:
            limit_point = oo
        elif to.lower() in ['-inf', '-oo', '-infinity']:
            limit_point = -oo
        else:
            limit_point = sympify(to)
        
        result = sympy_limit(sym_expr, x, limit_point)
        return {"ok": True, "result": str(result)}
    
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ==================== NDArray Engine (merged into math_tools) ====================

def _to_nested_list(arr: Any) -> List:
    """Convert numpy array to nested list recursively."""
    import numpy as np
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr

@math_tools.tool(description="NDArray engine: reshape array")
def reshape(
    a: List[Any] = Field(description="Input array"),
    shape: List[int] = Field(description="Target shape"),
    order: str = Field(default="C", description="Order: 'C' or 'F'")
) -> Dict[str, Any]:
    """Reshape array to target shape."""
    try:
        import numpy as np
        
        arr = np.array(a)
        if order not in ['C', 'F']:
            return {"ok": False, "error": f"Invalid order: {order}"}
        
        result = arr.reshape(tuple(shape), order=order)
        return {"ok": True, "result": _to_nested_list(result)}
    
    except Exception as e:
        return {"ok": False, "error": str(e)}

@math_tools.tool(description="NDArray engine: flatten array")
def flatten(
    a: List[Any] = Field(description="Input array"),
    order: str = Field(default="C", description="Order: 'C' or 'F'")
) -> Dict[str, Any]:
    """Flatten array to 1D."""
    try:
        import numpy as np
        
        arr = np.array(a)
        if order not in ['C', 'F']:
            return {"ok": False, "error": f"Invalid order: {order}"}
        
        result = arr.flatten(order=order)
        return {"ok": True, "result": _to_nested_list(result)}
    
    except Exception as e:
        return {"ok": False, "error": str(e)}

@math_tools.tool(description="NDArray engine: top-k elements")
def topk(
    a: List[Union[int, float]] = Field(description="Input array"),
    k: int = Field(description="Number of top elements"),
    largest: bool = Field(default=True, description="Largest or smallest")
) -> Dict[str, Any]:
    """Get top-k elements with indices."""
    try:
        import numpy as np
        
        arr = np.array(a)
        if k > len(arr):
            return {"ok": False, "error": f"k ({k}) > array length ({len(arr)})"}
        
        if largest:
            indices = np.argsort(arr)[-k:][::-1]
        else:
            indices = np.argsort(arr)[:k]
        
        values = arr[indices].tolist()
        indices_list = indices.tolist()
        
        return {"ok": True, "indices": indices_list, "values": values}
    
    except Exception as e:
        return {"ok": False, "error": str(e)}

@math_tools.tool(description="NDArray engine: 2D convolution")
def convolve2d(
    a: List[List[Union[int, float]]] = Field(description="Input 2D array"),
    k: List[List[Union[int, float]]] = Field(description="Kernel 2D array"),
    mode: str = Field(default="valid", description="Mode: 'valid', 'same', or 'full'")
) -> Dict[str, Any]:
    """2D convolution."""
    try:
        from scipy.signal import convolve2d
        import numpy as np
        
        arr = np.array(a)
        kernel = np.array(k)
        
        if mode not in ['valid', 'same', 'full']:
            return {"ok": False, "error": f"Invalid mode: {mode}"}
        
        result = convolve2d(arr, kernel, mode=mode)
        return {"ok": True, "result": _to_nested_list(result)}
    
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ==================== Number Theory (merged into math_tools) ====================

@math_tools.tool(description="Number theory: modular exponentiation")
def powmod(
    a: int = Field(description="Base"),
    b: int = Field(description="Exponent"),
    m: int = Field(description="Modulus")
) -> Dict[str, Any]:
    """Compute (a^b) mod m efficiently."""
    try:
        result = pow(a, b, m)
        return {"ok": True, "result": result}
    
    except Exception as e:
        return {"ok": False, "error": str(e)}

@math_tools.tool(description="Number theory: modular inverse")
def modinv(
    a: int = Field(description="Number"),
    m: int = Field(description="Modulus")
) -> Dict[str, Any]:
    """Compute modular inverse: a^(-1) mod m."""
    try:
        from sympy import mod_inverse
        
        result = mod_inverse(a, m)
        return {"ok": True, "result": result}
    
    except Exception as e:
        if "no inverse" in str(e).lower() or "not invertible" in str(e).lower():
            return {"ok": False, "error": "no_inverse"}
        return {"ok": False, "error": str(e)}

@math_tools.tool(description="Number theory: binomial coefficient")
def nCr(
    n: int = Field(description="n"),
    r: int = Field(description="r"),
    mod: Optional[int] = Field(default=None, description="Optional modulus")
) -> Dict[str, Any]:
    """Compute C(n, r) = n! / (r! * (n-r)!)."""
    try:
        from math import comb
        
        if mod is None:
            result = comb(n, r)
        else:
            # 使用 Lucas 定理或其他方法（简化实现）
            result = comb(n, r) % mod
        
        return {"ok": True, "result": result}
    
    except Exception as e:
        return {"ok": False, "error": str(e)}

@math_tools.tool(description="Number theory: Catalan number")
def catalan(
    n: int = Field(description="n"),
    mod: Optional[int] = Field(default=None, description="Optional modulus")
) -> Dict[str, Any]:
    """Compute n-th Catalan number."""
    try:
        from math import comb
        
        # C(n) = C(2n, n) / (n + 1)
        cat = comb(2 * n, n) // (n + 1)
        
        if mod is not None:
            cat = cat % mod
        
        return {"ok": True, "result": cat}
    
    except Exception as e:
        return {"ok": False, "error": str(e)}

@math_tools.tool(description="A tool that can calculate the value of pi.")
def calc_pi(prec: int = Field(description="how many decimal places")) -> float:
    import math
    from decimal import Decimal, getcontext

    getcontext().prec = prec
    x = 0
    for k in range(int(prec / 8) + 1):
        a = 2 * Decimal.sqrt(Decimal(2)) / 9801
        b = math.factorial(4 * k) * (1103 + 26390 * k)
        c = pow(math.factorial(k), 4) * pow(396, 4 * k)
        x = x + a * b / c
    return 1 / x

@math_tools.tool(description="A tool that applies a binary operation to corresponding elements of two lists.")
def list_operation(
    list1: list = Field(description="The first list"),
    list2: list = Field(description="The second list"),
    operation: str = Field(description="The operation to perform: 'add', 'subtract', 'multiply', 'divide', 'power', 'mod'")
) -> list:
    """
    Apply a binary operation element-wise between two lists.
    
    Args:
        list1: The first list
        list2: The second list  
        operation: The operation to perform ('add', 'subtract', 'multiply', 'divide', 'power', 'mod')
        
    Returns:
        A new list containing the results of the operation
        
    Raises:
        ValueError: If the lists have different lengths or operation is invalid
    """
    import operator
    if len(list1) != len(list2):
        raise ValueError(f"Lists must have the same length. Got {len(list1)} and {len(list2)}")
    
    # 操作符映射
    operations = {
        'add': operator.add,
        'subtract': operator.sub,
        'multiply': operator.mul,
        'divide': operator.truediv,
        'power': operator.pow,
        'mod': operator.mod
    }
    
    if operation not in operations:
        raise ValueError(f"Invalid operation '{operation}'. Supported operations: {list(operations.keys())}")
    
    op_func = operations[operation]
    
    try:
        return [op_func(a, b) for a, b in zip(list1, list2)]
    except ZeroDivisionError:
        raise ValueError("Division by zero encountered in the operation")
    except Exception as e:
        raise ValueError(f"Error performing {operation}: {str(e)}")
    

@math_tools.tool(description="A tool that evaluates mathematical expressions and returns the expression with its result.")
def calculate_expression(
    expression: str = Field(description="Mathematical expression to evaluate (e.g., '5+6', '10*3-2', '(4+5)*2')")
) -> str:
    """
    Evaluate a mathematical expression and return it with the result.
    Args:
        expression: Mathematical expression string to evaluate
    Returns:
        A string in the format "expression=result" (e.g., "5+6=11")
    Raises:
        ValueError: If the expression is invalid or contains unsafe operations
    """
    import ast
    import operator
    
    # Define allowed operators for safety
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def safe_eval(node):
        """Safely evaluate an AST node."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.BinOp):
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            op = allowed_operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = safe_eval(node.operand)
            op = allowed_operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operation: {type(node.op).__name__}")
            return op(operand)
        elif isinstance(node, ast.Expression):
            return safe_eval(node.body)
        else:
            raise ValueError(f"Unsupported AST node type: {type(node).__name__}")
    
    try:
        # Remove whitespace and validate the expression
        clean_expression = expression.strip()
        if not clean_expression:
            raise ValueError("Empty expression")
        
        # Parse the expression into an AST
        tree = ast.parse(clean_expression, mode='eval')
        
        # Evaluate the expression safely
        result = safe_eval(tree)
        
        # Format the result appropriately
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        
        return f"{clean_expression}={result}"
        
    except SyntaxError:
        raise ValueError(f"Invalid mathematical expression: {expression}")
    except ZeroDivisionError:
        raise ValueError(f"Division by zero in expression: {expression}")
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expression}': {str(e)}")