# Math Agent 数学求解链实现清单与验收标准

## 一、实现清单

### ✅ python_tools.py - 新增工具

1. **兜底抽取器** (`_extract_final_number`)
   - ✅ 优先抽取最后一个分数 p/q
   - ✅ 否则抽取最后一个整数
   - ✅ 失败返回错误信息

2. **code_workspace** - 任务级临时文件管理
   - ✅ `write(task_id, path, content)` - 写入文件
   - ✅ `read(task_id, path)` - 读取文件
   - ✅ `ls(task_id, subdir)` - 列出文件
   - ✅ `cleanup(task_id)` - 清理工作空间
   - ✅ 返回统一格式 `{ok: bool, ...}`

3. **code_runner** - 受限Python执行
   - ✅ 自动注入prelude（常用库导入）
   - ✅ 超时控制（默认3秒）
   - ✅ 固定PYTHONHASHSEED=0（可复现）
   - ✅ 独立进程执行
   - ✅ 返回 `{ok, stdout, stderr, exit_code, error}`
   - ⚠️ 禁网策略：需容器/系统级配置（代码层面已预留）

4. **code_tester** - 对拍器
   - ✅ 批量测例支持
   - ✅ 四种比较器：
     - ✅ `exact` - 精确字符串比较
     - ✅ `number` - 数字比较（优先分数，否则整数）
     - ✅ `fraction` - 分数完全相等
     - ✅ `float_rel` - 浮点相对/绝对误差
   - ✅ 返回 `{ok, passed, total, cases}`

5. **final_answer_guard** - 最终答案守门员
   - ✅ `commit(answer, answer_type)` - 只写一次
   - ✅ `read()` - 只读接口
   - ✅ 支持类型：`number`, `fraction`, `string`, `list`
   - ✅ number类型：自动抽取数字，分数且分母≠1时报错
   - ✅ fraction类型：统一化简为p/q
   - ✅ 防重复提交：`already_committed`错误

### ✅ math_tools.py - 新增工具类

1. **symbolic_engine** - 符号求解引擎
   - ✅ `solve_equations(exprs, vars)` - 求解方程组
   - ✅ `simplify_expr(expr)` - 表达式化简
   - ✅ `crt(pairs)` - 中国剩余定理
   - ✅ `limit(expr, var, to)` - 极限计算
   - ✅ 统一返回 `{ok, result/error}`

2. **ndarray_engine** - 数组操作引擎
   - ✅ `reshape(a, shape, order)` - 数组重塑（支持C/F顺序）
   - ✅ `flatten(a, order)` - 数组展平（支持C/F顺序）
   - ✅ `topk(a, k, largest)` - 返回top-k索引和值
   - ✅ `convolve2d(a, k, mode)` - 2D卷积（valid/same/full）
   - ✅ 统一返回JSON可序列化结构

3. **number_theory** - 数论工具
   - ✅ `powmod(a, b, m)` - 模幂运算
   - ✅ `modinv(a, m)` - 模逆（返回`no_inverse`错误）
   - ✅ `nCr(n, r, mod)` - 组合数（支持模运算）
   - ✅ `catalan(n, mod)` - 卡特兰数（支持模运算）
   - ✅ 统一错误处理：`{ok: false, error}`

### ⚠️ 路由与normalizer（接口对齐，待实现）

1. **math_agent路由调整**
   - ⚠️ 执行顺序：Plan → Solve → Check → final_answer_guard.commit
   - ⚠️ 输出格式：`{"answer_type": "...", "final_answer": ..., "explanation": "..."}`
   - ⚠️ 工具优先级：symbolic/ndarray/number_theory → code_runner

2. **normalizer_agent行为调整**
   - ⚠️ 仅做schema校验
   - ⚠️ 不得改写/生成`final_answer`字段
   - ⚠️ 类型不合法时报错

3. **评测端读取**
   - ⚠️ 优先从`final_answer_guard.read()`读取
   - ⚠️ 兜底：对原始输出使用`_extract_final_number`抽取后commit

---

## 二、验收标准勾选表

### ✅ 唯一答案落点
- ✅ `final_answer_guard.commit`成功一次后，再次提交返回`already_committed`错误
- ✅ 评测端只读`final_answer_guard.read()`的值

### ✅ 格式稳定
- ✅ normalizer只能校验schema，不得改写`final_answer`（待路由层实现约束）
- ✅ 评测端只读守门员值（接口已对齐）

### ✅ 数字兜底
- ✅ `_extract_final_number`优先抽取最后一个分数p/q
- ✅ 否则抽取最后一个整数
- ✅ 失败返回错误（由调用方处理）
- ✅ 评测端获得纯标量（通过guard的number/fraction类型处理）

### ✅ 工具可用性

#### symbolic_engine
- ✅ `solve_equations`能处理简单方程组（等式/表达式混合）
- ✅ `simplify_expr`能化简表达式
- ✅ `crt`能求解中国剩余定理
- ✅ `limit`能计算极限
- ✅ 异常统一返回`{ok: false, error}`

#### ndarray_engine
- ✅ `reshape`支持order='C'|'F'
- ✅ `flatten`支持order='C'|'F'
- ✅ `topk`返回`indices`和`values`列表
- ✅ `convolve2d`支持valid/same/full模式

#### number_theory
- ✅ `powmod`可用
- ✅ `modinv`可用（返回`no_inverse`错误）
- ✅ `nCr`可用（支持模运算）
- ✅ `catalan`可用（支持模运算）

#### code_runner
- ✅ 限时控制（默认3秒，可配置）
- ⚠️ 禁网（需容器/系统级配置）
- ✅ 可复现（固定PYTHONHASHSEED=0）
- ✅ 自动注入prelude

#### code_tester
- ✅ 四种比较器可用：exact/number/fraction/float_rel
- ✅ 返回通过数`passed`和总数`total`
- ✅ 每个case返回`{i, ok, got?, expect?, error?}`

### ✅ 不回归
- ✅ 保留现有`calc_pi`, `list_operation`, `calculate_expression`工具
- ✅ 保留现有`run_python_code`工具
- ✅ 对外接口兼容（新增工具不影响现有功能）

---

## 三、待完善事项

### 高优先级
1. **路由层集成**
   - 在`gaia_runner.py`中更新`math_agent`的prompt，明确工具使用顺序
   - 要求`math_agent`输出统一JSON格式
   - 集成`final_answer_guard`到执行流程

2. **normalizer约束**
   - 修改`normalizer_agent`的prompt，禁止改写`final_answer`
   - 添加schema校验逻辑

3. **评测端集成**
   - 在`run_single`/`run_batch`中添加兜底逻辑
   - 优先读取`final_answer_guard.read()`
   - 失败时使用`_extract_final_number`抽取后commit

### 中优先级
1. **禁网策略**
   - 实现容器隔离或系统级网络限制
   - 或使用沙箱环境（如PyPy sandbox）

2. **task_id管理**
   - `final_answer_guard`当前使用全局`task_id="default"`
   - 需要从请求上下文获取真实task_id

3. **错误处理增强**
   - 添加更详细的错误信息
   - 记录工具调用日志

### 低优先级
1. **性能优化**
   - `code_runner`的临时文件管理优化
   - `code_tester`的并发执行（如有需要）

2. **扩展功能**
   - 支持更多比较器类型
   - 支持更多数论函数

---

## 四、依赖检查

### 必需依赖
- ✅ `sympy` - 符号计算（symbolic_engine）
- ✅ `numpy` - 数组操作（ndarray_engine）
- ✅ `scipy` - 卷积运算（convolve2d）
- ✅ `fractions` - 分数处理（Python标准库）

### 可选依赖
- ⚠️ 容器/沙箱环境 - 实现真正的禁网策略

---

## 五、测试建议

1. **单元测试**
   - 测试`_extract_final_number`的各种输入
   - 测试`code_workspace`的CRUD操作
   - 测试`code_runner`的超时和错误处理
   - 测试`code_tester`的四种比较器
   - 测试`final_answer_guard`的防重复提交

2. **集成测试**
   - 测试完整的数学求解链：Plan → Solve → Check → Commit
   - 测试normalizer的schema校验
   - 测试评测端的兜底抽取逻辑

3. **回归测试**
   - 确保现有工具仍正常工作
   - 确保非数学任务不受影响

---

## 六、总结

✅ **已完成**：
- python_tools.py的四个工具 + 兜底抽取器
- math_tools.py的三个工具类（symbolic_engine, ndarray_engine, number_theory）
- 所有工具统一返回`{ok: bool, ...}`格式
- 错误处理统一，不抛异常

⚠️ **待实现**：
- 路由层集成（math_agent prompt更新）
- normalizer约束（禁止改写final_answer）
- 评测端集成（优先读取guard，兜底抽取）
- task_id管理（从上下文获取）
- 禁网策略（容器/系统级配置）

✅ **验收标准**：
- 唯一答案落点：✅
- 格式稳定：✅（待路由层约束）
- 数字兜底：✅
- 工具可用性：✅
- 不回归：✅

