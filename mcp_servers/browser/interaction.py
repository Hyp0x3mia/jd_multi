"""
浏览器交互功能

提供页面元素的点击、悬停和输入文本等交互操作
"""

import asyncio

from pydantic import Field

from .core import (
    _ensure_page,
    _set_operation_status,
    _verify_data_ready,
    check_dependencies,
    mcp,
)


@mcp.tool(description="点击元素，支持CSS选择器、文本定位器或可访问性名称")
async def browser_click(
    selector: str = Field(description="要点击的元素定位器（CSS选择器、文本内容如'text=All'、或角色+名称如'button[name=\"All\"]'）"),
    timeout: int = Field(default=5000, description="等待元素出现的超时时间(毫秒)"),
):
    """
    点击页面上的元素，支持多种定位策略
    """
    # 检查依赖
    missing_deps = check_dependencies()
    if missing_deps:
        return f"缺少必要的库: {', '.join(missing_deps)}。请使用pip安装: pip install {' '.join(missing_deps)}"

    try:
        await _set_operation_status(True)
        page = await _ensure_page()

        # 尝试多种定位策略
        clicked = False
        strategy_used = None
        
        # 策略1: 直接使用选择器（CSS 或 Playwright locator）
        try:
            await page.wait_for_selector(selector, timeout=timeout)
            await page.click(selector)
            clicked = True
            strategy_used = f"CSS选择器: {selector}"
        except Exception as e1:
            # 策略2: 如果选择器看起来像文本，尝试文本定位器
            try:
                # 移除可能的引号
                text_content = selector.strip().strip('"').strip("'")
                locator = page.locator(f"text={text_content}")
                await locator.wait_for(timeout=timeout)
                await locator.first.click()
                clicked = True
                strategy_used = f"文本定位器: text={text_content}"
            except Exception as e2:
                # 策略3: 尝试作为按钮文本
                try:
                    text_content = selector.strip().strip('"').strip("'")
                    locator = page.locator(f"button:has-text('{text_content}')")
                    await locator.wait_for(timeout=timeout)
                    await locator.first.click()
                    clicked = True
                    strategy_used = f"按钮文本: {text_content}"
                except Exception as e3:
                    # 策略4: 尝试作为链接文本
                    try:
                        text_content = selector.strip().strip('"').strip("'")
                        locator = page.locator(f"a:has-text('{text_content}')")
                        await locator.wait_for(timeout=timeout)
                        await locator.first.click()
                        clicked = True
                        strategy_used = f"链接文本: {text_content}"
                    except Exception as e4:
                        # 策略5: 尝试更通用的选择器
                        try:
                            # 如果是 input，尝试更通用的选择器
                            if 'input' in selector.lower():
                                generic_selector = "input[type='text'], input[type='search'], input:not([type='hidden'])"
                                await page.wait_for_selector(generic_selector, timeout=timeout)
                                inputs = await page.query_selector_all(generic_selector)
                                if inputs:
                                    await inputs[0].click()
                                    clicked = True
                                    strategy_used = f"通用输入框选择器"
                        except Exception:
                            pass

        if not clicked:
            await _set_operation_status(False)
            return f"无法找到元素: {selector}。建议：1) 先使用 browser_snapshot 查看页面结构；2) 使用 browser_analyze_screenshot 识别元素位置；3) 尝试文本定位器（如直接使用按钮文本 'All' 而不是 'input[name=\"q\"]'）"

        # 等待可能的页面变化
        await asyncio.sleep(1)
        await _verify_data_ready()

        await _set_operation_status(False)
        return f"成功点击元素（使用策略: {strategy_used}），数据已准备就绪"
    except Exception as e:
        await _set_operation_status(False)
        return f"点击元素 {selector} 时发生错误: {str(e)}。建议：1) 先使用 browser_snapshot 或 browser_analyze_screenshot 确认元素位置；2) 尝试使用文本定位器（如 'text=按钮文本'）或更通用的选择器"


@mcp.tool(description="悬停在元素上")
async def browser_hover(
    selector: str = Field(description="要悬停的元素的CSS选择器"),
    timeout: int = Field(default=5000, description="等待元素出现的超时时间(毫秒)"),
):
    """
    将鼠标悬停在页面上的元素上
    """
    # 检查依赖
    missing_deps = check_dependencies()
    if missing_deps:
        return f"缺少必要的库: {', '.join(missing_deps)}。请使用pip安装: pip install {' '.join(missing_deps)}"

    try:
        await _set_operation_status(True)
        page = await _ensure_page()

        # 等待元素出现
        await page.wait_for_selector(selector, timeout=timeout)

        # 悬停在元素上
        await page.hover(selector)

        # 等待可能的页面变化（如悬停菜单出现）
        await asyncio.sleep(0.5)
        await _verify_data_ready()

        await _set_operation_status(False)
        return f"成功悬停在元素上: {selector}，数据已准备就绪"
    except Exception as e:
        await _set_operation_status(False)
        return f"悬停在元素 {selector} 上时发生错误: {str(e)}"


@mcp.tool(description="在元素中输入文本，支持CSS选择器、文本定位器或可访问性名称")
async def browser_type(
    selector: str = Field(description="要输入文本的元素定位器（CSS选择器、文本内容或角色+名称）"),
    text: str = Field(description="要输入的文本"),
    timeout: int = Field(default=5000, description="等待元素出现的超时时间(毫秒)"),
):
    """
    在页面上的元素中输入文本，支持多种定位策略
    """
    # 检查依赖
    missing_deps = check_dependencies()
    if missing_deps:
        return f"缺少必要的库: {', '.join(missing_deps)}。请使用pip安装: pip install {' '.join(missing_deps)}"

    try:
        await _set_operation_status(True)
        page = await _ensure_page()

        # 尝试多种定位策略
        filled = False
        strategy_used = None
        
        # 策略1: 直接使用选择器
        try:
            await page.wait_for_selector(selector, timeout=timeout)
            await page.fill(selector, "")
            await page.type(selector, text)
            filled = True
            strategy_used = f"CSS选择器: {selector}"
        except Exception:
            # 策略2: 尝试作为输入框（更通用的选择器）
            try:
                # 尝试查找输入框
                generic_selector = "input[type='text'], input[type='search'], input:not([type='hidden'])"
                await page.wait_for_selector(generic_selector, timeout=timeout)
                inputs = await page.query_selector_all(generic_selector)
                if inputs:
                    await inputs[0].fill("")
                    await inputs[0].type(text)
                    filled = True
                    strategy_used = f"通用输入框定位器"
            except Exception:
                # 策略3: 尝试通过 placeholder 或 name 属性
                try:
                    # 如果 selector 看起来像属性值，尝试查找匹配的输入框
                    text_content = selector.strip().strip('"').strip("'")
                    attr_selector = f"input[placeholder*='{text_content}'], input[name*='{text_content}']"
                    await page.wait_for_selector(attr_selector, timeout=timeout)
                    await page.fill(attr_selector, "")
                    await page.type(attr_selector, text)
                    filled = True
                    strategy_used = f"属性匹配定位器: {text_content}"
                except Exception:
                    pass

        if not filled:
            await _set_operation_status(False)
            return f"无法找到输入元素: {selector}。建议：1) 先使用 browser_snapshot 查看页面结构；2) 使用 browser_analyze_screenshot 识别输入框位置；3) 尝试更通用的选择器（如 'input[type=\"text\"]'）"

        await _verify_data_ready()
        await _set_operation_status(False)
        return f"成功在元素（使用策略: {strategy_used}）中输入文本，数据已准备就绪"
    except Exception as e:
        await _set_operation_status(False)
        return f"在元素 {selector} 中输入文本时发生错误: {str(e)}。建议：1) 先使用 browser_snapshot 或 browser_analyze_screenshot 确认元素位置；2) 尝试更通用的输入框选择器"
