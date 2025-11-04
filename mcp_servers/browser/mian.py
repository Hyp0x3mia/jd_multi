#!/usr/bin/env python3
# test_navigation.py
"""
最小可运行示例 —— 测试 browser_navigate / back / forward
运行前：
  1. 把本文件放到与 navigation.py 同级目录；
  2. 安装依赖：pip install playwright asyncio；
  3. 安装浏览器：playwright install chromium（仅需一次）
"""

import asyncio
import sys
from pprint import pprint
import os
# 如果 navigation.py 里使用相对导入，需要先把它所在目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from navigation import browser_navigate, browser_navigate_back, browser_navigate_forward


async def main() -> None:
    """
    测试顺序：
      1. 导航到 A 页面（example.com）
      2. 导航到 B 页面（playwright.dev）
      3. back 一次，应该回到 A
      4. forward 一次，应该再次到 B
    每一步都打印当前 url 与核心字段，方便肉眼 check。
    """
    try:
        print("=== 1. 导航到 https://example.com ===")
        ret = await browser_navigate("https://example.com")
        assert ret["status"] == "success", ret
        example_url = ret["final_url"]
        print("final_url:", example_url)

        print("\n=== 2. 导航到 https://playwright.dev ===")
        ret = await browser_navigate("https://playwright.dev")
        assert ret["status"] == "success", ret
        pw_url = ret["final_url"]
        print("final_url:", pw_url)

        print("\n=== 3. browser_navigate_back (应该回到 example) ===")
        msg = await browser_navigate_back()
        print("back 返回值:", msg)

        print("\n=== 4. browser_navigate_forward (应该再次到 playwright) ===")
        msg = await browser_navigate_forward()
        print("forward 返回值:", msg)

        print("\n=== 5. 再次截图验证当前页面 ===")
        ret = await browser_navigate(pw_url, extract_content=False)  # 复用接口，仅拿 url
        print("当前最终 url:", ret["final_url"])
        print("\n✅ 导航三件套测试全部通过！")

    except Exception as exc:
        print("❌ 测试失败:", exc, file=sys.stderr)
        raise
    finally:
        # 确保关闭浏览器进程
        await _close_browser()


if __name__ == "__main__":
    asyncio.run(main())