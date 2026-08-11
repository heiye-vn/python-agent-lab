import asyncio
import os
import sys
import types
from pathlib import Path

from dotenv import load_dotenv
from playwright.async_api import async_playwright

# 引入 LangChain 1.x 官方推荐的 create_agent 入口 (基于 LangGraph 架构)
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.navigate import NavigateTool
from langchain_community.tools.playwright.utils import aget_current_page

# 解决 Windows 终端 GBK 编码导致的 UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(Path(__file__).parent / ".env")

# 1. 初始化 Chat Model
llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)


async def main():
    print("🚀 正在启动 Playwright 异步浏览器...")

    # 2. 在 main 内部初始化 Playwright (生命周期绑定同一 asyncio 事件循环)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        tools = toolkit.get_tools()

        # 3. 增强 NavigateTool 容错性：改用 domcontentloaded 策略 + 60s 超时
        for t in tools:
            if isinstance(t, NavigateTool):

                async def _safe_navigate_async(
                    self_tool, url: str, run_manager=None
                ) -> str:
                    page = await aget_current_page(self_tool.async_browser)
                    try:
                        response = await page.goto(
                            url, timeout=60000, wait_until="domcontentloaded"
                        )
                        status = response.status if response else "unknown"
                        return f"Navigating to {url} returned status code {status}"
                    except Exception as e:
                        return f"Navigating to {url} completed with warning: {str(e)}"

                t._arun = types.MethodType(_safe_navigate_async, t)

        # 4. 使用 LangChain 1.x 推荐的 create_agent 构建 Agent
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="你是浏览器自动化助手，请根据用户需求使用浏览器工具访问网站、获取网页内容并回答用户的问题。",
        )

        command = "访问这个网站 https://www.microsoft.com/en-us/microsoft-365/blog/2026/07/30/the-next-measure-of-ai-momentum-is-work-transformed/ 并帮我总结一下该网站的内容"

        print(
            "🔍 Agent 正在后台打开浏览器、导航网页并进行智能总结，请稍候 15~20 秒...\n"
        )

        # 5. 执行任务
        result = await agent.ainvoke({"messages": [("user", command)]})

        print("===================== 🎉 总结结果 =====================\n")
        print(result["messages"][-1].content)

        # 6. 关闭浏览器
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
