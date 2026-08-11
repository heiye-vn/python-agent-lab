import os
import sys
import types
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.navigate import NavigateTool
from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser,
    get_current_page,
)
from langchain_core.prompts import ChatPromptTemplate

# 解决 Windows 终端 GBK 编码导致的 UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(Path(__file__).parent / ".env")

llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)

# 初始化 Playwright 浏览器
sync_browser = create_sync_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = toolkit.get_tools()

# 优化 NavigateTool：改用 domcontentloaded 模式 + 60s 超时，防止因海外第三方追踪脚本加载缓慢而触发 30s TimeoutError
for t in tools:
    if isinstance(t, NavigateTool):
        def _safe_navigate(self_tool, url: str, run_manager=None) -> str:
            page = get_current_page(self_tool.sync_browser)
            try:
                response = page.goto(url, timeout=60000, wait_until="domcontentloaded")
                status = response.status if response else "unknown"
                return f"Navigating to {url} returned status code {status}"
            except Exception as e:
                return f"Navigating to {url} completed with warning: {str(e)}"

        t._run = types.MethodType(_safe_navigate, t)


# 构建 Agent 提示词模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是浏览器自动化助手，请根据用户需求使用浏览器工具访问网站、获取网页内容并回答用户的问题。",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    # 定义任务 (使用无访问限制且加载迅速的示例网站)
    # command = {
    #     "input": "访问这个网站 https://quotes.toscrape.com/ 并帮我总结页面上的主要名言内容"
    # }

    command = {
        "input": "访问这个网站 https://www.microsoft.com/en-us/microsoft-365/blog/2026/07/30/the-next-measure-of-ai-momentum-is-work-transformed/ 并帮我总结一下该网站的内容"
    }

    # 执行任务
    response = agent_executor.invoke(command)
    print(response)
