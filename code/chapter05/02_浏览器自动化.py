import asyncio
import os
import sys
import types
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from playwright.async_api import async_playwright
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.navigate import NavigateTool
from langchain_community.tools.playwright.utils import aget_current_page
from langchain_core.tools import tool

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

# 全局浏览器引用
_browser_instance = None


@tool
async def summarize_website(url: str) -> str:
    """访问指定网站 URL 提取网页关键文本，并使用大模型总结内容。

    :param url: 需要访问和总结的网站 URL 地址
    :return: 网页的核心总结文本
    """
    if _browser_instance is None:
        return "错误：浏览器实例尚未初始化。"

    page = await aget_current_page(_browser_instance)
    try:
        await page.goto(url, timeout=60000, wait_until="domcontentloaded")
        page_text = await page.evaluate("() => document.body.innerText")
    except Exception as e:
        page_text = f"网页访问提示: {str(e)}"

    truncated_text = page_text[:8000] if len(page_text) > 8000 else page_text
    prompt_msg = f"请简明扼要地总结以下网页的主要内容与核心要点：\n\n{truncated_text}"

    summary_response = await llm.ainvoke(prompt_msg)
    return summary_response.content


@tool
def generate_pdf(content: str, filename: str = "summary.pdf") -> str:
    """将给定的总结文本生成并保存为本地 PDF 文件。

    :param content: 需要生成 PDF 的文本总结内容
    :param filename: PDF 文件基础名称，默认为 summary.pdf（程序会自动追加当前时间戳防覆盖）
    :return: PDF 文件的保存路径及生成状态
    """
    output_dir = Path("py_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成当前格式化时间戳 (格式如: 20260811_145730)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(filename).stem if filename else "summary"
    filepath = output_dir / f"{stem}_{timestamp}.pdf"

    # 优先注册常见中文字体，无则降级使用系统默认字体 (Helvetica)
    font_name = "Helvetica"
    possible_fonts = [
        ("SimHei", "C:/Windows/Fonts/simhei.ttf"),
        ("MicrosoftYaHei", "C:/Windows/Fonts/msyh.ttc"),
        ("SimSun", "C:/Windows/Fonts/simsun.ttc"),
    ]
    for name, path in possible_fonts:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                font_name = name
                break
            except Exception:
                pass

    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    styles = getSampleStyleSheet()
    custom_style = ParagraphStyle(
        name="ChineseStyle",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=11,
        leading=16,
        spaceAfter=8,
    )

    story = []
    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        if line:
            clean_text = (
                line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            story.append(Paragraph(clean_text, custom_style))

    doc.build(story)
    return f"PDF 文件已成功生成并保存至：{filepath.resolve()}"


async def main():
    global _browser_instance
    print("🚀 正在启动 Playwright 异步浏览器...")

    async with async_playwright() as p:
        _browser_instance = await p.chromium.launch(headless=True)

        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=_browser_instance)
        pw_tools = toolkit.get_tools()

        for t in pw_tools:
            if isinstance(t, NavigateTool):
                async def _safe_navigate_async(self_tool, url: str, run_manager=None) -> str:
                    page = await aget_current_page(self_tool.async_browser)
                    try:
                        response = await page.goto(url, timeout=60000, wait_until="domcontentloaded")
                        status = response.status if response else "unknown"
                        return f"Navigating to {url} returned status code {status}"
                    except Exception as e:
                        return f"Navigating to {url} completed with warning: {str(e)}"

                t._arun = types.MethodType(_safe_navigate_async, t)

        all_tools = [summarize_website, generate_pdf] + pw_tools

        agent = create_agent(
            model=llm,
            tools=all_tools,
            system_prompt="你是智能浏览器与文档生成助手。请根据用户的请求依次调用网站总结工具 summarize_website 和 PDF 生成工具 generate_pdf，将网站的核心总结内容保存为本地 PDF 文件。",
        )

        command = "访问这个网站 https://www.microsoft.com/en-us/microsoft-365/blog/2026/07/30/the-next-measure-of-ai-momentum-is-work-transformed/ 并帮我总结该网站的内容，然后将总结结果生成并保存为 PDF 文件"

        print("🔍 Agent 开始执行多工具协作流程（总结网页 + 生成 PDF 报告），请稍候...\n")
        result = await agent.ainvoke({"messages": [("user", command)]})

        print("===================== 🎉 Agent 执行结果 =====================\n")
        print(result["messages"][-1].content)

        await _browser_instance.close()


if __name__ == "__main__":
    asyncio.run(main())
