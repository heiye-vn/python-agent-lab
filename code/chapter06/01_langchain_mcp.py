import asyncio
import os
import sys
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

# 解决 Windows 终端 GBK 编码导致的 UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8")

# 开启全局调试日志
# langchain.debug = True


class Configuration:
    def __init__(self) -> None:
        load_dotenv(Path(__file__).parent / ".env")
        self.api_key = os.getenv("ALI_BAILIAN_API_KEY")
        self.model = "qwen3.7-max-2026-05-20"
        self.base_url = os.getenv(
            "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    @staticmethod
    def load_servers(file_path=Path(__file__).parent / "servers_config.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f).get("mcpServers", {})


async def run_chat_loop():
    """启动 MCP-Agent 聊天循环"""
    cfg = Configuration()
    sercers_cfg = Configuration.load_servers()

    # 1️⃣ 连接多台 MCP 服务器
    mcp_client = MultiServerMCPClient(sercers_cfg)

    tools = await mcp_client.get_tools()

    logging.info(f"✅ 已加载 {len(tools)} 个 MCP 工具： {[t.name for t in tools]}")

    # 2️⃣ 初始化大模型
    llm = init_chat_model(
        model=cfg.model,
        model_provider="openai",
        base_url=cfg.base_url,
        api_key=cfg.api_key,
    )

    # 3️⃣ 构造 LangChain Agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="你是浏览器自动化助手，请根据用户需求使用浏览器工具访问网站、获取网页内容并回答用户的问题。",
    )

    # 4️⃣ CLI 聊天
    print("\n🤖 MCP Agent 已启动，输入 'quit' 退出")
    messages = []
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        messages.append(("user", user_input))
        try:
            last_messages = messages
            async for event in agent.astream(
                {"messages": messages}, stream_mode="values"
            ):
                # 打印最新产生的消息（包含 AI 的 Thought、Tool Call 和 Tool 返回结果）
                latest_msg = event["messages"][-1]
                latest_msg.pretty_print()
                last_messages = event["messages"]
            
            # 更新上下文列表，以便下一轮对话保持历史
            messages = last_messages

        except Exception as exc:
            print(f"\n⚠️  出错: {exc}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    asyncio.run(run_chat_loop())
