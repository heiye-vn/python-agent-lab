"""
LangGraph + Zep Cloud Demo
==========================
一个带长期记忆的 LangGraph Agent，使用 Zep Cloud 托管方案。

核心流程：
  1. 在 Zep Cloud 上创建用户和线程（thread）
  2. 每轮对话前，从 Zep 获取用户上下文（facts / entities / summary）注入 system prompt
  3. Agent 可主动调用 graph_search 工具搜索知识图谱
  4. 每轮对话后，将 user + assistant 消息持久化回 Zep，异步更新知识图谱

Zep 的核心价值：
  - 时序知识图谱：自动追踪事实的生效/失效时间
  - 跨会话记忆：一个用户的所有 thread 共享同一知识图谱
  - 异步提取：消息写入后 Zep 在后台提取实体和关系，不阻塞对话

前置条件：
  - Zep Cloud 账号 + API Key（https://app.getzep.com/）
  - OpenAI API Key
  - Python 3.11+
  - pip install zep-langgraph langchain-openai zep-cloud
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from zep_cloud import Message
from zep_cloud.client import AsyncZep
from zep_langgraph import (
    build_system_message,
    create_graph_search_tool,
    ensure_thread,
    ensure_user,
    persist_messages,
)

# ============================================================
# 1. 初始化
# ============================================================
load_dotenv()

ZEP_API_KEY = os.environ.get("ZEP_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not ZEP_API_KEY:
    raise RuntimeError("请在 .env 中设置 ZEP_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("请在 .env 中设置 OPENAI_API_KEY")

# AsyncZep 是异步客户端，推荐在 LangGraph 异步流程中使用
# 全局只创建一个实例，复用连接
zep = AsyncZep(api_key=ZEP_API_KEY)

# LLM
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 配置（实际应用中从你的用户系统获取）
USER_ID = "demo-user-001"
THREAD_ID = "demo-thread-001"
USER_FIRST_NAME = "Alice"
USER_LAST_NAME = "Smith"


# ============================================================
# 2. 构建 Agent
# ============================================================
async def build_agent():
    """
    创建一个 LangGraph ReAct Agent，集成 Zep 的三大能力：

    - build_system_message: 每轮调用前从 Zep 拉取用户上下文，拼接到 system prompt
    - create_graph_search_tool: 给 Agent 一个工具，可主动搜索知识图谱
    - persist_messages: 每轮结束后把对话写回 Zep
    """

    # 2a. 确保用户和线程存在（幂等操作，已存在则跳过）
    await ensure_user(
        zep,
        user_id=USER_ID,
        first_name=USER_FIRST_NAME,
        last_name=USER_LAST_NAME,
        email="alice@example.com",
    )
    await ensure_thread(zep, thread_id=THREAD_ID, user_id=USER_ID)

    # 2b. 定义 prompt 函数：每轮动态注入 Zep 上下文
    async def prompt(state):
        """
        build_system_message 会：
        1. 调用 thread.get_user_context() 获取用户的知识图谱上下文
        2. 将上下文包装成 <ZEP_CONTEXT>...</ZEP_CONTEXT> 标签
        3. 拼接到 base_instructions 之后，形成完整的 system message
        """
        system = await build_system_message(
            zep,
            thread_id=THREAD_ID,
            base_instructions=(
                "你是一个有帮助的 AI 助手。你可以记住用户的偏好和历史对话内容。"
                "如果用户提到了个人信息（如工作、偏好、计划），请记住它。"
                "如果不确定用户的信息，可以调用搜索工具查询知识图谱。"
            ),
        )
        return [system, *state["messages"]]

    # 2c. 创建 graph search 工具
    # Agent 可以在需要时调用此工具搜索用户的时序知识图谱
    # pinned_params 固定搜索参数，不让模型随意修改
    search_tool = create_graph_search_tool(
        zep,
        user_id=USER_ID,
        pinned_params={"limit": 10},  # 最多返回 10 条结果
    )

    # 2d. 构建 ReAct Agent
    agent = create_react_agent(
        model=model,
        tools=[search_tool],
        prompt=prompt,
    )

    return agent


# ============================================================
# 3. 对话循环
# ============================================================
async def chat_turn(agent, user_input: str) -> str:
    """
    执行一轮对话：
    1. 将用户消息交给 Agent
    2. Agent 自动注入 Zep 上下文 + 可能调用搜索工具 + 生成回复
    3. 将本轮 user + assistant 消息持久化回 Zep
    """
    result = await agent.ainvoke({
        "messages": [HumanMessage(content=user_input)],
    })

    reply = result["messages"][-1]

    # 持久化到 Zep
    # 注意：传入 name 字段帮助 Zep 正确识别用户身份
    await persist_messages(
        zep,
        thread_id=THREAD_ID,
        messages=[
            Message(
                role="user",
                content=user_input,
                name=f"{USER_FIRST_NAME} {USER_LAST_NAME}",
            ),
            Message(
                role="assistant",
                content=reply.content,
                name="AI Assistant",
            ),
        ],
    )

    return reply.content


# ============================================================
# 4. 交互式聊天
# ============================================================
async def main():
    print("=" * 60)
    print("  LangGraph + Zep Cloud  —  带长期记忆的 AI Agent")
    print("=" * 60)
    print(f"  用户: {USER_FIRST_NAME} {USER_LAST_NAME} ({USER_ID})")
    print(f"  线程: {THREAD_ID}")
    print("=" * 60)
    print()
    print("提示: 试试先告诉 Agent 你的信息，然后在新对话中问它是否记得。")
    print("输入 'quit' 或 'exit' 退出。")
    print()

    agent = await build_agent()

    # 演示：多轮对话展示记忆能力
    while True:
        user_input = input("🧑 You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        print("🤖 Assistant: ", end="", flush=True)
        try:
            reply = await chat_turn(agent, user_input)
            print(reply)
        except Exception as e:
            print(f"\n[错误] {e}")

        print()


if __name__ == "__main__":
    asyncio.run(main())
