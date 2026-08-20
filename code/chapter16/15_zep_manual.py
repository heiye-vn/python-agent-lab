"""
LangGraph + Zep Cloud Demo（手动集成版）
=========================================
不依赖 zep-langgraph 包，直接用 zep_cloud SDK 在 LangGraph 中手动管理记忆。
适合理解 Zep 的底层工作原理。

与官方 zep-langgraph 包的区别：
  - 官方包：封装好 build_system_message / persist_messages / create_graph_search_tool
  - 手动版：自己调 thread.get_user_context() / thread.add_messages() / graph.search()

前置条件：
  - pip install zep-cloud langchain-openai langgraph
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from zep_cloud import Message
from zep_cloud.client import Zep

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

# 同步 Zep 客户端
zep = Zep(api_key=ZEP_API_KEY)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

USER_ID = "demo-user-002"
THREAD_ID = f"thread-{uuid.uuid4().hex[:8]}"
USER_NAME = "Bob Johnson"


# ============================================================
# 2. 设置 Zep 用户和线程
# ============================================================
def setup_zep():
    """创建 Zep 用户和线程"""
    try:
        zep.user.add(
            user_id=USER_ID,
            first_name="Bob",
            last_name="Johnson",
            email="bob@example.com",
        )
        print(f"[Zep] 用户创建: {USER_ID}")
    except Exception:
        print(f"[Zep] 用户已存在: {USER_ID}")

    zep.thread.create(thread_id=THREAD_ID, user_id=USER_ID)
    print(f"[Zep] 线程创建: {THREAD_ID}")


# ============================================================
# 3. 定义 LangGraph State
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str
    thread_id: str


# ============================================================
# 4. 定义工具：搜索 Zep 知识图谱
# ============================================================
@tool
async def search_user_facts(query: str) -> str:
    """
    搜索用户的时序知识图谱，查找与查询相关的事实和实体。
    当需要回忆用户的个人信息、偏好或历史交互时使用此工具。
    """
    results = await zep.graph.search(
        user_id=USER_ID,
        query=query,
        scope="edges",  # edges = 事实关系；nodes = 实体节点
        limit=10,
    )

    if not results.edges:
        return "知识图谱中没有找到相关信息。"

    facts = []
    for edge in results.edges:
        # 每条 edge 包含 fact 文本、有效时间范围等信息
        time_range = ""
        if edge.valid_at:
            valid_str = edge.valid_at.strftime("%Y-%m-%d") if edge.valid_at else "?"
            if edge.invalid_at:
                invalid_str = edge.invalid_at.strftime("%Y-%m-%d")
                time_range = f" (有效: {valid_str} ~ {invalid_str})"
            else:
                time_range = f" (有效: {valid_str} ~ 现在)"
        facts.append(f"- {edge.fact}{time_range}")

    return "\n".join(facts)


# ============================================================
# 5. 定义图节点
# ============================================================
async def chatbot_with_tools(state: State) -> dict:
    """
    核心节点：
    1. 从 Zep 获取用户上下文（facts + summary + entities）
    2. 将上下文注入 system prompt
    3. 调用 LLM 生成回复（可能带 tool_calls）
    4. 将 user + assistant 消息写回 Zep
    """
    messages = state["messages"]

    # 步骤 1: 从 Zep 获取上下文
    user_context = await zep.thread.get_user_context(thread_id=state["thread_id"])
    context_block = user_context.context or ""

    # 步骤 2: 构建 system prompt
    system_prompt = (
        "你是一个有帮助的 AI 助手。\n\n"
        "以下是关于用户的已知信息，来自知识图谱：\n"
        f"{context_block}\n\n"
        "请基于这些信息回答用户的问题。"
        "如果需要查找用户的历史信息，可以调用 search_user_facts 工具。"
    )

    # 步骤 3: 裁剪消息 + 调用 LLM
    trimmed = trim_messages(
        messages,
        token_counter=model,
        max_tokens=4000,
        strategy="last",
        start_on="human",
        allow_partial=False,
        include_system=True,
    )

    # 绑定工具到模型
    model_with_tools = model.bind_tools([search_user_facts])
    full_messages = [SystemMessage(content=system_prompt)] + trimmed
    response = await model_with_tools.ainvoke(full_messages)

    # 步骤 4: 持久化到 Zep
    last_user_msg = messages[-1] if messages else None
    messages_to_save = []
    if last_user_msg and isinstance(last_user_msg, HumanMessage):
        messages_to_save.append(
            Message(
                created_at=datetime.now(timezone.utc).isoformat(),
                name=state["user_name"],
                role="user",
                content=last_user_msg.content,
            )
        )
    messages_to_save.append(
        Message(
            created_at=datetime.now(timezone.utc).isoformat(),
            name="AI Assistant",
            role="assistant",
            content=response.content,
        )
    )
    await zep.thread.add_messages(
        thread_id=state["thread_id"],
        messages=messages_to_save,
    )

    return {"messages": [response]}


def should_use_tool(state: State) -> str:
    """判断是否需要调用搜索工具"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "end"


# ============================================================
# 6. 构建图
# ============================================================
def build_graph():
    tools = [search_user_facts]
    tool_node = ToolNode(tools)

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot_with_tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        source="chatbot",
        path=should_use_tool,
        path_map={"tools": "tools", "end": END},
    )
    graph_builder.add_edge("tools", "chatbot")

    return graph_builder.compile()


# ============================================================
# 7. 交互式聊天
# ============================================================
async def main():
    setup_zep()

    print("=" * 60)
    print("  LangGraph + Zep Cloud（手动集成版）")
    print("=" * 60)
    print(f"  用户: {USER_NAME} ({USER_ID})")
    print(f"  线程: {THREAD_ID}")
    print("=" * 60)
    print()
    print("提示: 先告诉 Agent 你的信息（如工作、偏好），然后问它是否记得。")
    print("输入 'quit' 或 'exit' 退出。")
    print()

    graph = build_graph()

    while True:
        user_input = input("🧑 You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        print("🤖 Assistant: ", end="", flush=True)
        try:
            inputs = {
                "messages": [HumanMessage(content=user_input)],
                "user_name": USER_NAME,
                "thread_id": THREAD_ID,
            }
            async for output in graph.astream(inputs):
                for key, value in output.items():
                    if key == "chatbot":
                        last_msg = value["messages"][-1]
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            print(last_msg.content)
        except Exception as e:
            print(f"\n[错误] {e}")

        print()


if __name__ == "__main__":
    asyncio.run(main())
