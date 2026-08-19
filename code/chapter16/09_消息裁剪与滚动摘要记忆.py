import os
import sys
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    trim_messages,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# 避免 Windows 终端中文编码异常
sys.stdout.reconfigure(encoding="utf-8")

# 加载 .env 环境变量
load_dotenv(Path(__file__).parent / ".env")

# 校验 API Key
api_key = os.getenv("ALI_BAILIAN_API_KEY")
if not api_key:
    raise ValueError(
        "未检测到 ALI_BAILIAN_API_KEY，请检查 code/chapter16/.env 文件配置！"
    )

# 初始化统一的大模型客户端
llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=api_key,
)


# =====================================================================
# 方案一：基础消息裁剪 (trim_messages)
# 原理：按 Token/数量窗口裁剪老消息，始终锁定 SystemMessage，确保上下文不超限
# =====================================================================
def demo_trim_messages():
    print("=" * 65)
    print("【方案一：基于 trim_messages 的固定窗口裁剪】")
    print("=" * 65)

    # 模拟一段长对话历史
    raw_messages = [
        SystemMessage(content="你是一个专业的 Python 技术专家。"),
        HumanMessage(content="你好，我叫李雷。"),
        AIMessage(content="你好李雷！很高兴认识你，有什么 Python 问题我可以协助你？"),
        HumanMessage(content="什么是 Python 中的 GIL？"),
        AIMessage(
            content="GIL 是全局解释器锁，用于确保同一时刻只有一个线程执行 Python 字节码。"
        ),
        HumanMessage(content="那在多核 CPU 下如何利用好多进程？"),
        AIMessage(
            content="可以使用 multiprocessing 模块或 ProcessPoolExecutor 来规避 GIL。"
        ),
        HumanMessage(content="请问我叫什么名字？"),
    ]

    # 定义裁剪规则：保留最后 3 条对话消息，同时保留开头的 SystemMessage
    # token_counter 可以传入自定义函数或模型自带的 token 计算器，这里用 len 简化计数
    trimmed = trim_messages(
        raw_messages,
        max_tokens=4,  # 最多保留4条消息
        token_counter=len,  # 按消息条数计数
        strategy="last",  # 保留最新的消息
        include_system=True,  # 强制保留第一条 SystemMessage
        start_on="human",  # 确保裁剪后的第一条非系统消息是 HumanMessage
        allow_partial=False,
    )

    print(f"原始消息总数: {len(raw_messages)} 条")
    print(f"裁剪后消息数: {len(trimmed)} 条\n")
    for i, msg in enumerate(trimmed, start=1):
        print(f"  {i}. [{msg.__class__.__name__}]: {msg.content}")

    print("\n👉 注意：由于只保留了最近消息，前面的'我叫李雷'已被安全裁剪丢弃。")


# =====================================================================
# 方案二：带滚动摘要（Summarization）与自动修剪的生产级记忆图
# 原理：当对话消息超过阈值时，触发专门的摘要节点，将早期历史压缩为一段 Summary，
#       并使用 RemoveMessage 清除旧消息，既不爆 Context 又不丢失历史关键事实。
# =====================================================================


# 1. 定义图状态 State
class SummaryState(TypedDict):
    summary: str  # 累积历史摘要
    messages: Annotated[list[BaseMessage], add_messages]  # 消息列表（带 reducer）


# 2. 对话节点：拼接当前摘要 + 消息历史并调用模型
def chatbot_node(state: SummaryState):
    summary = state.get("summary", "")
    messages = state["messages"]

    # 如果有累积摘要，作为系统提示注入上下文头部
    if summary:
        system_content = (
            f"你是一个智能助手。\n"
            f"【过去对话的核心摘要】：\n{summary}\n"
            f"请结合以上历史摘要与当下的用户消息进行友好准确的回答。"
        )
        prompt = [SystemMessage(content=system_content)] + messages
    else:
        prompt = [SystemMessage(content="你是一个智能助手。")] + messages

    response = llm.invoke(prompt)
    return {"messages": [response]}


# 3. 摘要生成与历史清理节点
def summarize_conversation_node(state: SummaryState):
    summary = state.get("summary", "")
    messages = state["messages"]

    # 提取需要被压缩的消息（保留最近 2 条消息不参与压缩，以维持当期对话连贯性）
    messages_to_summarize = messages[:-2]
    # 需要删除的消息列表（使用 RemoveMessage）
    messages_to_remove = [RemoveMessage(id=m.id) for m in messages_to_summarize if m.id]

    # 构建摘要提取 Prompt
    if summary:
        summary_prompt = (
            f"这是之前的对话摘要：\n{summary}\n\n"
            f"请将上述摘要与以下新增的对话内容合并，更新为一个精炼、完整的最新中文摘要，"
            f"保留所有关键事实（如用户姓名、偏好、讨论的核心结论等）：\n"
        )
    else:
        summary_prompt = (
            "请为以下对话内容生成一段精炼、关键事实完整的中文摘要，"
            "提炼出用户的姓名、职业、提问的核心问题及结论：\n"
        )

    for msg in messages_to_summarize:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        summary_prompt += f"{role}: {msg.content}\n"

    # 调用模型生成新摘要
    response = llm.invoke([HumanMessage(content=summary_prompt)])
    new_summary = response.content

    print(
        f"\n⚡ [触发自动摘要] 已将 {len(messages_to_summarize)} 条老消息压缩为最新摘要："
    )
    print(f"   摘要内容: {new_summary}")
    print(f"   已从 State 移除 {len(messages_to_remove)} 条旧消息，释放上下文空间。\n")

    # 返回新 summary，并从 State 中移除已被摘要的消息
    return {
        "summary": new_summary,
        "messages": messages_to_remove,
    }


# 4. 条件路由：判断是否需要触发摘要（当未摘要消息超过 6 条时触发）
def should_summarize(state: SummaryState):
    messages = state["messages"]
    # 当累积消息超过 6 条时，触发摘要压缩；否则直接结束
    if len(messages) >= 6:
        return "summarize_node"
    return END


# 5. 构建与编译图
builder = StateGraph(SummaryState)
builder.add_node("chatbot", chatbot_node)
builder.add_node("summarize_node", summarize_conversation_node)

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", should_summarize, ["summarize_node", END])
builder.add_edge("summarize_node", END)

# 绑定 InMemorySaver 持久化
checkpointer = InMemorySaver()
summary_graph = builder.compile(checkpointer=checkpointer)


# 自动化多轮演练
def demo_summary_graph():
    print("\n" + "=" * 65)
    print("【方案二：滚动摘要（Summarization）状态机全流程测试】")
    print("=" * 65)

    config = {"configurable": {"thread_id": "summary_test_thread"}}

    dialogues = [
        "你好，我叫王小明，我是一名在深圳工作的自动驾驶算法工程师。",
        "我平时主要使用 C++ 和 Python，最近正在研究基于 LangGraph 的 Agent 开发。",
        "你觉得学习 Agent 最核心的三个能力模块是什么？",
        "我打算下个月考一个大模型算法工程师的专业认证。",
        "请问我是谁？我的职业和常用编程语言是什么？",
    ]

    for i, user_msg in enumerate(dialogues, start=1):
        print(f"\n--- 第 {i} 轮输入: '{user_msg}' ---")
        output = summary_graph.invoke(
            {"messages": [HumanMessage(content=user_msg)]},
            config=config,
        )
        ai_reply = output["messages"][-1].content
        print(f"AI 回复: {ai_reply}")

        # 查看当前 State 中的实际保留消息数与当前摘要
        current_state = summary_graph.get_state(config)
        retained_messages_count = len(current_state.values.get("messages", []))
        current_summary = current_state.values.get("summary", "（暂无）")
        print(
            f"[State 监控] 剩余未压缩消息数: {retained_messages_count} | 累计摘要长度: {len(current_summary)} 字"
        )


if __name__ == "__main__":
    # 1. 运行 trim_messages 演示
    demo_trim_messages()

    # 2. 运行带滚动摘要压缩与 RemoveMessage 的生产状态图演示
    demo_summary_graph()
