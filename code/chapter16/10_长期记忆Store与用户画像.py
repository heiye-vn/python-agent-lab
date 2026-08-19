import os
import sys
import uuid
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_store
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.memory import InMemoryStore
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
# 1. 定义热路径记忆工具 (In-the-loop Memory Tool)
# 原理：模型在对话中自主识别用户偏好/事实，通过工具调用将信息写入 Store
# =====================================================================
@tool
def save_user_memory(category: str, memory_text: str, config: RunnableConfig) -> str:
    """
    当用户在对话中透露出其姓名、个人画像、长期偏好（如饮食习惯、技术栈、城市、禁忌）等需要跨会话长期记住的信息时调用此工具。
    :param category: 记忆分类，例如 'profile' (基本信息)、'preferences' (偏好习惯)、'facts' (核心事实)
    :param memory_text: 记忆的具体内容，例如 '用户坚决不吃辣，极度偏好粤菜和清淡饮食' 或 '用户主语言是Python，常驻深圳'
    """
    # 从 config 中获取当前用户的唯一标识
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    store = get_store()

    if not store:
        return "错误：未配置全局 Store 存储器。"

    # 命名空间 Namespace 设计：("users", {user_id}, {category})
    namespace = ("users", str(user_id), category)
    memory_key = str(uuid.uuid4())[:8]

    # 将记忆持久化到 Store
    store.put(
        namespace,
        memory_key,
        {
            "content": memory_text,
            "category": category,
        },
    )

    print(
        f"\n💾 [Store 长期记忆写入] 用户: {user_id} | 分类: {category} | 键: {memory_key}"
    )
    print(f"   记忆内容: {memory_text}\n")
    return f"已成功将用户信息保存至长期记忆库 (Key: {memory_key})"


# 工具列表绑定
tools = [save_user_memory]
model_with_tools = llm.bind_tools(tools)


# =====================================================================
# 2. 定义图状态 State 与节点
# =====================================================================
class MemoryAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def recall_and_agent_node(state: MemoryAgentState, config: RunnableConfig):
    """
    Agent 核心节点：
    1. 动态从 Store 中搜索当前 user_id 的所有历史记忆画像；
    2. 将记忆作为 System Prompt 注入对话上下文；
    3. 调用绑定了记忆工具的大模型生成回复。
    """
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    store = get_store()

    # 从 Store 中检索该用户的所有长期记忆 (namespace 前缀检索)
    user_memories = []
    if store:
        # 搜索 ("users", user_id) 命名空间下的所有记录
        memories = store.search(("users", str(user_id)))
        for item in memories:
            val = item.value
            cat = item.namespace[-1]  # 获取最后一级分类名
            user_memories.append(f"- [{cat}] {val.get('content')}")

    # 组装 System Prompt
    if user_memories:
        memory_block = "\n".join(user_memories)
        system_content = (
            f"你是一个拥有长期记忆的智能助手。\n"
            f"【关于当前用户（ID: {user_id}）的长期画像与已知偏好】：\n"
            f"{memory_block}\n\n"
            f"请务必在回答中尊重并应用用户的已知偏好。如果用户在对话中提及了新的关键个人信息，请主动调用 save_user_memory 工具将其记录下来。"
        )
    else:
        system_content = (
            "你是一个拥有长期记忆的智能助手。\n"
            "如果用户在对话中提及了姓名、偏好、城市、技术栈等关键个人信息，请主动调用 save_user_memory 工具将其记录下来。"
        )

    prompt = [SystemMessage(content=system_content)] + state["messages"]
    response = model_with_tools.invoke(prompt, config=config)
    return {"messages": [response]}


# =====================================================================
# 3. 构建与编译图 (绑定 Checkpointer + Store)
# =====================================================================
builder = StateGraph(MemoryAgentState)

builder.add_node("agent", recall_and_agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
# 如果模型调用了工具（如 save_user_memory），跳转到 tools 节点，执行后再回 loop 到 agent
builder.add_conditional_edges("agent", tools_condition, ["tools", END])
builder.add_edge("tools", "agent")

# 初始化短期记忆 (Checkpointer) 与跨 Thread 长期记忆 (Store)
checkpointer = InMemorySaver()
store = InMemoryStore()

# 编译图：同时注入 checkpointer 与 store
agent_graph = builder.compile(checkpointer=checkpointer, store=store)


# =====================================================================
# 4. 自动化多会话演示（跨 Thread 记忆贯通测试）
# =====================================================================
def demo_cross_thread_memory():
    print("=" * 65)
    print("【实战：跨 Thread 长期记忆 Store 与用户画像系统】")
    print("=" * 65)

    user_id = "user_zhang_1001"

    # -------------------------------------------------------------
    # 场景一：周一会话 (Thread 1) —— 用户告诉 Agent 个人喜好与背景
    # -------------------------------------------------------------
    print("\n" + "-" * 55)
    print("【第一天：会话 ID = 'thread_monday_001'】")
    print("-" * 55)

    config_thread_1 = {
        "configurable": {
            "thread_id": "thread_monday_001",
            "user_id": user_id,
        }
    }

    user_input_1 = (
        "你好助手！我是小张，我常驻深圳，我平时极度偏好粤菜并且坚决不吃辣，"
        "我最常用的编程语言是 Python 和 Rust。"
    )
    print(f"[User (Thread 1)]: {user_input_1}")

    result_1 = agent_graph.invoke(
        {"messages": [HumanMessage(content=user_input_1)]},
        config=config_thread_1,
    )
    print(f"[AI (Thread 1)]: {result_1['messages'][-1].content}")

    # -------------------------------------------------------------
    # 场景二：周五会话 (Thread 2) —— 全新 Thread（短期 Checkpoint 已失效）
    # -------------------------------------------------------------
    print("\n" + "-" * 55)
    print("【第五天：开启全新会话 ID = 'thread_friday_002'】")
    print("（验证点：短期记忆已被隔离，Agent 是否能通过 Store 自动唤醒用户偏好？）")
    print("-" * 55)

    config_thread_2 = {
        "configurable": {
            "thread_id": "thread_friday_002",  # 全新的 thread_id！
            "user_id": user_id,
        }
    }

    user_input_2 = (
        "今晚下班后想在公司附近和朋友吃顿好的，请为我推荐 2 道特色美食并说明推荐原因。"
    )
    print(f"[User (Thread 2)]: {user_input_2}")

    result_2 = agent_graph.invoke(
        {"messages": [HumanMessage(content=user_input_2)]},
        config=config_thread_2,
    )
    print(f"[AI (Thread 2)]: {result_2['messages'][-1].content}")

    # -------------------------------------------------------------
    # 场景三：查看全局 Store 中沉淀的所有记忆条目
    # -------------------------------------------------------------
    print("\n" + "-" * 55)
    print(f"【长期记忆库审查：用户 {user_id} 的全部画像记录】")
    print("-" * 55)

    all_user_memories = store.search(("users", str(user_id)))
    print(f"共检索到 {len(all_user_memories)} 条持久化记忆：")
    for item in all_user_memories:
        print(f"  • Namespace: {item.namespace} | Key: {item.key}")
        print(f"    Value: {item.value}")
        print(f"    UpdatedAt: {item.updated_at}\n")


if __name__ == "__main__":
    demo_cross_thread_memory()
