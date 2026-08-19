import os
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv

# 避免 Windows 终端中文编码异常
sys.stdout.reconfigure(encoding="utf-8")

# 加载 .env 环境变量
load_dotenv(Path(__file__).parent / ".env")

# 规避系统代理阻断
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from mem0 import MemoryClient

# 校验 API Key
mem0_api_key = os.getenv("MEM0_API_KEY")
if not mem0_api_key:
    raise ValueError("未检测到 MEM0_API_KEY，请检查 code/chapter16/.env 文件配置！")

bailian_api_key = os.getenv("ALI_BAILIAN_API_KEY")
if not bailian_api_key:
    raise ValueError("未检测到 ALI_BAILIAN_API_KEY，请检查 code/chapter16/.env 文件配置！")

# 初始化 Mem0 客户端与 LLM
mem0_client = MemoryClient(api_key=mem0_api_key)

llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=bailian_api_key,
)

GLOBAL_APP_ID = "app_interactive_chat_kb"


# =====================================================================
# 1. 记忆管理工具 (Agent 在对话中按需自主调用)
# =====================================================================
@tool
def save_memory(
    fact_description: str,
    target_scope: str,
    config: RunnableConfig,
) -> str:
    """
    当用户在自然对话中透露出其个人偏好、背景、忌口、习惯，或者提出希望长期记住的规则时调用此工具。
    :param fact_description: 需要沉淀的明确事实或偏好语句，例如 '用户喜欢吃清淡粤菜，坚决不吃香菜' 或 '用户职业是自动驾驶工程师'
    :param target_scope: 存储范围，可选:
        - 'user': 用户画像（饮食偏好、常驻城市、技术栈、姓名等）
        - 'agent': Agent 自身的工作要求或 SOP
        - 'global': 全局公共业务知识
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", "default_user")
    agent_id = configurable.get("agent_id", "assistant")

    kwargs: Dict[str, Any] = {}
    if target_scope == "user":
        kwargs["user_id"] = user_id
    elif target_scope == "agent":
        kwargs["agent_id"] = agent_id
    else:
        kwargs["app_id"] = GLOBAL_APP_ID

    mem0_client.add([{"role": "user", "content": fact_description}], **kwargs)
    print(f"\n   💾 [Mem0 记忆已写入] 维度: {target_scope} | 内容: {fact_description}")
    return f"已成功将信息记入长期记忆库 ({target_scope})"


tools = [save_memory]
model_with_tools = llm.bind_tools(tools)


# =====================================================================
# 2. 定义状态机与自动召回节点
# =====================================================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def recall_and_chat_node(state: ChatState, config: RunnableConfig):
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", "default_user")
    agent_id = configurable.get("agent_id", "assistant")

    latest_msg = state["messages"][-1].content if state["messages"] else ""

    recalled_info: List[str] = []

    # 1. 自动检索用户专属画像
    try:
        u_hits = mem0_client.search(
            latest_msg, filters={"user_id": user_id}, limit=4
        )
        items = (
            u_hits.get("results", u_hits)
            if isinstance(u_hits, dict)
            else u_hits
        )
        memories = [it.get("memory") for it in items if it.get("memory")]
        if memories:
            recalled_info.append(
                f"【关于用户 (ID: {user_id}) 的已知长期画像】:\n"
                + "\n".join(f"  • {m}" for m in memories)
            )
    except Exception:
        pass

    # 2. 自动检索全局公共规则
    try:
        g_hits = mem0_client.search(
            latest_msg, filters={"app_id": GLOBAL_APP_ID}, limit=2
        )
        items = (
            g_hits.get("results", g_hits)
            if isinstance(g_hits, dict)
            else g_hits
        )
        memories = [it.get("memory") for it in items if it.get("memory")]
        if memories:
            recalled_info.append(
                "【全局公共业务规则】:\n" + "\n".join(f"  • {m}" for m in memories)
            )
    except Exception:
        pass

    system_prompt = (
        "你是一个拥有自主长期记忆能力的智能 AI 助手。\n"
        "你可以通过自然对话与用户交流。若用户在对话中透露了姓名、职业、喜好、禁忌或提出记忆需求，"
        "请主动调用 save_memory 工具将其记录下来。\n\n"
    )

    if recalled_info:
        system_prompt += (
            "【系统当前自动检索到的相关长期记忆】：\n"
            + "\n\n".join(recalled_info)
            + "\n\n请在回答中充分利用并尊重这些已知事实。\n"
        )

    prompt = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model_with_tools.invoke(prompt, config=config)
    return {"messages": [response]}


# 3. 构造并编译图
builder = StateGraph(ChatState)
builder.add_node("agent", recall_and_chat_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, ["tools", END])
builder.add_edge("tools", "agent")

checkpointer = InMemorySaver()
chatbot_graph = builder.compile(checkpointer=checkpointer)


# =====================================================================
# 4. 终端交互式对话循环
# =====================================================================
def run_interactive_memory_bot():
    current_user = "alice"
    current_agent = "assistant"
    current_thread = f"session_{uuid.uuid4().hex[:6]}"

    print("=" * 68)
    print("🤖 欢迎进入【LangGraph + Mem0 智能记忆对话系统】")
    print("=" * 68)
    print("💡 系统特性：")
    print("  1. 【自然添加】：直接在聊天中说出你的偏好或事实，Agent 会自动识别并保存到 Mem0；")
    print("  2. 【自然检索】：直接提问，Agent 会根据当前用户身份自动检索并回答；")
    print("  3. 【跨会话穿透】：切换新会话（/new）后，短期记忆重置，但长期记忆依然有效！")
    print("\n📌 快捷指令：")
    print("  • /user <name>   : 切换当前对话用户（测试多用户画像隔离）")
    print("  • /memories      : 查看当前用户在 Mem0 中的全部长期记忆")
    print("  • /new           : 开启全新会话（生成新 thread_id，验证跨会话长期记忆）")
    print("  • /help          : 查看指令帮助")
    print("  • exit / quit    : 退出程序")
    print("=" * 68)
    print(f"👉 当前状态: 用户 = [{current_user}] | 会话 ID = [{current_thread}]\n")

    while True:
        try:
            user_input = input(f"[{current_user}] > ").strip()
            if not user_input:
                continue

            # 指令处理
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 感谢使用，已退出对话。")
                break

            if user_input.startswith("/user "):
                new_user = user_input.split(" ", 1)[1].strip()
                if new_user:
                    current_user = new_user
                    current_thread = f"session_{uuid.uuid4().hex[:6]}"
                    print(
                        f"\n👤 [系统] 已切换当前用户为: [{current_user}]，并开启新会话: [{current_thread}]\n"
                    )
                    continue

            if user_input == "/new":
                current_thread = f"session_{uuid.uuid4().hex[:6]}"
                print(
                    f"\n🔄 [系统] 已重置短期会话！全新 Thread ID: [{current_thread}]（长期记忆依然保留）\n"
                )
                continue

            if user_input in ["/memories", "/view"]:
                print(f"\n🔍 正在查询用户 [{current_user}] 的全部 Mem0 长期记忆...")
                try:
                    all_m = mem0_client.get_all(filters={"user_id": current_user})
                    results = all_m.get("results", all_m) if isinstance(all_m, dict) else all_m
                    if not results:
                        print("  （暂无任何长期记忆记录）\n")
                    else:
                        print(f"  累计共 {len(results)} 条记忆：")
                        for i, it in enumerate(results, start=1):
                            print(f"   {i}. {it.get('memory')} (分类: {it.get('categories')})")
                        print()
                except Exception as e:
                    print(f"  查询失败: {e}\n")
                continue

            if user_input == "/help":
                print("\n📌 快捷指令列表：")
                print("  • /user <name>   : 切换当前对话用户")
                print("  • /memories      : 查看当前用户在 Mem0 中的全部长期记忆")
                print("  • /new           : 开启全新会话（测试跨 Thread 长期记忆）")
                print("  • exit / quit    : 退出程序\n")
                continue

            # 执行对话
            config = {
                "configurable": {
                    "thread_id": current_thread,
                    "user_id": current_user,
                    "agent_id": current_agent,
                }
            }

            response = chatbot_graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )

            ai_msg = response["messages"][-1].content
            print(f"\n[AI] > {ai_msg}\n")

        except KeyboardInterrupt:
            print("\n已退出对话。")
            break
        except Exception as e:
            print(f"\n❌ [执行异常]: {e}\n")


if __name__ == "__main__":
    run_interactive_memory_bot()
